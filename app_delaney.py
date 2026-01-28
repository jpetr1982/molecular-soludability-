import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="PharmaSol Predictor", layout="wide")

# --- Custom CSS (Blue/White) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; background-color: #007cc3; color: white; }
    .reportview-container .main .footer { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- Loading Assets (Cached για ταχύτητα) ---
@st.cache_resource
def load_assets():
    rf_model = joblib.load("solubility_model_rf.pkl")
    nn_model = tf.keras.models.load_model("solubility_model.keras")
    scaler = joblib.load("scaler.pkl")
    return rf_model, nn_model, scaler

try:
    rf_model, nn_model, scaler = load_assets()
except Exception as e:
    st.error(f"Σφάλμα κατά τη φόρτωση των μοντέλων: {e}")

# --- Helper Functions ---
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        num_atoms = mol.GetNumAtoms()
        aromatic_atoms = [mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in range(num_atoms)]
        desc_AromaticProportion = sum(aromatic_atoms) / num_atoms
        
        features = np.array([[desc_MolLogP, desc_MolWt, desc_NumRotatableBonds, desc_AromaticProportion]])
        return features, mol
    return None, None

# --- Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Pfizer_%282021%29.svg/1200px-Pfizer_%282021%29.svg.png", width=150)
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Επιλέξτε Σελίδα:", ["Predictor", "Model Insights", "About the Project"])

# --- PAGE 1: PREDICTOR ---
if app_mode == "Predictor":
    st.title("🧪 Molecular Solubility Predictor")
    st.write("Εργαλείο πρόβλεψης διαλυτότητας (LogS) για την επιτάχυνση του Drug Discovery.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Section")
        smiles_input = st.text_input("Εισάγετε το SMILES του μορίου:", "C1=CC=C(C=C1)O") # Default: Phenol
        model_choice = st.radio("Επιλογή Μοντέλου:", ("Random Forest (R²: 0.890)", "Neural Network (R²: 0.886)"))
        
        predict_btn = st.button("Calculate Solubility")

    with col2:
        st.subheader("Results")
        if predict_btn:
            features, mol = extract_features(smiles_input)
            
            if features is not None:
                # 1. Preprocessing
                features_scaled = scaler.transform(features)
                
                # 2. Prediction
                if "Random Forest" in model_choice:
                    prediction = rf_model.predict(features_scaled)[0]
                else:
                    prediction = nn_model.predict(features_scaled, verbose=0)[0][0]
                
                # 3. Display
                st.metric(label="Predicted LogS", value=f"{prediction:.3f}")
                
                # Interpretation
                if prediction > -2:
                    st.success("High Solubility: Likely good absorption.")
                elif prediction > -4:
                    st.warning("Moderate Solubility: Formulation optimization might be needed.")
                else:
                    st.error("Low Solubility: High risk of poor bioavailability.")
                
                # Μικρή απεικόνιση των χαρακτηριστικών
                st.write("**Physical Descriptors:**")
                df_feats = pd.DataFrame(features, columns=['LogP', 'Weight', 'RotBonds', 'AromaticProp'])
                st.table(df_feats)
            else:
                st.error("Invalid SMILES string. Please check the structure.")

# --- PAGE 2: MODEL INSIGHTS ---
elif app_mode == "Model Insights":
    st.title("📊 Model Analysis & Explainability")
    st.write("Γιατί το μοντέλο παίρνει αυτές τις αποφάσεις;")
    
    # Εδώ μπορείς να βάλεις ένα στατικό γράφημα που έσωσες από το notebook
    st.subheader("Feature Importance (Random Forest)")
    # Παράδειγμα στατικών τιμών που βρήκαμε στο notebook
    importance_data = pd.DataFrame({
        'Feature': ['MolLogP', 'MolWt', 'NumRotBonds', 'AromaticProp'],
        'Importance': [0.85, 0.08, 0.04, 0.03] # Αντικατάστησε με τα δικά σου νούμερα
    }).sort_values(by='Importance', ascending=True)
    
    fig, ax = plt.subplots()
    ax.barh(importance_data['Feature'], importance_data['Importance'], color='#007cc3')
    st.pyplot(fig)
    st.info("Το MolLogP (Hydrophobicity) είναι ο κυρίαρχος παράγοντας πρόβλεψης, επιβεβαιώνοντας τους κανόνες της Φαρμακοχημείας.")

# --- PAGE 3: ABOUT ---
elif app_mode == "About the Project":
    st.title("👨‍🔬 Project Background")
    st.markdown("""
    ### Ο Στόχος
    Αυτό το project αναπτύχθηκε για να δείξει πώς οι μέθοδοι της **Στατιστικής Φυσικής** και του **Machine Learning** μπορούν να επιταχύνουν τη διαδικασία επιλογής υποψήφιων φαρμάκων.
    
    ### Τεχνολογίες
    - **RDKit:** Χημική πληροφορική και εξαγωγή μοριακών περιγραφητών.
    - **Scikit-Learn:** Random Forest Regressor με Grid Search Optimization.
    - **TensorFlow/Keras:** Deep Neural Networks για μη-γραμμικές συσχετίσεις.
    - **Streamlit:** Deployment της εφαρμογής σε περιβάλλον cloud.
""")
    
  
st.sidebar.markdown("---")
st.sidebar.write("Developed by Petridis Dimitris (Physics & Data Science Graduate)")