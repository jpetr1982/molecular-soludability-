# 🧪 PharmaSol: Molecular Solubility Prediction using ML & Deep Learning

## 🎯 Project Overview
Αυτό το project αναπτύχθηκε με στόχο την πρόβλεψη της υδατοδιαλυτότητας (LogS) χημικών ενώσεων, έναν κρίσιμο παράγοντα στο Drug Discovery. Συνδυάζει τις αρχές της **Φυσικοχημείας** με προηγμένους αλγορίθμους Machine Learning.

## 🚀 Live Demo
https://soludability-dimipetr82.streamlit.app/

## 🛠 Tech Stack
* **Chemoinformatics:** RDKit (Molecular Descriptors extraction)
* **Machine Learning:** Scikit-Learn (Random Forest Regressor + GridSearchCV)
* **Deep Learning:** TensorFlow/Keras (Sequential Neural Networks)
* **Deployment:** Streamlit

## 🔬 Physics & Data Science Approach
Ως απόφοιτος **Φυσικής**, προσέγγισα το πρόβλημα εστιάζοντας στη σημασία των φυσικών ιδιοτήτων:
* **Feature Engineering:** Μετατροπή SMILES σε περιγραφητές όπως MolLogP (υδροφοβικότητα) και Μοριακό Βάρος.
* **Model Comparison:** Σύγκριση Random Forest (R²: 0.890) και Neural Networks (R²: 0.886).
* **Interpretability:** Χρήση Feature Importance για την επιβεβαίωση της φυσικής ορθότητας του μοντέλου.

## 📊 Key Results
* **Random Forest:** Επέδειξε την καλύτερη γενίκευση (R²: 0.890).
* **Interpretability:** Το MolLogP αναδείχθηκε ως ο σημαντικότερος προγνωστικός παράγοντας, συνάδοντας με τη θερμοδυναμική της διάλυσης.

## 💻 How to Run
1. Clone the repo: `git clone https://github.com/your-username/your-repo-name.git`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app_delaney.py`
