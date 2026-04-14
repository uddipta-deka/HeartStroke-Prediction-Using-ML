# Comparative Analysis of Machine Learning Models for Heart Stroke Prediction

A machine learning research project evaluating multiple classification algorithms for heart stroke risk prediction using rigorous validation techniques, hyperparameter tuning, and comprehensive performance benchmarking.

---

##  Motivation

Heart stroke is a leading cause of mortality worldwide. Early and accurate risk prediction can significantly improve clinical decision-making and preventive care.

This project evaluates classical machine learning models under rigorous validation frameworks to identify reliable approaches for real-world deployment.

---

## Project Highlights

- Benchmarking of **5 classification algorithms** under identical conditions  
- **5-fold Stratified Cross-Validation** to handle class imbalance  
- **GridSearchCV hyperparameter tuning** for optimal performance  
- Evaluation using **F1-score and ROC-AUC principles**  
- Focus on minimizing **false negatives**, critical in medical diagnosis  
- **Streamlit app** for real-time stroke risk prediction   

---

##  Models Evaluated

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Naive Bayes  

---

## Baseline Performance (5-Fold Cross-Validation)

| Model | Accuracy (Mean ± Std) | F1 Score (Mean ± Std) |
|------|----------------------|----------------------|
| Logistic Regression | 0.8638 ± 0.0167 | 0.8777 ± 0.0155 |
| KNN | 0.8365 ± 0.0148 | 0.8564 ± 0.0131 |
| Naive Bayes | 0.8501 ± 0.0162 | 0.8636 ± 0.0140 |
| Decision Tree | 0.7602 ± 0.0394 | 0.7816 ± 0.0353 |
| SVM | 0.8501 ± 0.0224 | 0.8675 ± 0.0194 |


---

##  Final Test Performance

| Model | Accuracy | F1 Score |
|------|----------|---------|
| Logistic Regression | 0.8641 | 0.8804 |
| KNN | 0.8587 | 0.8762 |
| Naive Bayes | 0.8533 | 0.8683 |
| Decision Tree | 0.7989 | 0.8230 |
| SVM | 0.8587 | 0.8750 |


##  Key Findings

- **Logistic Regression** achieved the most stable and consistent performance across evaluation metrics.  
- **KNN and Naive Bayes** provided competitive results but were sensitive to dataset characteristics.  
- **Decision Tree** showed signs of overfitting due to dataset imbalance.  
- **SVM** performed well but required careful hyperparameter tuning.  
- Proper validation strategies significantly improve the reliability of classical ML models in healthcare applications.  

---

## Evaluation Strategy

Traditional accuracy is misleading for imbalanced medical datasets. This project uses:

- **Stratified K-Fold Cross-Validation** to maintain class distribution  
- **F1-Score** as the primary metric (balances precision and recall)  
- **ROC-AUC** to measure discriminative performance across thresholds  
- **Confusion matrix** for detailed error understanding 

---


## Project Pipeline

```
Raw Data
↓
Preprocessing (label encoding, median imputation, stratified split)
↓
Feature Scaling (StandardScaler)
↓
Baseline Benchmarking (Stratified K-Fold)
↓
Hyperparameter Tuning (GridSearchCV)
↓
Final Evaluation (Accuracy, F1, ROC-AUC, Confusion Matrix)
↓
Streamlit Deployment 
```


## Project Structure

```
├── HeartStroke_Prediction.ipynb # Main notebook
├── App/ # Streamlit web application
├── Raw Data/ # Dataset
├── plots/ # Evaluation plots
│ ├── roc_curves.png
│ ├── confusion_matrix.png
│ └── cv_vs_tuned.png
└── README.md
```

## Technologies Used

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Streamlit  
- Jupyter Notebook  

---


## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/uddipta-deka/Comparative-study-of-classification-algorithms-for-Heart-Stroke-prediction-

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook HeartStroke_Prediction.ipynb

# Launch Streamlit app
cd App
streamlit run app.py
```


## Key Design Decisions

-Stratified sampling used to handle imbalance

-F1-score optimized instead of accuracy

-Median imputation for missing values

-Scaling applied after split to avoid data leakage
