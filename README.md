# loan-risk-and-amount-prediction-ml
Machine learning project that builds and evaluates classification models to predict credit risk. The project compares Decision Tree and Random Forest models, emphasizes recall and AUC for risk assessment, and demonstrates how predictive analytics can support real-world credit scoring decisions.


# Credit Score Classification Using Machine Learning

## Project Overview
This project focuses on building a machine learning classification model to assess the credit risk of loan applicants. The objective is to predict whether an applicant is likely to default using financial and demographic data, supporting data-driven decision-making in credit evaluation.

Credit scoring is a critical task in the financial sector, where incorrect classifications can lead to financial losses or missed business opportunities. For this reason, this project emphasizes evaluation metrics that are especially relevant for risk assessment.

---

## Dataset
The dataset contains historical information about loan applicants, including:

- Age  
- Annual income  
- Employment length  
- Loan interest rate  
- Loan-to-income ratio  

These variables reflect common indicators used in real-world credit risk models to evaluate financial stability and repayment capacity.

---

## Methodology
The project follows a structured machine learning workflow:

1. Data exploration and cleaning  
2. Feature selection and preprocessing  
3. Model training using:
   - Decision Tree Classifier  
   - Random Forest Classifier  
4. Model evaluation using:
   - Accuracy  
   - Recall  
   - Area Under the ROC Curve (AUC)  
5. Prediction on a new client profile to simulate a real business scenario  

Special attention is given to **recall**, as reducing false negatives is particularly important in credit risk classification.

---

## Models and Evaluation
Two supervised learning models were implemented and compared:

- **Decision Tree**: provides interpretability and transparency, which is valuable in regulated financial environments.
- **Random Forest**: improves predictive performance by reducing variance through ensemble learning.

Model performance was evaluated using multiple metrics to balance interpretability and predictive accuracy.

---

## Results
The Random Forest model achieved stronger overall performance compared to the baseline Decision Tree, particularly in terms of recall and AUC. This indicates a better ability to identify high-risk applicants while maintaining acceptable generalization.

The results highlight the trade-off between model interpretability and predictive power when selecting models for credit risk applications.

---

## Example Prediction
The project includes a prediction for a new loan applicant using the trained model. This demonstrates how the model can be applied to unseen data and used as a decision-support tool in real-world credit evaluation processes.

---

## Key Learnings
- Credit risk problems require careful selection of evaluation metrics beyond accuracy.
- Recall is a critical metric when the cost of false negatives is high.
- Ensemble methods such as Random Forest can significantly improve model stability.
- Model interpretability remains an important consideration in financial applications.

---

## Tools and Technologies
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Jupyter Notebook  

---

## Project Structure
credit-score-ml/
│
├── data/
│ └── credit_data.csv
│
├── notebooks/
│ └── credit_score_model.ipynb
│
├── README.md
│
└── requirements.txt



---

## Future Improvements
- Train the model on a larger and more diverse dataset
- Explore additional models such as Logistic Regression or Gradient Boosting
- Apply cross-validation for more robust performance estimation
- Incorporate feature importance analysis for improved interpretability
