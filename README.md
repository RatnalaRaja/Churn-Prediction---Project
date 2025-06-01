
#  Customer Churn Prediction Dashboard

An interactive web application that predicts the likelihood of a  customer churning using a machine learning model. Built with **Streamlit**, **Plotly**, and **scikit-learn**, this dashboard helps customer service teams proactively identify at-risk customers and take informed retention actions.


##  Project Summary

Customer churn is a critical metric for  companies. This project combines machine learning with a clean user interface to deliver:

- **Real-time churn predictions**
- **Visual feedback using a gauge chart**
- **Simple inputs to simulate customer profiles**
- A fully **deployable and customizable web dashboard**

---

##  Files Included

| File Name               | Description                                              |
|------------------------ |----------------------------------------------------------|
| `app.py`                | Streamlit app script that powers the dashboard           |
| `model.pkl`             | Trained machine learning model (binary classifier)       |
| `scaler.pkl`            | Fitted scaler for numerical features                     |
| `model_trained.pkl`     | Column order used during training (for input alignment)  |
| `Churn_prediction.ipynb`| Jupyter notebook containing the full training pipeline  |

---

##  Model & Features

###  Features Used

- **Numerical**: `tenure`, `MonthlyCharges`, `TotalCharges`
- **Categorical (one-hot encoded)**:
  - `gender`
  - `SeniorCitizen`
  - `Partner`
  - `Dependents`
  - `Contract`
  - `PaperlessBilling`
  - `PaymentMethod`

###  Model Pipeline

1. **Data Preprocessing**:
   - Categorical encoding (One-Hot)
   - Missing value handling
   - Feature scaling (`StandardScaler`)
2. **Modeling**:
   - Supervised classification model (likely Logistic Regression, Random Forest, etc.)
3. **Serialization**:
   - Saved using `joblib` for use in the app.

---

##  Getting Started

###  Requirements

- Python 3.7+
- Dependencies:

```bash
pip install streamlit pandas numpy scikit-learn joblib plotly
```

###  Run Locally

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

##  App Walkthrough
### 👤 Customer Profile Section

Users provide inputs for the following fields:
- Gender, Senior Citizen, Partner, Dependents
- Contract type, Billing method, Payment method
- Tenure, Monthly Charges, Total Charges

###  Prediction Output

Upon clicking **"Predict Now"**:
- The model processes the input and outputs:
  - **Prediction label** (High/Low Churn Risk)
  - **Churn Probability (%)**
  - A **Plotly gauge chart** indicating the confidence level.

###  Visualization

- Green (0–50%) → Low Risk  
- Yellow (50–75%) → Medium Risk  
- Red (75–100%) → High Risk

---

##  Example Use Case

-  A customer calls support.
-  Agent enters customer details in the dashboard.
-  Model predicts 82.5% churn probability.
-  The dashboard highlights **❌ High Risk of Churn**.
-  The team can now prioritize retention efforts.



## 📝 License

This project is free to use for **educational** and **research** purposes. 

