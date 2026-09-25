# 🏥 AI Healthcare Insurance Cost Prediction

An end-to-end **Machine Learning regression project** that predicts individual healthcare insurance charges based on demographic, lifestyle, and regional characteristics.

The project covers the complete machine learning workflow — from **data exploration and preprocessing to model evaluation, model serialization, and deployment using Streamlit**.

---

## 📌 Project Overview

Healthcare insurance costs can vary significantly depending on factors such as age, BMI, smoking status, number of dependents, and geographical region.

This project uses historical insurance data to build a machine learning model capable of estimating an individual's expected insurance charges.

### 🎯 Objective

Build a reliable regression model that can:

* Analyze factors influencing healthcare insurance costs
* Predict insurance charges for new customers
* Compare multiple regression algorithms
* Evaluate model performance using multiple metrics
* Provide an interactive interface for predictions

---

## 🚀 Features

* 📊 Exploratory Data Analysis (EDA)
* 🧹 Data preprocessing
* 🔢 Numerical feature scaling
* 🔤 Categorical feature encoding
* 🤖 Multiple machine learning models
* 📈 Model performance comparison
* 🔄 5-Fold Cross-Validation
* 💾 Model serialization using Joblib
* 🌐 Interactive Streamlit application
* 🔮 Real-time insurance cost prediction

---

## 🗂️ Dataset

The dataset contains information about individuals and their healthcare insurance charges.

### Features

| Feature    | Description                                 |
| ---------- | ------------------------------------------- |
| `age`      | Age of the individual                       |
| `sex`      | Gender                                      |
| `bmi`      | Body Mass Index                             |
| `children` | Number of dependents/children               |
| `smoker`   | Smoking status                              |
| `region`   | Residential region                          |
| `charges`  | Medical insurance charges (target variable) |

### Target Variable

```text
charges
```

The model learns the relationship between the input features and the individual's insurance charges.

---

## 🔬 Machine Learning Workflow

```text
                 Dataset
                    │
                    ▼
          Data Exploration & EDA
                    │
                    ▼
             Data Cleaning
                    │
                    ▼
          Feature / Target Split
                    │
                    ▼
             Train-Test Split
                    │
                    ▼
        ┌───────────────────────┐
        │   Preprocessing       │
        │                       │
        │ Numerical → Scaling   │
        │ Categorical → Encoding│
        └───────────────────────┘
                    │
                    ▼
          Model Training
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Linear     Ridge     Lasso
     Regression Regression Regression
          │         │         │
          └─────────┼─────────┘
                    ▼
           Model Evaluation
                    │
                    ▼
          5-Fold Cross Validation
                    │
                    ▼
          Final Model Selection
                    │
                    ▼
            Model Serialization
                    │
                    ▼
             Streamlit App
                    │
                    ▼
          Insurance Cost Prediction
```

---

## 🧹 Data Preprocessing

The project uses a Scikit-learn preprocessing pipeline to handle both numerical and categorical features.

### Numerical Features

Numerical variables are processed using:

* Missing-value imputation
* Standardization using `StandardScaler`

### Categorical Features

Categorical variables are processed using:

* Missing-value imputation
* One-hot encoding using `OneHotEncoder`

The preprocessing steps are integrated into a **Scikit-learn Pipeline / ColumnTransformer**, helping ensure that the same transformations are applied during both training and prediction.

---

## 🤖 Models Used

Three regression algorithms were evaluated:

### 1. Linear Regression

A baseline regression model used to establish the initial relationship between the input features and insurance charges.

### 2. Ridge Regression

Ridge Regression adds L2 regularization to Linear Regression, helping control model complexity and reduce the impact of multicollinearity.

### 3. Lasso Regression

Lasso Regression uses L1 regularization and can reduce the influence of less important features.

---

## 📊 Model Evaluation

The models were evaluated using:

* **R² Score**
* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**
* **5-Fold Cross-Validation**

### Test Set Performance

The Linear Regression model achieved an R² score of approximately:

```text
R² = 0.784
```

The models were also evaluated using 5-fold cross-validation. Ridge Regression achieved a cross-validation R² of approximately:

```text
CV R² ≈ 0.747
```

The final serialized model in the project is the **Ridge Regression pipeline**.

> Note: Test-set performance and cross-validation performance measure different things, so they should not be interpreted as directly interchangeable.

---

## 📈 Exploratory Data Analysis

EDA was performed using **Pandas, Matplotlib, and Seaborn**.

The analysis explored relationships between insurance charges and:

* Age
* BMI
* Smoking status
* Number of children
* Gender
* Region

Visualizations were used to understand patterns, distributions, correlations, and potential relationships between features and insurance costs.

---

## 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Supervised Learning
* Regression
* Train/Test Split
* Exploratory Data Analysis
* Feature Engineering
* Numerical Feature Scaling
* Categorical Feature Encoding
* Missing Value Handling
* Scikit-learn Pipelines
* `ColumnTransformer`
* Regularization
* Cross-Validation
* Model Evaluation
* Model Serialization
* ML Application Deployment

---

## 🌐 Streamlit Application

The trained model is integrated into a Streamlit application.

Users can enter information such as:

* Age
* Gender
* BMI
* Number of children
* Smoking status
* Region

The application then processes the inputs using the saved preprocessing pipeline and generates an estimated insurance charge.

### Application Flow

```text
User Input
    ↓
Streamlit Interface
    ↓
Saved Preprocessing Pipeline
    ↓
Ridge Regression Model
    ↓
Predicted Insurance Charges
```

---

## 📁 Project Structure

```text
AI-Healthcare-Insurance-Prediction/
│
├── app.py
│
├── data/
│   └── insurance.csv
│
├── models/
│   ├── preprocessor.pkl
│   └── ridge_model.pkl
│
├── notebook/
│   └── Insurance_Cost_Prediction.ipynb
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Healthcare-Insurance-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd AI-Healthcare-Insurance-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

Main libraries used in this project:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

---

## 💾 Model Serialization

The trained preprocessing pipeline and model are saved using **Joblib**.

Example:

```python
import joblib

joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(model, "models/ridge_model.pkl")
```

This allows the trained model to be loaded later without retraining it.

---

## 🔮 Future Improvements

Potential improvements for future versions include:

* Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
* Experimenting with tree-based models such as Random Forest and Gradient Boosting
* Adding XGBoost/LightGBM models
* Improving model interpretability with SHAP
* Adding prediction confidence/uncertainty estimates
* Deploying the Streamlit application to a cloud platform
* Adding automated model evaluation
* Implementing CI/CD for the application
* Adding unit tests for preprocessing and prediction

---

## 🛠️ Technologies Used

| Category          | Technologies                                  |
| ----------------- | --------------------------------------------- |
| Programming       | Python                                        |
| Data Processing   | Pandas, NumPy                                 |
| Visualization     | Matplotlib, Seaborn                           |
| Machine Learning  | Scikit-learn                                  |
| Models            | Linear, Ridge, Lasso Regression               |
| Model Persistence | Joblib                                        |
| Deployment/UI     | Streamlit                                     |
| Development       | Jupyter Notebook, VS Code, AI-assisted coding |

---

## 👨‍💻 Author

**Sahib Singh**

B.Tech Computer Science & Engineering — Cybersecurity
Post Graduate Diploma — Artificial Intelligence Development & Applications

Interested in **Artificial Intelligence, Machine Learning, Cybersecurity, and AI-powered applications**.

---

## ⭐ Acknowledgements

This project was developed as part of my practical learning journey in **Machine Learning and Artificial Intelligence**, with a focus on building an end-to-end ML application rather than only training a model.
