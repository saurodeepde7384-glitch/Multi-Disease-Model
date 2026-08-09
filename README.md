# Multi Disease Prediction System

A machine learning-based health risk prediction system that predicts the
likelihood of three diseases:

-   Heart Disease
-   Diabetes
-   Breast Cancer

The project includes exploratory data analysis, data preprocessing,
model training, model evaluation, model serialization, and a
Streamlit-based prediction interface.

> **Disclaimer:** This project is intended for educational and
> demonstration purposes only. It is not a medical diagnostic system and
> should not be used as a substitute for professional medical advice.

------------------------------------------------------------------------

## Project Overview

The Multi Disease Prediction System uses supervised machine learning
models to analyze patient-related clinical and lifestyle information and
generate a binary prediction along with a model probability.

The project follows an end-to-end machine learning workflow:

``` text
Raw Dataset
     ↓
Exploratory Data Analysis
     ↓
Data Cleaning
     ↓
Train/Test Split
     ↓
Preprocessing Pipeline
     ↓
Multiple Model Training
     ↓
Cross-Validation
     ↓
Model Selection
     ↓
Final Model Training
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
Streamlit Application
     ↓
Prediction + Probability
```

## Diseases and Models

  Disease         Final Model                      Test Accuracy
  --------------- ------------------------------ ---------------
  Heart Disease   Calibrated Decision Tree                97.73%
  Diabetes        Gradient Boosting Classifier            92.00%
  Breast Cancer   Calibrated Decision Tree                99.80%

The final models were selected after comparing multiple classification
algorithms using cross-validation and evaluation metrics such as
Accuracy, Precision, Recall, and F1 Score.

## Features

### Heart Disease

The Heart Disease model uses clinical measurements including:

-   Age
-   Biological Sex
-   Resting Heart Rate
-   Systolic Blood Pressure
-   Diastolic Blood Pressure
-   Blood Glucose
-   CK-MB
-   Troponin

### Diabetes

The Diabetes model uses:

-   Age
-   Biological Sex
-   BMI
-   Blood Pressure
-   Glucose
-   HbA1c
-   Insulin
-   Physical Activity
-   Family History
-   Smoking Status

### Breast Cancer

The Breast Cancer model uses:

-   Age
-   Gender
-   Family History
-   BMI
-   Smoking Status
-   Alcohol Consumption
-   Physical Activity
-   Breastfeeding History
-   Hormone Therapy
-   Menopausal Status
-   Tumor Size
-   Palpable Lump Presence

## Machine Learning Workflow

### Data Cleaning

The datasets are loaded using Pandas and basic cleaning operations are
performed.

The workflow includes:

-   Removing duplicate records
-   Cleaning column names
-   Separating features and target variables
-   Identifying numerical and categorical features
-   Handling missing values

### Train/Test Split

The datasets are divided into training and testing sets using an 80/20
split with `random_state=42`.

### Data Preprocessing

Separate preprocessing pipelines are created for numerical and
categorical features.

**Numerical features:**

``` text
Missing Value Imputation
        ↓
Mean Imputation
        ↓
StandardScaler
```

**Categorical features:**

``` text
Missing Value Imputation
        ↓
Most Frequent Value
        ↓
One-Hot Encoding
```

A `ColumnTransformer` combines both preprocessing pipelines.

This keeps preprocessing inside the machine learning pipeline so the
same transformations are applied during prediction.

## Models Evaluated

The following classification algorithms were evaluated:

-   Logistic Regression
-   K-Nearest Neighbors
-   Support Vector Machine
-   Decision Tree
-   Random Forest
-   Gradient Boosting

The models were evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1 Score

A 5-fold `StratifiedKFold` cross-validation strategy was used for model
comparison.

## Final Model Performance

### Heart Disease

  Metric         Score
  ----------- --------
  Accuracy      97.73%
  Precision     98.16%
  Recall        98.16%
  F1 Score      98.16%

### Diabetes

  Metric         Score
  ----------- --------
  Accuracy      92.00%
  Precision     91.30%
  Recall        91.30%
  F1 Score      91.30%

### Breast Cancer

  Metric         Score
  ----------- --------
  Accuracy      99.80%
  Precision     99.86%
  Recall        99.86%
  F1 Score      99.86%

These values represent the final held-out test-set results reported by
the training notebooks.

## Project Structure

``` text
MULTI DISEASE MODEL/
│
├── Data/
│   ├── Breast_Cancer_Data.xlsx
│   ├── Diabetes_Data.xlsx
│   └── Heart_Attack_Data.csv
│
├── EDA/
│   ├── breast_cancer_EDA.ipynb
│   ├── diabetes_EDA.ipynb
│   └── heart_EDA.ipynb
│
├── Models/
│   ├── model_breast_cancer.pkl
│   ├── model_diabetes.pkl
│   └── model_heart_disease.pkl
│
├── Training Notebooks/
│   ├── breast_cancer.ipynb
│   ├── diabetes_training.ipynb
│   └── heart_disease_training.ipynb
│
│
├── .gitignore
│
├── README.md
│
│
├── requirements.txt
│
└── app.py
```

## Directory Description

### `Data/`

Contains the datasets used for training the three machine learning
models.

### `EDA/`

Contains separate exploratory data analysis notebooks for each disease
dataset.

### `Training Notebooks/`

Contains the complete model development notebooks, including:

-   Dataset loading
-   Data cleaning
-   Feature/target separation
-   Train/test splitting
-   Preprocessing
-   Pipeline creation
-   Model comparison
-   Cross-validation
-   Final model training
-   Test-set evaluation
-   Model serialization

### `Models/`

Contains the trained machine learning pipelines saved using Joblib.

### `app.py`

The Streamlit application that provides the user interface for making
predictions.

## Streamlit Application

The application provides a user interface for selecting one of the three
prediction systems:

``` text
Multi Disease Prediction System

├── Heart Disease
├── Diabetes
└── Breast Cancer
```

Each disease provides:

-   Patient input form
-   Prediction result
-   Model probability
-   Risk category
-   Patient input summary

The application loads the serialized models using Joblib instead of
retraining them every time the application starts.

## Prediction Output

After submitting patient information, the application displays the
predicted result and model probability.

Example:

``` text
Heart Disease Detected
```

or:

``` text
No Heart Disease Detected
```

The application also displays a probability associated with the
prediction and categorizes it into a risk level:

``` text
0% - 33%     → Low Risk
34% - 66%    → Moderate Risk
67% - 100%   → High Risk
```

## Technologies Used

### Programming Language

-   Python

### Data Processing

-   Pandas
-   NumPy

### Machine Learning

-   Scikit-learn

### Model Persistence

-   Joblib

### Application

-   Streamlit

### Data Formats

-   CSV
-   Excel

## Installation

Clone the repository:

``` bash
git clone https://github.com/saurodeepde7384-glitch/Multi-Disease-Model.git
```

Navigate into the project:

``` bash
cd "MULTI DISEASE MODEL"
```

Create a virtual environment:

``` bash
python -m venv venv
```

Activate the virtual environment on Windows:

``` bash
venv\Scripts\activate
```

Install the required dependencies:

``` bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

``` bash
streamlit run app.py
```

The application will open in your browser.

## How It Works

``` text
User selects disease
        ↓
Corresponding model is loaded
        ↓
User enters patient information
        ↓
Input converted into DataFrame
        ↓
Preprocessing pipeline transforms the input
        ↓
Trained classifier makes prediction
        ↓
Prediction probability is generated
        ↓
Risk category is calculated
        ↓
Result displayed in Streamlit
```

## Model Persistence

The trained models are saved using Joblib:

``` python
joblib.dump(model, "Models/model_heart_disease.pkl")
```

``` python
joblib.dump(model, "Models/model_diabetes.pkl")
```

``` python
joblib.dump(model, "Models/model_breast_cancer.pkl")
```

This allows the application to load already-trained models instead of
retraining them every time.

## Key Machine Learning Concepts Demonstrated

-   Supervised Learning
-   Binary Classification
-   Exploratory Data Analysis
-   Data Cleaning
-   Missing Value Imputation
-   Feature Scaling
-   Categorical Feature Encoding
-   Pipeline
-   ColumnTransformer
-   Train/Test Split
-   Stratified K-Fold Cross-Validation
-   Model Comparison
-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   Probability Prediction
-   Model Calibration
-   Model Serialization
-   Streamlit Application Development

## Limitations

-   Predictions depend on the datasets used for training.
-   High test accuracy does not guarantee real-world clinical
    performance.
-   The datasets may not represent the full diversity of real patient
    populations.
-   Model probabilities should not be interpreted as medical certainty.
-   The system has not been clinically validated.
-   Predictions should not be used to make medical decisions.

## Future Improvements

Possible improvements include:

-   Hyperparameter tuning
-   ROC-AUC evaluation
-   Confusion matrices
-   Precision-Recall curves
-   Feature importance analysis
-   Probability calibration analysis
-   Better class-imbalance handling
-   Larger and more clinically representative datasets
-   External validation datasets
-   Model explainability using SHAP
-   Improved input validation
-   Production deployment
-   Automated model retraining pipeline

## Disclaimer

This project is developed for educational and machine learning
demonstration purposes.

It is **not a medical diagnostic tool**.

The predictions generated by this application should not be considered
medical advice, diagnosis, or treatment recommendations. Always consult
a qualified healthcare professional for medical decisions.

## Author

**Saurodeep De**

BTech CSE (AI/ML)

Machine Learning \| Python \| Data Science \| AI
