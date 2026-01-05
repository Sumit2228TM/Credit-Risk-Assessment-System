# 🏦 Credit Risk Assessment System

A sophisticated machine learning system for predicting credit default risk using ensemble methods and advanced data balancing techniques.

## 📋 Overview

This project implements a comprehensive credit risk assessment model that predicts the likelihood of loan default based on socio-economic, financial, and credit history features. The system uses a stacked ensemble approach combining multiple state-of-the-art algorithms to achieve robust and stable predictions.

## ✨ Key Features

- **Advanced Ensemble Learning**: Stacked classifier combining XGBoost, LightGBM, and Random Forest
- **Class Imbalance Handling**: SMOTE-based oversampling with optimized parameters
- **Threshold Optimization**: Dynamic threshold tuning for balanced precision-recall trade-off
- **Hyperparameter Tuning**: Randomized search for optimal model configuration
- **Feature Importance Analysis**: Identification of primary credit risk drivers
- **Stability Analysis**: Cross-validation based model reliability assessment
- **Comprehensive Visualization**: ROC curves, correlation matrices, and feature importance plots

## 🎯 Model Performance

- **ROC-AUC Score**: Optimized for high discrimination capability
- **Threshold-Optimized Predictions**: Balanced performance on minority class
- **Cross-Validated Stability**: Enhanced prediction consistency across data splits

## 📊 Dataset Features

### Socio-Economic Features
- Age, Income, Employment Years
- Education Level, Marital Status
- Home Ownership, Number of Dependents
- Geographic Region

### Financial & Credit History
- Loan Amount, Credit Score
- Credit History Length
- Number of Existing Loans, Total Debt
- Account Balances (Checking & Savings)
- Previous Defaults, Credit Card Utilization
- Late Payments, Credit Inquiries
- Co-signer Status, Loan Purpose

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/credit-risk-assessment.git
cd credit-risk-assessment
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook
```bash
jupyter notebook "Credit Risk Assessment System New.ipynb"
```

## 📈 Usage

The notebook is organized into clear sections:

1. **Data Generation**: Synthetic dataset creation with realistic features
2. **EDA**: Exploratory data analysis with visualizations
3. **Preprocessing**: Feature encoding and train-test split
4. **Class Balancing**: SMOTE application for imbalanced data
5. **Model Training**: Stacked ensemble model fitting
6. **Threshold Optimization**: Finding optimal prediction threshold
7. **Evaluation**: Performance metrics and ROC curve
8. **Hyperparameter Tuning**: Model optimization
9. **Feature Analysis**: Identifying key risk drivers
10. **Stability Testing**: Cross-validation analysis

## 🔍 Model Architecture

```
Base Models:
├── XGBoost Classifier (scale_pos_weight optimized)
├── LightGBM Classifier (class_weight balanced)
└── Random Forest Classifier (class_weight balanced)
```

## 📊 Results

The system provides:
- **Classification Report**: Precision, recall, F1-score for both classes
- **ROC Curve**: Visual representation of model performance
- **Feature Importance**: Top 10 drivers of credit risk
- **Stability Metrics**: Model consistency across folds
- **Processed Dataset**: CSV export for further analysis

## 🛠️ Technologies Used

- **Python 3.12.7**
- **scikit-learn**: ML algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Gradient boosting framework
- **imbalanced-learn**: SMOTE implementation
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Visualization

## 📝 Output Files

- `Processed_Credit_Risk_Data.csv`: Processed dataset with all features and predictions

Author - Sumit Gatade
Linkedin - https://www.linkedin.com/in/sumit-gatade-b30142295/
