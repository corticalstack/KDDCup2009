# 🔍 KDD Cup 2009 Customer Relationship Management Prediction

A machine learning project for predicting customer churn, appetency, and upselling using the Orange dataset from the KDD Cup 2009 competition.

## 📋 Description

This repository contains a comprehensive machine learning solution for customer relationship management (CRM) prediction tasks. It focuses on three key prediction challenges from the KDD Cup 2009 competition:

- **Churn**: Predicting which customers are likely to leave
- **Appetency**: Predicting which customers are likely to buy a product
- **Upselling**: Predicting which customers are likely to buy more products

The project implements multiple machine learning models, preprocessing strategies, and evaluation techniques to achieve high-performance predictions on these tasks.

## ✨ Features

- **Multiple Classifier Support**: Implementation of 6 different classifiers:
  - Random Forest (RFC)
  - Decision Tree (DTC)
  - AdaBoost (ABC)
  - Gradient Boosting (GBC)
  - Bagging (BGC)
  - Voting Classifier (VTC)

- **Advanced Preprocessing**:
  - Three different preprocessing strategies (DS01, DS02, DS03)
  - Handling of missing values with various imputation techniques
  - Feature selection and transformation
  - Categorical feature encoding

- **Hyperparameter Tuning**:
  - Single parameter grid search
  - Multi-parameter grid search
  - Optimized configurations for each target variable

- **Comprehensive Evaluation**:
  - ROC AUC scoring
  - Confusion matrix analysis
  - Threshold adjustment for classification
  - Feature importance ranking

- **Visualization**:
  - Missing data visualization
  - Confusion matrix plots
  - Performance comparison charts
  - Decision tree visualization

## 🛠️ Setup Guide

### Prerequisites

- Python 3.x
- Required Python packages (install via pip):
  ```
  pandas
  numpy
  scikit-learn
  matplotlib
  seaborn
  pydotplus
  missingno
  ```
- Graphviz (for decision tree visualization)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/KDDCup2009.git
   cd KDDCup2009
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the Orange dataset from the KDD Cup 2009 competition and place it in the `data` directory.

## 🚀 Usage

The main script accepts various command-line arguments to control the execution flow:

```bash
python main.py PrepEnabled=1 ProcessDS01=1 ProcessDS02=1 ProcessDS03=1 PredictChurn=1 PredictAppetency=1 PredictUpselling=1 GridSearchSingleRFC=0 GridSearchSingleDTC=0 GridSearchSingleABC=0 GridSearchSingleGBC=0 GridSearchSingleBGC=0 GridSearchMultiRFC=0 GridSearchMultiDTC=0 GridSearchMultiABC=0 GridSearchMultiGBC=0 GridSearchMultiBGC=0 FinalRFC=1 FinalDTC=1 FinalABC=1 FinalGBC=1 FinalBGC=1 FinalVTC=1 BaselineRFC=1 BaselineDTC=1 BaselineABC=1 BaselineGBC=1 BaselineBGC=1 BaselineVTC=1 PlotGraphs=1
```

### Command-line Arguments

- **Data Preparation**:
  - `PrepEnabled=1`: Enable data preprocessing
  - `ProcessDS01=1`: Process dataset with strategy 1 (mean imputation)
  - `ProcessDS02=1`: Process dataset with strategy 2 (zero imputation)
  - `ProcessDS03=1`: Process dataset with strategy 3 (special value imputation + one-hot encoding)

- **Prediction Targets**:
  - `PredictChurn=1`: Enable churn prediction
  - `PredictAppetency=1`: Enable appetency prediction
  - `PredictUpselling=1`: Enable upselling prediction

- **Grid Search**:
  - `GridSearchSingleXXX=1`: Enable single parameter grid search for classifier XXX
  - `GridSearchMultiXXX=1`: Enable multi-parameter grid search for classifier XXX

- **Model Evaluation**:
  - `BaselineXXX=1`: Evaluate baseline model for classifier XXX
  - `FinalXXX=1`: Evaluate final (optimized) model for classifier XXX

- **Visualization**:
  - `PlotGraphs=1`: Generate visualization plots

## 📊 Project Structure

```
KDDCup2009/
├── main.py                 # Main entry point
├── preprocessor.py         # Data preprocessing functionality
├── filehandler.py          # File I/O operations
├── modeller.py             # Machine learning models implementation
├── visualizer.py           # Visualization functions
├── logging.conf            # Logging configuration
├── data/                   # Data directory
│   ├── orange_small_train.csv                    # Training data
│   ├── orange_small_test.csv                     # Test data
│   ├── orange_small_train_churn.labels.csv       # Churn labels
│   ├── orange_small_train_appetency.labels.csv   # Appetency labels
│   ├── orange_small_train_upselling.labels.csv   # Upselling labels
│   └── ...                                       # Other data files
└── graphs/                 # Generated visualizations
    ├── Data Completion Categorical Features.png
    ├── Graph - Baseline vs Final - Churn Scores - DS02.png
    └── ...                 # Other visualization files
```

## 📈 Results

The project evaluates multiple classifiers on three prediction tasks (churn, appetency, upselling) using ROC AUC as the primary evaluation metric. Results are saved in CSV files in the `data` directory and visualized in the `graphs` directory.

Key findings:
- Gradient Boosting and Random Forest classifiers generally perform best
- Feature importance varies significantly between prediction tasks
- Ensemble methods (Voting Classifier) can improve performance by combining multiple models

## 🔧 Advanced Usage

### Custom Preprocessing

You can modify the preprocessing strategies in `preprocessor.py` to experiment with different approaches:

- Change imputation strategies for missing values
- Adjust feature selection criteria
- Implement additional feature engineering techniques

### Model Tuning

To tune a specific model for a particular prediction task:

1. Enable the appropriate grid search parameters
2. Run the main script
3. Analyze the results in the output CSV files
4. Update the final model parameters in `modeller.py`

## 📚 Resources

- [KDD Cup 2009 Competition](https://www.kdd.org/kdd-cup/view/kdd-cup-2009)
- [Orange Dataset Description](https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
