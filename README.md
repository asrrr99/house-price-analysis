# house-price-analysis
A comprehensive machine learning project designed to predict residential real estate prices using regression techniques. Includes data cleaning, exploratory data analysis (EDA), and model evaluation using Scikit-Learn
## Overview
This project focuses on predicting house prices using machine learning models. The dataset used is the King County House Sales dataset (kc_house_data.csv).
The workflow includes data preprocessing, visualization, model training, and performance evaluation.
Dataset
File: kc_house_data.csv
Contains features like:
sqft_living (living area)
bedrooms, bathrooms
floors, condition
price (target variable)
Steps Performed
1. Import Libraries
Used essential Python libraries:
pandas, numpy → data handling
matplotlib, seaborn → visualization
sklearn → machine learning
2. Data Loading
Python
df = pd.read_csv("kc_house_data.csv")
3. Data Exploration
Checked structure using:
Python
df.info()
Checked missing values:
Python
df.isnull().sum()
4. Data Cleaning
Filled missing values (note: your code has a mistake here — should be df.mean() not df.mean)
Dropped unnecessary columns:
Python
df.drop(['id', 'date'], axis=1, inplace=True)
5. Feature Scaling
Applied StandardScaler to normalize features:
Python
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('price', axis=1))
y = df['price']
6. Outlier Removal
Used IQR method to remove extreme values:
Python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
7. Data Visualization
Histogram of price distribution
Scatter plot (sqft_living vs price)
Boxplot for detecting outliers
8. Train-Test Split
Python
train_test_split(X, y, test_size=0.2)
9. Models Used
Linear Regression
Decision Tree Regressor
Random Forest Regressor
10. Model Evaluation
Metrics used:
MAE (Mean Absolute Error)
MSE (Mean Squared Error)
R² Score
Python
def evaluate(y_test, y_pred):
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))
Results
Compared performance of all three models
Random Forest typically gives better accuracy (depending on data)
