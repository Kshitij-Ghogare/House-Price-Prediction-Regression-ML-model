# imports

import numpy as np
import pandas as pd
from IPython.display import display
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

import seaborn as sns
import matplotlib.pyplot as plt

# Load File

dataset = pd.read_csv(r"C:\Users\Kshitij\Downloads\Pune_House_Prices_10000_rows.csv")

print(dataset.head(3))

# Data cleaning

print(dataset.isnull().sum())

dataset = dataset.dropna(subset = ['price'])

dataset.rename(columns={'has_garage': 'parking_available'}, inplace=True)


dataset['area'] = dataset['area'].astype('category')

numeric_cols = ['square_feet', 'num_bedrooms', 'num_bathrooms', 'year_built', 'parking_available', 'price']
dataset[numeric_cols] = dataset[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Feature Engineering

dataset['property_age'] = 2025 - dataset['year_built']

dataset = dataset.drop(columns=['id'])

dataset = dataset.drop(columns=['year_built'])

print(dataset.head(3))

# Preprocessing

numeric_cols = ['square_feet', 'num_bedrooms', 'num_bathrooms','property_age']
categorical_cols = ['area','parking_available']

target = 'price'

numeric_transformer = Pipeline(steps = [
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

categorical_transformer = Pipeline(steps= [
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown = "ignore" ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num",numeric_transformer,numeric_cols),
        ("cat",categorical_transformer,categorical_cols)
    ]
)

# Split feature and check relation

X = dataset.drop(columns = 'price')
Y = dataset['price']

# Train Test Split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

"""
corr = dataset.corr(numeric_only=True)

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5, annot_kws={"size": 8})
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()

sns.pairplot(data = dataset)
plt.show()
"""

# model1 : LinearRegression
model = Pipeline( steps=[
    ("preprocessor",preprocessor),
    ("regressor",LinearRegression())
])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")
Train1 = model.score(x_train, y_train)*100
Test1 = model.score(x_test, y_test)*100
print ("Train score :",Train1)
print ("Test score :", Test1)

"""
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonal
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Linear Regression: Actual vs Predicted")
plt.tight_layout()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True, bins=10)
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution (Linear Regression)")
plt.xlabel("Error = Actual - Predicted")
plt.tight_layout()
plt.show()
"""

# model 2 :  Decision Tree

model2 = Pipeline( steps=[
    ("preprocessor",preprocessor),
    ("regressor",DecisionTreeRegressor(random_state=42,  max_depth=5,        # limit tree depth
        min_samples_split=10, # minimum samples to split a node
        min_samples_leaf=5   # minimum samples in a leaf
         ))
])

# Train
model2.fit(x_train, y_train)

# Predict
y_pred2 = model2.predict(x_test)

# Evaluate
mae2 = mean_absolute_error(y_test, y_pred2)
rmse2 = np.sqrt(mean_squared_error(y_test, y_pred2))
r2_2 = r2_score(y_test, y_pred2)

print("Decision Tree Results:")
print(f"MAE:  {mae2:.2f}")
print(f"RMSE: {rmse2:.2f}")
print(f"R²:   {r2_2:.4f}")
Train2 = model2.score(x_train, y_train)*100
Test2 = model2.score(x_test, y_test)*100
print ("Train score :",Train2)
print ("Test score :", Test2)

"""
# Compare visually
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred2, color="green", label="Decision Tree")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Decision Tree: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

# Residuals plot
residuals2 = y_test - y_pred2
plt.figure(figsize=(6,4))
sns.histplot(residuals2, kde=True, bins=10, color="green")
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution (Decision Tree)")
plt.xlabel("Error = Actual - Predicted")
plt.tight_layout()
plt.show()
"""

# model 3 Random Forest

model3 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,      # number of trees
        max_depth=None,        # let trees grow fully (can tune later)
        min_samples_split=5,   # prevent tiny splits
        min_samples_leaf=2,    # prevent leaves with single points
        random_state=42,
        n_jobs=-1              # use all cores for speed
    ))
])

# Train
model3.fit(x_train, y_train)

y_pred3 = model3.predict(x_test)

joblib.dump(model3, "house_price_model.pkl")

mae3 = mean_absolute_error(y_test, y_pred3)
rmse3 = np.sqrt(mean_squared_error(y_test, y_pred3))
r2_3 = r2_score(y_test, y_pred3)

print("Random Forest Results:")
print(f"MAE:  {mae3:.2f}")
print(f"RMSE: {rmse3:.2f}")
print(f"R²:   {r2_3:.4f}")
Train3 = model3.score(x_train, y_train)*100
Test3 = model3.score(x_test, y_test)*100
print ("Train score :",Train3)
print ("Test score :", Test3)

"""
# Scatter plot (Actual vs Predicted)
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred3, color="orange", label="Random Forest")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

# Residuals
residuals3 = y_test - y_pred3
plt.figure(figsize=(6,4))
sns.histplot(residuals3, kde=True, bins=15, color="orange")
plt.axvline(0, color='red', linestyle='--')
plt.title("Residuals Distribution (Random Forest)")
plt.xlabel("Error = Actual - Predicted")
plt.tight_layout()
plt.show()
"""

# model 4 : XGBoost

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
])

xgb_pipeline.fit(x_train, y_train)

y_pred_xgb = xgb_pipeline.predict(x_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\nXGBoost Results:")
print(f"MAE:  {mae_xgb:.2f}")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"R²:   {r2_xgb:.4f}")
Train4 = xgb_pipeline.score(x_train, y_train)*100
Test4 = xgb_pipeline.score(x_test, y_test)*100
print ("Train score :",Train4)
print ("Test score :", Test4)

# comparison

# Collect results in a list of dicts
results = [
    {"Model": "Linear Regression", "MAE": mae, "RMSE": rmse, "R²": r2, "Train_Score" : Train1,"Test_Score" : Test1},
    {"Model": "Decision Tree", "MAE": mae2, "RMSE": rmse2, "R²": r2_2, "Train_Score" : Train2,"Test_Score" : Test2},
    {"Model": "Random Forest", "MAE": mae3, "RMSE": rmse3, "R²": r2_3, "Train_Score" : Train3,"Test_Score" : Test3},
    {"Model": "XGBoost", "MAE": mae_xgb, "RMSE": rmse_xgb, "R²": r2_xgb, "Train_Score" : Train4,"Test_Score" : Test4}
]

comparison_df = pd.DataFrame(results)

print("\nModel Comparison Table:")
display(comparison_df)