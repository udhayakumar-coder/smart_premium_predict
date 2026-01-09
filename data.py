import pandas as pd
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

import joblib

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
df = pd.read_csv(r"D:\project\smart_premium\data\train.csv", index_col="id")
df.drop_duplicates(inplace=True)

# -----------------------------------------------------------
# Independent and Dependent features
# -----------------------------------------------------------
       
target = "Premium Amount"
x = df.drop(target, axis=1)             #Independent features
y = df[target]                          #Dependent features

# -----------------------------------------------------------
# Identify numerical and categorical columns
# -----------------------------------------------------------

num_cols = x.select_dtypes(include=["float64", "int64"]).columns
cat_cols = x.select_dtypes(include=["object"]).columns

# -----------------------------------------------------------
# Preprocessing pipeline
# -----------------------------------------------------------

num_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),      #Median imputer for numerical features
    ("scaler", StandardScaler())])                      #StandardScaler for numerical features


cat_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),   #Most frequent imputer for categorical features
    ("encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value",                 #OrdinalEncoder for categorical features
        unknown_value=-1
    ))])

preprocess = ColumnTransformer([
    ("num", num_tf, num_cols),              #Numerical features
    ("cat", cat_tf, cat_cols),              #Categorical features 
])

# -----------------------------------------------------------
# Split the data into training and testing sets
# -----------------------------------------------------------

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# -----------------------------------------------------------
# Import evaluation metrics
# -----------------------------------------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------
# mlflow multiple model registry
# -----------------------------------------------------------

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

models = [
    (
        "Random Forest Regressor",
        {"n_estimators": 15, "max_depth": 5},
        RandomForestRegressor(),
        (x_train, y_train),
        (x_test, y_test)
    ),
    (
        "Gradient Boosting Regressor",
        {"n_estimators": 15, "learning_rate": 0.1},
        GradientBoostingRegressor(),
        (x_train, y_train),
        (x_test, y_test)
    ),
    (
        "XGB Regressor",
        {"n_estimators": 15, "learning_rate": 0.1},
        XGBRegressor(),
        (x_train, y_train),
        (x_test, y_test)
    )
]

# -----------------------------------------------------------
# MLflow experiment setup
# -----------------------------------------------------------

mlflow.set_tracking_uri("https://dagshub.com/udhayakumar24092/smart_premium")
mlflow.set_experiment("DS_Smart_Premium_Experiments")

# -----------------------------------------------------------
#Train, predict and evaluate each model
# -----------------------------------------------------------

reports = []
for model_name, params, model_instance, train_data, test_data in models:
    with mlflow.start_run(run_name=model_name):
        x_train = train_data[0]
        y_train = train_data[1]
        x_test = test_data[0]
        y_test = test_data[1]
        
        #Apply hyperparameters and train the model
        model_instance.set_params(**params)
        model_instance.fit(x_train, y_train)
        
        #Make predictions
        y_pred = model_instance.predict(x_test)

        #Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        #Store the results
        reports.append({
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "R2_Score": r2
        })
        # Save model
        mlflow.sklearn.log_model(model_instance, "models")

# -----------------------------------------------------------
#	Model Registration with MLflow
# -----------------------------------------------------------

model_name = "XGB_Regressor"
run_id = "dbef6a8758444a38b1b212a62eadbfac"             

# Correct model URI
model_uri = f"runs:/{run_id}/models"

# Register model
with mlflow.start_run(run_id=run_id):
    mlflow.register_model(model_uri=model_uri, name=model_name )

# -----------------------------------------------------------
# Promote model to Production stage
# -----------------------------------------------------------

from mlflow.tracking import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name="XGB_Regressor",
    version=5,
    stage="Production",
    archive_existing_versions=True
)

# -----------------------------------------------------------
# Load model from MLflow Model Registry
# -----------------------------------------------------------

import mlflow.pyfunc

model = mlflow.pyfunc.load_model(
    "models:/XGB_Regressor/Production"
)

predictions = model.predict(x_test)
predictions

print("done")