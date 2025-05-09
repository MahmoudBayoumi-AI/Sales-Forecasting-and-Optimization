
# Import required libraries
import mlflow.catboost
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# Define a function to evaluate model metrics
def evaluate_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Main script
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("Preprocessed_Data.csv")

    # Split features and target variable
    X = df.drop(['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Item_Outlet_Sales'], axis=1)
    y = df["Item_Outlet_Sales"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define CatBoost model hyperparameters
    iterations = 700
    depth = 4
    learning_rate = 0.1
    l2_leaf_reg = 7

    # Set MLflow experiment name
    mlflow.set_experiment("CatBoost_Regression_Experiment")

    # Start MLflow run
    with mlflow.start_run(run_name='first_run'):

        # Initialize CatBoost Regressor model
        model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_state=42,
            verbose=0,
            early_stopping_rounds=20
        )

        # Train the model
        model.fit(X_train, y_train, eval_set=(X_test, y_test))  # Fixed: use X_test/y_test as eval_set (you had X_valid which was undefined)

        # Predict on the test set
        y_pred_cat = model.predict(X_test)

        # Compute evaluation metrics
        rmse, mae, r2 = evaluate_metrics(y_test, y_pred_cat)

        # Log model hyperparameters
        mlflow.log_param("iterations", iterations)
        mlflow.log_param("depth", depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("l2_leaf_reg", l2_leaf_reg)

        # Log evaluation metrics
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)

        # Infer model signature (input/output schema)
        signature = infer_signature(X, y)

        # Log model along with signature
        mlflow.catboost.log_model(model, "catboost_model", signature=signature)

        # Print the evaluation results
        print(f"Model successfully logged to MLflow.")
        print(f"R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")


        


