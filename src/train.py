import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

X_TRAIN_PATH = PROCESSED_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"


# --------------------------------------------------
# 2. MLflow configuration
# --------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Customer_Churn_Prediction")


# --------------------------------------------------
# 3. Load processed data
# --------------------------------------------------

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

print("Processed data loaded successfully!")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# --------------------------------------------------
# 4. Define models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
}


# --------------------------------------------------
# 5. Train each model
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training: {model_name}")
    print("=" * 50)

    # --------------------------------------------------
    # Start separate MLflow run
    # --------------------------------------------------

    with mlflow.start_run(run_name=model_name):

        # --------------------------------------------------
        # Log model parameters
        # --------------------------------------------------

        mlflow.log_param("model_type", model_name)

        if model_name == "Logistic Regression":

            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("random_state", 42)

        elif model_name == "Decision Tree":

            mlflow.log_param("max_depth", 5)
            mlflow.log_param("random_state", 42)

        # --------------------------------------------------
        # Train model
        # --------------------------------------------------

        model.fit(X_train, y_train)

        print("Model training completed!")

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        y_pred = model.predict(X_test)

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # Calculate metrics
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_pred_proba
        )

        # --------------------------------------------------
        # Log metrics to MLflow
        # --------------------------------------------------

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # --------------------------------------------------
        # Save model locally
        # --------------------------------------------------

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_filename = (
            "logistic_regression.pkl"
            if model_name == "Logistic Regression"
            else "decision_tree.pkl"
        )

        model_path = MODEL_DIR / model_filename

        joblib.dump(model, model_path)

        # --------------------------------------------------
        # Log model to MLflow
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model"
        )

        # --------------------------------------------------
        # Display results
        # --------------------------------------------------

        print("\nModel Evaluation")
        print("-------------------------")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        print("\nLocal model saved:")
        print(model_path)

        print("\nMLflow Run ID:")
        print(mlflow.active_run().info.run_id)


print("\n" + "=" * 50)
print("All models trained successfully!")
print("=" * 50)