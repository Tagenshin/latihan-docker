import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Ambil file dataset dari argumen atau default ke train_pca.csv
    file_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")
    )
    data = pd.read_csv(file_path)

    # Pisahkan data fitur dan target
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2
    )

    # Ambil parameter dari argumen
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    # Contoh input untuk logging model
    input_example = X_train.head(5)

    with mlflow.start_run():
        # Buat model dan latih
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Hitung metrik evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_prob)

        # Cetak hasil ke terminal
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {loss:.4f}")

        # Log parameter dan metrik ke MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("log_loss", loss)

        # Simpan model ke MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
