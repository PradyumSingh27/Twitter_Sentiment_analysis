import re
import joblib
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from pathlib import Path

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import emoji

import mlflow

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -------------------------------
# Paths
# -------------------------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "final_data_pipeline.joblib"


# -------------------------------
# Load Params
# -------------------------------
def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


# -------------------------------
# Main
# -------------------------------
def main():
    config = load_params()

    DATA_PATH = Path(config["data"]["path"])
    TEXT_COL = config["data"]["text_col"]
    LABEL_COL = config["data"]["label_col"]

    max_features = config["tfidf"]["max_features"]
    ngram_range = (config["tfidf"]["ngram_min"], config["tfidf"]["ngram_max"])

    test_size = config["train"]["test_size"]
    random_state = config["train"]["random_state"]

    lgb_cfg = config["lgbm"]

    experiment_name = config["mlflow"]["experiment_name"]
    run_name = config["mlflow"]["run_name"]

    df = pd.read_csv(DATA_PATH)

    X = df[TEXT_COL].astype(str)
    y = df[LABEL_COL].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ✅ Full pipeline (tfidf + lgbm)
    pipeline = Pipeline(
        [
            
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=lgb_cfg["n_estimators"],
                    learning_rate=lgb_cfg["learning_rate"],
                    num_leaves=lgb_cfg["num_leaves"],
                    max_depth=lgb_cfg["max_depth"],
                    subsample=lgb_cfg["subsample"],
                    colsample_bytree=lgb_cfg["colsample_bytree"],
                    random_state=random_state,
                ),
            ),
        ]
    )

    # ✅ MLflow tracking
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        f1_weighted = f1_score(y_test, preds, average="weighted")

        report = classification_report(y_test, preds)

        print("✅ Accuracy:", acc)
        print("✅ F1 Macro:", f1_macro)
        print("✅ F1 Weighted:", f1_weighted)
        print("\n✅ Report:\n", report)

        # log params
        mlflow.log_param("classifier", "LightGBM")
        mlflow.log_param("tfidf_max_features", max_features)
        mlflow.log_param("ngram_range", f"{ngram_range[0]}-{ngram_range[1]}")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        for k, v in lgb_cfg.items():
            mlflow.log_param(f"lgb_{k}", v)

        # log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)

        # ✅ Save pipeline model
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))

        # ✅ Save classification report
        report_path = MODEL_DIR / "classification_report_final_datapipline.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(str(report_path))

        # ✅ Confusion matrix
        labels_sorted = sorted(y.unique())
        cm = confusion_matrix(y_test, preds, labels=labels_sorted)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, xticks_rotation=45, values_format="d")
        plt.title("Confusion Matrix - Final Data Pipeline")
        plt.tight_layout()

        cm_path = MODEL_DIR / "confusion_matrix_final_datapipline.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()

        mlflow.log_artifact(str(cm_path))

        print("\n✅ Saved Pipeline:", MODEL_PATH)
        print("✅ Saved CM:", cm_path)
        print("✅ Logged everything to MLflow")


if __name__ == "__main__":
    main()

