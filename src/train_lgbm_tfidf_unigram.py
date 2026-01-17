import pandas as pd
import yaml
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_params()

    data_path = config["data"]["path"]
    text_col = config["data"]["text_col"]
    label_col = config["data"]["label_col"]

    test_size = config["train"]["test_size"]
    random_state = config["train"]["random_state"]

    max_features = config["tfidf"]["max_features"]

    lgb_params = config["lgbm"]

    df = pd.read_csv(data_path)
    X = df[text_col].astype(str)
    y = df[label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 1))),
        ("clf", LGBMClassifier(
            n_estimators=lgb_params["n_estimators"],
            learning_rate=lgb_params["learning_rate"],
            num_leaves=lgb_params["num_leaves"],
            max_depth=lgb_params["max_depth"],
            subsample=lgb_params["subsample"],
            colsample_bytree=lgb_params["colsample_bytree"],
            random_state=random_state
        ))
    ])

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="LGBM_TFIDF_UNIGRAM"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        # log params
        mlflow.log_param("classifier", "LightGBM")
        mlflow.log_param("tfidf_max_features", max_features)
        mlflow.log_param("ngram_range", "1-1")

        for k, v in lgb_params.items():
            mlflow.log_param(f"lgb_{k}", v)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # save model
        Path("models").mkdir(exist_ok=True)
        model_path = "models/lgbm_tfidf_unigram.joblib"
        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path)

        
        print("Accuracy:", acc)
        print("F1:", f1)


if __name__ == "__main__":
    main()
