# app.py
import io
import re
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import emoji
from functools import lru_cache

import mlflow
from pathlib import Path


app = FastAPI(title="Twitter Sentiment FastAPI")


# --------------------------------------------------
# ✅ Enable/Disable MLflow using Environment Variable
# --------------------------------------------------
# default: false (for deployment safety)
ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "false").lower() == "true"


# --------------------------------------------------
# ✅ Model path (inside backend/models)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
MODEL_PATH = BASE_DIR / "models" / "final_data_pipeline.joblib"


# -----------------------------
# ✅ Global objects (fast)
# -----------------------------
STOP_WORDS = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
LEMMATIZER = WordNetLemmatizer()

chat_word = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATM": "At The Moment",
    "BRB": "Be Right Back",
    "BTW": "By The Way",
    "FYI": "For Your Information",
    "GG": "Good Game",
    "GN": "Good Night",
    "IDK": "I Don’t Know",
    "IMO": "In My Opinion",
    "IMHO": "In My Honest Opinion",
    "IRL": "In Real Life",
    "LMAO": "Laugh My Ass Off",
    "LOL": "Laughing Out Loud",
    "ROFL": "Rolling On The Floor Laughing",
    "THX": "Thank You",
    "TTYL": "Talk To You Later",
    "U": "You",
    "WB": "Welcome Back",
    "WTF": "What The F...",
}


def chat_conversion(text: str) -> str:
    new_text = []
    for w in str(text).split():
        if w.upper() in chat_word:
            new_text.append(chat_word[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)


def get_wordnet_pos(word: str):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


@lru_cache(maxsize=50000)
def preprocess_comment(text: str) -> str:
    """
    ✅ Preprocessing logic optimized for API
    """
    text = str(text).lower().strip()

    # normalize repeated chars (coooool -> cool)
    text = " ".join([re.sub(r"(.)\1{2,}", r"\1\1", w) for w in text.split()])

    # chat conversion (LOL etc)
    text = chat_conversion(text)

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove mentions
    text = re.sub(r"@\w+", "", text)

    # hashtags symbol remove but keep words
    text = re.sub(r"#", "", text)

    # remove emojis
    text = emoji.replace_emoji(text, replace="")

    # keep only letters + spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # tokenize + stopwords + lemmatize (POS)
    words = []
    for w in text.split():
        if w in STOP_WORDS:
            continue
        w = LEMMATIZER.lemmatize(w, get_wordnet_pos(w))
        words.append(w)

    return " ".join(words)


# -----------------------------
# ✅ Load Model Once
# -----------------------------
@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded from:", MODEL_PATH)

    # ✅ MLflow - Enable/Disable
    if ENABLE_MLFLOW:
        # If your mlflow.db is in project root (outside backend)
        ROOT_DIR = BASE_DIR.parent  # project root
        mlflow_db_path = ROOT_DIR / "mlflow.db"

        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
        mlflow.set_experiment("FastAPI_Predictions")

        print("✅ MLflow Enabled | Tracking URI:", mlflow_db_path)
    else:
        print("⚠️ MLflow Disabled (ENABLE_MLFLOW=false)")


# -----------------------------
# ✅ Request Models
# -----------------------------
class PredictRequest(BaseModel):
    comments: List[str]


class ChartRequest(BaseModel):
    sentiment_counts: dict  # {"positive": 10, "neutral": 5, "negative": 3}


class WordCloudRequest(BaseModel):
    comments: List[str]


class TrendItem(BaseModel):
    timestamp: str
    sentiment: str


class TrendRequest(BaseModel):
    sentiment_data: List[TrendItem]


# -----------------------------
# ✅ Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI sentiment api"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# -----------------------------
# ✅ Predict
# -----------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    if not req.comments:
        return JSONResponse({"error": "No comments provided"}, status_code=400)

    try:
        cleaned = [preprocess_comment(c) for c in req.comments]

        # ✅ model is pipeline: TF-IDF + LightGBM
        preds = model.predict(cleaned).tolist()

        # ✅ Optional MLflow logging
        if ENABLE_MLFLOW:
            with mlflow.start_run(run_name="api_prediction"):
                mlflow.log_param("num_comments", len(req.comments))

        response = [
            {"comment": c, "cleaned": cl, "sentiment": p}
            for c, cl, p in zip(req.comments, cleaned, preds)
        ]
        return response

    except Exception as e:
        return JSONResponse({"error": f"Prediction failed: {str(e)}"}, status_code=500)


# -----------------------------
# ✅ Generate Pie Chart
# -----------------------------
@app.post("/generate_chart")
def generate_chart(req: ChartRequest):
    try:
        counts = req.sentiment_counts

        labels = ["positive", "neutral", "negative"]
        sizes = [
            int(counts.get("positive", 0)),
            int(counts.get("neutral", 0)),
            int(counts.get("negative", 0)),
        ]

        if sum(sizes) == 0:
            return JSONResponse({"error": "Sentiment counts sum to zero"}, status_code=400)

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=[x.title() for x in labels], autopct="%1.1f%%", startangle=140)
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": f"Chart generation failed: {str(e)}"}, status_code=500)


# -----------------------------
# ✅ Wordcloud
# -----------------------------
@app.post("/generate_wordcloud")
def generate_wordcloud(req: WordCloudRequest):
    try:
        if not req.comments:
            return JSONResponse({"error": "No comments provided"}, status_code=400)

        cleaned = [preprocess_comment(c) for c in req.comments]
        text = " ".join(cleaned)

        wc = WordCloud(
            width=800,
            height=400,
            background_color="black",
            stopwords=STOP_WORDS,
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wc.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": f"Word cloud generation failed: {str(e)}"}, status_code=500)


# -----------------------------
# ✅ Trend Graph (Monthly %)
# -----------------------------
@app.post("/generate_trend_graph")
def generate_trend_graph(req: TrendRequest):
    try:
        if not req.sentiment_data:
            return JSONResponse({"error": "No sentiment data provided"}, status_code=400)

        df = pd.DataFrame([x.model_dump() for x in req.sentiment_data])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        mapping = {"negative": -1, "neutral": 0, "positive": 1}
        df["sentiment_num"] = df["sentiment"].map(mapping)

        monthly_counts = df.resample("M")["sentiment_num"].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for col in [-1, 0, 1]:
            if col not in monthly_percentages.columns:
                monthly_percentages[col] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        series_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        for val in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[val],
                marker="o",
                linestyle="-",
                label=series_labels[val],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage (%)")
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": f"Trend graph generation failed: {str(e)}"}, status_code=500)
