import io
import os
import re
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from functools import lru_cache
from typing import List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse

from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import emoji
import mlflow
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")


# =========================================================
# ✅ App Init
# =========================================================
app = FastAPI(title="Sentiment Analyzer API (YouTube + Reddit)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ for local development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# ✅ MLflow Toggle
# =========================================================
ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "false").lower() == "true"

# =========================================================
# ✅ Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
MODEL_PATH = BASE_DIR / "models" / "final_data_pipeline.joblib"


# =========================================================
# ✅ Globals
# =========================================================
try:
    STOP_WORDS = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
except:
    STOP_WORDS = set()

LEMMATIZER = WordNetLemmatizer()

CHAT_WORDS = {
    "LOL": "Laughing Out Loud",
    "LMAO": "Laugh My Ass Off",
    "BRB": "Be Right Back",
    "AFK": "Away From Keyboard",
    "BTW": "By The Way",
    "IDK": "I Don’t Know",
    "IMO": "In My Opinion",
    "IMHO": "In My Honest Opinion",
    "FYI": "For Your Information",
}


def chat_conversion(text: str) -> str:
    out = []
    for w in str(text).split():
        out.append(CHAT_WORDS.get(w.upper(), w))
    return " ".join(out)


def get_wordnet_pos(word: str):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


@lru_cache(maxsize=50000)
def preprocess_comment(text: str) -> str:
    """
    ✅ Production optimized preprocessing
    """
    text = str(text).lower().strip()

    # normalize repeated chars
    text = " ".join([re.sub(r"(.)\1{2,}", r"\1\1", w) for w in text.split()])

    # chat conversion
    text = chat_conversion(text)

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove mentions
    text = re.sub(r"@\w+", "", text)

    # remove hashtag symbol only
    text = text.replace("#", "")

    # emojis remove
    text = emoji.replace_emoji(text, replace="")

    # keep only letters+spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = []
    for w in text.split():
        if w in STOP_WORDS:
            continue
        w = LEMMATIZER.lemmatize(w, get_wordnet_pos(w))
        words.append(w)

    return " ".join(words)


# =========================================================
# ✅ Load Model
# =========================================================
@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded:", MODEL_PATH)

    if ENABLE_MLFLOW:
        ROOT_DIR = BASE_DIR.parent
        mlflow_db_path = ROOT_DIR / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
        mlflow.set_experiment("FastAPI_Predictions")
        print("✅ MLflow ENABLED")
    else:
        print("⚠️ MLflow DISABLED")


# =========================================================
# ✅ Request Schemas
# =========================================================
class PredictRequest(BaseModel):
    comments: List[str]


class ChartRequest(BaseModel):
    sentiment_counts: Dict[str, int]


class WordCloudRequest(BaseModel):
    comments: List[str]


class LengthChartRequest(BaseModel):
    sentiments: List[str]
    lengths: List[int]


# =========================================================
# ✅ Routes
# =========================================================
@app.get("/")
def home():
    return {"message": "Sentiment Analyzer API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# =========================================================
# ✅ Prediction Endpoint
# =========================================================
@app.post("/predict")
def predict(req: PredictRequest):
    if not req.comments:
        return JSONResponse({"error": "No comments provided"}, status_code=400)

    try:
        cleaned = [preprocess_comment(c) for c in req.comments]
        preds = model.predict(cleaned).tolist()

        if ENABLE_MLFLOW:
            with mlflow.start_run(run_name="predict"):
                mlflow.log_param("num_comments", len(req.comments))

        return [
            {"comment": c, "cleaned": cl, "sentiment": p}
            for c, cl, p in zip(req.comments, cleaned, preds)
        ]

    except Exception as e:
        return JSONResponse({"error": f"Prediction failed: {e}"}, status_code=500)


# =========================================================
# ✅ Pie Chart
# =========================================================
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
            return JSONResponse({"error": "Counts sum to zero"}, status_code=400)

        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#111827")
        ax.set_facecolor("#111827")

        wedges, _, _ = ax.pie(
            sizes,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.7,
            textprops={"color": "white", "fontsize": 12, "fontweight": "bold"},
        )

        legend = ax.legend(
            wedges,
            [x.title() for x in labels],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False
        )
        for t in legend.get_texts():
            t.set_color("white")

        ax.axis("equal")

        img_io = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close(fig)

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": f"Chart generation failed: {e}"}, status_code=500)


# =========================================================
# ✅ Wordcloud
# =========================================================
@app.post("/generate_wordcloud")
def generate_wordcloud(req: WordCloudRequest):
    try:
        if not req.comments:
            return JSONResponse({"error": "No comments provided"}, status_code=400)

        cleaned = [preprocess_comment(c) for c in req.comments]
        text = " ".join(cleaned)

        wc = WordCloud(
            width=900,
            height=450,
            background_color="black",
            stopwords=STOP_WORDS,
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wc.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        return JSONResponse({"error": f"Wordcloud generation failed: {e}"}, status_code=500)

