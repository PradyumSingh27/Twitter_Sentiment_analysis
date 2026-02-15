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
import requests


# =========================================================
# ✅ App Init
# =========================================================
app = FastAPI(title="Sentiment Analyzer API (YouTube + Reddit)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# ✅ Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
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

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MAX_COMMENTS = 1500


# =========================================================
# ✅ YouTube Comments Route
# =========================================================
@app.get("/youtube-comments")
def youtube_comments(video_id: str):
    if not YOUTUBE_API_KEY:
        return JSONResponse({"error": "YouTube API key missing"}, status_code=500)

    comments = []
    page = ""

    try:
        while len(comments) < MAX_COMMENTS:
            url = (
                "https://www.googleapis.com/youtube/v3/commentThreads"
                f"?part=snippet&videoId={video_id}"
                f"&maxResults=100&key={YOUTUBE_API_KEY}"
            )

            if page:
                url += f"&pageToken={page}"

            r = requests.get(url)
            data = r.json()

            if "items" not in data:
                break

            for item in data["items"]:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": snippet.get("textOriginal", ""),
                    "author": snippet.get("authorChannelId", {}).get("value", "yt"),
                    "likes": snippet.get("likeCount", 0),
                    "replies": item["snippet"].get("totalReplyCount", 0)
                })

            page = data.get("nextPageToken")
            if not page:
                break

        return comments[:MAX_COMMENTS]

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================================================
# ✅ Preprocessing
# =========================================================
def chat_conversion(text: str) -> str:
    return " ".join(CHAT_WORDS.get(w.upper(), w) for w in str(text).split())


def get_wordnet_pos(word: str):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


@lru_cache(maxsize=50000)
def preprocess_comment(text: str) -> str:
    text = str(text).lower().strip()
    text = " ".join([re.sub(r"(.)\1{2,}", r"\1\1", w) for w in text.split()])
    text = chat_conversion(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")
    text = emoji.replace_emoji(text, replace="")
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
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    print("Model loaded:", MODEL_PATH)


# =========================================================
# ✅ Schemas
# =========================================================
class PredictRequest(BaseModel):
    comments: List[str]


class ChartRequest(BaseModel):
    sentiment_counts: Dict[str, int]


class WordCloudRequest(BaseModel):
    comments: List[str]


# =========================================================
# ✅ Basic Routes
# =========================================================
@app.get("/")
def home():
    return {"message": "Sentiment Analyzer API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# =========================================================
# ✅ Prediction
# =========================================================
@app.post("/predict")
def predict(req: PredictRequest):
    if not req.comments:
        return JSONResponse({"error": "No comments provided"}, status_code=400)

    try:
        cleaned = [preprocess_comment(c) for c in req.comments]
        preds = model.predict(cleaned).tolist()

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
