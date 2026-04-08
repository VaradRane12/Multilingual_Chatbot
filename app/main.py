from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipline import EXAMPLE_PROMPTS, LANG_MAP, predict_chat

app = FastAPI(title="Multilingual Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    user_message: str
    detected_lang: str
    preprocessed_text: str
    pivot_text: str
    pivot_response: str
    translated_to_pivot: bool
    translated_from_pivot: bool
    translation_backend: str
    intent: str
    confidence: float
    assistant_message: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "supported_languages": sorted(LANG_MAP.keys()),
    }


@app.get("/languages")
def get_languages() -> dict:
    return {
        "languages": LANG_MAP,
    }


@app.get("/examples")
def get_examples() -> dict:
    return {
        "examples": EXAMPLE_PROMPTS,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    clean_message = payload.message.strip()
    prediction = predict_chat(clean_message)

    return ChatResponse(
        user_message=clean_message,
        detected_lang=prediction["detected_lang"],
        preprocessed_text=prediction["preprocessed_text"],
        pivot_text=prediction["pivot_text"],
        pivot_response=prediction["pivot_response"],
        translated_to_pivot=prediction["translated_to_pivot"],
        translated_from_pivot=prediction["translated_from_pivot"],
        translation_backend=prediction["translation_backend"],
        intent=prediction["intent"],
        confidence=prediction["confidence"],
        assistant_message=prediction["assistant_message"],
    )
