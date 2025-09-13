from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from functools import lru_cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import os, json, time, jwt
from .settings import get_settings, setup_logging


settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(title="AI Integration Skeleton", version="0.2.0")

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

def verify_jwt(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    secret = os.getenv("JWT_SECRET", "dev-secret")
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@lru_cache
def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)

class Ask(BaseModel):
    question: str

@app.post("/ask")
@limiter.limit("20/minute")
def ask(request: Request, body: Ask, user=Depends(verify_jwt)):
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"]
            }
        }
    }
    try:
        resp = get_openai_client().responses.create(
            model="gpt-4.1-mini",
            input=f"Ответь кратко и строго в JSON по схеме: вопрос: {body.question}",
            response_format=schema,
        )
        raw = resp.output[0].content[0].text
        payload = json.loads(raw)
        assert isinstance(payload, dict) and "answer" in payload
        return payload
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")