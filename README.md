# AI Integration Skeleton (OpenAI / Hugging Face / LangChain) 

Готовый каркас для безопасных AI‑интеграций на **FastAPI** с упором на:
- **Structured Outputs** и **tool calling** (OpenAI)
- **RAG** на базе **LangChain** (эмбеддинги HF, локальная векторка)
- Варианты инференса: **облако (HF Inference Endpoints)** и **локально (`transformers`)**
- Политики безопасности: строгая валидация, песочница инструментов, egress‑allowlist
- Готовые профили запуска: `uvicorn`, `Dockerfile`, `docker-compose`, `Kubernetes` (+ `NetworkPolicy` для egress‑allowlist)

- ## 📌 Ключевые возможности
- **Единый API** поверх нескольких провайдеров LLM/Embeddings.
- **Эндпоинт /ask** с потоковой выдачей (SSE/Chunked), контекстом и защитами.
- **Rate limiting, Idempotency-Key, retries, circuit breaker.**
- **JWT** с ролями/скоупами; CORS, CSRF (для cookie‑режима).
- **Кэш** ответов и эмбеддингов (Redis).
- **Очереди** для тяжелых задач (Celery/Redis).
- **RFC 7807** формат ошибок, **ETag/Cache-Control**, **cursor‑pagination**.
- **Observability**: логи JSON, Prometheus метрики, OpenTelemetry трассы.


---

## 0. Требования и переменные окружения

- Python 3.10+
- Переменные окружения:
  - `OPENAI_API_KEY` — ключ OpenAI (хранить ТОЛЬКО на бэке)
  - `HF_TOKEN` — токен Hugging Face (для InferenceClient/эндпоинтов)
  - `LOG_LEVEL` — уровень логов (по умолчанию `INFO`)

Создай `.env` (локально и/или для Docker Compose):

```env
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
LOG_LEVEL=INFO
```

---

## 1. Структура проекта

```
ai_integration_skeleton/
├─ app/
│  ├─ main.py                  # FastAPI-приложение (эндпоинт /ask, безопасные дефолты)
│  ├─ ai_openai.py             # OpenAI: structured outputs + tool calling
│  ├─ ai_hf_cloud.py           # Hugging Face InferenceClient (облако)
│  ├─ ai_hf_local.py           # Hugging Face transformers (локально)
│  ├─ rag_chain.py             # LangChain RAG (JSON Output + парсер)
│  ├─ safe_tools.py            # Декоратор safe_tool (валидация и политики)
│  └─ settings.py              # Загрузка .env, конфиг логирования
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ k8s/
│  ├─ deployment.yaml
│  ├─ service.yaml
│  ├─ configmap.yaml
│  ├─ secret.yaml
│  └─ networkpolicy-egress-allowlist.yaml
└─ README.md 
```

Установи зависимости:
```bash
pip install -r requirements.txt
```

Запуск локально (dev):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

---

## 2. Код: базовые модули

### 2.1 `app/settings.py` — конфиг/логи
```python
import os
from functools import lru_cache
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

load_dotenv()

class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    hf_token: str | None = os.getenv("HF_TOKEN")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

@lru_cache
def get_settings() -> Settings:
    return Settings()

def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
```

### 2.2 `app/ai_openai.py` — Structured Outputs + tool calling
```python
import os
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class FlightSearch(BaseModel):
    origin: str = Field(..., min_length=3, max_length=64)
    destination: str = Field(..., min_length=3, max_length=64)
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")  # YYYY-MM-DD

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "flight_query",
        "schema": FlightSearch.model_json_schema()
    }
}

def search_flights(origin: str, destination: str, date: str):
    return [{"origin": origin, "destination": destination, "date": date, "price": 1234}]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "Search real flights safely",
        "parameters": FlightSearch.model_json_schema()
    }
}]

def ask_openai_flights(query: str):
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=query,
        response_format=SCHEMA,
        tools=TOOLS,
    )
    calls = []
    for item in (resp.output or []):
        for piece in (item.content or []):
            if getattr(piece, "type", None) == "tool_call" and piece.tool.name == "search_flights":
                try:
                    args = FlightSearch(**piece.tool.arguments)
                except ValidationError as e:
                    raise ValueError(f"Bad tool args: {e}")
                calls.append(search_flights(args.origin, args.destination, args.date))
    return calls
```

### 2.3 `app/ai_hf_cloud.py` — HF InferenceClient
```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=os.environ.get("HF_TOKEN")
)

def hf_generate_json(prompt: str, max_new_tokens: int = 128) -> str:
    return client.text_generation(prompt, max_new_tokens=max_new_tokens, details=False)
```

### 2.4 `app/ai_hf_local.py` — локально через transformers
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID)
_pipe = pipeline("text-generation", model=_mdl, tokenizer=_tok)

def local_generate(prompt: str, max_new_tokens: int = 64) -> str:
    return _pipe(prompt, max_new_tokens=max_new_tokens, temperature=0.2)[0]["generated_text"]
```

### 2.5 `app/rag_chain.py` — LangChain RAG (JSON Output)
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
  ("system", "Ты отвечаешь строго в JSON: {format}"),
  ("user", "Ответь на вопрос на основе контекста: {question}\nКонтекст:\n{context}")
])

parser = JsonOutputParser()
chain = (prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | parser)

def rag_answer(question: str, context: str) -> dict:
    return chain.invoke({"question": question, "context": context, "format": parser.get_format_instructions()})
```

### 2.6 `app/safe_tools.py` — пример песочницы
```python
from functools import wraps
from pydantic import BaseModel, ValidationError

def safe_tool(schema: type[BaseModel]):
    def deco(fn):
        def wrapper(**kwargs):
            try:
                args = schema(**kwargs)
            except ValidationError as e:
                raise ValueError(f"Bad args: {e}")
            if hasattr(args, "url"):
                allow = ("https://api.example.com/", "https://internal.service/")
                if not str(args.url).startswith(allow):
                    raise PermissionError("URL not allowed")
            return fn(**args.dict())
        return wrapper
    return deco
```

### 2.7 `app/main.py` — FastAPI каркас
```python
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os, json
from .settings import get_settings, setup_logging

settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(title="AI Integration Skeleton", version="0.1.0")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class Ask(BaseModel):
    question: str

@app.post("/ask")
def ask(body: Ask):
    schema = { "type":"json_schema","json_schema":{"name":"answer","schema":{
        "type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]
    }}}
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=f"Ответь кратко и строго в JSON по схеме: вопрос: {body.question}",
        response_format=schema,
    )
    raw = resp.output[0].content[0].text
    payload = json.loads(raw)
    assert isinstance(payload, dict) and "answer" in payload
    return payload
```

---

## 3. Запуск: Uvicorn

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 4. Docker

### 4.1 `requirements.txt`
```txt
fastapi==0.112.2
uvicorn[standard]==0.30.6
python-dotenv==1.0.1
pydantic==2.8.2
openai==1.40.6
huggingface_hub==0.25.1
transformers==4.44.2
langchain==0.2.14
langchain-openai==0.1.22
```

### 4.2 `Dockerfile`
```Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.3 `docker-compose.yml`
```yaml
version: "3.9"
services:
  ai-service:
    build: .
    image: ai-integration:latest
    ports:
      - "8000:8000"
    environment:
      OPENAI_API_KEY: "${OPENAI_API_KEY}"
      HF_TOKEN: "${HF_TOKEN}"
      LOG_LEVEL: "${LOG_LEVEL:-INFO}"
    restart: unless-stopped
```

Запуск:
```bash
docker compose up --build -d
```

---

## 5. Kubernetes (с egress‑allowlist)

### 5.1 `k8s/configmap.yaml`
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-config
data:
  LOG_LEVEL: "INFO"
```

### 5.2 `k8s/secret.yaml`
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-secrets
type: Opaque
stringData:
  OPENAI_API_KEY: "replace_me"
  HF_TOKEN: "replace_me"
```

### 5.3 `k8s/deployment.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-integration
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ai-integration
  template:
    metadata:
      labels:
        app: ai-integration
    spec:
      containers:
        - name: api
          image: ai-integration:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: ai-config
            - secretRef:
                name: ai-secrets
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
```

### 5.4 `k8s/service.yaml`
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-integration-svc
spec:
  type: ClusterIP
  selector:
    app: ai-integration
  ports:
    - name: http
      port: 80
      targetPort: 8000
```

### 5.5 `k8s/networkpolicy-egress-allowlist.yaml`
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ai-egress-allowlist
spec:
  podSelector:
    matchLabels:
      app: ai-integration
  policyTypes:
    - Egress
  egress:
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
```
