import os
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class FlightSearch(BaseModel):
    origin: str = Field(..., min_length=3, max_length=64)
    destination: str = Field(..., min_length=3, max_length=64)
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")

SCHEMA = { "type": "json_schema", "json_schema": { "name": "flight_query", "schema": FlightSearch.model_json_schema() } }

def search_flights(origin: str, destination: str, date: str):
    return [{"origin": origin, "destination": destination, "date": date, "price": 1234}]

TOOLS = [{ "type": "function", "function": { "name": "search_flights", "description": "Search real flights safely", "parameters": FlightSearch.model_json_schema() } }]

def ask_openai_flights(query: str):
    resp = client.responses.create(model="gpt-4.1-mini", input=query, response_format=SCHEMA, tools=TOOLS)
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
