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
