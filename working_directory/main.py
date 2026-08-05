from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class SumRequest(BaseModel):
    a: float
    b: float


@app.get("/health")
async def health():
    return {"status": "ok", "message": "Service is healthy"}


@app.post("/sum")
async def sum_endpoint(request: SumRequest):
    return {"a": request.a, "b": request.b, "sum": request.a + request.b}