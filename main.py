from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import pinecone
import os

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "computeruserag"

client = OpenAI(api_key=OPENAI_API_KEY)
pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(INDEX_NAME)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_pinecone(data: QueryRequest):
    embedding = client.embeddings.create(
        input=data.query,
        model="text-embedding-3-large"
    ).data[0].embedding

    result = index.query(vector=embedding, top_k=5, include_metadata=True)
    chunks = [match["metadata"]["text"] for match in result["matches"]]
    return {"chunks": chunks}
