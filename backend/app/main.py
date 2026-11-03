"""AARYX backend entry point.

The FastAPI application will be implemented in Milestone M1.
"""

from fastapi import FastAPI

app = FastAPI(
    title="AARYX API",
    version="0.1.0",
    description="Personal AI Companion API",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "scaffolded", "app": "AARYX"}
