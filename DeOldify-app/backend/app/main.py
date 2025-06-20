from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.data_processor import router as data_router

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
]

app.add_middleware(CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health():
    return {
        "Response":"200",
        "Status": "OK",
    }

app.include_router(router=data_router)
