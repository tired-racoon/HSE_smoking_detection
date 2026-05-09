import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers import ping, streaming, frontend, heatmap, video_processing, analytics
from routers import auth as auth_router

_BASE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(
    title="Smoking Detection Service",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(_BASE, "static")), name="static")

app.include_router(auth_router.router)
app.include_router(frontend.router)
app.include_router(ping.router)
app.include_router(streaming.router)
app.include_router(streaming.websocket_router)
app.include_router(heatmap.router)
app.include_router(video_processing.router)
app.include_router(analytics.router)