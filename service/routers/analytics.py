import json
import os

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["Analytics"])

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_PATH = os.path.join(_BASE, "data", "demo_stats.json")
SCREENSHOTS_PATH = os.path.join(_BASE, "data", "screenshots.json")


@router.get("/analytics")
async def get_analytics():
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@router.get("/screenshots")
async def get_screenshots():
    try:
        with open(SCREENSHOTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


@router.get("/heatmap-data")
async def get_heatmap_data():
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("by_camera", [])
    except FileNotFoundError:
        return []