import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from auth import get_current_user

router = APIRouter()

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(_BASE, "templates"))


def _auth_or_redirect(request: Request):
    user = get_current_user(request)
    if not user:
        return None, RedirectResponse(url="/login", status_code=302)
    return user, None


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse(url="/dashboard", status_code=302)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user, redir = _auth_or_redirect(request)
    if redir:
        return redir
    return templates.TemplateResponse(request, "dashboard.html", {"user": user})


@router.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request):
    user, redir = _auth_or_redirect(request)
    if redir:
        return redir
    return templates.TemplateResponse(request, "analytics.html", {"user": user})


@router.get("/heatmap", response_class=HTMLResponse)
async def heatmap(request: Request):
    user, redir = _auth_or_redirect(request)
    if redir:
        return redir
    return templates.TemplateResponse(request, "heatmap.html", {"user": user})


@router.get("/screenshots", response_class=HTMLResponse)
async def screenshots(request: Request):
    user, redir = _auth_or_redirect(request)
    if redir:
        return redir
    return templates.TemplateResponse(request, "screenshots.html", {"user": user})


@router.get("/video-check", response_class=HTMLResponse)
async def video_check(request: Request):
    user, redir = _auth_or_redirect(request)
    if redir:
        return redir
    return templates.TemplateResponse(request, "video_check.html", {"user": user})