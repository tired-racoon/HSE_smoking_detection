import os
from typing import Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from auth import create_session, delete_session, get_current_user, verify_user

router = APIRouter()

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(_BASE, "templates"))


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None):
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": error})


@router.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    user = verify_user(username, password)
    if not user:
        return RedirectResponse(url="/login?error=invalid", status_code=302)
    token = create_session(user["username"], user["role"], user["display_name"])
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=28800
    )
    return response


@router.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session_token")
    if token:
        delete_session(token)
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session_token")
    return response