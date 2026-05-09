from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBasic
import hashlib
import hmac
import json
import os
import secrets

SESSION_SECRET = os.environ.get("SESSION_SECRET", "changeme-in-production-please")
SESSION_EXPIRE_HOURS = 8

USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "display_name": "Administrator"
    },
    "operator": {
        "password_hash": hashlib.sha256("operator123".encode()).hexdigest(),
        "role": "user",
        "display_name": "Operator"
    }
}

active_sessions: dict = {}


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_user(username: str, password: str) -> Optional[dict]:
    user = USERS.get(username)
    if not user:
        return None
    if not hmac.compare_digest(user["password_hash"], hash_password(password)):
        return None
    return {"username": username, "role": user["role"], "display_name": user["display_name"]}


def create_session(username: str, role: str, display_name: str) -> str:
    token = secrets.token_urlsafe(32)
    active_sessions[token] = {
        "username": username,
        "role": role,
        "display_name": display_name,
        "expires_at": (datetime.utcnow() + timedelta(hours=SESSION_EXPIRE_HOURS)).isoformat()
    }
    return token


def get_session(token: str) -> Optional[dict]:
    session = active_sessions.get(token)
    if not session:
        return None
    expires_at = datetime.fromisoformat(session["expires_at"])
    if datetime.utcnow() > expires_at:
        del active_sessions[token]
        return None
    return session


def delete_session(token: str):
    active_sessions.pop(token, None)


def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get("session_token")
    if not token:
        return None
    return get_session(token)


def require_auth(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"}
        )
    return user


def require_admin(request: Request) -> dict:
    user = require_auth(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
