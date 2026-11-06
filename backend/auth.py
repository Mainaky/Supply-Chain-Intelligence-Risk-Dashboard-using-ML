"""Password-based accounts backed by Render Postgres and signed short-lived JWTs."""

from __future__ import annotations

import logging
import os
import hmac
import hashlib
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import jwt
import psycopg
from fastapi import HTTPException, status
from jwt.exceptions import InvalidTokenError
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from pwdlib import PasswordHash


LOGGER = logging.getLogger("supply_chain_auth")
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "supply-chain-risk-api"
JWT_AUDIENCE = "supply-chain-dashboard"
ACCESS_TOKEN_MINUTES = 30
LOCAL_DB = Path(os.getenv("LOCAL_AUTH_DB_PATH", Path(__file__).resolve().parents[1] / "Data" / "auth_local.db"))
PASSWORDS = PasswordHash.recommended()
DUMMY_PASSWORD_HASH = PASSWORDS.hash("nonexistent-account-dummy-secret")


class SignUpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)

    @field_validator("username")
    @classmethod
    def clean_username(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: EmailStr) -> str:
        return str(value).strip().lower()


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    expires_in: int
    message: str


def configured_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "")
    if len(secret) < 32:
        raise RuntimeError("Set JWT_SECRET_KEY to a random secret with at least 32 characters")
    return secret


def _database_url() -> str | None:
    url = os.getenv("DATABASE_URL")
    if url:
        # Render also accepts postgres:// URLs; psycopg expects postgresql://.
        return "postgresql://" + url.removeprefix("postgres://") if url.startswith("postgres://") else url
    if os.getenv("RENDER"):
        raise RuntimeError("DATABASE_URL must point to persistent Postgres on Render")
    return None


@contextmanager
def database_connection() -> Iterator[object]:
    url = _database_url()
    if url:
        with psycopg.connect(url, row_factory=dict_row, connect_timeout=5) as connection:
            yield connection
        return

    LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(LOCAL_DB, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    """Create the small account table; Render uses persistent managed Postgres."""
    configured_jwt_secret()
    _database_url()
    with database_connection() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY,
                username VARCHAR(40) NOT NULL UNIQUE,
                email VARCHAR(320) NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
            if _database_url()
            else """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS auth_rate_limits (
                bucket_key TEXT PRIMARY KEY,
                window_start BIGINT NOT NULL,
                attempts INTEGER NOT NULL,
                blocked_until BIGINT NOT NULL DEFAULT 0
            )"""
        )


def _rate_key(bucket: str, identity: str) -> str:
    message = f"{bucket}:{identity.strip().lower()}".encode("utf-8")
    return hmac.new(configured_jwt_secret().encode("utf-8"), message, hashlib.sha256).hexdigest()


def enforce_rate_limit(
    bucket: str,
    identity: str,
    *,
    max_attempts: int,
    window_seconds: int = 900,
    block_seconds: int = 900,
) -> None:
    """Database-backed throttle keyed by HMAC so usernames and IPs aren't stored."""
    now = int(time.time())
    key = _rate_key(bucket, identity)
    placeholders = "%s" if _database_url() else "?"
    with database_connection() as connection:
        row = connection.execute(
            f"""INSERT INTO auth_rate_limits (bucket_key, window_start, attempts, blocked_until)
            VALUES ({placeholders}, {placeholders}, 1, 0)
            ON CONFLICT(bucket_key) DO UPDATE SET
                window_start = CASE
                    WHEN excluded.window_start - auth_rate_limits.window_start >= {int(window_seconds)}
                    THEN excluded.window_start ELSE auth_rate_limits.window_start END,
                attempts = CASE
                    WHEN excluded.window_start - auth_rate_limits.window_start >= {int(window_seconds)}
                    THEN 1 ELSE auth_rate_limits.attempts + 1 END,
                blocked_until = CASE
                    WHEN (CASE
                        WHEN excluded.window_start - auth_rate_limits.window_start >= {int(window_seconds)}
                        THEN 1 ELSE auth_rate_limits.attempts + 1 END) > {int(max_attempts)}
                    THEN excluded.window_start + {int(block_seconds)} ELSE 0 END
            RETURNING window_start, attempts, blocked_until""",
            (key, now),
        ).fetchone()
        attempts = int(row["attempts"])
        blocked_until = int(row["blocked_until"])
        exceeded = attempts > max_attempts
        retry_after = max(1, blocked_until - now) if blocked_until else block_seconds
    if exceeded:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts. Wait before trying again.",
            headers={"Retry-After": str(retry_after)},
        )


def reset_rate_limit(bucket: str, identity: str) -> None:
    key = _rate_key(bucket, identity)
    placeholder = "%s" if _database_url() else "?"
    with database_connection() as connection:
        connection.execute(
            f"DELETE FROM auth_rate_limits WHERE bucket_key = {placeholder}",
            (key,),
        )


def _find_user(identifier: str) -> dict | None:
    with database_connection() as connection:
        row = connection.execute(
            "SELECT username, email, password_hash FROM users WHERE username = %s OR email = %s"
            if _database_url()
            else "SELECT username, email, password_hash FROM users WHERE username = ? OR email = ?",
            (identifier, identifier),
        ).fetchone()
        return dict(row) if row else None


def _create_user(username: str, email: str, password_hash: str) -> None:
    with database_connection() as connection:
        connection.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)"
            if _database_url()
            else "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )


def _access_token(username: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": username,
            "iat": now,
            "exp": now + timedelta(minutes=ACCESS_TOKEN_MINUTES),
            "iss": JWT_ISSUER,
            "aud": JWT_AUDIENCE,
        },
        configured_jwt_secret(),
        algorithm=JWT_ALGORITHM,
    )


def register_user(request: SignUpRequest) -> AuthResponse:
    username = request.username
    email = str(request.email)
    try:
        _create_user(username, email, PASSWORDS.hash(request.password))
    except (sqlite3.IntegrityError, psycopg.errors.UniqueViolation) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email is already registered",
        ) from exc
    LOGGER.info("New account registered")
    return AuthResponse(
        access_token=_access_token(username),
        username=username,
        expires_in=ACCESS_TOKEN_MINUTES * 60,
        message="Account created. Please note that this demo does not verify email ownership.",
    )


def authenticate_user(request: LoginRequest) -> AuthResponse:
    identifier = request.username.strip().lower()
    user = _find_user(identifier)
    stored_hash = user["password_hash"] if user else DUMMY_PASSWORD_HASH
    verified, replacement_hash = PASSWORDS.verify_and_update(request.password, stored_hash)
    if not user or not verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if replacement_hash:
        _update_password_hash(identifier, replacement_hash)
    LOGGER.info("Account authenticated")
    return AuthResponse(
        access_token=_access_token(user["username"]),
        username=user["username"],
        expires_in=ACCESS_TOKEN_MINUTES * 60,
        message="Signed in successfully",
    )


def _update_password_hash(username: str, password_hash: str) -> None:
    with database_connection() as connection:
        connection.execute(
            "UPDATE users SET password_hash = %s WHERE username = %s"
            if _database_url()
            else "UPDATE users SET password_hash = ? WHERE username = ?",
            (password_hash, username),
        )


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            configured_jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
    except (InvalidTokenError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired sign-in session",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
