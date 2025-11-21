"""
Authentication service with Access Token + Refresh Token + Google OAuth
"""

import jwt
import hashlib
import secrets
import os
import httpx
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from fastapi import HTTPException, Header
from src.infrastructure.persistence.database import get_db_connection
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "festmoment_secret_key_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # 15 minutes (short-lived)
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 7 days (long-lived)

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")


def hash_password(password: str) -> str:
    """Hash a password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(plain_password) == hashed_password


def create_access_token(data: dict) -> str:
    """Create a JWT access token (15 minutes)"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(user_id: int) -> Tuple[str, datetime]:
    """Create a refresh token and store it in database"""
    # Generate random token
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    # Store in database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO refresh_tokens (user_id, token, expires_at)
        VALUES (?, ?, ?)
        """,
        (user_id, token, expires_at),
    )
    conn.commit()
    conn.close()

    return token, expires_at


def verify_refresh_token(token: str) -> Optional[int]:
    """Verify refresh token and return user_id if valid"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT user_id, expires_at, revoked
        FROM refresh_tokens
        WHERE token = ?
        """,
        (token,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    user_id, expires_at, revoked = row

    # Check if revoked
    if revoked:
        return None

    # Check if expired
    expires_at_dt = datetime.fromisoformat(expires_at)
    if datetime.utcnow() > expires_at_dt:
        return None

    return user_id


def revoke_refresh_token(token: str) -> bool:
    """Revoke a refresh token (logout)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE refresh_tokens SET revoked = 1 WHERE token = ?",
        (token,),
    )
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0


def revoke_all_user_tokens(user_id: int) -> bool:
    """Revoke all refresh tokens for a user (logout all devices)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE refresh_tokens SET revoked = 1 WHERE user_id = ? AND revoked = 0",
        (user_id,),
    )
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0


def cleanup_expired_tokens():
    """Clean up expired and revoked tokens (run periodically)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM refresh_tokens
        WHERE revoked = 1 OR datetime(expires_at) < datetime('now')
        """
    )
    conn.commit()
    deleted_count = cursor.rowcount
    conn.close()
    return deleted_count


def decode_access_token(token: str) -> Optional[Dict]:
    """Decode and verify a JWT access token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user_from_token(authorization: Optional[str] = Header(None)) -> Optional[Dict]:
    """Extract user info from Authorization header"""
    if not authorization:
        return None

    try:
        # Format: "Bearer <token>"
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None

        payload = decode_access_token(token)
        return payload
    except:
        return None


def require_auth(authorization: Optional[str] = Header(None)) -> Dict:
    """Require authentication - raise exception if not authenticated"""
    user = get_current_user_from_token(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def require_admin(authorization: Optional[str] = Header(None)) -> Dict:
    """Require admin role - raise exception if not admin"""
    user = require_auth(authorization)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ===== Google OAuth Functions =====

async def verify_google_token(token: str) -> Optional[Dict]:
    """Verify Google ID token and return user info"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": token}
            )

            if response.status_code != 200:
                return None

            data = response.json()

            # Verify the token is for our app
            if data.get("aud") != GOOGLE_CLIENT_ID:
                return None

            return {
                "oauth_id": data.get("sub"),  # Google user ID
                "email": data.get("email"),
                "name": data.get("name"),
                "picture": data.get("picture"),
                "email_verified": data.get("email_verified") == "true"
            }
    except Exception as e:
        print(f"Error verifying Google token: {e}")
        return None


def get_or_create_google_user(google_user_info: Dict) -> Dict:
    """Get existing Google user or create new one"""
    conn = get_db_connection()
    cursor = conn.cursor()

    oauth_id = google_user_info["oauth_id"]
    email = google_user_info["email"]
    name = google_user_info.get("name", email.split("@")[0])
    picture = google_user_info.get("picture")

    # Check if user exists by oauth_id
    cursor.execute(
        """
        SELECT id, username, email, full_name, role, oauth_picture
        FROM users
        WHERE oauth_id = ? AND auth_provider = 'google'
        """,
        (oauth_id,)
    )
    row = cursor.fetchone()

    if row:
        # Update last login and picture
        user_id = row[0]
        cursor.execute(
            """
            UPDATE users
            SET last_login = CURRENT_TIMESTAMP, oauth_picture = ?
            WHERE id = ?
            """,
            (picture, user_id)
        )
        conn.commit()
        conn.close()

        return {
            "id": row[0],
            "username": row[1],
            "email": row[2],
            "full_name": row[3],
            "role": row[4],
            "oauth_picture": picture,
            "auth_provider": "google"
        }

    # Check if email already exists with local account
    cursor.execute(
        "SELECT id FROM users WHERE email = ? AND auth_provider = 'local'",
        (email,)
    )
    if cursor.fetchone():
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="이메일이 이미 일반 계정으로 등록되어 있습니다. 일반 로그인을 사용해주세요."
        )

    # Create new Google user
    username = email.split("@")[0] + "_google"
    # Make username unique
    cursor.execute("SELECT COUNT(*) FROM users WHERE username LIKE ?", (f"{username}%",))
    count = cursor.fetchone()[0]
    if count > 0:
        username = f"{username}{count + 1}"

    cursor.execute(
        """
        INSERT INTO users (username, email, password_hash, full_name, role, auth_provider, oauth_id, oauth_picture)
        VALUES (?, ?, ?, ?, 'user', 'google', ?, ?)
        """,
        (username, email, "", name, oauth_id, picture)
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()

    return {
        "id": user_id,
        "username": username,
        "email": email,
        "full_name": name,
        "role": "user",
        "oauth_picture": picture,
        "auth_provider": "google"
    }


def delete_user_account(user_id: int) -> bool:
    """
    Delete user account and all associated data (CASCADE)
    Works for both local and Google OAuth accounts
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get user info first
        cursor.execute("SELECT username, auth_provider FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            return False

        username, auth_provider = user

        # Delete all associated data
        # 1. Revoke all refresh tokens
        cursor.execute("DELETE FROM refresh_tokens WHERE user_id = ?", (user_id,))

        # 2. Delete Q&A answers
        cursor.execute("DELETE FROM answers WHERE user_id = ?", (user_id,))

        # 3. Delete Q&A questions
        cursor.execute("DELETE FROM questions WHERE user_id = ?", (user_id,))

        # 4. Delete feedback (set user_id to NULL to keep analytics data)
        cursor.execute("UPDATE feedback SET user_id = NULL WHERE user_id = ?", (user_id,))

        # 5. Delete user events (set user_id to NULL to keep analytics data)
        cursor.execute("UPDATE user_events SET user_id = NULL WHERE user_id = ?", (user_id,))

        # 6. Delete feature ratings (set user_id to NULL)
        cursor.execute("UPDATE feature_ratings SET user_id = NULL WHERE user_id = ?", (user_id,))

        # 7. Finally, delete the user
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))

        conn.commit()
        conn.close()

        print(f"[OK] Deleted {auth_provider} account: {username} (ID: {user_id})")
        return True

    except Exception as e:
        conn.rollback()
        conn.close()
        print(f"[ERROR] Failed to delete user account: {e}")
        return False
