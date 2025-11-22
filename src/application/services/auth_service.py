"""
Authentication service with Access Token + Refresh Token + Google OAuth
PostgreSQL 버전
"""

import jwt
import hashlib
import secrets
import os
import httpx
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from fastapi import HTTPException, Header
from src.infrastructure.persistence.database import get_db_connection, release_connection, get_cursor
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
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    # Store in database
    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute(
            """
            INSERT INTO refresh_tokens (user_id, token_hash, expires_at)
            VALUES (%s, %s, %s)
            """,
            (user_id, token_hash, expires_at),
        )
        conn.commit()
    finally:
        cursor.close()
        release_connection(conn)

    return token, expires_at


def verify_refresh_token(token: str) -> Optional[int]:
    """Verify refresh token and return user_id if valid"""
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        cursor.execute(
            """
            SELECT user_id, expires_at, is_revoked
            FROM refresh_tokens
            WHERE token_hash = %s
            """,
            (token_hash,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        user_id = row["user_id"]
        expires_at = row["expires_at"]
        is_revoked = row["is_revoked"]

        # Check if revoked
        if is_revoked:
            return None

        # Check if expired
        if datetime.utcnow() > expires_at:
            return None

        return user_id
    finally:
        cursor.close()
        release_connection(conn)


def revoke_refresh_token(token: str) -> bool:
    """Revoke a refresh token (logout)"""
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute(
            "UPDATE refresh_tokens SET is_revoked = TRUE WHERE token_hash = %s",
            (token_hash,),
        )
        conn.commit()
        rows_affected = cursor.rowcount
        return rows_affected > 0
    finally:
        cursor.close()
        release_connection(conn)


def revoke_all_user_tokens(user_id: int) -> bool:
    """Revoke all refresh tokens for a user (logout all devices)"""
    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute(
            "UPDATE refresh_tokens SET is_revoked = TRUE WHERE user_id = %s AND is_revoked = FALSE",
            (user_id,),
        )
        conn.commit()
        rows_affected = cursor.rowcount
        return rows_affected > 0
    finally:
        cursor.close()
        release_connection(conn)


def cleanup_expired_tokens():
    """Clean up expired and revoked tokens (run periodically)"""
    conn = get_db_connection()
    cursor = get_cursor(conn)
    try:
        cursor.execute(
            """
            DELETE FROM refresh_tokens
            WHERE is_revoked = TRUE OR expires_at < CURRENT_TIMESTAMP
            """
        )
        conn.commit()
        deleted_count = cursor.rowcount
        return deleted_count
    finally:
        cursor.close()
        release_connection(conn)


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
    cursor = get_cursor(conn)

    try:
        oauth_id = google_user_info["oauth_id"]
        email = google_user_info["email"]
        name = google_user_info.get("name", email.split("@")[0])
        picture = google_user_info.get("picture")

        # Check if user exists by google_id
        cursor.execute(
            """
            SELECT id, username, email, full_name, role, profile_picture
            FROM users
            WHERE google_id = %s AND auth_provider = 'google'
            """,
            (oauth_id,)
        )
        row = cursor.fetchone()

        if row:
            # Update last login and picture
            user_id = row["id"]
            cursor.execute(
                """
                UPDATE users
                SET last_login = CURRENT_TIMESTAMP, profile_picture = %s
                WHERE id = %s
                """,
                (picture, user_id)
            )
            conn.commit()

            return {
                "id": row["id"],
                "username": row["username"],
                "email": row["email"],
                "full_name": row["full_name"],
                "role": row["role"],
                "profile_picture": picture,
                "auth_provider": "google"
            }

        # Check if email already exists with local account
        cursor.execute(
            "SELECT id FROM users WHERE email = %s AND auth_provider = 'local'",
            (email,)
        )
        if cursor.fetchone():
            raise HTTPException(
                status_code=400,
                detail="이메일이 이미 일반 계정으로 등록되어 있습니다. 일반 로그인을 사용해주세요."
            )

        # Create new Google user
        username = email.split("@")[0] + "_google"
        # Make username unique
        cursor.execute("SELECT COUNT(*) as cnt FROM users WHERE username LIKE %s", (f"{username}%",))
        result = cursor.fetchone()
        count = result["cnt"] if result else 0
        if count > 0:
            username = f"{username}{count + 1}"

        cursor.execute(
            """
            INSERT INTO users (username, email, password_hash, full_name, role, auth_provider, google_id, profile_picture)
            VALUES (%s, %s, %s, %s, 'user', 'google', %s, %s)
            RETURNING id
            """,
            (username, email, "", name, oauth_id, picture)
        )
        result = cursor.fetchone()
        conn.commit()
        user_id = result["id"]

        return {
            "id": user_id,
            "username": username,
            "email": email,
            "full_name": name,
            "role": "user",
            "profile_picture": picture,
            "auth_provider": "google"
        }
    finally:
        cursor.close()
        release_connection(conn)


def delete_user_account(user_id: int) -> bool:
    """
    Delete user account and all associated data (CASCADE)
    Works for both local and Google OAuth accounts
    """
    conn = get_db_connection()
    cursor = get_cursor(conn)

    try:
        # Get user info first
        cursor.execute("SELECT username, auth_provider FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            return False

        username = user["username"]
        auth_provider = user["auth_provider"]

        # Delete all associated data
        # 1. Revoke all refresh tokens
        cursor.execute("DELETE FROM refresh_tokens WHERE user_id = %s", (user_id,))

        # 2. Delete Q&A answers
        cursor.execute("DELETE FROM answers WHERE user_id = %s", (user_id,))

        # 3. Delete Q&A questions
        cursor.execute("DELETE FROM questions WHERE user_id = %s", (user_id,))

        # 4. Delete feedback (set user_id to NULL to keep analytics data)
        cursor.execute("UPDATE feedback SET user_id = NULL WHERE user_id = %s", (user_id,))

        # 5. Delete feature ratings (set user_id to NULL)
        cursor.execute("UPDATE feature_ratings SET user_id = NULL WHERE user_id = %s", (user_id,))

        # 6. Finally, delete the user
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))

        conn.commit()

        print(f"[OK] Deleted {auth_provider} account: {username} (ID: {user_id})")
        return True

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to delete user account: {e}")
        return False
    finally:
        cursor.close()
        release_connection(conn)
