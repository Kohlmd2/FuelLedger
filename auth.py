from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from pathlib import Path

import streamlit as st


DATA_DIR = Path(".fuel_profit_data")
DATA_DIR.mkdir(exist_ok=True)

AUTH_DB = DATA_DIR / "auth.db"
REMEMBER_FILE = DATA_DIR / "remember_login.json"


def _auth_conn():
    conn = sqlite3.connect(AUTH_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    with _auth_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _normalize_username(username: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", username.strip().lower())


def _hash_password(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return dk.hex()


def _create_user(username: str, email: str, password: str, is_admin: bool = False) -> tuple[bool, str]:
    username_n = _normalize_username(username)
    if not username_n:
        return False, "Username must be letters, numbers, or underscores."
    if "@" not in email or "." not in email:
        return False, "Enter a valid email."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    salt_hex = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt_hex)
    try:
        with _auth_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, salt, is_admin, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (username_n, email.strip().lower(), pw_hash, salt_hex, 1 if is_admin else 0),
            )
            conn.commit()
        return True, "User created."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def _get_user_by_login(login: str):
    login_l = login.strip().lower()
    with _auth_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (login_l, login_l),
        ).fetchone()
    return row


def _verify_user(login: str, password: str):
    row = _get_user_by_login(login)
    if not row:
        return None
    expected = row["password_hash"]
    salt = row["salt"]
    actual = _hash_password(password, salt)
    if hmac.compare_digest(expected, actual):
        return row
    return None


def _admin_exists() -> bool:
    with _auth_conn() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE is_admin = 1 LIMIT 1").fetchone()
    return row is not None


def _load_remembered_user():
    if not REMEMBER_FILE.exists():
        return None
    try:
        data = json.loads(REMEMBER_FILE.read_text())
    except Exception:
        return None
    user_id = data.get("user_id")
    if not user_id:
        return None
    with _auth_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return row


def _save_remembered_user(user_id: int) -> None:
    REMEMBER_FILE.write_text(json.dumps({"user_id": int(user_id)}))


def _clear_remembered_user() -> None:
    if REMEMBER_FILE.exists():
        REMEMBER_FILE.unlink()


def require_login() -> None:
    init_auth_db()

    if st.session_state.get("user_id"):
        return

    remembered = _load_remembered_user()
    if remembered:
        st.session_state["user_id"] = int(remembered["id"])
        st.session_state["username"] = remembered["username"]
        st.session_state["is_admin"] = bool(remembered["is_admin"])
        return

    if not _admin_exists():
        st.warning("Set up your first admin account to secure this app.")
        with st.form("admin_setup"):
            admin_user = st.text_input("Admin username")
            admin_email = st.text_input("Admin email")
            admin_pw = st.text_input("Admin password", type="password")
            admin_pw2 = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create admin")
        if submitted:
            if admin_pw != admin_pw2:
                st.error("Passwords do not match.")
            else:
                ok, msg = _create_user(admin_user, admin_email, admin_pw, is_admin=True)
                if ok:
                    st.success("Admin created. Please log in.")
                    st.rerun()
                else:
                    st.error(msg)
        st.stop()

    st.markdown(
        """
        <div style="height: 20vh;"></div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.subheader("Login")
        with st.form("login_form"):
            login = st.text_input("Username or email")
            password = st.text_input("Password", type="password")
            remember_me = st.checkbox("Remember me on this device", value=True)
            st.caption("Use remember-me only on a trusted/private device.")
            submitted = st.form_submit_button("Log in")
    if submitted:
        row = _verify_user(login, password)
        if row:
            st.session_state["user_id"] = int(row["id"])
            st.session_state["username"] = row["username"]
            st.session_state["is_admin"] = bool(row["is_admin"])
            if remember_me:
                _save_remembered_user(int(row["id"]))
            for k in [
                "pricebook_df",
                "pricebook_loaded_at",
                "fixed_costs_df",
                "fixed_costs_month",
            ]:
                st.session_state.pop(k, None)
            st.rerun()
        else:
            st.error("Invalid login.")

    st.stop()
