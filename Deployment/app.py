"""Streamlit application entrypoint for local and cloud deployment.

This bootstrap makes ``app.py`` the single entrypoint by:
- normalizing the working directory to ``Deployment/``
- starting the FastAPI backend in-process when needed
- exposing the backend URL through Streamlit session state
- executing the dashboard script as the root Streamlit page
"""

from __future__ import annotations

import os
import runpy
import socket
import sys
import threading
import time
from pathlib import Path

import streamlit as st
import uvicorn


BASE_DIR = Path(__file__).resolve().parent
BACKEND_HOST = os.getenv("API_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("PORT_API", os.getenv("API_PORT", "8000")))
BACKEND_URL = os.getenv("API_BASE_URL", f"http://{BACKEND_HOST}:{BACKEND_PORT}")
API_KEY = os.getenv("API_KEY", "xgold-forecast-key-2026-3b7f8a9c2d1e5f6g")
BACKEND_THREAD_KEY = "_fastapi_backend_thread_started"


def _prepare_environment() -> None:
	os.chdir(BASE_DIR)
	base_dir_str = str(BASE_DIR)
	if base_dir_str not in sys.path:
		sys.path.insert(0, base_dir_str)


def _is_port_open(host: str, port: int) -> bool:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
		client.settimeout(0.5)
		return client.connect_ex((host, port)) == 0


def _run_backend() -> None:
	from utils.fastapi_backend import app as fastapi_app

	config = uvicorn.Config(
		fastapi_app,
		host=BACKEND_HOST,
		port=BACKEND_PORT,
		log_level="info",
		reload=False,
	)
	server = uvicorn.Server(config)
	server.run()


def _ensure_backend_running() -> None:
	if _is_port_open(BACKEND_HOST, BACKEND_PORT):
		return

	if not st.session_state.get(BACKEND_THREAD_KEY):
		backend_thread = threading.Thread(target=_run_backend, name="fastapi-backend", daemon=True)
		backend_thread.start()
		st.session_state[BACKEND_THREAD_KEY] = True

	for _ in range(20):
		if _is_port_open(BACKEND_HOST, BACKEND_PORT):
			return
		time.sleep(0.25)

	st.warning(
		"FastAPI backend did not start in time. "
		"Pages that depend on API calls may show connection errors."
	)


def main() -> None:
	_prepare_environment()
	st.session_state.setdefault("api_base_url", BACKEND_URL)
	st.session_state.setdefault("api_key", API_KEY)
	_ensure_backend_running()
	runpy.run_path(str(BASE_DIR / "streamlit_dashboard.py"), run_name="__main__")


main()
