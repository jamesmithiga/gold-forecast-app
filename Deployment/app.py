"""Compatibility Streamlit entrypoint.

Some deployment targets default to app.py. This module forwards to the
actual dashboard script.
"""

from streamlit_dashboard import *  # noqa: F401,F403
