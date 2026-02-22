#!/usr/bin/env python3
"""
run.py - Bleeper Flask application entry point.

Usage:
    python run.py                     # development (debug mode)
    gunicorn -w 1 -b 0.0.0.0:5000 run:app   # production

Note: Use a single Gunicorn worker (-w 1) because job_status is an
in-process dict.  For multi-worker deployments, replace job_status with
Redis or a similar shared store.
"""

from app import app
from app.bleeper_backend import *  # register all routes

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False,   # reloader conflicts with background threads
    )
