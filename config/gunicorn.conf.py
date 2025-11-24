# Gunicorn configuration for GCP Cloud Run deployment
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Worker processes
workers = int(os.environ.get('WORKERS', min(max(multiprocessing.cpu_count(), 2), 4)))
worker_class = "uvicorn.workers.UvicornWorker"
threads = 2

# Timeout settings (long for AI analysis tasks)
timeout = 300  # 5 minutes for long-running analysis
graceful_timeout = 120
keepalive = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get('LOG_LEVEL', 'info')

# Process naming
proc_name = "festmoment-api"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = None
# certfile = None

# Restart workers periodically to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Preload app for better startup performance
preload_app = True

def on_starting(server):
    print("FestMoment API Server starting...")

def on_exit(server):
    print("FestMoment API Server shutting down...")
