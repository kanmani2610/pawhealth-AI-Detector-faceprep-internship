
import multiprocessing
workers = 1
timeout = 120
keepalive = 5
import os
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
accesslog = "-"
errorlog  = "-"
loglevel  = "info"
preload_app = False