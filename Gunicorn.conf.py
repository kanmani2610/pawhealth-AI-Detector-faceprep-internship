
import multiprocessing
timeout = 120
keepalive = 5
workers = max(2, multiprocessing.cpu_count())
bind = "0.0.0.0:5000"
accesslog  = "-"
errorlog   = "-"
loglevel   = "info"
preload_app = True