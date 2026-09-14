import sys
import os
import signal
import time
from flask import Flask, request
import shlex

# Import config first (parses args and sets up device)
from configurations import config

# Set up logging BEFORE importing detector
from utils import Tee

os.makedirs(config.results_path, exist_ok=True)

if config.checkpoint_round is not None:
    # resume logging instead of overwriting
    log_file = open(os.path.join(config.results_path, 'app.log'), 'a')
else:
    log_file = open(os.path.join(config.results_path, 'app.log'), 'w')
    
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# Import detector and routes
from detector import SLDetector
from routes import api_bp, views_bp
from routes.api import init_api
from routes.views import init_views

# Create Flask app
app = Flask(__name__)

invoked_command = shlex.join([sys.executable, *sys.argv])
print("Invoked command:")
print(invoked_command)

sl_detector = SLDetector()

# Initialize routes with detector
init_api(sl_detector)
init_views(sl_detector)

# Register blueprints
app.register_blueprint(api_bp)
app.register_blueprint(views_bp)


@app.after_request
def _no_store_api_responses(response):
    """Prevent stale JSON from browser/CDN caches (breaks scoring poll / status)."""
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
    return response


_last_sigint_time = 0.0


def install_signal_guard():
    """Ignore sporadic SIGINT once; require a second one to stop."""
    global _last_sigint_time

    if os.environ.get('JWST_IGNORE_SINGLE_SIGINT', '1') != '1':
        return

    def _sigint_handler(signum, frame):
        del signum, frame  # unused
        global _last_sigint_time
        now = time.monotonic()
        if now - _last_sigint_time <= 2.0:
            print('Second SIGINT received, shutting down.')
            signal.signal(signal.SIGINT, signal.default_int_handler)
            signal.raise_signal(signal.SIGINT)
            return
        _last_sigint_time = now
        print('SIGINT received and ignored once. Press Ctrl+C again within 2s to stop.')

    signal.signal(signal.SIGINT, _sigint_handler)


if __name__ == '__main__':
    install_signal_guard()
    app.run(debug=False, host=config.host, port=config.port, threaded=True)
    log_file.close()
