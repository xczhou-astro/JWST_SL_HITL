from flask import Blueprint

api_bp = Blueprint('api', __name__, url_prefix='/api')
views_bp = Blueprint('views', __name__)

# Import routes after blueprints are created
from . import api, views
