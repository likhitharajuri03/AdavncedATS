from prometheus_flask_exporter import PrometheusMetrics
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_monitoring(app):
    # Setup Prometheus metrics
    metrics = PrometheusMetrics(app)
    
    # Setup basic metrics
    metrics.info('app_info', 'Application info', version='1.0.0')
    
    # Setup custom metrics
    by_path_counter = metrics.counter(
        'by_path_counter', 'Request count by request paths',
        labels={'path': lambda: request.path}
    )
    
    # Setup Sentry for error tracking
    sentry_sdk.init(
        dsn=os.getenv('SENTRY_DSN'),
        integrations=[FlaskIntegration()],
        traces_sample_rate=1.0,
        environment=os.getenv('FLASK_ENV', 'production')
    )
    
    # Setup logging
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    file_handler = RotatingFileHandler(
        'logs/app.log', 
        maxBytes=10240, 
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    
    app.logger.setLevel(logging.INFO)
    app.logger.info('ATS startup')
    
    return metrics

def log_event(category, action, label=None, value=None):
    """
    Log custom events for analytics
    """
    app.logger.info(f"Event: {category} - {action} - {label} - {value}")

# Custom metrics tracking
from functools import wraps
from time import time

def track_time(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        
        # Log execution time
        app.logger.info(f"{f.__name__} took {end-start:.2f} seconds")
        return result
    return wrap