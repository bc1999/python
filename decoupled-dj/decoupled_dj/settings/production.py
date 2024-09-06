from .base import *  # noqa

SECURE_SSL_REDIRECT = True # Ensures that every request via HTTP gets redirected to HTTPS
ALLOWED_HOSTS = env.list("ALLOWED_HOSTS") # Drives what hostnames Django will serve
STATIC_ROOT = env("STATIC_ROOT") # Is where Django will look for static files

CSRF_COOKIE_SECURE = True # Securing authentication cookies by disabling transmission of 
# csrftoken over plain HTTP. Thus, only transmits over HTTPS
SESSION_COOKIE_SECURE = True # Securing authentication cookies by disabling transmission of 
# sessionid over plain HTTP. Thus, only transmits over HTTPS


REST_FRAMEWORK = {
    **REST_FRAMEWORK,
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    # change behavior by exposing only JSONRenderer to prevent leak data and expose too many details
}

CORS_ALLOWED_ORIGINS = env.list("CORS_ALLOWED_ORIGINS", default=[])

INSTALLED_APPS = INSTALLED_APPS + ["monitus"]

MIDDLEWARE = MIDDLEWARE + [
    "monitus.middleware.FailedLoginMiddleware",
    "monitus.middleware.Error403EmailsMiddleware",
]

EMAIL_HOST = env("EMAIL_HOST")
EMAIL_PORT = env("EMAIL_PORT")
EMAIL_HOST_USER = env("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = env("EMAIL_HOST_PASSWORD")
EMAIL_USE_TLS = True


ADMINS = [("Your name here", "your@email.here")]  # CHANGE THIS WITH YOUR EMAIL