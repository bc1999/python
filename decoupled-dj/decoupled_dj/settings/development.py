from .base import * #noqa
INSTALLED_APPS = INSTALLED_APPS + ["django_extensions"]

export DJANGO_SETTINGS_MODULE=decoupled_dj.settings.development

