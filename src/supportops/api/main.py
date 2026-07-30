"""ASGI application entry point."""

from supportops.api.application import create_application

app = create_application()
