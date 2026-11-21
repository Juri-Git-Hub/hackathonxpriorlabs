"""FastAPI WebSocket server and judge control handler."""

def create_app(*args, **kwargs):
    # Keep python -m server.app from pre-importing its own entry point.
    from server.app import create_app as factory
    return factory(*args, **kwargs)

__all__ = ["create_app"]
