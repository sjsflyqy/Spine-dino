from app import create_app
from app.config import HOST, PORT, get_ssl_context


app = create_app()


if __name__ == "__main__":
    ssl_ctx = get_ssl_context()
    proto = "https" if ssl_ctx else "http"
    print(f"Starting server: {proto}://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, ssl_context=ssl_ctx)
