from app import app
from asgiref.wsgi import WsgiToAsgi

# Wrap with ASGI handler for Vercel
handler = WsgiToAsgi(app)
