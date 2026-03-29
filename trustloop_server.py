#!/usr/bin/env python3
"""
TrustLoop local server — serves static files + proxies chat via Anthropic SDK.
Uses API key directly — no CLAUDE.md loaded, no filesystem access, no secrets exposed.

Usage: python3 trustloop_server.py [port]
Requires: pip install anthropic
API key: set ANTHROPIC_API_KEY env var or enter in browser settings
"""

import json
import sys
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check for anthropic SDK
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: 'anthropic' package not found. Run: pip install anthropic")

# API key: env var takes priority, browser can also send one
ENV_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

ALLOWED_ORIGIN = 'http://localhost:' + str(PORT)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ('200', '304'):
            super().log_message(fmt, *args)

    def _cors_headers(self):
        # Strict: only allow same localhost origin
        origin = self.headers.get('Origin', '')
        if origin.startswith('http://localhost:') or origin.startswith('http://127.0.0.1:'):
            self.send_header('Access-Control-Allow-Origin', origin)
        else:
            self.send_header('Access-Control-Allow-Origin', ALLOWED_ORIGIN)
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/chat':
            self._handle_chat()
        else:
            self.send_error(404)

    def _handle_chat(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except Exception:
            self.send_error(400, 'Bad JSON')
            return

        prompt = data.get('prompt', '').strip()
        system = data.get('system', '').strip()
        api_key = data.get('api_key', '').strip() or ENV_API_KEY

        if not prompt:
            self._json_response({'error': 'No prompt'}, 400)
            return

        if not ANTHROPIC_AVAILABLE:
            self._json_response({'error': 'anthropic package not installed. Run: pip install anthropic'}, 500)
            return

        if not api_key:
            self._json_response({'error': 'No API key. Set ANTHROPIC_API_KEY env var or enter in Settings.'}, 401)
            return

        try:
            client = anthropic.Anthropic(api_key=api_key)
            kwargs = dict(
                model='claude-sonnet-4-6',
                max_tokens=2048,
                messages=[{'role': 'user', 'content': prompt}],
            )
            if system:
                kwargs['system'] = system
            msg = client.messages.create(**kwargs)
            reply = msg.content[0].text if msg.content else '(empty)'
        except anthropic.AuthenticationError:
            self._json_response({'error': 'Invalid API key.'}, 401)
            return
        except Exception as e:
            self._json_response({'error': str(e)}, 500)
            return

        self._json_response({'text': reply})

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    os.chdir(BASE_DIR)
    httpd = HTTPServer(('127.0.0.1', PORT), Handler)  # 127.0.0.1 only, not 0.0.0.0
    print(f'TrustLoop: http://localhost:{PORT}/trustloop_viewer.html')
    print(f'API key: {"set via ANTHROPIC_API_KEY env" if ENV_API_KEY else "enter in browser Settings"}')
    print(f'No CLAUDE.md loaded — clean context only')
    print('Ctrl+C to stop')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
