#!/usr/bin/env python3
"""
TrustLoop local server — serves static files, trace API, and forensic chat.

Usage:
    # Without traces (original behavior)
    python3 trustloop_server.py [port]

    # With trace data (enables forensic queries)
    python3 trustloop_server.py --traces /tmp/rrma_traces_test.jsonl [port]

Requires: pip install anthropic
API key: set ANTHROPIC_API_KEY env var or enter in browser settings
"""

import json
import sys
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ── Args ──────────────────────────────────────────────────────────────────────
traces_path = None
port = 8765
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--traces" and i < len(sys.argv) - 1:
        traces_path = sys.argv[i + 1]
    elif arg.isdigit():
        port = int(arg)

PORT = port
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Anthropic SDK ─────────────────────────────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: 'anthropic' package not found. Run: pip install anthropic")

ENV_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
ALLOWED_ORIGIN = 'http://localhost:' + str(PORT)

# ── Trace store ───────────────────────────────────────────────────────────────
TRACE_STORE = None
if traces_path:
    sys.path.insert(0, os.path.join(BASE_DIR, 'tools'))
    from trace_forensics import TraceStore, forensic_query, handle_api_request, TOOL_DEFINITIONS, SYSTEM_PROMPT_TEMPLATE
    TRACE_STORE = TraceStore(traces_path)
    print(f"Loaded traces: {len(TRACE_STORE.traces)} traces, {len(TRACE_STORE.traces_by_agent)} agents")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ('200', '304'):
            super().log_message(fmt, *args)

    def _cors_headers(self):
        origin = self.headers.get('Origin', '')
        if origin.startswith('http://localhost:') or origin.startswith('http://127.0.0.1:'):
            self.send_header('Access-Control-Allow-Origin', origin)
        else:
            self.send_header('Access-Control-Allow-Origin', ALLOWED_ORIGIN)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        # Trace API endpoints
        if path.startswith('/api/traces/') and TRACE_STORE:
            result = handle_api_request(TRACE_STORE, path, params)
            self._json_response(result)
            return

        # Status endpoint
        if path == '/api/status':
            self._json_response({
                'traces_loaded': TRACE_STORE is not None,
                'trace_count': len(TRACE_STORE.traces) if TRACE_STORE else 0,
                'agent_count': len(TRACE_STORE.traces_by_agent) if TRACE_STORE else 0,
                'anthropic_available': ANTHROPIC_AVAILABLE,
                'api_key_set': bool(ENV_API_KEY),
            })
            return

        # Fall through to static file serving
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/chat':
            self._handle_chat()
        elif path == '/api/forensic' and TRACE_STORE:
            self._handle_forensic()
        else:
            self.send_error(404)

    def _handle_chat(self):
        data = self._read_json()
        if data is None:
            return

        prompt = data.get('prompt', '').strip()
        system = data.get('system', '').strip()
        api_key = data.get('api_key', '').strip() or ENV_API_KEY

        if not prompt:
            self._json_response({'error': 'No prompt'}, 400)
            return
        if not ANTHROPIC_AVAILABLE:
            self._json_response({'error': 'anthropic package not installed'}, 500)
            return
        if not api_key:
            self._json_response({'error': 'No API key. Set ANTHROPIC_API_KEY or enter in Settings.'}, 401)
            return

        # If traces are loaded, inject index into system prompt
        if TRACE_STORE and not system:
            index = TRACE_STORE.build_index()
            system = (
                "You are a forensic analyst for multi-agent AI research systems. "
                "You have the following trace index available. When answering, "
                "reference specific agents, steps, and artifacts.\n\n" + index
            )

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

    def _handle_forensic(self):
        """Full forensic query with tool-use loop."""
        data = self._read_json()
        if data is None:
            return

        question = data.get('question', '').strip()
        api_key = data.get('api_key', '').strip() or ENV_API_KEY

        if not question:
            self._json_response({'error': 'No question'}, 400)
            return
        if not ANTHROPIC_AVAILABLE:
            self._json_response({'error': 'anthropic package not installed'}, 500)
            return
        if not api_key:
            self._json_response({'error': 'No API key'}, 401)
            return

        # Set API key for the forensic query
        os.environ['ANTHROPIC_API_KEY'] = api_key

        try:
            answer = forensic_query(TRACE_STORE, question, verbose=True)
            self._json_response({'answer': answer})
        except Exception as e:
            self._json_response({'error': str(e)}, 500)

    def _read_json(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            return json.loads(body)
        except Exception:
            self.send_error(400, 'Bad JSON')
            return None

    def _json_response(self, data, status=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    os.chdir(BASE_DIR)
    httpd = HTTPServer(('127.0.0.1', PORT), Handler)
    print(f'TrustLoop: http://localhost:{PORT}/trustloop_viewer.html')
    if TRACE_STORE:
        print(f'Traces: {len(TRACE_STORE.traces)} loaded from {traces_path}')
        print(f'  Forensic API: POST /api/forensic  (tool-use loop)')
        print(f'  Trace API:    GET  /api/traces/*   (direct queries)')
    print(f'API key: {"set via ANTHROPIC_API_KEY" if ENV_API_KEY else "enter in browser Settings"}')
    print('Ctrl+C to stop')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
