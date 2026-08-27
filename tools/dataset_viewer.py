"""
Visualizzatore web per i dataset di camminate a N passi.

    python tools/dataset_viewer.py                       # dataset_ris + dataset_risv2
    python tools/dataset_viewer.py dataset_ris --port 8123
    python tools/dataset_viewer.py dataset_ris --host 0.0.0.0   # accesso da remoto

Poi apri http://localhost:8000 nel browser.
Nessuna dipendenza oltre a numpy: usa solo la stdlib (http.server).
"""
import argparse
import io
import json
import mimetypes
import os
import posixpath
import re
import socketserver
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_index import build_index, compute_stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, 'dataset_viewer.html')

# Popolato all'avvio: {nome_dataset: {'root':..., 'samples':[...], 'stats':{...}}}
DATASETS = {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *a):  # meno rumore in console
        pass

    # ---------- routing ----------
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ('/', '/index.html'):
                self._send_file(INDEX_HTML, 'text/html; charset=utf-8')
            elif path == '/api/datasets':
                self._send_json({
                    name: {'meta': d['stats']['meta'], 'stats': d['stats']}
                    for name, d in DATASETS.items()
                })
            elif path == '/api/samples':
                name = urlparse(self.path).query.split('ds=')[-1] or next(iter(DATASETS))
                name = unquote(name)
                if name not in DATASETS:
                    self.send_error(404, 'dataset sconosciuto')
                    return
                self._send_json(DATASETS[name]['samples'])
            elif path.startswith('/media/'):
                self._send_media(path[len('/media/'):])
            else:
                self.send_error(404)
        except BrokenPipeError:
            pass  # il browser ha chiuso la connessione (normale sui video)
        except ConnectionResetError:
            pass

    # ---------- helper ----------
    def _send_json(self, obj):
        body = json.dumps(obj, default=str).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath, ctype):
        if not os.path.exists(filepath):
            self.send_error(404, f'manca {os.path.basename(filepath)}')
            return
        with open(filepath, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def _resolve_media(self, rel):
        """/media/<dataset>/<path relativo alla root del dataset>, con
        protezione contro il path traversal."""
        rel = unquote(rel)
        parts = [p for p in posixpath.normpath(rel).split('/') if p not in ('', '.')]
        if not parts or '..' in parts:
            return None
        name, rest = parts[0], parts[1:]
        if name not in DATASETS or not rest:
            return None
        root = os.path.realpath(DATASETS[name]['root'])
        target = os.path.realpath(os.path.join(root, *rest))
        if os.path.commonpath([root, target]) != root or not os.path.isfile(target):
            return None
        return target

    def _send_media(self, rel):
        """Serve i file media con supporto alle Range request: senza di esse
        Chrome/Firefox non riescono a fare seek nei <video>."""
        target = self._resolve_media(rel)
        if target is None:
            self.send_error(404)
            return

        ctype = mimetypes.guess_type(target)[0] or 'application/octet-stream'
        size = os.path.getsize(target)
        rng = self.headers.get('Range')
        start, end = 0, size - 1
        partial = False

        if rng:
            m = re.match(r'bytes=(\d*)-(\d*)', rng.strip())
            if m:
                s, e = m.group(1), m.group(2)
                if s:
                    start, end = int(s), (int(e) if e else size - 1)
                elif e:  # suffisso: ultimi N byte
                    start = max(0, size - int(e))
                end = min(end, size - 1)
                partial = start <= end

        if not partial:
            start, end = 0, size - 1

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', ctype)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(length))
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()

        with open(target, 'rb') as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', nargs='*', default=['dataset_ris', 'dataset_risv2'],
                    help='cartelle dataset da caricare')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1',
                    help="usa 0.0.0.0 per esporre il viewer sulla rete")
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()

    for root in args.roots:
        if not os.path.isdir(root):
            print(f'[SKIP] {root}: non esiste')
            continue
        print(f'Indicizzo {root} ...', flush=True)
        samples, meta = build_index(root)
        DATASETS[os.path.basename(os.path.normpath(root))] = {
            'root': os.path.abspath(root),
            'samples': samples,
            'stats': compute_stats(samples, meta),
        }
        print(f'  {len(samples)} clip')

    if not DATASETS:
        sys.exit('Nessun dataset caricato.')

    url = f'http://{"localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host}:{args.port}'
    with Server((args.host, args.port), Handler) as httpd:
        print(f'\nViewer attivo su {url}   (Ctrl-C per fermare)')
        if not args.no_browser:
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nChiuso.')


if __name__ == '__main__':
    main()
