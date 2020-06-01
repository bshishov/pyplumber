from typing import Dict, Any, Callable, Optional, Tuple, List
import threading
import logging
import re
import json
import mimetypes
import urllib.parse
import os
from wsgiref import simple_server, util
import traceback
import tracemalloc
import pickle

import io

import jinja2

try:
    import objgraph
    import graphviz
    OBJGRAPH = True
except ImportError:
    OBJGRAPH = False

from pyplumber.tracemalloc_utils import *
from pyplumber.gc_utils import *


__all__ = ['DEFAULT_PORT', 'run_wsgi_server', 'PyPlumberApp']

DEFAULT_PORT = 9123
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(MODULE_DIR, 'media')
STATISTICS_DEFAULT_TOP_N = 50

logger = logging.getLogger()


class Request:
    def __init__(self, env):
        self.env = env
        self.qs = urllib.parse.parse_qs(self.query_string)

    @property
    def method(self) -> str:
        return self.env['REQUEST_METHOD']

    @property
    def path(self) -> str:
        return self.env['PATH_INFO']

    @property
    def query_string(self) -> str:
        return self.env['QUERY_STRING']

    def get_arg(self, key, default=None) -> Optional[str]:
        values = self.qs.get(key)
        if values:
            return values[0]
        return default

    def get_args(self, key, default=None) -> List[str]:
        return self.qs.get(key) or []


class Response:
    def __init__(self, data, status: Optional[str] = None, headers=None):
        self.status = status or '200 OK'
        self.headers = headers or []
        self.data = data

    @classmethod
    def file(cls, fp, name: str, status=None, headers=None, attachment: bool = False) -> 'Response':
        h = {'Content-Type': mimetypes.guess_type(name)[0] or 'application/octet-stream'}

        if attachment:
            h['Content-Disposition'] = f'attachment; filename="{name}"'

        if headers:
            h.update(dict(headers))
        return Response(
            data=util.FileWrapper(fp),
            status=status,
            headers=list(h.items())
        )

    @classmethod
    def json(cls, obj, status=None, headers=None) -> 'Response':
        serialized = json.dumps(obj, indent=4, default=repr)
        return Response(
            data=[serialized.encode('utf-8')],
            status=status,
            headers=headers or [('Content-Type', 'application/json')]
        )

    @classmethod
    def text(cls, text: str, status=None, headers=None) -> 'Response':
        return Response(
            data=[text.encode('utf-8')],
            status=status,
            headers=headers or [('Content-Type', 'text/plain')]
        )

    @classmethod
    def html(cls, html: str, status=None, headers=None) -> 'Response':
        return Response(
            data=[html.encode('utf-8')],
            status=status,
            headers=headers or [('Content-Type', 'text/html')]
        )

    @classmethod
    def redirect(cls, location: str):
        logger.info(f'Redirecting to {location}')
        return Response(b'', status='302 Found', headers=[('Location', location)])


def route(path, methods=('GET', 'POST')):
    def _wrapper(fn):
        fn.route_info = {
            'path': path,
            'path_re': re.compile(path),
            'methods': methods
        }
        return fn
    return _wrapper


class PyPlumberApp:
    def __init__(self):
        self.routes = []
        self.manager = MemoryManager()
        for attr in dir(self):
            obj = getattr(self, attr)
            if hasattr(obj, 'route_info'):
                self.routes.append({
                    **getattr(obj, 'route_info'),
                    **{'handler': obj}
                })

        self.template_env = jinja2.Environment(
            loader=jinja2.PackageLoader('pyplumber', 'templates'),
        )

    def template_context(self) -> Dict[str, Any]:
        traced_memory_current, traced_memory_peak = self.manager.get_traced_memory()
        return {
            'snapshots': self.manager.snapshots,
            'is_tracing': self.manager.is_tracing(),
            'traced_memory_peak': traced_memory_peak,
            'traced_memory_current': traced_memory_current,
            'tracemalloc_memory': self.manager.get_tracemalloc_memory()
        }

    def render_template(self, path: str, *args, **kwargs):
        self.template_env.globals.update(self.template_context())
        template = self.template_env.get_template(path)
        rendered = template.render(*args, **kwargs)
        return Response.html(rendered)

    @route('^/$')
    def index(self, request: Request) -> Response:
        return Response.redirect('/snapshots')

    @route('^/snapshots$')
    def snapshots(self, request: Request) -> Response:
        return self.render_template('tracemalloc/snapshots.html')

    @route('^/env$')
    def env(self, request: Request) -> Response:
        return Response.json(request.env)

    @route('^/api/routes$')
    def routes(self, request: Request) -> Response:
        return Response.json(self.routes)

    @route('^/start_tracing$')
    def start_tracing(self, request: Request) -> Response:
        n_frames = request.get_arg('n_frames')
        n_frames = int(n_frames) if n_frames else None
        self.manager.start_tracing(n_frames)
        return Response.redirect('/snapshots')

    @route('^/stop_tracing$')
    def stop_tracing(self, request: Request) -> Response:
        self.manager.stop_tracing()
        return Response.redirect(f'/snapshots?{request.query_string}')

    @route('^/take_snapshot$')
    def take_snapshot(self, request: Request) -> Response:
        def _qs_to_filter(names: List[str], inclusive: bool) -> List[tracemalloc.Filter]:
            return [tracemalloc.Filter(inclusive, name) for name in names if name]

        include_filters = _qs_to_filter(request.get_args('include'), inclusive=True)
        exclude_filters = _qs_to_filter(request.get_args('exclude'), inclusive=False)
        self.manager.take_snapshot(include_filters + exclude_filters)
        return Response.redirect(f'/snapshots?{request.query_string}')

    @route('^/dump')
    def dump(self, request: Request) -> Response:
        dump = MemoryDump()
        dump.sort_by_size()
        return self.render_template('dump.html', dump=dump)

    @route('^/snapshots/(?P<snapshot_id>[a-z0-9]+)$')
    def snapshot_view(self, request: Request, snapshot_id) -> Response:
        snapshot = self.manager.get_snapshot(snapshot_id)
        if not snapshot:
            raise KeyError(f'No such snapshot: {snapshot_id}')

        stats_key_type = request.get_arg('key_type', 'filename')
        stats_cumulative = bool(request.get_arg('cumulative', False))
        top = int(request.get_arg('top', STATISTICS_DEFAULT_TOP_N))

        stats = snapshot.snapshot.statistics(stats_key_type, stats_cumulative)[:top]

        context = {
            'snapshot': snapshot,
            'stats': stats
        }
        return self.render_template('tracemalloc/snapshot.html', **context)

    @route('^/snapshots/(?P<snapshot_id1>[a-z0-9]+)/diff/(?P<snapshot_id2>[a-z0-9]+)$')
    def snapshots_diff(self, request: Request, snapshot_id1: str, snapshot_id2: str) -> Response:
        new_snapshot: Optional[SnapshotRecord] = self.manager.get_snapshot(snapshot_id1)
        old_snapshot: Optional[SnapshotRecord] = self.manager.get_snapshot(snapshot_id2)
        assert new_snapshot, f'No snapshot with id {snapshot_id1}'
        assert old_snapshot, f'No snapshot with id {snapshot_id2}'

        stats_key_type = request.get_arg('key_type', 'filename')
        stats_cumulative = bool(request.get_arg('cumulative', False))
        top = int(request.get_arg('top', STATISTICS_DEFAULT_TOP_N))

        diff_stats = new_snapshot.snapshot.compare_to(old_snapshot.snapshot,
                                                      stats_key_type,
                                                      stats_cumulative)[:top]

        context = {
            'new_snapshot': new_snapshot,
            'old_snapshot': old_snapshot,
            'diff_stats': diff_stats
        }
        return self.render_template('tracemalloc/diff.html', **context)

    @route('^/snapshots/(?P<snapshot_id>[a-z0-9]+)/delete$')
    def snapshot_delete(self, request: Request, snapshot_id) -> Response:
        self.manager.delete_snapshot(snapshot_id)
        return Response.redirect('/snapshots')

    @route('^/snapshots/clear$')
    def snapshots_clear(self, request: Request, snapshot_id) -> Response:
        self.manager.clear()
        return Response.redirect('/snapshots')

    @route('^/snapshots/(?P<snapshot_id>[a-z0-9]+)/download$')
    def snapshot_download(self, request: Request, snapshot_id) -> Response:
        snapshot_record = self.manager.get_snapshot(snapshot_id)
        assert snapshot_record, f'No such snapshot {snapshot_id}'
        fp = io.BytesIO()
        pickle.dump(snapshot_record, fp)
        fp.seek(0)
        return Response.file(fp, name=f'snapshot_record_{snapshot_id}.pickle', attachment=True)

    @route('^/obj/(?P<obj_id>[0-9]+)$')
    def obj(self, request: Request, obj_id: int) -> Response:
        assert OBJGRAPH, 'Python module graphviz is required'
        obj_id = int(obj_id)
        obj = objgraph.at(obj_id)
        if obj is not None:
            raise KeyError(f'No object with id: {obj_id}')
        context = {
            'obj': obj_info(obj)
        }
        return self.render_template('obj.html', **context)

    @route('^/objgraph/obj/(?P<obj_id>[0-9]+).(?P<fmt>[a-z]+)')
    def objgraph(self, request: Request, obj_id: int, fmt: str) -> Response:
        assert OBJGRAPH, 'Python module graphviz is required'
        assert fmt in {'png', 'pdf'}, 'Only png and pdf are supported'
        obj_id = int(obj_id)
        obj = objgraph.at(obj_id)
        if not obj:
            raise KeyError(f'No object with id: {obj_id}')
        max_depth = int(request.get_arg('depth', 10))

        f = io.StringIO()
        objgraph.show_backrefs(obj, max_depth=max_depth, output=f)
        f.seek(0)
        s = graphviz.Source(f.getvalue())
        filename = f'{obj_id}.{fmt}'
        c_type = mimetypes.guess_type(filename)[0]
        return Response([s.pipe(format=fmt)], status='200 OK', headers=[
            ('Content-Type', c_type)
        ])

    @route('^/media/(?P<file_path>.*)$')
    def _media(self, request: Request, file_path: str) -> Response:
        full_path = os.path.join(MEDIA_DIR, file_path)
        file_name = os.path.basename(full_path)
        logger.debug(f'Opening file: {full_path} (name: {file_name})')
        return Response.file(open(full_path, 'rb'), name=file_name)

    def get_handler(self, request: Request) -> Tuple[Optional[Callable[[Request], Response]], Dict]:
        for r in self.routes:
            match = re.match(r['path'], request.path)
            if match:
                methods = r.get('methods')
                if methods is not None and request.method in methods:
                    return r['handler'], match.groupdict()
        return None, {}

    def handle_request(self, environ, start_response):
        env = {k: v for k, v in environ.items() if not k.startswith('wsgi.')}
        request = Request(env)
        handler, kwargs = self.get_handler(request)
        if not handler:
            response = Response.text('404\nPage not found', '404 Not Found')
        else:
            try:
                response = handler(request, **kwargs)
            except Exception as err:
                response = Response.text(f'500 Internal Server Error\n{err}',
                                         status='500 Internal Server Error')
                traceback.print_exc()
        start_response(response.status, response.headers)
        return response.data

    def __call__(self, environ, start_response):
        return self.handle_request(environ, start_response)


def _wsgi_thread_worker(host, port):
    app = PyPlumberApp()
    with simple_server.make_server(host, port, app) as httpd:
        logging.info(f'Serving PyPlumberApp on http://{host}:{port}')
        httpd.serve_forever()


def run_wsgi_server(port: int = DEFAULT_PORT, host: str = '0.0.0.0'):
    threading.Thread(target=_wsgi_thread_worker, daemon=True, kwargs={
        'host': host,
        'port': port
    }).start()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    run_wsgi_server()
    import time

    class MyClass:
        def __init__(self, children):
            self.data = ['hello'] * 10000
            if children > 0:
                self.child = MyClass(children - 1)
            else:
                self.child = 0

    root = MyClass(4)
    time.sleep(10000)
