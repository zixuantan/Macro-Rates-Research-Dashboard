from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from datetime import date


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))


def _context_stub() -> object:
	class _StubContext:
		def __enter__(self):
			return self

		def __exit__(self, exc_type, exc, tb):
			return False

	return _StubContext()


if "dotenv" not in sys.modules:
	sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

if "certifi" not in sys.modules:
	sys.modules["certifi"] = SimpleNamespace(where=lambda: "/tmp/cert.pem")

if "fredapi" not in sys.modules:
	sys.modules["fredapi"] = SimpleNamespace(Fred=object)

if "streamlit" not in sys.modules:
	streamlit_stub = ModuleType("streamlit")
	streamlit_stub.cache_data = lambda *dargs, **dkwargs: (lambda func: func) if dargs and callable(dargs[0]) else (lambda func: func)
	streamlit_stub.cache_resource = lambda *dargs, **dkwargs: (lambda func: func) if dargs and callable(dargs[0]) else (lambda func: func)
	streamlit_stub.session_state = {}
	streamlit_stub.caption = lambda *args, **kwargs: None
	streamlit_stub.info = lambda *args, **kwargs: None
	streamlit_stub.markdown = lambda *args, **kwargs: None
	streamlit_stub.subheader = lambda *args, **kwargs: None
	streamlit_stub.text_area = lambda *args, **kwargs: ""
	streamlit_stub.text_input = lambda *args, **kwargs: ""
	streamlit_stub.selectbox = lambda *args, **kwargs: ""
	streamlit_stub.checkbox = lambda *args, **kwargs: False
	streamlit_stub.button = lambda *args, **kwargs: False
	streamlit_stub.columns = lambda *args, **kwargs: (_context_stub(), _context_stub(), _context_stub())
	streamlit_stub.expander = lambda *args, **kwargs: _context_stub()
	streamlit_stub.form = lambda *args, **kwargs: _context_stub()
	streamlit_stub.form_submit_button = lambda *args, **kwargs: False
	streamlit_stub.date_input = lambda *args, **kwargs: date(2026, 1, 1)
	streamlit_stub.rerun = lambda *args, **kwargs: None
	streamlit_stub.column_config = SimpleNamespace(DateColumn=lambda *args, **kwargs: None)
	components_stub = ModuleType("streamlit.components.v1")
	components_stub.html = lambda *args, **kwargs: None
	components_pkg = ModuleType("streamlit.components")
	components_pkg.v1 = components_stub
	streamlit_stub.components = components_pkg
	sys.modules["streamlit"] = streamlit_stub
	sys.modules["streamlit.components"] = components_pkg
	sys.modules["streamlit.components.v1"] = components_stub

if "plotly" not in sys.modules:
	plotly_pkg = ModuleType("plotly")
	graph_objects = ModuleType("plotly.graph_objects")
	graph_objects.Figure = object
	graph_objects.Scatter = object
	express = ModuleType("plotly.express")
	subplots = ModuleType("plotly.subplots")
	subplots.make_subplots = lambda *args, **kwargs: None
	plotly_pkg.graph_objects = graph_objects
	plotly_pkg.express = express
	plotly_pkg.subplots = subplots
	sys.modules["plotly"] = plotly_pkg
	sys.modules["plotly.graph_objects"] = graph_objects
	sys.modules["plotly.express"] = express
	sys.modules["plotly.subplots"] = subplots
