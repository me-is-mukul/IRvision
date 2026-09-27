"""Smoke test: the Streamlit app renders without exceptions (no data needed)."""

import pytest

from irvision.utils.config import PROJECT_ROOT

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def test_app_renders():
    # absolute path: AppTest resolves relative paths from this test file's folder
    app = str(PROJECT_ROOT / "app" / "streamlit_app.py")
    at = streamlit_testing.AppTest.from_file(app, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.startswith("IRVision")
