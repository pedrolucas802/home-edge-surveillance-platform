import shutil
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from app.config.hub_config import alert_kind_for, in_quiet_hours, is_alert, load_hub_config, save_hub_config
from app.devices.registry import device_to_dict, load_registry, save_registry

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "configs" / "devices.demo.yaml"
APP = ROOT / "app" / "dashboard" / "streamlit_app.py"


def test_hub_config_defaults_and_round_trip(tmp_path):
    path = tmp_path / "hub_config.yaml"
    config = load_hub_config(path)
    assert config["alerts"]["enabled"] is True
    assert config["alerts"]["kinds"]["tripwire_beam"] is True

    config["user"]["name"] = "Pedro"
    config["alerts"]["kinds"]["radar_presence"] = False
    save_hub_config(path, config)
    loaded = load_hub_config(path)
    assert loaded["user"]["name"] == "Pedro"
    assert loaded["alerts"]["kinds"]["radar_presence"] is False
    # Keys added later still get their defaults.
    assert loaded["alerts"]["kinds"]["camera_detection"] is True


def test_alert_rules():
    config = load_hub_config(Path("/nonexistent/hub_config.yaml"))
    beam = {"device_id": "tw-1", "device_type": "tripwire", "event_type": "beam_broken"}
    assert alert_kind_for("tripwire", "beam_restored") is None
    assert is_alert(beam, config)
    assert not is_alert({**beam, "event_type": "beam_restored"}, config)

    config["alerts"]["muted_devices"] = ["tw-1"]
    assert not is_alert(beam, config)
    config["alerts"]["muted_devices"] = []
    config["alerts"]["kinds"]["tripwire_beam"] = False
    assert not is_alert(beam, config)
    config["alerts"]["kinds"]["tripwire_beam"] = True
    config["alerts"]["enabled"] = False
    assert not is_alert(beam, config)


def test_quiet_hours_cross_midnight():
    quiet = {"enabled": True, "start": "23:00", "end": "06:00"}
    assert in_quiet_hours(quiet, datetime(2026, 1, 1, 23, 30))
    assert in_quiet_hours(quiet, datetime(2026, 1, 1, 5, 59))
    assert not in_quiet_hours(quiet, datetime(2026, 1, 1, 12, 0))
    assert not in_quiet_hours({**quiet, "enabled": False}, datetime(2026, 1, 1, 23, 30))


def test_save_registry_keeps_options_and_validates(tmp_path):
    registry = load_registry(DEMO)
    entries = [device_to_dict(device) for device in registry.devices]
    entries[0]["host"] = "10.0.0.50"
    path = tmp_path / "devices.yaml"
    saved = save_registry(path, registry.site_id, entries)
    assert saved.get("cam-front").option("host") == "10.0.0.50"
    assert load_registry(path).get("radar-porch").option("model") == "RD-03D"

    with pytest.raises(ValueError):
        save_registry(path, registry.site_id, entries + [dict(entries[0])])
    # A rejected save leaves the file as it was.
    assert len(yaml.safe_load(path.read_text())["devices"]) == len(entries)


@pytest.fixture
def demo_env(tmp_path, monkeypatch):
    devices = tmp_path / "devices.yaml"
    shutil.copy(DEMO, devices)
    monkeypatch.setenv("DEVICES_CONFIG", str(devices))
    monkeypatch.setenv("DASHBOARD_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HUB_CONFIG", str(tmp_path / "hub_config.yaml"))
    return tmp_path


def test_app_starts_on_home(demo_env):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    assert not app.exception, app.exception
    assert any("Jump to" in str(block.value) for block in app.markdown)


def _render_page(page: str) -> None:
    import importlib

    importlib.import_module(f"app.dashboard.views.{page}").render()


@pytest.mark.parametrize("page", ["cameras", "radars", "tripwires", "devices", "configuration"])
def test_every_page_renders(demo_env, page):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_render_page, args=(page,), default_timeout=30)
    app.run()
    assert not app.exception, app.exception
