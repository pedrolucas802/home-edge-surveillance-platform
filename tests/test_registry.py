from pathlib import Path

import pytest

from app.devices.registry import load_registry, parse_registry
from app.devices.types import CAMERA, RADAR, TRIPWIRE, get_device_type

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "devices.example.yaml"


def test_example_config_loads_n_devices_per_type():
    registry = load_registry(EXAMPLE)
    grouped = registry.by_type()
    assert registry.site_id == "home"
    assert len(grouped[CAMERA]) == 2
    assert len(grouped[RADAR]) == 2
    assert len(grouped[TRIPWIRE]) == 3
    assert registry.camera_hosts() == {"cam1": "192.168.0.90", "cam2": "192.168.0.89"}


def test_missing_file_falls_back_to_camera_hosts(tmp_path):
    registry = load_registry(tmp_path / "nope.yaml", fallback_camera_hosts={"cam1": "10.0.0.2"})
    assert [device.id for device in registry.of_type(CAMERA)] == ["cam1"]
    assert registry.source == "environment"


def test_unknown_type_is_generic_mqtt_device():
    registry = parse_registry(
        {"site": "home", "devices": [{"id": "door-1", "type": "door_contact", "pin": 4}]}
    )
    device = registry.get("door-1")
    assert device is not None
    assert device.device_type.transport == "mqtt"
    assert device.device_type.label == "Door Contact"
    assert device.option("pin") == 4
    assert get_device_type("RADAR").key == RADAR


@pytest.mark.parametrize(
    "devices",
    [
        [{"id": "a", "type": "radar"}, {"id": "a", "type": "tripwire"}],
        [{"id": "Bad Id", "type": "radar"}],
        [{"id": "system", "type": "radar"}],
        [{"id": "x"}],
    ],
)
def test_invalid_devices_are_rejected(devices):
    with pytest.raises(ValueError):
        parse_registry({"site": "home", "devices": devices})


def test_disabled_devices_are_skipped():
    registry = parse_registry(
        {"devices": [{"id": "r1", "type": "radar", "enabled": False}, {"id": "r2", "type": "radar"}]}
    )
    assert [device.id for device in registry.of_type(RADAR)] == ["r2"]
