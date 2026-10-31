import tomllib
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    ("package", "remediated"),
    [
        ("certifi", "2026.10.23"),
        ("cryptography", "50.0.2"),
        ("pyjwt", "2.15.1"),
        ("urllib3", "2.8.0"),
    ],
)
def test_action_runtime_rejects_vulnerable_preinstalled_transitive_packages(
    package: str, remediated: str
) -> None:
    runtime = Path("requirements.txt").read_text().casefold().splitlines()
    locked = {
        entry["name"]: entry["version"]
        for entry in tomllib.loads(Path("uv.lock").read_text())["package"]
    }
    assert f"{package}>={remediated}" in runtime
    assert locked[package] == remediated


def test_ci_trivy_scan_is_pinned_and_blocks_high_severity_findings() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["security-scan"]
    checkout, scan = job["steps"]

    assert workflow["permissions"] == {"contents": "read"}
    assert checkout["with"]["persist-credentials"] is False
    source, revision = scan["uses"].split("@")
    assert source == "aquasecurity/trivy-action"
    assert len(revision) == 40 and all(char in "0123456789abcdef" for char in revision)
    assert scan["with"]["version"] == "v0.74.0"
    assert scan["with"]["scan-type"] == "fs"
    assert scan["with"]["scan-ref"] == "."
    assert set(scan["with"]["scanners"].split(",")) == {"vuln", "misconfig", "secret"}
    assert scan["with"]["severity"] == "HIGH,CRITICAL"
    assert scan["with"]["exit-code"] == "1"
    assert scan["env"]["TRIVY_INCLUDE_DEV_DEPS"] == "true"
    assert scan.get("continue-on-error", False) is False
    assert job.get("continue-on-error", False) is False
