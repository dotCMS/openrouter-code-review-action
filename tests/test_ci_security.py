from pathlib import Path

import yaml


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
    assert scan.get("continue-on-error", False) is False
    assert job.get("continue-on-error", False) is False
