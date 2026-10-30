"""Tests for the release workflows.

These execute the real workflow scripts (the Bash step in ``auto-release.yml``
and the ``actions/github-script`` body in ``release.yml``) with stubbed
``gh``/GitHub API clients so the version-bump and tag-retag logic is covered
without touching a network.

Covered:

* the next version is ``major`` for Conventional Commit ``!`` or a
  ``BREAKING CHANGE:`` footer, ``minor`` for an ordinary ``feat``, and
  ``patch`` otherwise;
* ``auto-release`` moves only the release's own major tag (``v2`` for a v2
  release) and ``latest`` — it never repoints ``v1`` at a v2 release, so
  existing ``@v1`` consumers are preserved;
* the release published/edited handler skips non-semver tags (no legacy
  placeholder promotion), only moves refs forward, and does not let editing
  an older release downgrade ``latest`` (or ``v<major>``);
* it compares against published (non-draft, non-prerelease) releases, so a
  higher unpublished/orphan tag cannot block promotion;
* both workflows share the same serializing ``auto-release`` concurrency
  group with ``cancel-in-progress: false``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
AUTO_WORKFLOW = ROOT / ".github/workflows/auto-release.yml"
RELEASE_WORKFLOW = ROOT / ".github/workflows/release.yml"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")

GH_STUB = """#!/usr/bin/env python3
import os
import subprocess
import sys

args = sys.argv[1:]
log = os.environ.get("GH_LOG")
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(" ".join(args) + "\\n")

if args and args[0] == "api":
    ref = None
    sha = None
    for arg in args:
        if "/refs/tags/" in arg:
            ref = arg.split("/refs/tags/", 1)[1]
        elif arg.startswith("ref=refs/tags/"):
            ref = arg[len("ref=refs/tags/"):]
        elif arg.startswith("sha="):
            sha = arg[len("sha="):]
    if ref and sha:
        subprocess.run(
            ["git", "-C", os.environ["GH_REPO_DIR"], "tag", "-f", ref, sha],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
sys.exit(0)
"""


def _auto_script() -> str:
    workflow = yaml.safe_load(AUTO_WORKFLOW.read_text(encoding="utf-8"))
    for step in workflow["jobs"]["release"]["steps"]:
        if str(step.get("name", "")).startswith("Cut next release"):
            return str(step["run"])
    raise AssertionError(f"no 'Cut next release' step in {AUTO_WORKFLOW}")


def _release_script() -> str:
    workflow = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))
    for step in workflow["jobs"]["retag"]["steps"]:
        if str(step.get("name", "")).startswith("Update the moving major"):
            return str(step["with"]["script"])
    raise AssertionError(f"no retag script step in {RELEASE_WORKFLOW}")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed, repo-owned git invocation
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _commit(repo: Path, ref: str = "HEAD") -> str:
    result = _git(repo, "rev-parse", f"{ref}^{{commit}}")
    return result.stdout.strip()


def _tags(repo: Path) -> set[str]:
    return set(_git(repo, "tag", "--list").stdout.split())


@dataclass
class AutoRun:
    repo: Path
    base_sha: str
    head_sha: str
    result: subprocess.CompletedProcess[str]
    gh_log: str


def _run_auto(
    tmp_path: Path,
    base_tag: str,
    commit_messages: list[str],
    seeded_tags: dict[str, str] | None = None,
) -> AutoRun:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Release Test")
    (repo / "code.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "code.txt")
    _git(repo, "commit", "-q", "-m", "chore: base")
    _git(repo, "tag", base_tag)
    base_sha = _commit(repo)
    for name, ref in (seeded_tags or {}).items():
        _git(repo, "tag", "-f", name, ref if ref else base_sha)

    for index, message in enumerate(commit_messages):
        (repo / "code.txt").write_text(f"change {index}\n", encoding="utf-8")
        _git(repo, "add", "code.txt")
        _git(repo, "commit", "-q", "-m", message)

    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)  # noqa: S603
    _git(repo, "remote", "add", "origin", str(origin))

    bindir = tmp_path / "bin"
    bindir.mkdir()
    gh = bindir / "gh"
    gh.write_text(GH_STUB, encoding="utf-8")
    gh.chmod(0o755)
    gh_log = tmp_path / "gh.log"

    env = dict(os.environ)
    env.update(
        {
            "GH_TOKEN": "test-token",
            "GITHUB_REPOSITORY": "owner/repo",
            "GH_REPO_DIR": str(repo),
            "GH_LOG": str(gh_log),
        }
    )
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    result = subprocess.run(  # noqa: S603 - repo-owned workflow script under test
        ["bash", "-c", _auto_script()],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return AutoRun(
        repo=repo,
        base_sha=base_sha,
        head_sha=_commit(repo),
        result=result,
        gh_log=gh_log.read_text(encoding="utf-8") if gh_log.exists() else "",
    )


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (["fix: correct typo"], "v1.2.4"),
        (["docs: describe release flow"], "v1.2.4"),
        (["feat: add review panel"], "v1.3.0"),
        (["feat!: drop legacy review paths"], "v2.0.0"),
        (["feat(core)!: drop legacy review paths"], "v2.0.0"),
        (["feat: drop legacy review paths\n\nBREAKING CHANGE: legacy review removed"], "v2.0.0"),
        (["chore: cleanup\n\nBREAKING-CHANGE: legacy review removed"], "v2.0.0"),
    ],
)
def test_auto_release_next_version(tmp_path: Path, messages: list[str], expected: str) -> None:
    run = _run_auto(tmp_path, "v1.2.3", messages)
    assert run.result.returncode == 0, run.result.stderr
    assert expected in _tags(run.repo)
    assert f"release create {expected} " in run.gh_log


@pytest.mark.parametrize("checkout", ["HEAD", "HEAD^"])
def test_queued_or_repeated_release_never_retags_an_already_released_commit(
    tmp_path: Path, checkout: str
) -> None:
    run = _run_auto(tmp_path, "v1.2.3", ["fix: newest main commit"])
    assert run.result.returncode == 0
    tags_before = _tags(run.repo)
    _git(run.repo, "checkout", "--detach", checkout)
    log = tmp_path / "retry-gh.log"
    env = {
        **os.environ,
        "GH_TOKEN": "test",
        "GITHUB_REPOSITORY": "owner/repo",
        "GH_REPO_DIR": str(run.repo),
        "GH_LOG": str(log),
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    result = subprocess.run(
        ["bash", "-c", _auto_script()],
        cwd=run.repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "supersedes this queued push" in result.stdout
    assert not log.exists()
    assert _tags(run.repo) == tags_before
    assert _commit(run.repo, "latest") == run.head_sha
    assert _commit(run.repo, "v1") == run.head_sha


def test_auto_release_breaking_keeps_v1_at_previous_commit(tmp_path: Path) -> None:
    run = _run_auto(
        tmp_path,
        "v1.0.0",
        ["feat!: remove legacy review paths"],
        seeded_tags={"v1": ""},
    )
    assert run.result.returncode == 0, run.result.stderr
    tags = _tags(run.repo)
    assert "v2.0.0" in tags
    # The v1 moving tag must stay on the previous v1 commit, not jump to v2.
    assert _commit(run.repo, "v1") == run.base_sha
    assert _commit(run.repo, "v2") == run.head_sha
    assert _commit(run.repo, "latest") == run.head_sha
    assert "refs/tags/v1" not in run.gh_log


def test_auto_release_feat_moves_only_v1_and_latest(tmp_path: Path) -> None:
    run = _run_auto(tmp_path, "v1.0.0", ["feat: add panel"], seeded_tags={"v1": ""})
    assert run.result.returncode == 0, run.result.stderr
    assert "v1.1.0" in _tags(run.repo)
    assert _commit(run.repo, "v1") == run.head_sha
    assert _commit(run.repo, "latest") == run.head_sha
    assert "refs/tags/v2" not in run.gh_log


def _run_release_script(
    tmp_path: Path,
    refs: dict[str, str],
    tags: list[str],
    release_tag: str,
    releases: list[str] | None = None,
    *,
    draft: bool = False,
    prerelease: bool = False,
) -> dict[str, Any]:
    if shutil.which("node") is None:
        pytest.skip("node not available")

    harness = """
const refs = __REFS__;
const tags = __TAGS__;
const releases = __RELEASES__;
const calls = [];
let failed = null;
const core = {
  info: (message) => calls.push(["info", String(message)]),
  setFailed: (message) => { failed = String(message); },
};
const context = { repo: { owner: "o", repo: "r" }, payload: { release: __RELEASE__ } };
const github = {
  paginate: async (fn, opts) => (await fn(opts)).data,
  rest: {
    git: {
      getRef: async ({ ref }) => {
        const name = ref.replace(/^tags\\//, "");
        if (!(name in refs)) {
          const err = new Error("Not Found");
          err.status = 404;
          throw err;
        }
        return { data: { object: { sha: refs[name], type: "commit" } } };
      },
      getTag: async () => { throw new Error("getTag not stubbed"); },
      updateRef: async ({ ref, sha, force }) => {
        calls.push(["updateRef", ref, sha, force]);
        refs[ref.replace(/^tags\\//, "")] = sha;
      },
      createRef: async ({ ref, sha }) => {
        calls.push(["createRef", ref, sha]);
        refs[ref.replace(/^refs\\/tags\\//, "")] = sha;
      },
    },
    repos: {
      listTags: async () => ({ data: tags.map((name) => ({ name })) }),
      listReleases: async () => ({
        data: releases.map((name) => ({ tag_name: name, draft: false, prerelease: false })),
      }),
      updateRelease: async (args) => { calls.push(["updateRelease", args.tag_name]); },
    },
  },
};

(async () => {
  try {
    await (async () => {
      __SCRIPT__
    })();
  } catch (err) {
    failed = String((err && err.message) || err);
  }
  process.stdout.write(JSON.stringify({ refs, calls, failed }));
})();
"""
    harness = harness.replace("__REFS__", json.dumps(refs))
    harness = harness.replace("__TAGS__", json.dumps(tags))
    harness = harness.replace("__RELEASES__", json.dumps(tags if releases is None else releases))
    harness = harness.replace(
        "__RELEASE__",
        json.dumps({"tag_name": release_tag, "id": 1, "draft": draft, "prerelease": prerelease}),
    )
    harness = harness.replace("__SCRIPT__", _release_script())

    script_path = tmp_path / "release_harness.js"
    script_path.write_text(harness, encoding="utf-8")
    result = subprocess.run(  # noqa: S603 - repo-owned workflow script under test
        ["node", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_release_handler_supports_v2_and_preserves_v1(tmp_path: Path) -> None:
    payload = _run_release_script(
        tmp_path,
        refs={"v1.5.0": "sha_150", "v2.0.0": "sha_200", "v1": "sha_150", "latest": "sha_150"},
        tags=["v1.5.0", "v2.0.0"],
        release_tag="v2.0.0",
    )
    assert payload["failed"] is None
    refs = payload["refs"]
    assert refs["v2"] == "sha_200"
    assert refs["latest"] == "sha_200"
    assert refs["v1"] == "sha_150"
    updated = [call[1] for call in payload["calls"] if call[0] == "updateRef"]
    assert "tags/v2" in updated
    assert "tags/latest" in updated
    assert "tags/v1" not in updated


def test_release_handler_does_not_downgrade_latest_on_edit(tmp_path: Path) -> None:
    payload = _run_release_script(
        tmp_path,
        refs={"v2.0.0": "sha_200", "v2.1.0": "sha_210", "v2": "sha_210", "latest": "sha_210"},
        tags=["v2.0.0", "v2.1.0"],
        release_tag="v2.0.0",
    )
    assert payload["failed"] is None
    refs = payload["refs"]
    assert refs["v2"] == "sha_210"
    assert refs["latest"] == "sha_210"
    assert not [call for call in payload["calls"] if call[0] in {"updateRef", "createRef"}]


def test_release_handler_skips_non_semver_without_placeholder_promotion(tmp_path: Path) -> None:
    payload = _run_release_script(
        tmp_path,
        refs={},
        tags=[],
        release_tag="nightly",
    )
    assert payload["failed"] is None
    assert not [
        call for call in payload["calls"] if call[0] in {"updateRef", "createRef", "updateRelease"}
    ]


def test_release_handler_ignores_unpublished_higher_tag(tmp_path: Path) -> None:
    # v2.9.0 exists only as an orphan git tag (no published release). It must
    # not hold back promotion of the published v2.1.0 release.
    payload = _run_release_script(
        tmp_path,
        refs={
            "v2.1.0": "sha_210",
            "v2.9.0": "sha_290",
            "v2": "sha_200",
            "latest": "sha_200",
        },
        tags=["v2.1.0", "v2.9.0"],
        releases=["v2.1.0"],
        release_tag="v2.1.0",
    )
    assert payload["failed"] is None
    refs = payload["refs"]
    assert refs["v2"] == "sha_210"
    assert refs["latest"] == "sha_210"


@pytest.mark.parametrize("flag", ["draft", "prerelease"])
def test_release_handler_never_promotes_nonstable_release_events(tmp_path: Path, flag: str) -> None:
    payload = _run_release_script(
        tmp_path,
        refs={"v2.0.0": "stable", "v2.1.0": "preview", "v2": "stable", "latest": "stable"},
        tags=["v2.0.0", "v2.1.0"],
        releases=["v2.0.0"],
        release_tag="v2.1.0",
        draft=flag == "draft",
        prerelease=flag == "prerelease",
    )
    assert payload["failed"] is None
    assert payload["refs"]["v2"] == "stable"
    assert payload["refs"]["latest"] == "stable"
    assert not [call for call in payload["calls"] if call[0] in {"updateRef", "createRef"}]


def test_release_workflows_share_serializing_concurrency() -> None:
    auto = yaml.safe_load(AUTO_WORKFLOW.read_text(encoding="utf-8"))["concurrency"]
    release = yaml.safe_load(RELEASE_WORKFLOW.read_text(encoding="utf-8"))["concurrency"]
    assert auto == {"group": "auto-release", "cancel-in-progress": False, "queue": "max"}
    assert release == auto
