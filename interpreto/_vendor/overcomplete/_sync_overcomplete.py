#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

UPSTREAM_REPO_DEFAULT = "https://github.com/KempnerInstitute/overcomplete.git"
UPSTREAM_REF_DEFAULT = "main"

# Upstream directories -> destination directories (relative to your repo root).
# NOTE: verify whether the upstream path is "overcomplete/optimization" or "overcomplete/optimizations".
VENDOR_DIRS: list[tuple[str, str]] = [
    ("overcomplete/optimization", "interpreto/_vendor/overcomplete/optimization"),
    ("overcomplete/sae", "interpreto/_vendor/overcomplete/sae"),
]

# Upstream files -> destination files (relative to your repo root).
VENDOR_FILES: list[tuple[str, str]] = [
    ("overcomplete/base.py", "interpreto/_vendor/overcomplete/base.py"),
    ("overcomplete/metrics.py", "interpreto/_vendor/overcomplete/metrics.py"),
]

VENDORED_FROM_PATH = "interpreto/_vendor/overcomplete/VENDORED_FROM"


@dataclass(frozen=True)
class GitInfo:
    repo_url: str
    ref: str
    commit: str


def run(cmd: list[str], cwd: str | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}\n"
        )
    return p.stdout.strip()


def ensure_repo_root() -> Path:
    return Path(run(["git", "rev-parse", "--show-toplevel"]))


def sparse_checkout(repo_url: str, ref: str, upstream_paths: Iterable[str], workdir: str) -> GitInfo:
    """Checkout only the requested upstream paths into a temporary directory."""
    run(["git", "init"], cwd=workdir)
    run(["git", "remote", "add", "origin", repo_url], cwd=workdir)
    run(["git", "fetch", "--depth", "1", "origin", ref], cwd=workdir)

    # Non-cone mode supports sparse checkout of individual files.
    run(["git", "sparse-checkout", "init", "--no-cone"], cwd=workdir)
    run(["git", "sparse-checkout", "set", *list(upstream_paths)], cwd=workdir)

    run(["git", "checkout", "FETCH_HEAD"], cwd=workdir)
    commit = run(["git", "rev-parse", "HEAD"], cwd=workdir)
    return GitInfo(repo_url=repo_url, ref=ref, commit=commit)


def mirror_copy_dir(src: Path, dst: Path) -> None:
    """Mirror-copy src -> dst (dst is replaced entirely)."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def mirror_copy_file(src: Path, dst: Path) -> None:
    """Copy file src -> dst (dst is replaced)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def write_vendored_from(
    repo_root: Path,
    gi: GitInfo,
    dir_map: list[tuple[str, str]],
    file_map: list[tuple[str, str]],
) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: list[str] = [
        f"repo: {gi.repo_url}",
        f"ref: {gi.ref}",
        f"commit: {gi.commit}",
        f"vendored_at_utc: {now}",
        "directories:",
    ]
    for src, dst in dir_map:
        lines.append(f"  - {src} -> {dst}")
    lines.append("files:")
    for src, dst in file_map:
        lines.append(f"  - {src} -> {dst}")

    out = repo_root / VENDORED_FROM_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Vendor a subset of overcomplete into this repo.")
    ap.add_argument("--repo-url", default=UPSTREAM_REPO_DEFAULT)
    ap.add_argument("--ref", default=UPSTREAM_REF_DEFAULT, help="branch, tag, or commit SHA")
    args = ap.parse_args()

    repo_root = ensure_repo_root()

    upstream_paths = [src for src, _ in VENDOR_DIRS] + [src for src, _ in VENDOR_FILES]

    with tempfile.TemporaryDirectory() as td:
        gi = sparse_checkout(args.repo_url, args.ref, upstream_paths, td)

        for src_dir, dst_dir in VENDOR_DIRS:
            src_path = Path(td) / src_dir
            if not src_path.exists():
                raise RuntimeError(f"Upstream directory not found at {gi.commit}: {src_dir}")
            mirror_copy_dir(src_path, repo_root / dst_dir)

        for src_file, dst_file in VENDOR_FILES:
            src_path = Path(td) / src_file
            if not src_path.exists():
                raise RuntimeError(f"Upstream file not found at {gi.commit}: {src_file}")
            mirror_copy_file(src_path, repo_root / dst_file)

        write_vendored_from(repo_root, gi, VENDOR_DIRS, VENDOR_FILES)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
