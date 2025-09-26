#!/usr/bin/env bash
set -euo pipefail

# Make ERRs visible (even inside functions/command substitutions)
set -E -o errtrace
trap 'rc=$?; echo -e "${RED}Error:${RESET} command \"${BASH_COMMAND}\" failed with exit $rc at line $LINENO"; exit $rc' ERR


RED="\033[31m"
GREEN="\033[32m"
RESET="\033[0m"

error() {
    echo -e "${RED}Error:${RESET} $1" >&2
    exit 1
}

if [[ $# -ne 1 ]]; then
    error "Usage: $0 {major|minor|patch}"
fi

part=$1
case "$part" in
    major|minor|patch)
    ;;
    
    *)
        error "Invalid argument '$part'. Expected one of: major, minor, patch."
    ;;
esac

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    error "This script must be run inside a git repository."
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

if [[ -n $(git status --porcelain) ]]; then
    error "Working tree must be clean before bumping the version."
fi

if ! git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
    error "Current branch does not track a remote branch. Set an upstream with 'git push -u'."
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" == "HEAD" ]]; then
    error "Detached HEAD state is not supported. Checkout a branch before bumping the version."
fi

if ! git fetch origin main; then
    error "Failed to fetch origin/main."
fi

if ! git merge-base --is-ancestor origin/main HEAD; then
    error "Current branch is not up to date with origin/main. Please rebase or merge the latest changes. 'git pull origin main'"
fi

version_line=$(grep -E '^version = "[0-9]+\.[0-9]+\.[0-9]+"' pyproject.toml || true)
if [[ -z "$version_line" ]]; then
    error "Could not find a semantic version in pyproject.toml."
fi
current_version=${version_line#version = \"}
current_version=${current_version%\"}
IFS='.' read -r major minor patch <<<"$current_version"

case "$part" in
  major) ((++major)); minor=0; patch=0 ;;
  minor) ((++minor)); patch=0 ;;
  patch) ((++patch)) ;;
esac

next_version="${major}.${minor}.${patch}"

if [[ "$current_version" == "$next_version" ]]; then
    error "Next version is identical to current version."
fi

tag_name="v${next_version}"

if git show-ref --tags --verify --quiet "refs/tags/${tag_name}"; then
    error "Tag ${tag_name} already exists locally."
fi

if ! remote_tag_output=$(git ls-remote --tags origin "refs/tags/${tag_name}"); then
    error "Failed to query tags from origin."
fi

if [[ -n "$remote_tag_output" ]]; then
    error "Tag ${tag_name} already exists on origin."
fi

committed=false
cleanup() {
    if ! $committed; then
        git checkout -- pyproject.toml >/dev/null 2>&1 || true
    fi
}
trap cleanup ERR INT

if ! NEXT_VERSION="$next_version" python - <<'PY'
from pathlib import Path
import os, re

path = Path("pyproject.toml")
text = path.read_text()

pattern = re.compile(r'^(version\s*=\s*")(\d+\.\d+\.\d+)(")$', re.MULTILINE)
next_version = os.environ["NEXT_VERSION"]

if not pattern.search(text):
    raise SystemExit(1)

updated, count = pattern.subn(r'\g<1>' + next_version + r'\g<3>', text, count=1)
if count != 1:
    raise SystemExit(1)

path.write_text(updated)
PY
then
    error "Failed to update version string in pyproject.toml."
fi

if ! git diff --quiet -- pyproject.toml; then
    git add pyproject.toml
else
    error "Version number was not updated."
fi

git commit -m "${current_version} -> ${next_version}"
committed=true

if ! git push; then
    error "Failed to push commit to upstream."
fi

if ! git tag "${tag_name}"; then
    error "Failed to create tag ${tag_name}."
fi

if ! git push origin "${tag_name}"; then
    error "Failed to push tag ${tag_name} to origin."
fi

echo -e "${GREEN}Version bumped from ${current_version} to ${next_version}.${RESET}"
