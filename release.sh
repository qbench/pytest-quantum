#!/usr/bin/env bash
# Usage: ./release.sh 0.2.0
# Bumps version in pyproject.toml, commits, tags, and pushes.
# The tag push triggers the publish workflow automatically.

set -euo pipefail

VERSION="${1:-}"

if [[ -z "$VERSION" ]]; then
  echo "Usage: ./release.sh <version>  e.g. ./release.sh 0.2.0"
  exit 1
fi

# Validate semver-ish format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: version must be X.Y.Z (e.g. 0.2.0), got: $VERSION"
  exit 1
fi

TAG="v${VERSION}"

# Make sure we're on master and up to date
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "master" ]]; then
  echo "Error: must be on master branch (currently on '$BRANCH')"
  exit 1
fi

git pull --ff-only origin master

# Check tag doesn't already exist
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "Error: tag $TAG already exists. Did you mean a different version?"
  exit 1
fi

# Bump version in pyproject.toml
CURRENT=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo "Bumping $CURRENT → $VERSION"
sed -i.bak "s/^version = \"${CURRENT}\"/version = \"${VERSION}\"/" pyproject.toml
rm -f pyproject.toml.bak

# Commit + tag + push
git add pyproject.toml
git commit -m "chore: release v${VERSION}"
git tag "$TAG"
git push origin master
git push origin "$TAG"

echo ""
echo "Done. Tag $TAG pushed — watch CI at:"
echo "  https://github.com/qbench/pytest-quantum/actions"
echo ""
echo "PyPI page (live in ~2 min after CI passes):"
echo "  https://pypi.org/project/pytest-quantum/${VERSION}/"
