#!/usr/bin/env bash
# Copies docs/wiki/ into the repository's GitHub Wiki.
#
# One-time setup: on GitHub, open the repository's Settings, tick "Wikis", then open the Wiki tab and
# click "Create the first page" (any content; this script replaces it).
#
# Usage: docs/publish_wiki.sh [owner/repo]     (default: thomasmd321/Image-Classifier)
set -euo pipefail

repo="${1:-thomasmd321/Image-Classifier}"
source_dir="$(cd "$(dirname "$0")/wiki" && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

git clone --quiet "https://github.com/${repo}.wiki.git" "$work_dir/wiki"
cd "$work_dir/wiki"
# Replace the wiki's pages with ours
find . -maxdepth 1 -name '*.md' -delete
cp "$source_dir"/*.md .
# The pages link to each other as Page.md so they also work when browsing the repository;
# the wiki wants plain page names: [Text](Page.md#part) -> [Text](Page#part)
sed -i.bak -E 's/\]\(([A-Za-z0-9_-]+)\.md(#[^)]*)?\)/](\1\2)/g' ./*.md
rm -f ./*.bak

git add -A
if git diff --cached --quiet; then
    echo "The wiki is already up to date."
    exit 0
fi
git commit --quiet -m "Update wiki from docs/wiki"
git push --quiet
echo "Published: https://github.com/${repo}/wiki"
