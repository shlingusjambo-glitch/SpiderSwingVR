#!/usr/bin/env bash
# Builds the game into docs/ (GitHub Pages source) and pushes it.
set -euo pipefail
cd "$(dirname "$0")"
../vapour-engine-0.2.5-alpha/vapour validate . --json
rm -rf docs && ../vapour-engine-0.2.5-alpha/vapour build . --out docs --json
# The generated service worker pins the first bundle it ever cached, so every later deploy would serve stale code.
rm docs/service-worker.js docs/headers.txt
sed -i '' 's#<script>if("serviceWorker" in navigator).*</script>##' docs/index.html
touch docs/.nojekyll
git add -A && git commit -m "Deploy $(date +%F)" && git push
