#!/usr/bin/env bash
# Builds both games into docs/ (GitHub Pages source: /monke and /bone) and pushes.
set -euo pipefail
cd "$(dirname "$0")"
V=./vapour-engine-0.2.5-alpha/vapour
rm -rf docs && mkdir docs && touch docs/.nojekyll
for pair in "SpiderSwing monke" "BoneVR bone"; do
  set -- $pair
  $V validate "./$1" --json
  $V build "./$1" --out "docs/$2" --json
  # The generated service worker pins the first bundle it ever cached, so every later deploy would serve stale code.
  rm "docs/$2/service-worker.js" "docs/$2/headers.txt"
  sed -i '' 's#<script>if("serviceWorker" in navigator).*</script>##' "docs/$2/index.html"
done
cat > docs/index.html <<'HTML'
<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vapour VR games</title>
<body style="margin:0;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:24px;font:20px system-ui;background:#0b0d12;color:#fff">
<h1 style="margin:0">Vapour VR games</h1>
<a href="monke/" style="font-size:28px;padding:18px 40px;border-radius:14px;background:#d1202a;color:#fff;text-decoration:none">🦍 Monke Swing</a>
<a href="bone/" style="font-size:28px;padding:18px 40px;border-radius:14px;background:#c9a227;color:#000;text-decoration:none">🦴 BoneVR</a>
<p style="opacity:.6;font-size:14px">Open on a Quest in the Meta Browser, then tap Enter VR.</p>
HTML
git add -A && git commit -m "Deploy $(date +%F)" && git push
