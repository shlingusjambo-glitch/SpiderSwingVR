#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cp engine-patches/skinned-shadows/shadow.rs engine-patches/skinned-shadows/mod.rs EngineSource/crates/vapour-render/src/surface/
cp engine-patches/skinned-shadows/shadow_depth.wgsl EngineSource/crates/vapour-render/shaders/
cp engine-patches/static-snapshots/physics3d.rs EngineSource/crates/vapour-core/src/
(cd EngineSource && cargo build --release --target wasm32-unknown-unknown -p vapour-wasm)
out=$(mktemp -d)
wasm-bindgen --target web --out-dir "$out" --out-name vapour_runtime EngineSource/target/wasm32-unknown-unknown/release/vapour_wasm.wasm
cp "$out/vapour_runtime_bg.wasm" vapour-engine-0.2.5-alpha/runtime/vapour_runtime_bg.wasm
echo "installed $(wc -c < vapour-engine-0.2.5-alpha/runtime/vapour_runtime_bg.wasm) bytes"
cp engine-patches/text-layout-cache/text-2d.ts vapour-engine-0.2.5-alpha/sdk/engine-src/text/text-2d.ts
