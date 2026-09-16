# Engine patches

Vapour 0.2.5-alpha's shadow pass ignored bone weights, so skinned characters cast bind-pose shadows.
`skinned-shadows/` holds the three patched files (vapour-render `surface/shadow.rs`, `surface/mod.rs`,
`shaders/shadow_depth.wgsl`) that bind the frame's bone palette into the shadow pipeline and skin the
caster's position exactly like `standard_pbr.wgsl` does.

`static-snapshots/physics3d.rs` (vapour-core) stops `step()` from serialising every fixed body: a level made of
thousands of static colliders otherwise spends ~7 ms per step building JSON for bodies that never move.

Rebuild the runtime (Rust 1.98 + wasm32 target + wasm-bindgen 0.2.127, both installed here) with:

```sh
./engine-patches/build-runtime.sh
```

It copies the patched files into `EngineSource`, builds `vapour-wasm` for wasm32, runs wasm-bindgen and
installs `vapour_runtime_bg.wasm` into `vapour-engine-0.2.5-alpha/runtime/` (the JS bindings are unchanged).

`text-layout-cache/text-2d.ts` (sdk/engine-src/text, the source the CLI bundles) caches each font atlas's validated glyph table and reuses one
grapheme segmenter: `layoutText2D` validated the whole atlas — allocating an `Intl.Segmenter` per glyph — on every
call, which made a 40-label HUD cost ~60 ms per frame. Copy it over `vapour-engine-0.2.5-alpha/sdk/engine-src/text/`.

`skinned-shadows/mod.rs` also carries the XR multiview fix: the layered PBR pass used a load-op clear per layer, which
ignores the scissor, so the second eye of a side-by-side XR frame wiped the first. A viewport layer that shares its
destination with an earlier layer now loads instead of clearing, so both eyes render in one submission
(`XrPresentation` multiview mode) instead of two full passes per frame.
