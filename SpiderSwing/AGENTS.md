# Vapour project guidance

Gameplay code is Vapour Script (`.vs`) and/or TypeScript in `Scripts/`. Both
languages use the same engine APIs. Keep scenes and settings deterministic JSON,
preserve stable UUID references, and never edit `.vapour/`, `dist/`, or
generated WebAssembly bindings.

Before handing work back, run:

```sh
pnpm vapour validate . --json
pnpm vapour build . --json
```

Use Vapour engine APIs instead of sample-only browser code. Do not add a WebGL
primary path.
