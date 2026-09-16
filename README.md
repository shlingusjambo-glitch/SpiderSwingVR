# Vapour VR games

Two WebXR games for the Quest 2 built on the Vapour engine (0.2.5-alpha), playable at
https://shlingusjambo-glitch.github.io/SpiderSwingVR/

- `SpiderSwing/` — **Monke Swing**: Gorilla Tag fan game with web swinging (`/monke`).
- `BoneVR/` — **BoneVR**: Bonelab-style physics sandbox with Ford and a KMP-60 (`/bone`).
- `shared/` — XR stereo presentation and skinned-GLB avatar code both games use.
- `engine-patches/` — the engine changes the games depend on (XR multiview clear fix, skinned shadows, static-snapshot physics).

The engine SDK and its Rust sources are not in this repo. Build and deploy with `./deploy.sh`.

Credits: Gorilla Tag map by familycucui4, Gorilla Tag player model by S3ntaGT, Ford (Bonelab) model and
Signalis KMP-60 from Sketchfab (CC-BY-4.0). Fan projects, not affiliated with Another Axiom, Stress Level Zero or rose-engine.
