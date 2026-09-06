# Vendored runtime assets

## three.global.js

- Upstream: three.js **0.170.0** (r170), `build/three.module.min.js`
- Source URL: <https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js>
- Upstream SHA-256 (module build): `08fd7545d13d2c7fb65ab691530a802dafefd638596501854f267d0fb13c39e7`
- Conversion: the single terminal `export{...}` statement was rewritten to a
  `window.THREE = {...}` assignment so the runtime loads as a classic script.
  This keeps portable artifacts fully functional under `file://` (ES module
  imports are blocked there) with no CDN dependency and no build toolchain.
- Upstream license: MIT (c) three.js authors.
- Vendored file SHA-256: `c7b180b6913e6881e82ce3d455d5a74879f498ad4db13659f28158fb066d4e53`

Orbit/pan/zoom controls are implemented directly in `viewer.js` (no
OrbitControls dependency); label overlays use HTML, not CSS2DRenderer.
