/* =============================================================================
 * MathKernel Viz - portable artifact viewer runtime (schema 2.0)
 * Copyright (c) 2026 Maarten Boone
 * SPDX-License-Identifier: MIT
 *
 * Framework-free classic script (file:// safe: no modules, no fetch, no
 * workers, no network).  Every artifact is a composition of building blocks;
 * this viewer keeps a registry of block renderers and wires select controls
 * to bound blocks.  All embedded strings are untrusted display data:
 * textContent only, never innerHTML, never eval.
 * ===========================================================================*/
(function (global) {
  "use strict";

  var SCHEMA = "mathkernel-viz/2.0";
  var TRUST_COLORS = {
    formal: "#b18cff", exact: "#7ce38b", symbolic: "#5aa9e6",
    interval_certified: "#5ad4d4", numeric_high_precision: "#efb34c",
    numeric: "#f0924c", empirical: "#e86f6f", heuristic: "#e86f9e",
    unknown: "#8a97a3"
  };
  var PALETTE = ["#35a9ef", "#efb34c", "#7ce38b", "#e86f9e", "#b18cff",
                 "#5ad4d4", "#f28b82", "#a8c7fa"];
  var BG = "#0b0e11", PANEL_BG = "#0b1014", GRID_LINE = "#1d2831",
      TEXT = "#dbe8f3", DIM = "#7790a2", ACCENT = "#35a9ef";

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }

  /* ---- mountable instance -------------------------------------------------
   * mount(root, payloadOrDoc) renders one VisualizationDocument inside root.
   * root === document keeps the legacy single-artifact behaviour (global element
   * ids); any other root element is scoped through [data-mkv="<id>"] hooks so
   * several instances can coexist in one page (multimodal artifacts).
   * ------------------------------------------------------------------------*/

  function mount(root, payloadOrDoc) {

  function byId(id) {
    if (root === document) return document.getElementById(id);
    return root.querySelector('[data-mkv="' + id + '"]');
  }

  function fail(msg) {
    var box = byId("mkv-view");
    if (!box) {
      box = el("div");
      box.setAttribute("data-mkv", "mkv-view");
      (root === document ? document.body : root).appendChild(box);
    }
    box.textContent = "";
    box.appendChild(el("div", "mkv-error", msg));
  }

  /* ---- payload ---------------------------------------------------------- */

  var payloadEl = null, payloadText = null, doc = null;
  if (payloadOrDoc && payloadOrDoc.nodeType === 1) {
    payloadEl = payloadOrDoc;
    payloadText = payloadEl.textContent;
    try { doc = JSON.parse(payloadText); }
    catch (e) { fail("Artifact payload is not valid JSON."); return; }
  } else {
    doc = payloadOrDoc;
  }
  if (!doc) { fail("Artifact payload missing."); return; }

  if (doc.artifact_schema !== SCHEMA) {
    fail("Unsupported artifact schema: " + String(doc.artifact_schema) +
         " (this viewer understands " + SCHEMA + ").");
    return;
  }

  if (!byId("mkv-tip")) {
    var tipEl = el("div", "mkv-tip");
    tipEl.id = root === document ? "mkv-tip" : "";
    tipEl.setAttribute("data-mkv", "mkv-tip");
    (root === document ? document.body : root).appendChild(tipEl);
  }

  /* ---- integrity -------------------------------------------------------- */

  function sha256Hex(buffer) {
    if (!window.crypto || !crypto.subtle) return Promise.resolve(null);
    return crypto.subtle.digest("SHA-256", buffer).then(function (hash) {
      return Array.prototype.map.call(new Uint8Array(hash), function (b) {
        return ("0" + b.toString(16)).slice(-2);
      }).join("");
    }).catch(function () { return null; });
  }

  var integrityLines = [];
  function checkIntegrity() {
    var expected = payloadEl ? (payloadEl.getAttribute("data-sha256") || null)
                             : null;
    var hashed = payloadText !== null
      ? sha256Hex(new TextEncoder().encode(payloadText))
      : Promise.resolve(null);
    return hashed.then(function (actual) {
      if (expected && actual) {
        integrityLines.push(["Result payload",
          actual === expected ? "VERIFIED" : "MISMATCH", actual]);
      } else if (expected) {
        integrityLines.push(["Result payload", "hash present (WebCrypto unavailable)", expected]);
      }
      var dh = (doc.manifest && doc.manifest.dataset_hashes) || {};
      Object.keys(dh).forEach(function (k) {
        integrityLines.push(["Dataset " + k, "SHA-256", dh[k]]);
      });
    });
  }

  /* ---- dataset decoding ------------------------------------------------- */

  function b64ToBytes(b64) {
    var bin = atob(b64), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function inflate(bytes) {
    if (!window.DecompressionStream) return Promise.reject(new Error("no DecompressionStream"));
    var ds = new DecompressionStream("deflate");
    var stream = new Blob([bytes]).stream().pipeThrough(ds);
    return new Response(stream).arrayBuffer();
  }

  function decodeDataset(ds) {
    if (ds.encoding === "inline") return Promise.resolve(ds.data || []);
    var chunks = ds.encoding === "zlib+base64+chunked" ? ds.data : [ds.data];
    var parts = [];
    var p = Promise.resolve();
    chunks.forEach(function (c, i) {
      p = p.then(function () {
        return inflate(b64ToBytes(c)).then(function (buf) {
          var arr;
          if (ds.kind === "i64") {
            var big = new BigInt64Array(buf);
            arr = new Array(big.length);
            for (var j = 0; j < big.length; j++) arr[j] = Number(big[j]);
          } else {
            arr = Array.prototype.slice.call(new Float64Array(buf));
          }
          parts[i] = arr;
        });
      });
    });
    return p.then(function () {
      var out = [];
      parts.forEach(function (a) { out = out.concat(a); });
      return out;
    });
  }

  /* ---- formatting / ticks ----------------------------------------------- */

  function fmtTick(v) {
    if (v === 0) return "0";
    if (!isFinite(v)) return String(v);
    var av = Math.abs(v);
    if (av >= 1e6 || av < 1e-4) return v.toExponential(2);
    return String(Math.round(v * 1e6) / 1e6);
  }

  function niceTicks(lo, hi, n) {
    n = n || 6;
    if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
    var span = hi - lo;
    var step = Math.pow(10, Math.floor(Math.log10(span / n)));
    var mults = [1, 2, 5, 10];
    for (var i = 0; i < mults.length; i++) {
      if (step * mults[i] * n >= span) { step *= mults[i]; break; }
    }
    var start = Math.floor(lo / step) * step;
    var ticks = [], t = start;
    while (t <= hi + 1e-12) {
      if (t >= lo - 1e-12) ticks.push(Math.round(t * 1e12) / 1e12);
      t += step;
    }
    return ticks;
  }

  function sizeCanvas(c) {
    var r = c.getBoundingClientRect();
    var d = Math.min(window.devicePixelRatio || 1, 2);
    var w = Math.max(60, Math.round(r.width * d));
    var h = Math.max(60, Math.round(r.height * d));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return { w: w, h: h, d: d };
  }

  function seriesXY(s, decoded) {
    if (s.x && s.y && decoded[s.x] && decoded[s.y]) {
      return [decoded[s.x], decoded[s.y]];
    }
    if (s.points) {
      return [s.points.map(function (p) { return p[0]; }),
              s.points.map(function (p) { return p[1]; })];
    }
    return null;
  }

  function seriesColor(s, i) {
    return (s.style && s.style.color) || PALETTE[i % PALETTE.length];
  }

  /* ---- generic client-side reshaping (mirrors transforms.py) ------------ */

  function lagEmbed(vals, cfg) {
    var lags = (cfg.lags || [0, 1, 2]).slice(0, 3);
    while (lags.length < 3) lags.push(0);
    var stride = Math.max(1, cfg.stride || 1), offset = cfg.offset || 0;
    var diff = !!cfg.differences;
    var extra = diff ? 1 : 0;
    var maxLag = Math.max.apply(null, lags);
    var count = vals.length - maxLag - extra;
    var lo = Infinity, hi = -Infinity, i;
    if (cfg.normalize !== false) {
      for (i = 0; i < vals.length; i++) {
        if (vals[i] < lo) lo = vals[i];
        if (vals[i] > hi) hi = vals[i];
      }
    }
    var center = (lo + hi) / 2, span = Math.max(hi - lo, 1e-300);
    function map(v) { return cfg.normalize === false ? v : (v - center) / span; }
    var out = [];
    for (var n = offset; n < count; n += stride) {
      var row = [];
      for (var j = 0; j < 3; j++) {
        var k = lags[j];
        row.push(map(diff ? vals[n + k + 1] - vals[n + k] : vals[n + k]));
      }
      out.push(row);
    }
    return out;
  }

  function basicStats(vals, want) {
    var out = [];
    var n = vals.length;
    if (!n) return [["count", "0"]];
    var lookup = {
      count: function () { return n.toLocaleString(); },
      mean: function () { var s = 0; for (var i = 0; i < n; i++) s += vals[i]; return fmtTick(s / n); },
      std: function () {
        var s = 0, i; for (i = 0; i < n; i++) s += vals[i];
        var m = s / n, v = 0;
        for (i = 0; i < n; i++) v += (vals[i] - m) * (vals[i] - m);
        return fmtTick(Math.sqrt(Math.max(0, v / n)));
      },
      min: function () { return fmtTick(Math.min.apply(null, vals)); },
      max: function () { return fmtTick(Math.max.apply(null, vals)); },
      distinct: function () { return new Set(vals).size.toLocaleString(); }
    };
    (want || ["count", "mean", "std", "min", "max"]).forEach(function (k) {
      if (lookup[k]) out.push([k, lookup[k]()]);
    });
    return out;
  }

  /* ---- 3D orbit controls (damped, pointer-capture based) ---------------- */

  function makeOrbit(camera, dom, opts) {
    var target = new THREE.Vector3().fromArray(opts.target || [0, 0, 0]);
    var home = {
      radius: opts.radius || 2.2,
      theta: opts.theta != null ? opts.theta : 0.72,
      phi: opts.phi != null ? opts.phi : 1.05,
      target: target.clone()
    };
    var radius = home.radius, theta = home.theta, phi = home.phi;
    var vTheta = 0, vPhi = 0, vRadius = 0;
    var panDelta = new THREE.Vector3();
    var DAMP = 0.08;
    var pointers = new Map();
    var mode = 0, lastX = 0, lastY = 0, pinchDist = 0;

    dom.style.touchAction = "none";
    dom.addEventListener("contextmenu", function (e) { e.preventDefault(); });
    dom.addEventListener("pointerdown", function (e) {
      dom.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (pointers.size === 1) {
        mode = (e.button === 2 || e.shiftKey) ? 2 : 1;
        lastX = e.clientX; lastY = e.clientY;
      } else if (pointers.size === 2) {
        var pts = Array.from(pointers.values());
        pinchDist = Math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]);
        mode = 3;
      }
    });
    dom.addEventListener("pointermove", function (e) {
      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (mode === 3 && pointers.size === 2) {
        var pts = Array.from(pointers.values());
        var d = Math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]);
        if (pinchDist > 0) vRadius += (pinchDist - d) * 0.004;
        pinchDist = d;
        return;
      }
      if (mode === 0) return;
      var dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (mode === 2) {
        var panScale = radius * 0.0016;
        var fwd = new THREE.Vector3();
        camera.getWorldDirection(fwd);
        var right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
        var up = new THREE.Vector3().crossVectors(right, fwd).normalize();
        panDelta.addScaledVector(right, -dx * panScale)
                .addScaledVector(up, dy * panScale);
      } else {
        vTheta -= dx * 0.005;
        vPhi -= dy * 0.005;
      }
    });
    function release(e) {
      pointers.delete(e.pointerId);
      if (pointers.size === 0) mode = 0;
    }
    dom.addEventListener("pointerup", release);
    dom.addEventListener("pointercancel", release);
    dom.addEventListener("wheel", function (e) {
      e.preventDefault();
      vRadius += e.deltaY * 0.0012;
    }, { passive: false });
    dom.addEventListener("dblclick", function () { api.reset(); });

    var api = {
      reset: function () {
        radius = home.radius; theta = home.theta; phi = home.phi;
        target.copy(home.target);
        vTheta = vPhi = vRadius = 0; panDelta.set(0, 0, 0);
      },
      update: function () {
        theta += vTheta;
        phi = Math.min(Math.PI - 0.01, Math.max(0.01, phi + vPhi));
        radius = Math.min(1e7, Math.max(1e-6, radius * (1 + vRadius)));
        target.add(panDelta);
        vTheta *= (1 - DAMP); vPhi *= (1 - DAMP); vRadius *= (1 - DAMP);
        panDelta.multiplyScalar(1 - DAMP);
        camera.position.set(
          target.x + radius * Math.sin(phi) * Math.sin(theta),
          target.y + radius * Math.cos(phi),
          target.z + radius * Math.sin(phi) * Math.cos(theta));
        camera.lookAt(target);
      },
      isInteracting: function () { return pointers.size > 0; }
    };
    api.update();
    return api;
  }

  /* ---- block renderers -------------------------------------------------- */
  /* Each renderer: (container, block, ctx) -> optional cleanup function.    */

  var RENDERERS = {};

  function blockCanvas(container, minHeight) {
    var c = el("canvas", "mkv-block-canvas");
    c.style.height = (minHeight || 320) + "px";
    container.appendChild(c);
    return c;
  }

  /* -- plot2d -- */
  RENDERERS.plot2d = function (container, block, ctx) {
    var canvas = blockCanvas(container, 340);
    var tip = byId("mkv-tip");
    function draw() {
      var s = sizeCanvas(canvas), g = canvas.getContext("2d");
      var W = s.w, H = s.h, pad = 46 * s.d, cfg = block.config || {};
      g.fillStyle = PANEL_BG; g.fillRect(0, 0, W, H);
      var logY = !!cfg.log_y, logX = !!cfg.log_x;
      var all = [];
      block.series.forEach(function (sid, i) {
        var sr = ctx.doc.series[sid];
        if (!sr) return;
        var xy = seriesXY(sr, ctx.decoded);
        if (!xy) return;
        var xs = logX ? xy[0].map(function (v) { return Math.log10(Math.max(v, 1e-300)); }) : xy[0];
        var ys = logY ? xy[1].map(function (v) { return Math.log10(Math.max(v, 1e-300)); }) : xy[1];
        all.push([sr, xs, ys, i]);
      });
      if (!all.length) {
        g.fillStyle = DIM; g.font = (12 * s.d) + "px monospace";
        g.fillText("no plottable series", W / 2 - 70 * s.d, H / 2);
        return;
      }
      var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      all.forEach(function (t) {
        t[1].forEach(function (v) { if (v < x0) x0 = v; if (v > x1) x1 = v; });
        t[2].forEach(function (v) { if (v < y0) y0 = v; if (v > y1) y1 = v; });
      });
      if (x0 === x1) { x0 -= 1; x1 += 1; }
      if (y0 === y1) { y0 -= 1; y1 += 1; }
      var hasBar = all.some(function (t) { return (t[0].style || {}).marker === "bar"; });
      var mx = (x1 - x0) * 0.04, my = hasBar ? 0 : (y1 - y0) * 0.07;
      x0 -= mx; x1 += mx; y0 -= my; y1 += my;
      if (hasBar) y0 = Math.min(y0, 0);
      function X(x) { return pad + (x - x0) / (x1 - x0) * (W - 2 * pad); }
      function Y(y) { return H - pad - (y - y0) / (y1 - y0) * (H - 2 * pad); }

      g.strokeStyle = GRID_LINE; g.fillStyle = DIM;
      g.font = (10 * s.d) + "px monospace"; g.lineWidth = 1;
      niceTicks(x0, x1, 6).forEach(function (t) {
        g.beginPath(); g.moveTo(X(t), pad * 0.6); g.lineTo(X(t), H - pad); g.stroke();
        g.fillText(logX ? "1e" + fmtTick(t) : fmtTick(t), X(t) - 12 * s.d, H - pad + 14 * s.d);
      });
      niceTicks(y0, y1, 6).forEach(function (t) {
        g.beginPath(); g.moveTo(pad, Y(t)); g.lineTo(W - pad * 0.6, Y(t)); g.stroke();
        g.fillText(logY ? "1e" + fmtTick(t) : fmtTick(t), 4 * s.d, Y(t) + 3 * s.d);
      });
      g.strokeStyle = "#33424e"; g.lineWidth = 1.2 * s.d;
      g.strokeRect(pad, pad * 0.6, W - pad - pad * 0.6, H - pad - pad * 0.6);
      g.fillStyle = DIM; g.font = (11 * s.d) + "px monospace";
      if (cfg.x_label) g.fillText(cfg.x_label, W / 2 - 20 * s.d, H - 6 * s.d);
      if (cfg.y_label) {
        g.save(); g.translate(11 * s.d, H / 2 + 24 * s.d); g.rotate(-Math.PI / 2);
        g.fillText(cfg.y_label, 0, 0); g.restore();
      }

      var hit = [];
      var stackOffsets = {};
      var maxN = Math.max.apply(null, all.map(function (t) { return t[1].length; }));
      all.forEach(function (t) {
        var sr = t[0], xs = t[1], ys = t[2], st = sr.style || {};
        var color = seriesColor(sr, t[3]);
        g.strokeStyle = color; g.fillStyle = color;
        g.lineWidth = 1.4 * s.d;
        g.setLineDash(sr.role === "prediction" ? [6, 4] : []);
        var marker = st.marker || (xs.length > 64 ? "line" : "points");
        if (marker === "bar") {
          var bw = Math.max(s.d, (W - 2 * pad) / Math.max(maxN, 1) * 0.7);
          var stack = logY ? null : st.stack;
          g.globalAlpha = 0.85;
          for (var b = 0; b < xs.length; b++) {
            var base = stack ? (stackOffsets[stack + "|" + xs[b]] || 0) : 0;
            if (stack) stackOffsets[stack + "|" + xs[b]] = base + ys[b];
            g.fillRect(X(xs[b]) - bw / 2, Y(base + ys[b]), bw, Y(base) - Y(base + ys[b]));
          }
          g.globalAlpha = 1;
        } else if (marker === "line") {
          g.beginPath();
          for (var i = 0; i < xs.length; i++) {
            if (i === 0) g.moveTo(X(xs[i]), Y(ys[i])); else g.lineTo(X(xs[i]), Y(ys[i]));
          }
          g.stroke();
        } else {
          var r = (st.size || (xs.length > 2000 ? 1.1 : 2.6)) * s.d;
          for (var j = 0; j < xs.length; j++) {
            g.beginPath(); g.arc(X(xs[j]), Y(ys[j]), r, 0, 6.2832); g.fill();
            if (xs.length <= 2000) hit.push([X(xs[j]), Y(ys[j]), sr, xs[j], ys[j]]);
          }
        }
        g.setLineDash([]);
      });
      if (all.length > 1) {
        g.font = (10 * s.d) + "px monospace";
        all.forEach(function (t, i) {
          g.fillStyle = seriesColor(t[0], t[3]);
          g.fillRect(pad + 8 * s.d + i * 170 * s.d, 8 * s.d, 9 * s.d, 9 * s.d);
          g.fillStyle = TEXT;
          g.fillText((t[0].label || t[0].series_id).slice(0, 24),
                     pad + 22 * s.d + i * 170 * s.d, 16 * s.d);
        });
      }
      canvas._mkvHit = hit;
      canvas._mkvScale = s.d;
    }
    draw();
    canvas.addEventListener("mousemove", function (e) {
      var rect = canvas.getBoundingClientRect();
      var d = canvas._mkvScale || 1;
      var mx = (e.clientX - rect.left) * d, my = (e.clientY - rect.top) * d;
      var best = null, bd = 400 * d * d;
      (canvas._mkvHit || []).forEach(function (h) {
        var dd = (h[0] - mx) * (h[0] - mx) + (h[1] - my) * (h[1] - my);
        if (dd < bd) { bd = dd; best = h; }
      });
      if (best) {
        var sr = best[2];
        var txt = (sr.label || sr.series_id) + "\n(" + fmtTick(best[3]) + ", " +
          fmtTick(best[4]) + ")\ntrust: " + sr.trust + " | role: " + sr.role +
          (sr.source ? " | source: " + sr.source : "");
        if (sr.enclosure) txt += "\ncertified: [" + sr.enclosure.lower + ", " + sr.enclosure.upper + "]";
        tip.style.display = "block";
        tip.style.left = (e.clientX + 14) + "px";
        tip.style.top = (e.clientY + 14) + "px";
        tip.textContent = txt;
      } else tip.style.display = "none";
    });
    canvas.addEventListener("mouseleave", function () { tip.style.display = "none"; });
    var ro = new ResizeObserver(draw); ro.observe(canvas);
    return function () { ro.disconnect(); };
  };

  /* -- histogram (bins computed client-side; bins stay interactive) -- */
  RENDERERS.histogram = function (container, block, ctx) {
    var canvas = blockCanvas(container, 300);
    var tip = byId("mkv-tip");
    function draw() {
      var s = sizeCanvas(canvas), g = canvas.getContext("2d");
      var W = s.w, H = s.h, pad = 40 * s.d, cfg = block.config || {};
      g.fillStyle = PANEL_BG; g.fillRect(0, 0, W, H);
      var vals = ctx.decoded[cfg.dataset];
      if (!vals || !vals.length) {
        g.fillStyle = DIM; g.font = (12 * s.d) + "px monospace";
        g.fillText("dataset unavailable", W / 2 - 60 * s.d, H / 2);
        return;
      }
      var bins = Math.max(1, Math.min(2048, cfg.bins || 64));
      var lo = cfg.range ? cfg.range[0] : Math.min.apply(null, vals);
      var hi = cfg.range ? cfg.range[1] : Math.max.apply(null, vals);
      if (lo === hi) hi = lo + 1;
      var width = (hi - lo) / bins;
      var counts = new Uint32Array(bins);
      for (var i = 0; i < vals.length; i++) {
        var v = vals[i];
        if (v < lo || v > hi) continue;
        counts[Math.min(bins - 1, Math.floor((v - lo) / width))]++;
      }
      var maxC = Math.max.apply(null, Array.from(counts));
      var logY = !!cfg.log_y;
      var top = logY ? Math.log10(Math.max(maxC, 1)) : maxC;
      var color = cfg.color || ACCENT;
      var bw = (W - 2 * pad) / bins;
      var hit = [];
      g.fillStyle = color;
      for (var b = 0; b < bins; b++) {
        var cval = logY ? (counts[b] ? Math.log10(counts[b]) : 0) : counts[b];
        var hgt = top ? (cval / top) * (H - 2 * pad) : 0;
        var x = pad + b * bw, y = H - pad - hgt;
        g.globalAlpha = counts[b] ? 0.9 : 0.15;
        g.fillRect(x, y, Math.max(1, bw - 0.5), hgt);
        hit.push([x, y, Math.max(1, bw - 0.5), hgt, b, counts[b]]);
      }
      g.globalAlpha = 1;
      g.fillStyle = DIM; g.font = (10 * s.d) + "px monospace";
      g.fillText(fmtTick(lo), pad, H - pad + 14 * s.d);
      g.fillText(fmtTick(hi), W - pad - 30 * s.d, H - pad + 14 * s.d);
      g.fillText((logY ? "log10 " : "") + "max " + fmtTick(maxC), pad, pad * 0.5);
      g.strokeStyle = "#33424e"; g.lineWidth = 1.2 * s.d;
      g.strokeRect(pad, pad * 0.6, W - 2 * pad, H - pad - pad * 0.6);
      canvas._mkvHist = { hit: hit, lo: lo, width: width, d: s.d };
    }
    draw();
    canvas.addEventListener("mousemove", function (e) {
      var info = canvas._mkvHist;
      if (!info) return;
      var rect = canvas.getBoundingClientRect();
      var mx = (e.clientX - rect.left) * info.d, my = (e.clientY - rect.top) * info.d;
      var best = null;
      info.hit.forEach(function (h) {
        if (mx >= h[0] && mx <= h[0] + h[2] && my >= h[1]) best = h;
      });
      if (best) {
        tip.style.display = "block";
        tip.style.left = (e.clientX + 14) + "px";
        tip.style.top = (e.clientY + 14) + "px";
        tip.textContent = "[" + fmtTick(info.lo + best[4] * info.width) + ", " +
          fmtTick(info.lo + (best[4] + 1) * info.width) + ")\ncount: " + best[5];
      } else tip.style.display = "none";
    });
    canvas.addEventListener("mouseleave", function () { tip.style.display = "none"; });
    var ro = new ResizeObserver(draw); ro.observe(canvas);
    return function () { ro.disconnect(); };
  };

  /* -- heatmap -- */
  RENDERERS.heatmap = function (container, block, ctx) {
    var canvas = blockCanvas(container, 340);
    function draw() {
      var s = sizeCanvas(canvas), g = canvas.getContext("2d");
      var W = s.w, H = s.h, pad = 40 * s.d, cfg = block.config || {};
      var vals = ctx.decoded[cfg.dataset];
      if (!vals) { g.fillStyle = DIM; g.fillText("dataset unavailable", 20, 20); return; }
      var n = cfg.rows || 1, m = cfg.cols || 1;
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
      var span = (hi - lo) || 1;
      g.fillStyle = PANEL_BG; g.fillRect(0, 0, W, H);
      var cw = (W - 2 * pad) / m, ch = (H - 2 * pad) / n;
      for (var i = 0; i < n; i++) {
        for (var j = 0; j < m; j++) {
          var t = (vals[i * m + j] - lo) / span;
          g.fillStyle = "rgb(" + Math.round(10 + t * 43) + "," +
            Math.round(20 + t * 149) + "," + Math.round(30 + t * 209) + ")";
          g.fillRect(pad + j * cw, pad * 0.6 + i * ch, cw + 0.5, ch + 0.5);
        }
      }
      g.fillStyle = DIM; g.font = (10 * s.d) + "px monospace";
      g.fillText(n + " x " + m + "  min " + fmtTick(lo) + "  max " + fmtTick(hi),
                 pad, H - 8 * s.d);
    }
    draw();
    var ro = new ResizeObserver(draw); ro.observe(canvas);
    return function () { ro.disconnect(); };
  };

  /* -- dag -- */
  RENDERERS.dag = function (container, block, ctx) {
    var canvas = blockCanvas(container, 340);
    function draw() {
      var s = sizeCanvas(canvas), g = canvas.getContext("2d");
      var W = s.w, H = s.h, pad = 40 * s.d;
      g.fillStyle = PANEL_BG; g.fillRect(0, 0, W, H);
      var cfg = block.config || {};
      var steps = cfg.nodes ||
        (((ctx.doc.provenance || {}).steps) || []).map(function (st) {
          return { id: st.step_id, label: st.operation, parents: st.parents,
                   trust: st.trust, engine: st.engine };
        });
      if (!steps.length) {
        g.fillStyle = DIM; g.font = (12 * s.d) + "px monospace";
        g.fillText("no graph recorded", W / 2 - 60 * s.d, H / 2);
        return;
      }
      var depth = {}, layersMap = {};
      steps.forEach(function (st) {
        var d = 0;
        (st.parents || []).forEach(function (p) {
          if (depth[p] !== undefined) d = Math.max(d, depth[p] + 1);
        });
        depth[st.id] = d;
        (layersMap[d] = layersMap[d] || []).push(st);
      });
      var maxD = Math.max.apply(null, Object.keys(layersMap).map(Number));
      var pos = {};
      Object.keys(layersMap).forEach(function (d) {
        var grp = layersMap[d];
        grp.forEach(function (st, i) {
          pos[st.id] = [pad + (maxD ? d / maxD : 0) * (W - 2 * pad - 120 * s.d),
                        pad + (i + 1) / (grp.length + 1) * (H - 2 * pad)];
        });
      });
      g.strokeStyle = "#33424e"; g.lineWidth = 1.2 * s.d;
      steps.forEach(function (st) {
        var p2 = pos[st.id];
        (st.parents || []).forEach(function (p) {
          if (pos[p]) {
            g.beginPath();
            g.moveTo(pos[p][0] + 110 * s.d, pos[p][1]);
            g.lineTo(p2[0], p2[1]); g.stroke();
          }
        });
      });
      steps.forEach(function (st) {
        var p = pos[st.id];
        var color = TRUST_COLORS[st.trust] || "#8a97a3";
        g.strokeStyle = color; g.lineWidth = 2 * s.d;
        g.beginPath(); g.rect(p[0], p[1] - 14 * s.d, 110 * s.d, 28 * s.d); g.stroke();
        g.fillStyle = TEXT; g.font = (10 * s.d) + "px monospace";
        g.fillText(String(st.label).slice(0, 18), p[0] + 6 * s.d, p[1] + 4 * s.d);
      });
    }
    draw();
    var ro = new ResizeObserver(draw); ro.observe(canvas);
    return function () { ro.disconnect(); };
  };

  /* -- 3D (vendored THREE global) -- */
  function render3d(container, block, ctx, geometry) {
    if (typeof THREE === "undefined") {
      container.appendChild(el("div", "mkv-error",
        "Interactive 3D rendering unavailable (Three.js runtime not embedded). " +
        "The embedded result and metadata remain accessible in the inspector."));
      return null;
    }
    var probe = document.createElement("canvas");
    if (!probe.getContext("webgl2")) {
      container.appendChild(el("div", "mkv-error",
        "Interactive 3D rendering unavailable (WebGL2 not supported). " +
        "The embedded result and metadata remain accessible in the inspector."));
      return null;
    }
    var cfg = block.config || {};
    var wrap = el("div", "mkv-3d");
    wrap.style.height = (cfg.height || 460) + "px";
    container.appendChild(wrap);
    var hint = el("div", "mkv-hint",
      "drag to rotate · right-drag / shift-drag to pan · wheel to zoom · double-click to reset");
    wrap.appendChild(hint);

    var W = wrap.clientWidth || 640, H = wrap.clientHeight || 460;
    var renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(W, H);
    wrap.appendChild(renderer.domElement);
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(cfg.background || BG);
    var camera = new THREE.PerspectiveCamera(45, W / H, 1e-4, 1e9);
    scene.add(new THREE.AmbientLight(0xffffff, 0.9));

    /* gather point sets per series (raw coordinates; camera fits the bbox) */
    var sets = [];
    if (cfg.embed) {
      var src = ctx.decoded[cfg.embed.dataset];
      if (src) {
        sets.push({ rows: lagEmbed(src, cfg.embed),
                    meta: { label: block.title || "embedding", trust: "unknown",
                            style: {} } });
      }
    }
    (block.series || []).forEach(function (sid, i) {
      var sr = ctx.doc.series[sid];
      if (!sr || !sr.x || !ctx.decoded[sr.x]) return;
      var raw = ctx.decoded[sr.x], rows = [];
      for (var k = 0; k + 2 < raw.length; k += 3) rows.push([raw[k], raw[k + 1], raw[k + 2]]);
      sets.push({ rows: rows, meta: sr, idx: i });
    });

    var bbox = new THREE.Box3();
    sets.forEach(function (st) {
      st.rows.forEach(function (r) {
        bbox.expandByPoint(new THREE.Vector3(r[0], r[1], r[2]));
      });
    });
    if (geometry === "surface_3d" || geometry === "vector_field_3d") {
      (block.datasets || []).forEach(function (did) {
        var raw = ctx.decoded[did];
        if (!raw) return;
        for (var k = 0; k + 2 < raw.length; k += 3) {
          bbox.expandByPoint(new THREE.Vector3(raw[k], raw[k + 1], raw[k + 2]));
        }
      });
    }
    if (bbox.isEmpty()) bbox.setFromPoints([new THREE.Vector3(-0.5, -0.5, -0.5),
                                            new THREE.Vector3(0.5, 0.5, 0.5)]);
    var center = new THREE.Vector3(); bbox.getCenter(center);
    var size = new THREE.Vector3(); bbox.getSize(size);
    var extent = Math.max(size.x, size.y, size.z) || 1;

    scene.add(new THREE.AxesHelper(extent * 0.6));
    var boxEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x || extent, size.y || extent, size.z || extent)),
      new THREE.LineBasicMaterial({ color: 0x263640, transparent: true, opacity: 0.8 }));
    boxEdges.position.copy(center);
    scene.add(boxEdges);

    var pointObjects = [];
    if (geometry === "point_cloud_3d" || geometry === "trajectory_3d") {
      sets.forEach(function (st) {
        var pos = new Float32Array(st.rows.length * 3);
        for (var i = 0; i < st.rows.length; i++) {
          pos[i * 3] = st.rows[i][0]; pos[i * 3 + 1] = st.rows[i][1]; pos[i * 3 + 2] = st.rows[i][2];
        }
        var g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        var color = (st.meta.style && st.meta.style.color) || PALETTE[(st.idx || 0) % PALETTE.length];
        if (geometry === "trajectory_3d") {
          scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: color })));
        }
        if (geometry === "point_cloud_3d" || cfg.show_points) {
          var mat = new THREE.PointsMaterial({
            color: color, size: cfg.point_size || extent * 0.005,
            sizeAttenuation: true, transparent: true,
            opacity: cfg.opacity != null ? cfg.opacity : 0.74
          });
          var pts = new THREE.Points(g, mat);
          pts.userData.series = st.meta;
          scene.add(pts);
          pointObjects.push(pts);
        }
      });
    } else if (geometry === "surface_3d") {
      var did0 = (block.datasets || [])[0];
      var vals = ctx.decoded[did0] || [];
      var rows = cfg.rows || 1, cols = cfg.cols || 1;
      var xr = cfg.x_range || [0, cols - 1], yr = cfg.y_range || [0, rows - 1];
      var sg = new THREE.BufferGeometry();
      var sp = new Float32Array(rows * cols * 3);
      var q = 0;
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          sp[q++] = (c / (cols - 1 || 1)) * (xr[1] - xr[0]) + xr[0];
          sp[q++] = vals[r * cols + c] || 0;
          sp[q++] = (r / (rows - 1 || 1)) * (yr[1] - yr[0]) + yr[0];
        }
      }
      var idx = [];
      for (var r2 = 0; r2 < rows - 1; r2++) {
        for (var c2 = 0; c2 < cols - 1; c2++) {
          var a = r2 * cols + c2, b = a + 1, d2 = a + cols, e2 = d2 + 1;
          idx.push(a, d2, b, b, d2, e2);
        }
      }
      sg.setAttribute("position", new THREE.BufferAttribute(sp, 3));
      sg.setIndex(idx);
      sg.computeVertexNormals();
      scene.add(new THREE.Mesh(sg, new THREE.MeshLambertMaterial({
        color: 0x176795, side: THREE.DoubleSide })));
      scene.add(new THREE.Mesh(sg, new THREE.MeshBasicMaterial({
        color: 0x35a9ef, wireframe: true, transparent: true, opacity: 0.18 })));
      var light = new THREE.DirectionalLight(0xffffff, 1.2);
      light.position.set(1, 2, 1);
      scene.add(light);
      bbox.setFromBufferAttribute(sg.getAttribute("position"));
      bbox.getCenter(center); bbox.getSize(size);
      extent = Math.max(size.x, size.y, size.z) || 1;
    } else if (geometry === "vector_field_3d") {
      var didO = (block.datasets || [])[0], didV = (block.datasets || [])[1];
      var o = ctx.decoded[didO] || [], v = ctx.decoded[didV] || [];
      var nArrows = Math.min(Math.floor(o.length / 3), 5000);
      for (var i2 = 0; i2 < nArrows; i2++) {
        var start = new THREE.Vector3(o[i2 * 3], o[i2 * 3 + 1], o[i2 * 3 + 2]);
        var dir = new THREE.Vector3(v[i2 * 3], v[i2 * 3 + 1], v[i2 * 3 + 2]);
        var len = dir.length() || 1;
        scene.add(new THREE.ArrowHelper(dir.normalize(), start,
          extent * 0.06 * Math.min(len, 2), 0x35a9ef, extent * 0.02, extent * 0.012));
      }
    }

    var cam = cfg.camera || {};
    var controls = makeOrbit(camera, renderer.domElement, {
      target: cam.target || [center.x, center.y, center.z],
      radius: cam.radius || extent * 1.9,
      theta: cam.theta, phi: cam.phi
    });

    /* hover inspection: raycast points, show mathematical identity */
    var raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: Math.max(extent * 0.012, 1e-9) };
    var mouse = new THREE.Vector2();
    var mouseMoved = false;
    var tip = byId("mkv-tip");
    if (cfg.inspect !== false && pointObjects.length) {
      renderer.domElement.addEventListener("pointermove", function (e) {
        var rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        mouseMoved = true;
        tip.style.left = (e.clientX + 14) + "px";
        tip.style.top = (e.clientY + 14) + "px";
      });
      renderer.domElement.addEventListener("pointerleave", function () {
        tip.style.display = "none"; mouseMoved = false;
      });
    }
    function hover() {
      if (!mouseMoved || controls.isInteracting()) return;
      mouseMoved = false;
      raycaster.setFromCamera(mouse, camera);
      var hits = raycaster.intersectObjects(pointObjects);
      if (hits.length) {
        var h = hits[0];
        var sr = h.object.userData.series || {};
        var p = h.object.geometry.getAttribute("position");
        var txt = (sr.label || "point") + " #" + h.index +
          "\n(" + fmtTick(p.getX(h.index)) + ", " + fmtTick(p.getY(h.index)) +
          ", " + fmtTick(p.getZ(h.index)) + ")" +
          "\ntrust: " + (sr.trust || "unknown") +
          (sr.source ? " | source: " + sr.source : "");
        tip.style.display = "block";
        tip.textContent = txt;
      } else {
        tip.style.display = "none";
      }
    }

    var raf = 0, disposed = false;
    (function loop() {
      if (disposed) return;
      controls.update();
      hover();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    })();
    var ro = new ResizeObserver(function () {
      var w = wrap.clientWidth, h = wrap.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(wrap);
    container._mkvRenderer = renderer;
    return function () {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.dispose();
      if (renderer.forceContextLoss) renderer.forceContextLoss();
    };
  }

  RENDERERS.point_cloud_3d = function (c, b, x) { return render3d(c, b, x, "point_cloud_3d"); };
  RENDERERS.trajectory_3d = function (c, b, x) { return render3d(c, b, x, "trajectory_3d"); };
  RENDERERS.surface_3d = function (c, b, x) { return render3d(c, b, x, "surface_3d"); };
  RENDERERS.vector_field_3d = function (c, b, x) { return render3d(c, b, x, "vector_field_3d"); };

  /* -- metric grid -- */
  RENDERERS.metric_grid = function (container, block, ctx) {
    var cfg = block.config || {};
    var entries = cfg.entries ? cfg.entries.map(function (e) {
      return [e.label, String(e.value), e.hint || ""];
    }) : [];
    if (cfg.dataset && ctx.decoded[cfg.dataset]) {
      basicStats(ctx.decoded[cfg.dataset], cfg.stats).forEach(function (row) {
        entries.push([row[0], row[1], "computed from " + cfg.dataset]);
      });
    }
    var grid = el("div", "mkv-metrics");
    entries.forEach(function (e) {
      var card = el("div", "mkv-metric");
      card.appendChild(el("span", null, e[0]));
      card.appendChild(el("strong", null, e[1]));
      if (e[2]) card.appendChild(el("em", null, e[2]));
      grid.appendChild(card);
    });
    if (!entries.length) grid.appendChild(el("p", "mkv-dim", "no metrics"));
    container.appendChild(grid);
    return null;
  };

  /* -- data table -- */
  RENDERERS.data_table = function (container, block, ctx) {
    var cfg = block.config || {};
    var maxRows = cfg.max_rows || 100;
    var table = el("table", "mkv-table");
    var thead = el("thead"), tbody = el("tbody");
    var headRow = el("tr");
    var cols = cfg.columns || [];
    var colVals = cols.map(function (c) {
      headRow.appendChild(el("th", null, c.label || c.dataset || ""));
      if (c.dataset && ctx.decoded[c.dataset]) return ctx.decoded[c.dataset];
      return c.values || [];
    });
    var rows = cfg.rows || [];
    var nRows = rows.length ||
      Math.max.apply(null, colVals.map(function (v) { return v.length; }).concat([0]));
    thead.appendChild(headRow);
    table.appendChild(thead);
    var shown = Math.min(nRows, maxRows);
    for (var i = 0; i < shown; i++) {
      var tr = el("tr");
      if (rows.length) {
        rows[i].forEach(function (v) { tr.appendChild(el("td", null, v)); });
      } else {
        colVals.forEach(function (v) {
          tr.appendChild(el("td", null, i < v.length ? fmtTick(v[i]) : ""));
        });
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    container.appendChild(table);
    if (nRows > shown) {
      container.appendChild(el("p", "mkv-dim",
        "showing " + shown.toLocaleString() + " of " + nRows.toLocaleString() + " rows"));
    }
    return null;
  };

  /* -- text -- */
  RENDERERS.text = function (container, block) {
    var cfg = block.config || {};
    var box = el("div", "mkv-text");
    String(cfg.text || "").split("\n").forEach(function (line) {
      box.appendChild(el("p", null, line));
    });
    container.appendChild(box);
    return null;
  };

  /* -- select control: publishes a parameter, re-renders bound blocks -- */
  RENDERERS.select = function (container, block, ctx) {
    var cfg = block.config || {};
    var field = el("div", "mkv-field");
    field.appendChild(el("label", null, cfg.label || cfg.param));
    var sel = el("select");
    (cfg.options || []).forEach(function (o, i) {
      var opt = el("option", null, o.label);
      opt.value = String(i);
      sel.appendChild(opt);
    });
    sel.value = String(Math.min(cfg.default || 0, (cfg.options || []).length - 1));
    field.appendChild(sel);
    container.appendChild(field);
    function publish() {
      var opt = (cfg.options || [])[Number(sel.value)];
      if (opt) ctx.publish(cfg.param, opt.value);
    }
    sel.addEventListener("change", publish);
    if (cfg.options && cfg.options.length) publish();
    return null;
  };

  /* ---- layout / composition ---------------------------------------------- */

  var params = {};
  var mounted = {};   /* block_id -> {container, block, cleanup} */
  var blockClickCbs = [];
  var highlighted = [];

  function setPath(obj, path, value) {
    var keys = path.split(".");
    var o = obj;
    for (var i = 0; i < keys.length - 1; i++) {
      if (typeof o[keys[i]] !== "object" || o[keys[i]] === null) o[keys[i]] = {};
      o = o[keys[i]];
    }
    o[keys[keys.length - 1]] = value;
  }

  function deepMerge(dst, src) {
    Object.keys(src).forEach(function (k) {
      if (src[k] && typeof src[k] === "object" && !Array.isArray(src[k]) &&
          dst[k] && typeof dst[k] === "object" && !Array.isArray(dst[k])) {
        deepMerge(dst[k], src[k]);
      } else {
        dst[k] = src[k];
      }
    });
  }

  function applyParams(block) {
    Object.keys(block.bindings || {}).forEach(function (target) {
      var param = block.bindings[target];
      if (!(param in params)) return;
      if (target === "*") {
        var v = params[param];
        if (v && typeof v === "object") deepMerge(block.config, v);
      } else {
        setPath(block.config, target, params[param]);
      }
    });
  }

  function renderBlock(shell, block) {
    var prev = mounted[block.block_id];
    if (prev && prev.cleanup) prev.cleanup();
    var content = shell.querySelector(".mkv-block-body");
    content.textContent = "";
    var renderer = RENDERERS[block.kind];
    var cleanup = null;
    if (renderer) {
      cleanup = renderer(content, block, { doc: doc, decoded: decodedValues,
                                           publish: publish });
    } else {
      content.appendChild(el("div", "mkv-error", "unknown block kind: " + block.kind));
    }
    mounted[block.block_id] = { cleanup: cleanup };
  }

  function publish(param, value) {
    params[param] = value;
    (doc.blocks || []).forEach(function (b) {
      var bound = Object.keys(b.bindings || {}).some(function (t) {
        return b.bindings[t] === param;
      });
      if (bound) {
        applyParams(b);
        var shell = byId("mkv-block-" + b.block_id);
        if (shell) renderBlock(shell, b);
      }
    });
  }

  var decodedValues = {};

  function buildLayout() {
    var view = byId("mkv-view");
    view.textContent = "";
    var grid = el("div", "mkv-grid");
    var cols = (doc.layout && doc.layout.cols) || 2;
    grid.style.gridTemplateColumns = "repeat(" + cols + ", minmax(0, 1fr))";
    view.appendChild(grid);
    (doc.blocks || []).forEach(function (b) {
      applyParams(b);
      var shell = el("section", "mkv-block");
      shell.id = "mkv-block-" + b.block_id;
      shell.setAttribute("data-mkv", "mkv-block-" + b.block_id);
      shell.setAttribute("data-mkv-block", b.block_id);
      if (b.span > 1) shell.style.gridColumn = "span " + Math.min(b.span, cols);
      var head = el("header", "mkv-block-head");
      head.appendChild(el("h3", null, b.title || b.kind));
      var tb = el("span", "mkv-block-trust", b.trust);
      tb.style.color = TRUST_COLORS[b.trust] || "#8a97a3";
      head.appendChild(tb);
      head.addEventListener("click", function () {
        blockClickCbs.forEach(function (cb) { cb(b.block_id); });
      });
      shell.appendChild(head);
      shell.appendChild(el("div", "mkv-block-body"));
      grid.appendChild(shell);
      renderBlock(shell, b);
    });
  }

  /* ---- inspector ---------------------------------------------------------- */

  function fillInspector() {
    function tab(id, build) {
      var pane = byId(id);
      if (!pane) return;
      pane.textContent = "";
      build(pane);
    }
    tab("mkv-tab-result", function (p) {
      p.appendChild(el("h3", null, doc.title));
      p.appendChild(el("p", null, "Engine: " + (doc.engine || "unknown")));
      var badge = el("span", "mkv-trust", "trust: " + doc.trust);
      badge.style.color = TRUST_COLORS[doc.trust] || "#8a97a3";
      p.appendChild(badge);
      (doc.assumptions || []).forEach(function (a) {
        p.appendChild(el("p", "mkv-dim", "assumption: " + a));
      });
      p.appendChild(el("h4", null, "Blocks"));
      (doc.blocks || []).forEach(function (b) {
        p.appendChild(el("p", null, b.kind + "  [" + b.trust + "]" +
          (b.title ? "  — " + b.title : "")));
      });
      var skeys = Object.keys(doc.series || {});
      if (skeys.length) {
        p.appendChild(el("h4", null, "Series"));
        skeys.forEach(function (k) {
          var s = doc.series[k];
          var line = (s.label || k) + "  [" + s.role + ", " + s.trust + "]";
          if (s.enclosure) {
            line += "  enclosure [" + s.enclosure.lower + ", " + s.enclosure.upper + "]";
          }
          p.appendChild(el("p", null, line));
        });
      }
    });
    tab("mkv-tab-evidence", function (p) {
      integrityLines.forEach(function (l) {
        p.appendChild(el("p", null, l[0] + ": " + l[1]));
        if (l[2]) p.appendChild(el("p", "mkv-dim mkv-hash", l[2]));
      });
      var linked = doc.linked_result || {};
      var bundle = linked.evidence_bundle || {};
      var claims = linked.claim_evidence || {};
      if (Object.keys(bundle).length) {
        p.appendChild(el("h4", null, "Evidence bundle"));
        p.appendChild(el("pre", "mkv-json", JSON.stringify(bundle, null, 2)));
      }
      if (Object.keys(claims).length) {
        p.appendChild(el("h4", null, "Claim-specific evidence"));
        p.appendChild(el("pre", "mkv-json", JSON.stringify(claims, null, 2)));
      }
      if (!integrityLines.length && !Object.keys(bundle).length &&
          !Object.keys(claims).length) {
        p.appendChild(el("p", null, "No evidence embedded."));
      }
    });
    tab("mkv-tab-provenance", function (p) {
      var steps = (doc.provenance && doc.provenance.steps) || [];
      if (!steps.length) { p.appendChild(el("p", null, "No provenance recorded.")); return; }
      steps.forEach(function (s) {
        var d = el("div", "mkv-step");
        d.appendChild(el("b", null, s.operation));
        d.appendChild(el("span", "mkv-dim",
          "  " + s.step_id + " | engine=" + (s.engine || "-") + " | trust=" + s.trust));
        if (s.output) d.appendChild(el("div", "mkv-dim", "-> " + s.output));
        p.appendChild(d);
      });
    });
    tab("mkv-tab-data", function (p) {
      Object.keys(doc.datasets || {}).forEach(function (k) {
        var ds = doc.datasets[k];
        p.appendChild(el("p", null, k + ": " + ds.kind + " " +
          JSON.stringify(ds.shape) + " (" + ds.encoding + ", " +
          ds.byte_length + " bytes, trust=" + ds.trust + ", role=" + ds.role + ")"));
      });
    });
    tab("mkv-tab-repro", function (p) {
      var m = doc.manifest || {};
      Object.keys(m).forEach(function (k) {
        if (k === "dataset_hashes") return;
        p.appendChild(el("p", null, k + ": " + JSON.stringify(m[k])));
      });
    });
  }

  function wireExport() {
    var pngBtn = byId("mkv-exp-png");
    var jsonBtn = byId("mkv-exp-json");
    var csvBtn = byId("mkv-exp-csv");
    var svgBtn = byId("mkv-exp-svg");
    if (!pngBtn && !jsonBtn && !csvBtn && !svgBtn) return;
    if (pngBtn) pngBtn.addEventListener("click", function () {
      var src = null;
      var view = byId("mkv-view");
      var three = view.querySelector(".mkv-3d");
      if (three && three.parentElement && three.parentElement._mkvRenderer) {
        src = three.parentElement._mkvRenderer.domElement;
      }
      if (!src) src = view.querySelector("canvas");
      if (!src) return;
      var a = document.createElement("a");
      a.download = "mathkernel-artifact.png";
      a.href = src.toDataURL("image/png");
      a.click();
    });
    if (jsonBtn) jsonBtn.addEventListener("click", function () {
      var a = document.createElement("a");
      a.download = "mathkernel-artifact.json";
      var raw = payloadText !== null ? payloadText : JSON.stringify(doc);
      a.href = "data:application/json;charset=utf-8," + encodeURIComponent(raw);
      a.click();
    });
    if (csvBtn) csvBtn.addEventListener("click", function () {
      var rows = [];
      Object.keys(doc.series || {}).forEach(function (k) {
        var s = doc.series[k];
        var xy = seriesXY(s, decodedValues);
        if (!xy) return;
        rows.push("# " + (s.label || k) + " [" + s.role + ", " + s.trust +
          ", source=" + (s.source || "MathKernel") + "]");
        for (var i = 0; i < xy[0].length; i++) rows.push(xy[0][i] + "," + xy[1][i]);
      });
      var a = document.createElement("a");
      a.download = "mathkernel-artifact.csv";
      a.href = "data:text/csv;charset=utf-8," + encodeURIComponent(rows.join("\n"));
      a.click();
    });
    var embedded = byId("mathkernel-static-svg");
    if (svgBtn && embedded) {
      svgBtn.addEventListener("click", function () {
        var a = document.createElement("a");
        a.download = "mathkernel-artifact.svg";
        a.href = "data:image/svg+xml;charset=utf-8," +
          encodeURIComponent(embedded.innerHTML);
        a.click();
      });
    } else if (svgBtn) {
      svgBtn.disabled = true;
    }
  }

  /* ---- cross-modal sync API ---------------------------------------------- */

  function blockSources() {
    var map = {};
    (doc.blocks || []).forEach(function (b) {
      var refs = [];
      (b.series || []).forEach(function (sid) {
        var s = (doc.series || {})[sid];
        if (s && s.source_ref && s.source_ref.source_id) {
          refs.push(s.source_ref.source_id);
        } else if (s && s.source) {
          refs.push(s.source);
        }
      });
      (b.datasets || []).forEach(function (did) {
        var d = (doc.datasets || {})[did];
        if (d && d.source_ref && d.source_ref.source_id) {
          refs.push(d.source_ref.source_id);
        } else if (d && d.source) {
          refs.push(d.source);
        }
      });
      map[b.block_id] = refs;
    });
    return map;
  }

  function clearHighlight() {
    highlighted.forEach(function (s) { s.classList.remove("mkv-sync-hl"); });
    highlighted = [];
  }

  function highlight(blockIds) {
    clearHighlight();
    (blockIds || []).forEach(function (id) {
      var shell = byId("mkv-block-" + id);
      if (shell) {
        shell.classList.add("mkv-sync-hl");
        highlighted.push(shell);
      }
    });
  }

  var instance = {
    doc: doc,
    root: root,
    blockSources: blockSources,
    highlight: highlight,
    clearHighlight: clearHighlight,
    onBlockClick: function (cb) {
      blockClickCbs.push(cb);
      if (root !== document) root.classList.add("mkv-syncable");
    }
  };

  /* ---- boot --------------------------------------------------------------- */

  instance.ready = checkIntegrity().then(function () {
    var names = Object.keys(doc.datasets || {});
    var p = Promise.resolve();
    names.forEach(function (name) {
      p = p.then(function () {
        return decodeDataset(doc.datasets[name]).then(function (vals) {
          decodedValues[name] = vals;
        }).catch(function () {
          decodedValues[name] = null;
        });
      });
    });
    return p.then(function () {
      buildLayout();
      fillInspector();
      wireExport();
      return instance;
    });
  });

  return instance;
  }

  /* ---- public registry + legacy single-artifact boot ---------------------- */

  global.MathKernelViz = { mount: mount, SCHEMA: SCHEMA };

  var bootPayload = document.getElementById("mathkernel-data");
  if (bootPayload && document.getElementById("mkv-view")) {
    mount(document, bootPayload);
  }
})(window);
