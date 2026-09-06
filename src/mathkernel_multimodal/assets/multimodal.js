/* =============================================================================
 * MathKernel Multimodal - unified research artifact runtime
 * Copyright (c) 2026 Maarten Boone
 * SPDX-License-Identifier: MIT
 *
 * Hosts any number of visualization documents (via MathKernelViz.mount) and
 * sonification documents (fixed WebAudio interpreter) in one offline page,
 * wired together through the artifact's SynchronizationLink records:
 *
 *   audio time -> active events -> source refs -> highlighted visual blocks
 *   visual block selection -> synchronization link -> audio seek
 *
 * Framework-free classic script (file:// safe).  All embedded strings are
 * untrusted display data: textContent only, never innerHTML, never eval.
 * ===========================================================================*/
(function (global) {
  "use strict";

  var ARTIFACT_SCHEMA = "mathkernel-artifact/1.0";
  var TRUST_COLORS = {
    formal: "#b18cff", exact: "#7ce38b", symbolic: "#5aa9e6",
    interval_certified: "#5ad4d4", numeric_high_precision: "#efb34c",
    numeric: "#f0924c", empirical: "#e86f6f", heuristic: "#e86f9e",
    unknown: "#8a97a3"
  };

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }

  function sha256Hex(buffer) {
    if (!global.crypto || !crypto.subtle) return Promise.resolve(null);
    return crypto.subtle.digest("SHA-256", buffer).then(function (hash) {
      return Array.prototype.map.call(new Uint8Array(hash), function (b) {
        return ("0" + b.toString(16)).slice(-2);
      }).join("");
    }).catch(function () { return null; });
  }

  function fmt(t) { return Number(t).toFixed(3); }

  /* ---- payload ------------------------------------------------------------ */

  var payloadEl = document.getElementById("mathkernel-artifact");
  var view = document.getElementById("mkm-content");
  if (!payloadEl || !view) return;
  var artifact;
  try { artifact = JSON.parse(payloadEl.textContent); }
  catch (e) {
    view.appendChild(el("div", "mkm-error", "Artifact payload is not valid JSON."));
    return;
  }
  if (artifact.artifact_schema !== ARTIFACT_SCHEMA) {
    view.appendChild(el("div", "mkm-error",
      "Unsupported artifact schema: " + String(artifact.artifact_schema) +
      " (this viewer understands " + ARTIFACT_SCHEMA + ")."));
    return;
  }

  var integrityLines = [];
  var expected = payloadEl.getAttribute("data-sha256") || null;
  sha256Hex(new TextEncoder().encode(payloadEl.textContent)).then(function (actual) {
    if (expected && actual) {
      integrityLines.push(["Artifact payload",
        actual === expected ? "VERIFIED" : "MISMATCH", actual]);
    } else if (expected) {
      integrityLines.push(["Artifact payload",
        "hash present (WebCrypto unavailable)", expected]);
    }
    fillInspectors();
  });

  /* ---- fixed declarative mapping interpreter (parity with ------------------
   * mathkernel_sonify.mapping.apply_transform; data in, numbers out, no eval) */

  function applyTransform(value, spec) {
    var t = (spec && spec.type) || "identity";
    var x = Number(value);
    if (t === "identity") return x;
    if (t === "linear") return (Number(spec.offset) || 0) +
      (spec.scale === undefined ? 1 : Number(spec.scale)) * x;
    if (t === "abs") return Math.abs(x);
    if (t === "log") return Math.log(Math.max(x, Number(spec.floor) || 1e-300)) /
      Math.log(spec.base === undefined ? Math.E : Number(spec.base));
    if (t === "clamp") return Math.min(Number(spec.max), Math.max(Number(spec.min), x));
    if (t === "normalize") {
      var lo = Number(spec.source_min) || 0, hi = spec.source_max === undefined ? 1 : Number(spec.source_max);
      var a = Number(spec.target_min) || 0, b = spec.target_max === undefined ? 1 : Number(spec.target_max);
      return hi === lo ? (a + b) / 2 : a + (x - lo) * (b - a) / (hi - lo);
    }
    if (t === "log_map") {
      var lo2 = Number(spec.source_min) || 0, hi2 = spec.source_max === undefined ? 1 : Number(spec.source_max);
      var a2 = Number(spec.target_min), b2 = Number(spec.target_max);
      var u = hi2 === lo2 ? 0.5 : Math.max(0, Math.min(1, (x - lo2) / (hi2 - lo2)));
      return a2 * Math.pow(b2 / a2, u);
    }
    if (t === "harmonic") return (spec.fundamental === undefined ? 110 : Number(spec.fundamental)) * x;
    throw new Error("unsupported sonification transform " + t);
  }

  /* ---- sonification players ------------------------------------------------ */

  var players = [];
  var sessionAnnotations = [];

  function eventParams(track, ev, mappings) {
    var f = ev.values.frequency !== undefined ? Number(ev.values.frequency) : 440;
    var g = ev.values.gain !== undefined ? Number(ev.values.gain) : 1;
    var p = ev.values.coefficient_phase !== undefined ? Number(ev.values.coefficient_phase)
      : (ev.values.phase !== undefined ? Number(ev.values.phase) : 0);
    var pan = track.pan || 0;
    (track.mapping_refs || []).forEach(function (id) {
      var m = mappings[id];
      if (!m || !(m.source in ev.values)) return;
      var v = applyTransform(ev.values[m.source], m.transform);
      if (m.target === "frequency") f = v;
      else if (m.target === "gain") g = v;
      else if (m.target === "phase") p = v;
      else if (m.target === "pan") pan = Math.max(-1, Math.min(1, v));
    });
    return { f: f, g: g, p: p, pan: pan };
  }

  function makePlayer(section, sonDoc, index) {
    var player = {
      index: index, doc: sonDoc, ctx: null, nodes: [], base: 0, offset: 0,
      playing: false, raf: 0, muted: {}, solo: {}, dur: 0
    };
    sonDoc.tracks.forEach(function (tr) {
      tr.events.forEach(function (ev) {
        player.dur = Math.max(player.dur, ev.time + ev.duration);
      });
    });

    var transport = el("div", "mkm-transport");
    var playBtn = el("button", null, "Play");
    var stopBtn = el("button", null, "Stop");
    var seek = el("input", "mkm-seek");
    seek.type = "range"; seek.min = "0"; seek.max = "1000"; seek.value = "0";
    var time = el("span", "mkm-time", "0.000 / " + fmt(player.dur) + " s");
    var mark = el("button", null, "Mark candidate");
    var note = el("input", "mkm-note");
    note.type = "text"; note.placeholder = "perceptual note (stays a candidate)";
    transport.appendChild(playBtn); transport.appendChild(stopBtn);
    transport.appendChild(seek); transport.appendChild(time);
    transport.appendChild(mark); transport.appendChild(note);
    section.appendChild(transport);

    var tracksBox = el("div", "mkm-tracks");
    sonDoc.tracks.forEach(function (tr, ti) {
      var row = el("label", "mkm-track");
      var mute = el("input"); mute.type = "checkbox";
      var solo = el("input"); solo.type = "checkbox";
      mute.addEventListener("change", function () {
        player.muted[ti] = mute.checked;
        if (player.playing) playFrom(currentTime());
      });
      solo.addEventListener("change", function () {
        player.solo[ti] = solo.checked;
        if (player.playing) playFrom(currentTime());
      });
      row.appendChild(el("span", "mkm-track-name",
        (tr.label || tr.track_id) + " [" + tr.role + ", " + tr.trust + "]"));
      row.appendChild(el("span", null, "mute")); row.appendChild(mute);
      row.appendChild(el("span", null, "solo")); row.appendChild(solo);
      tracksBox.appendChild(row);
    });
    section.appendChild(tracksBox);

    function currentTime() {
      if (!player.playing || !player.ctx) return player.offset;
      return Math.min(player.dur,
        player.offset + player.ctx.currentTime - player.base);
    }

    function updateTime() {
      var t = currentTime();
      time.textContent = fmt(t) + " / " + fmt(player.dur) + " s";
      seek.value = String(Math.round(1000 * (player.dur ? t / player.dur : 0)));
      onPlayerTime(index, t);
    }

    function tick() {
      if (!player.playing) return;
      updateTime();
      if (currentTime() >= player.dur) { stop(); return; }
      player.raf = requestAnimationFrame(tick);
    }

    function stop() {
      cancelAnimationFrame(player.raf);
      player.nodes.forEach(function (n) { try { n.stop(); } catch (e) { } });
      player.nodes = [];
      if (player.ctx) { player.ctx.close(); player.ctx = null; }
      player.playing = false;
      player.offset = 0;
      updateTime();
    }

    function playFrom(offset) {
      var i;
      cancelAnimationFrame(player.raf);
      player.nodes.forEach(function (n) { try { n.stop(); } catch (e) { } });
      player.nodes = [];
      if (player.ctx) { player.ctx.close(); player.ctx = null; }
      var render = sonDoc.render || {};
      var ctx = new AudioContext({ sampleRate: render.sample_rate || 48000 });
      if (ctx.state === "suspended") ctx.resume();
      player.ctx = ctx;
      player.offset = Math.max(0, Math.min(player.dur, offset || 0));
      player.base = ctx.currentTime + 0.05;
      var peak = render.peak_limit === undefined ? 0.9 : Number(render.peak_limit);
      var anySolo = Object.keys(player.solo).some(function (k) { return player.solo[k]; });
      sonDoc.tracks.forEach(function (tr, ti) {
        if (player.muted[ti]) return;
        if (anySolo && !player.solo[ti]) return;
        tr.events.forEach(function (ev) {
          var start = ev.time, end = ev.time + ev.duration;
          if (end <= player.offset) return;
          var prm;
          try { prm = eventParams(tr, ev, sonDoc.mappings || {}); }
          catch (e) { return; }
          if (!(prm.f > 0) || prm.f >= ctx.sampleRate / 2) return;  /* no silent folding */
          var o = ctx.createOscillator();
          var gn = ctx.createGain();
          var pn = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
          o.frequency.value = prm.f;
          /* OscillatorNode cannot set start phase; phase is preserved in the
             document/mappings and applied by the deterministic PCM renderer. */
          gn.gain.value = Math.min(peak,
            Math.abs(prm.g * (tr.gain === undefined ? 1 : tr.gain)) * 0.25);
          if (pn) { pn.pan.value = prm.pan; o.connect(gn).connect(pn).connect(ctx.destination); }
          else { o.connect(gn).connect(ctx.destination); }
          o.start(player.base + Math.max(0, start - player.offset));
          o.stop(player.base + (end - player.offset));
          player.nodes.push(o);
        });
      });
      player.playing = true;
      player.raf = requestAnimationFrame(tick);
    }

    playBtn.addEventListener("click", function () { playFrom(player.offset); });
    stopBtn.addEventListener("click", stop);
    seek.addEventListener("input", function () {
      var t = player.dur * Number(seek.value) / 1000;
      if (player.playing) playFrom(t);
      else { player.offset = t; updateTime(); }
    });
    mark.addEventListener("click", function () {
      var t = currentTime();
      var refs = [];
      sonDoc.tracks.forEach(function (tr) {
        tr.events.forEach(function (ev) {
          if (ev.source_ref && t >= ev.time && t <= ev.time + ev.duration &&
              refs.indexOf(ev.source_ref) < 0) refs.push(ev.source_ref);
        });
      });
      sessionAnnotations.push({
        text: note.value || "perceptual candidate",
        kind: "perceptual", claim_status: "candidate", author_type: "human",
        sonification: "s" + index, time_range: [t, t], source_refs: refs
      });
      note.value = "";
      fillAnnotations();
    });

    player.seekAndPlay = function (t) { playFrom(t); };
    player.currentTime = currentTime;
    players[index] = player;
    return player;
  }

  /* ---- cross-modal synchronization ----------------------------------------- */

  var vizInstances = [];
  var links = artifact.synchronization || [];

  function onPlayerTime(sonIndex, t) {
    var targets = {};
    links.forEach(function (L) {
      if (L.sonification_ref && L.sonification_ref !== "s" + sonIndex) return;
      if (L.time_range && (t < L.time_range[0] || t > L.time_range[1])) return;
      var m = /^v(\d+):(.+)$/.exec(L.visual_ref || "");
      if (!m) return;
      var vi = Number(m[1]);
      (targets[vi] = targets[vi] || []).push(m[2]);
    });
    vizInstances.forEach(function (inst, vi) {
      if (inst) inst.highlight(targets[vi] || []);
    });
    var live = document.getElementById("mkm-sync-live");
    if (live) {
      var n = Object.keys(targets).reduce(function (a, k) {
        return a + targets[k].length; }, 0);
      live.textContent = n
        ? ("s" + sonIndex + " @ " + fmt(t) + " s -> " + n + " visual block(s)")
        : "no active link";
    }
  }

  function onBlockClick(vizIndex, blockId) {
    var ref = "v" + vizIndex + ":" + blockId;
    for (var i = 0; i < links.length; i++) {
      var L = links[i];
      if (L.visual_ref !== ref || !L.time_range) continue;
      var m = /^s(\d+)$/.exec(L.sonification_ref || "");
      if (m && players[Number(m[1])]) {
        players[Number(m[1])].seekAndPlay(L.time_range[0]);
        var live = document.getElementById("mkm-sync-live");
        if (live) {
          live.textContent = ref + " -> s" + m[1] + " @ " +
            fmt(L.time_range[0]) + " s";
        }
        return;
      }
    }
  }

  /* ---- layout -------------------------------------------------------------- */

  (artifact.visualizations || []).forEach(function (vizDoc, i) {
    var section = el("section", "mkm-viz");
    section.setAttribute("data-mkm-viz", String(i));
    var head = el("div", "mkm-sec-head");
    head.appendChild(el("h2", null, vizDoc.title || ("Visualization " + (i + 1))));
    var badge = el("span", "mkm-badge", "trust: " + (vizDoc.trust || "unknown"));
    badge.style.color = TRUST_COLORS[vizDoc.trust] || "#8a97a3";
    head.appendChild(badge);
    section.appendChild(head);
    var viewEl = el("div", "mkv-view");
    viewEl.setAttribute("data-mkv", "mkv-view");
    section.appendChild(viewEl);
    view.appendChild(section);
    if (global.MathKernelViz && MathKernelViz.mount) {
      var inst = MathKernelViz.mount(section, vizDoc);
      inst.onBlockClick(function (blockId) { onBlockClick(i, blockId); });
      vizInstances[i] = inst;
    } else {
      section.appendChild(el("div", "mkm-error",
        "Visualization runtime unavailable; embedded data remains inspectable."));
    }
  });

  (artifact.sonifications || []).forEach(function (sonDoc, j) {
    var section = el("section", "mkm-audio");
    section.setAttribute("data-mkm-audio", String(j));
    var head = el("div", "mkm-sec-head");
    head.appendChild(el("h2", null, sonDoc.title || ("Sonification " + (j + 1))));
    var badge = el("span", "mkm-badge", "trust: " + (sonDoc.trust || "unknown"));
    badge.style.color = TRUST_COLORS[sonDoc.trust] || "#8a97a3";
    head.appendChild(badge);
    section.appendChild(head);
    section.appendChild(el("p", "mkm-warn",
      "Audible patterns are candidate observations, not proof."));
    view.appendChild(section);
    makePlayer(section, sonDoc, j);
  });

  /* ---- inspectors ------------------------------------------------------------ */

  function kv(p, k, v) {
    var row = el("p");
    row.appendChild(el("strong", null, k + ": "));
    row.appendChild(el("span", null, v));
    p.appendChild(row);
  }

  function pre(p, obj) {
    var node = el("pre", "mkm-pre");
    node.textContent = JSON.stringify(obj, null, 2);
    p.appendChild(node);
  }

  function fillAnnotations() {
    var pane = document.getElementById("mkm-tab-annotations");
    if (!pane) return;
    pane.textContent = "";
    pane.appendChild(el("h3", null, "Annotations"));
    (artifact.annotations || []).forEach(function (a) {
      pane.appendChild(el("p", null, "[" + (a.kind || "note") + "/" +
        (a.claim_status || "none") + "] " + (a.text || "")));
    });
    sessionAnnotations.forEach(function (a) {
      pane.appendChild(el("p", "mkm-candidate",
        "[candidate @ " + fmt(a.time_range[0]) + " s, " + a.sonification +
        ", sources: " + (a.source_refs.join(", ") || "none") + "] " + a.text));
    });
    var exp = el("button", null, "Export annotations JSON");
    exp.addEventListener("click", function () {
      var a = document.createElement("a");
      a.download = "annotations.json";
      a.href = "data:application/json;charset=utf-8," + encodeURIComponent(
        JSON.stringify({ artifact_id: artifact.artifact_id,
                         stored: artifact.annotations || [],
                         session_candidates: sessionAnnotations }, null, 2));
      a.click();
    });
    pane.appendChild(exp);
  }

  function fillInspectors() {
    function tab(id, build) {
      var pane = document.getElementById(id);
      if (!pane) return;
      pane.textContent = "";
      build(pane);
    }
    tab("mkm-tab-result", function (p) {
      p.appendChild(el("h3", null, artifact.title || "MathKernel artifact"));
      kv(p, "artifact", artifact.artifact_id || "(unassigned)");
      kv(p, "trust", artifact.trust || "unknown");
      kv(p, "schema", artifact.artifact_schema);
      if (artifact.summary) p.appendChild(el("p", null, artifact.summary));
      if (artifact.result) pre(p, artifact.result);
      if ((artifact.assumptions || []).length) {
        p.appendChild(el("h4", null, "Assumptions"));
        artifact.assumptions.forEach(function (a) {
          p.appendChild(el("p", null, a));
        });
      }
    });
    tab("mkm-tab-evidence", function (p) {
      var ev = artifact.evidence || {};
      var keys = Object.keys(ev);
      keys.forEach(function (k) {
        var e = ev[k];
        p.appendChild(el("p", null, "[" + (e.kind || "other") + ", trust=" +
          (e.trust || "unknown") + (e.engine ? ", " + e.engine : "") + "] " +
          (e.claim || k)));
      });
      var bundle = artifact.evidence_bundle || {};
      var claims = artifact.claim_evidence || {};
      if (Object.keys(bundle).length) {
        p.appendChild(el("h4", null, "Evidence bundle"));
        pre(p, bundle);
      }
      if (Object.keys(claims).length) {
        p.appendChild(el("h4", null, "Claim-specific evidence"));
        pre(p, claims);
      }
      if (!keys.length && !Object.keys(bundle).length &&
          !Object.keys(claims).length) {
        p.appendChild(el("p", null, "No evidence items embedded."));
      }
    });
    tab("mkm-tab-provenance", function (p) {
      (artifact.transformations || []).forEach(function (t) {
        p.appendChild(el("p", null, t.operation + " (" + (t.purpose || "presentation") +
          "): " + (t.inputs || []).join(", ") + " -> " + (t.outputs || []).join(", ") +
          " " + JSON.stringify(t.parameters || {})));
      });
      if (!(artifact.transformations || []).length) {
        p.appendChild(el("p", null, "No presentation transformations recorded."));
      }
    });
    tab("mkm-tab-data", function (p) {
      var src = artifact.sources || {};
      Object.keys(src).forEach(function (k) {
        var s = src[k];
        p.appendChild(el("p", null, k + " [" + (s.kind || "other") + ", trust=" +
          (s.trust || "unknown") + (s.sha256 ? ", sha256=" + s.sha256.slice(0, 16) + "…" : "") + "]"));
      });
      if (!Object.keys(src).length) p.appendChild(el("p", null, "No sources recorded."));
    });
    tab("mkm-tab-repro", function (p) {
      pre(p, artifact.reproducibility || {});
      p.appendChild(el("h4", null, "Integrity"));
      integrityLines.forEach(function (row) {
        p.appendChild(el("p", null, row[0] + ": " + row[1] +
          (row[2] ? " (" + row[2] + ")" : "")));
      });
      var integ = artifact.integrity || {};
      Object.keys(integ).forEach(function (k) {
        p.appendChild(el("p", "mkm-hash", k + ": SHA-256 " + integ[k]));
      });
    });
    tab("mkm-tab-visual", function (p) {
      (artifact.visualizations || []).forEach(function (v, i) {
        p.appendChild(el("h4", null, "v" + i + ": " + (v.title || "visualization") +
          " (" + (v.artifact_schema || "?") + ")"));
        (v.blocks || []).forEach(function (b) {
          p.appendChild(el("p", null, b.block_id + ": " + b.kind +
            (b.title ? " — " + b.title : "") + " [trust=" + b.trust + "]"));
        });
      });
      if (!(artifact.visualizations || []).length) {
        p.appendChild(el("p", null, "No visualizations embedded."));
      }
    });
    tab("mkm-tab-audio", function (p) {
      (artifact.sonifications || []).forEach(function (s, j) {
        p.appendChild(el("h4", null, "s" + j + ": " + (s.title || "sonification") +
          " (" + (s.artifact_schema || "?") + ")"));
        var maps = s.mappings || {};
        Object.keys(maps).forEach(function (id) {
          var m = maps[id];
          p.appendChild(el("p", null, id + ": " + m.source + " -> " + m.target +
            " " + JSON.stringify(m.transform || {})));
        });
        (s.tracks || []).forEach(function (tr) {
          p.appendChild(el("p", null, "track " + (tr.label || tr.track_id) +
            " [" + tr.role + ", trust=" + tr.trust + ", " +
            (tr.events || []).length + " events]"));
        });
        kv(p, "render", JSON.stringify(s.render || {}));
      });
      if (!(artifact.sonifications || []).length) {
        p.appendChild(el("p", null, "No sonifications embedded."));
      }
    });
    tab("mkm-tab-sync", function (p) {
      p.appendChild(el("p", null, links.length + " synchronization link(s)."));
      links.forEach(function (L) {
        p.appendChild(el("p", "mkm-hash", L.link_id + ": " +
          (L.source_ref || "?") + " <-> " + (L.visual_ref || "-") + " <-> " +
          (L.sonification_ref || "-") +
          (L.time_range ? " [" + fmt(L.time_range[0]) + "…" + fmt(L.time_range[1]) + " s]" : "")));
      });
      var live = el("p", "mkm-live", "no active link");
      live.id = "mkm-sync-live";
      p.appendChild(el("h4", null, "Live"));
      p.appendChild(live);
    });
    fillAnnotations();
  }

  fillInspectors();
})(window);
