'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(Number(v))) ? 'n/a' : Number(v).toFixed(d);
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let m = r.statusText;
    try { m = (await r.json()).detail || m; } catch (e) { /* ignore */ }
    throw new Error(m);
  }
  return r.json();
}
const post = (p, b) => api(p, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(b || {})});

/* ------------------------------------------------------------------ header */
function setPill(id, text, cls) {
  const el = $('#' + id);
  el.textContent = text;
  el.className = 'pill ' + cls;
}
const modeClass = s => (String(s).toUpperCase().includes('OFFLINE') ? 'off' : 'ok');
async function loadHealth() {
  try {
    const h = await api('/api/health');
    setPill('pill-nebius', 'Nebius: ' + h.mode.nebius, modeClass(h.mode.nebius));
    setPill('pill-tavily', 'Tavily: ' + h.mode.tavily, modeClass(h.mode.tavily));
    setHazPill(h.hazards_active);
  } catch (e) {
    setPill('pill-nebius', 'API unreachable', 'bad');
  }
}
function setHazPill(active) {
  setPill('pill-haz', 'Live hazards: ' + (active ? 'ACTIVE' : 'off'), active ? 'ok' : 'off');
  const cb = $('#use-live');
  cb.disabled = !active;
  if (!active) cb.checked = false;
}

/* -------------------------------------------------------------------- tabs */
const loaded = {t3: false, t4: false};
$$('.tab').forEach(b => b.addEventListener('click', () => {
  $$('.tab').forEach(x => x.classList.toggle('active', x === b));
  $$('.panel').forEach(p => p.classList.toggle('active', p.id === b.dataset.tab));
  if (b.dataset.tab === 't1' && plotReady) Plotly.Plots.resize($('#plot'));
  if (b.dataset.tab === 't3') { if (!loaded.t3) { loaded.t3 = true; loadBenchmark('serverless'); } else resizeCharts(); }
  if (b.dataset.tab === 't4' && !loaded.t4) { loaded.t4 = true; loadInspector(); }
}));

/* --------------------------------------------------------- TAB 1: 3D view */
const TRACE = {terrain: 0, pillars: 1, smoke: 2, ring: 3, dTrail: 4, rTrail: 5, drone: 6, rover: 7, target: 8, tracks: 9, truth: 10};
const LABEL_COLOR = {survivor: '#ff3b3b', uncertain: '#ffb020', decoy: '#7d8590', rescued: '#3ddc84'};
let plotReady = false, pulse = 0, ws = null, paused = false, seenLog = new Set();

function pillarLines(w) {
  const xs = [], ys = [], zs = [];
  const H = w.size_m[2];
  for (let j = 0; j < w.pillars.length; j++) {
    for (let i = 0; i < w.pillars[j].length; i++) {
      if (w.pillars[j][i] > 0.5) {
        xs.push(w.x[i], w.x[i], null); ys.push(w.y[j], w.y[j], null);
        zs.push(w.elevation[j][i], H, null);
      }
    }
  }
  return {x: xs, y: ys, z: zs};
}
function pts(arr) { return {x: arr.map(p => p[0]), y: arr.map(p => p[1]), z: arr.map(p => p[2])}; }

function initPlot(w) {
  const pl = pillarLines(w);
  const sm = pts(w.smoke_points || []);
  const data = [
    {type: 'surface', name: 'Rubble (instability)', x: w.x, y: w.y, z: w.elevation, surfacecolor: w.instability, cmin: 0, cmax: 1,
      colorscale: [[0, '#2b9348'], [0.5, '#f2c14e'], [1, '#d62828']], showscale: true,
      colorbar: {title: 'Instab.', len: 0.45, thickness: 10, x: 0.98}, opacity: 0.96, showlegend: true, hoverinfo: 'z'},
    {type: 'scatter3d', mode: 'lines', name: 'Pillars', x: pl.x, y: pl.y, z: pl.z, line: {color: '#6e7681', width: 4}, hoverinfo: 'skip'},
    {type: 'scatter3d', mode: 'markers', name: 'Smoke', x: sm.x, y: sm.y, z: sm.z, marker: {size: 5, color: '#b8bec8', opacity: 0.12}, hoverinfo: 'skip'},
    {type: 'scatter3d', mode: 'lines', name: 'LiDAR range', x: [], y: [], z: [], line: {color: 'rgba(76,201,240,0.45)', width: 2}, hoverinfo: 'skip'},
    {type: 'scatter3d', mode: 'lines', name: 'Drone path', x: [], y: [], z: [], line: {color: '#4cc9f0', width: 5}, hoverinfo: 'skip'},
    {type: 'scatter3d', mode: 'lines', name: 'Rover path', x: [], y: [], z: [], line: {color: '#ff9f1c', width: 6}, hoverinfo: 'skip'},
    {type: 'scatter3d', mode: 'markers', name: 'Micro-drone', x: [], y: [], z: [], marker: {size: 7, color: '#4cc9f0', symbol: 'diamond', line: {color: '#fff', width: 1}}},
    {type: 'scatter3d', mode: 'markers', name: 'Quadruped rover', x: [], y: [], z: [], marker: {size: 8, color: '#ff9f1c', symbol: 'square', line: {color: '#fff', width: 1}}},
    {type: 'scatter3d', mode: 'markers', name: 'Rover target', x: [], y: [], z: [], marker: {size: 6, color: '#ffffff', symbol: 'x'}},
    {type: 'scatter3d', mode: 'markers', name: 'Thermal beacons', x: [], y: [], z: [], text: [], hoverinfo: 'text', marker: {size: 9, color: [], opacity: 0.95, line: {color: '#000', width: 1}}},
    {type: 'scatter3d', mode: 'markers', name: 'Ground truth', x: [], y: [], z: [], text: [], hoverinfo: 'text', marker: {size: 5, color: [], symbol: 'diamond-open', line: {width: 2}}},
  ];
  const sx = w.size_m[0], sy = w.size_m[1], sz = w.size_m[2];
  const ax = (t, r) => ({title: t, range: r, backgroundcolor: '#0d1117', gridcolor: '#2a313c', color: '#8b95a3', showbackground: true});
  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)', font: {color: '#c9d1d9', size: 11}, margin: {l: 0, r: 0, t: 0, b: 0}, uirevision: 'keep',
    legend: {orientation: 'h', y: 0.0, font: {size: 10}, bgcolor: 'rgba(13,17,23,0.6)'},
    scene: {aspectmode: 'manual', aspectratio: {x: 1, y: sy / sx, z: 0.4}, xaxis: ax('x (m)', [0, sx]), yaxis: ax('y (m)', [0, sy]), zaxis: ax('z (m)', [0, sz]),
      camera: {eye: {x: 1.3, y: -1.5, z: 0.95}}},
  };
  Plotly.react('plot', data, layout, {responsive: true, displaylogo: false});
  plotReady = true;
}
function updateTerrain(w) {
  Plotly.restyle('plot', {z: [w.elevation], surfacecolor: [w.instability]}, [TRACE.terrain]);
  const pl = pillarLines(w);
  Plotly.restyle('plot', {x: [pl.x], y: [pl.y], z: [pl.z]}, [TRACE.pillars]);
  const sm = pts(w.smoke_points || []);
  Plotly.restyle('plot', {x: [sm.x], y: [sm.y], z: [sm.z]}, [TRACE.smoke]);
}
function ringPoints(cx, cy, cz, r) {
  const x = [], y = [], z = [];
  for (let k = 0; k <= 36; k++) { const a = (k / 36) * 2 * Math.PI; x.push(cx + r * Math.cos(a)); y.push(cy + r * Math.sin(a)); z.push(cz); }
  for (let k = 0; k < 8; k++) { const a = (k / 8) * 2 * Math.PI; x.push(null, cx, cx + r * Math.cos(a)); y.push(null, cy, cy + r * Math.sin(a)); z.push(null, cz, cz); }
  return {x, y, z};
}
function updateFrame(fr) {
  if (!plotReady) return;
  pulse++;
  const r = fr.rover, d = fr.drone;
  const ring = d.lost ? {x: [], y: [], z: []} : ringPoints(d.x, d.y, d.z, 15);
  const dT = pts(d.trail || []), rT = pts(r.trail || []);
  const one = (o, ok) => ok ? {x: [o.x], y: [o.y], z: [o.z + 0.6]} : {x: [], y: [], z: []};
  const dM = one(d, !d.lost), rM = one(r, !r.lost);
  const tg = r.target ? {x: [r.target[0]], y: [r.target[1]], z: [2.0]} : {x: [], y: [], z: []};
  const idx = [TRACE.ring, TRACE.dTrail, TRACE.rTrail, TRACE.drone, TRACE.rover, TRACE.target];
  const L = [ring, dT, rT, dM, rM, tg];
  Plotly.restyle('plot', {x: L.map(o => o.x), y: L.map(o => o.y), z: L.map(o => o.z)}, idx);

  const tr = fr.tracks || [];
  Plotly.restyle('plot', {
    x: [tr.map(t => t.x)], y: [tr.map(t => t.y)], z: [tr.map(t => t.z + 1.0)],
    text: [tr.map(t => `T${t.tid} ${t.label}<br>conf ${t.confidence}<br>${t.est_temp_c}°C snr ${t.snr_db}dB`)],
    'marker.color': [tr.map(t => LABEL_COLOR[t.label] || '#ffb020')],
    'marker.size': [tr.map(t => t.label === 'survivor' ? 9 + 4 * Math.sin(pulse / 2) : 8)],
  }, [TRACE.tracks]);

  const sv = fr.survivors || [];
  Plotly.restyle('plot', {
    x: [sv.map(s => s.x)], y: [sv.map(s => s.y)], z: [sv.map(s => s.z + 0.5)],
    text: [sv.map(s => `S${s.sid} vitality ${fmt(s.vitality, 2)}${s.rescued ? ' (rescued)' : s.located ? ' (located)' : ''}`)],
    'marker.color': [sv.map(s => s.rescued ? '#3ddc84' : s.located ? '#ff4d4d' : 'rgba(255,255,255,0.45)')],
  }, [TRACE.truth]);

  const k = fr.kpis;
  $('#k-clock').textContent = `t=${fmt(fr.t, 1)}s  step ${fr.step}`;
  $('#k-located').textContent = `${k.located}/${k.total}`;
  $('#k-rescued').textContent = `${k.rescued}/${k.total}`;
  $('#k-risk').textContent = fmt(k.risk_exposure, 1);
  $('#k-explored').textContent = fmt(k.map_explored * 100, 0) + '%';
  $('#k-aft').textContent = k.aftershocks;
  $('#k-reflex').textContent = fr.reflex_calls ?? 0;
  setBat('r', r.battery_pct, r.lost);
  setBat('d', d.battery_pct, d.lost);
  $('#w-alpha').textContent = fmt(fr.weights.alpha_slope, 2);
  $('#w-beta').textContent = fmt(fr.weights.beta_collapse, 2);
  $('#w-gamma').textContent = fmt(fr.weights.gamma_smoke, 2);
  if (fr.last_plan) {
    const p = fr.last_plan;
    $('#plan-box').innerHTML = `Plan #${fr.plans_count} · trigger <b>${esc(p.trigger)}</b> · ${esc(p.mode)}/${esc(p.model)} · ${fmt(p.latency_ms, 0)} ms<br>${esc(p.rationale)}`;
  }
  (fr.log || []).forEach(addLog);
}
function setBat(who, pct, lost) {
  const v = Math.max(0, Math.min(100, Number(pct) || 0));
  $('#bat-' + who).style.width = v + '%';
  $('#bat-' + who).style.background = lost ? '#ff4d4d' : v < 25 ? '#ffb020' : '#76b900';
  $('#bat-' + who + '-v').textContent = lost ? 'LOST' : fmt(v, 0) + '%';
}
function addLog(e) {
  const key = `${e.t}|${e.src}|${e.msg}`;
  if (seenLog.has(key)) return;
  seenLog.add(key);
  const box = $('#log');
  const row = document.createElement('div');
  row.innerHTML = `<span class="t">[${fmt(e.t, 1).padStart(6)}]</span> <b class="src-${esc(e.src)}">${esc(e.src)}</b> ${esc(e.msg)}`;
  box.appendChild(row);
  while (box.childElementCount > 300) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}
function renderFinal(m) {
  $('#final-card').classList.remove('hidden');
  $('#final-box').innerHTML = `<div class="kv">
    <span>Outcome</span><b>${esc(m.end_reason)}</b>
    <span>Rescued</span><b>${m.survivors_rescued}/${m.survivors_total} (${fmt(m.rescue_rate * 100, 0)}%)</b>
    <span>Time</span><b>${fmt(m.time_s, 0)} s</b>
    <span>First located</span><b>${fmt(m.first_located_s, 0)} s</b>
    <span>Energy</span><b>${fmt(m.energy_wh, 1)} Wh (${fmt(m.energy_wh_per_rescued, 1)} Wh/rescued)</b>
    <span>Robots lost</span><b>${m.rover_lost ? 'rover (' + esc(m.rover_loss_reason) + ') ' : ''}${m.drone_lost ? 'drone' : ''}${!m.robot_loss ? 'none' : ''}</b>
    <span>Planner calls</span><b>${m.plans} (${esc(JSON.stringify(m.llm_mode))}) · reflex ${m.reflex_calls}</b></div>`;
}

function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/sim`);
  ws.onopen = () => setPill('pill-conn', 'WS: connected', 'ok');
  ws.onclose = () => { setPill('pill-conn', 'WS: reconnecting', 'off'); setTimeout(connect, 1500); };
  ws.onmessage = ev => handle(JSON.parse(ev.data));
}
function handle(m) {
  if (m.type === 'init') {
    seenLog = new Set(); $('#log').innerHTML = ''; $('#final-card').classList.add('hidden');
    initPlot(m.world);
    setPill('pill-nebius', 'Nebius: ' + m.mode.nebius, modeClass(m.mode.nebius));
    setPill('pill-tavily', 'Tavily: ' + m.mode.tavily, modeClass(m.mode.tavily));
    addLog({t: 0, src: 'ui', msg: `hazards: aftershock p=${fmt(m.hazards.seismic_aftershock_prob, 2)} wind=${fmt(m.hazards.wind_turbulence_mps, 1)} m/s ambient=${fmt(m.hazards.ambient_temp_c, 1)}C toxicity=${fmt(m.hazards.smoke_toxicity_index, 2)} FEMA margin=${fmt(m.hazards.fema_safety_margin_m, 1)}m`});
    updateFrame(m.frame);
    $('#c-truth').dispatchEvent(new Event('change'));
  } else if (m.type === 'frame') {
    if (m.world) updateTerrain(m.world);
    if (m.smoke) { const s = pts(m.smoke); Plotly.restyle('plot', {x: [s.x], y: [s.y], z: [s.z]}, [TRACE.smoke]); }
    if (m.quake) addLog({t: m.t, src: 'world', msg: `aftershock injected M${fmt(m.quake.magnitude, 1)}; corridors re-planned`});
    updateFrame(m);
  } else if (m.type === 'done') {
    renderFinal(m.metrics);
    $('#b-start').textContent = '▶ Start mission';
  } else if (m.type === 'error') {
    addLog({t: 0, src: 'ui', msg: 'ERROR ' + m.message});
  }
}
const send = o => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); };
$('#b-start').onclick = () => {
  paused = false; $('#b-pause').textContent = '⏸ Pause';
  send({cmd: 'start', policy: $('#c-policy').value, seed: +$('#c-seed').value, survivors: +$('#c-surv').value, use_live_hazards: $('#use-live').checked});
  send({cmd: 'speed', value: +$('#c-speed').value});
  $('#b-start').textContent = '↻ Restart';
};
$('#b-pause').onclick = () => { paused = !paused; send({cmd: paused ? 'pause' : 'resume'}); $('#b-pause').textContent = paused ? '▶ Resume' : '⏸ Pause'; };
$('#b-stop').onclick = () => { send({cmd: 'stop'}); $('#b-start').textContent = '▶ Start mission'; };
$('#c-speed').onchange = e => send({cmd: 'speed', value: +e.target.value});
$('#c-mag').oninput = e => { $('#c-mag-v').textContent = e.target.value; };
$('#b-quake').onclick = () => send({cmd: 'aftershock', magnitude: +$('#c-mag').value});
$('#c-truth').onchange = e => { if (plotReady) Plotly.restyle('plot', {visible: [e.target.checked]}, [TRACE.truth]); };

/* ------------------------------------------------------ TAB 2: Tavily intel */
function table(rows, head) {
  return `<table>${head ? `<tr>${head.map(h => `<th>${h}</th>`).join('')}</tr>` : ''}${rows.map(r => `<tr>${r.map((c, i) => `<td${i > 0 && /^-?[\d.]+$/.test(String(c).replace(/<[^>]*>/g, '')) ? ' class="num"' : ''}>${c}</td>`).join('')}</tr>`).join('')}</table>`;
}
async function fetchHazards() {
  const btn = $('#h-fetch');
  btn.disabled = true;
  $('#h-status').textContent = 'Searching and extracting...';
  try {
    const r = await post('/api/hazards', {location: $('#h-loc').value || null, query: $('#h-query').value || null, seed: +$('#c-seed').value});
    const i = r.intel;
    $('#h-status').innerHTML = `Mode <b>${esc(i.mode)}</b> · extraction <b>${esc(i.extraction)}</b> · queries: ${i.queries.map(esc).join(' | ')}` +
      `${i.error ? ` · <span class="err">${esc(i.error)}</span>` : ''} · latency ${esc(JSON.stringify(i.latency_ms))}`;
    const ev = i.evidence || {};
    $('#h-mods').innerHTML = table(Object.entries(r.modifiers).filter(([k]) => k !== 'source_urls').map(([k, v]) =>
      [esc(k), fmt(v, 2), `<span class="muted small">${esc(ev[k] || '')}</span>`]), ['Parameter', 'Value', 'Evidence']);
    const keys = Object.keys(r.physics_after);
    $('#h-phys').innerHTML = table(keys.map(k => {
      const b = r.physics_before[k], a = r.physics_after[k], dl = a - b;
      const cls = Math.abs(dl) < 1e-9 ? '' : dl > 0 ? 'up' : 'down';
      return [esc(k), fmt(b, 3), fmt(a, 3), `<span class="${cls}">${dl > 0 ? '+' : ''}${fmt(dl, 3)}</span>`];
    }), ['Engine parameter', 'Before', 'After', 'Δ']);
    $('#h-hits').innerHTML = (i.hits || []).map(h => `<div class="hit"><a href="${esc(h.url)}" target="_blank" rel="noopener">${esc(h.title || h.url)}</a> <span class="muted small">score ${fmt(h.score, 2)}</span><p>${esc(h.content)}</p></div>`).join('') || '<span class="muted">No sources (offline profile in use).</span>';
    setHazPill(true);
    $('#use-live').checked = true;
  } catch (e) {
    $('#h-status').innerHTML = `<span class="err">${esc(e.message)}</span>`;
  } finally { btn.disabled = false; }
}
$('#h-fetch').onclick = fetchHazards;
$('#h-clear').onclick = async () => { await post('/api/hazards/clear'); setHazPill(false); $('#h-status').textContent = 'Live hazards cleared; simulation uses seeded hazard sampling.'; };

/* -------------------------------------------------------- TAB 3: benchmark */
const POL = [['greedy', 'Greedy', '#7d8590'], ['static', 'Static A*', '#4cc9f0'], ['adaptive', 'Adaptive', '#76b900']];
const CHARTS = [
  ['ch-rescue', 'Survivor rescue rate (%)', s => ({v: s.rescue_rate_pct.mean, ci: s.rescue_rate_pct.ci95})],
  ['ch-loss', 'Robot loss / collision rate (%)', s => ({v: s.robot_loss_rate_pct.mean, ci: s.robot_loss_rate_pct.ci95})],
  ['ch-ttfs', 'Mean time-to-first-survivor (s)', s => ({v: s.time_to_first_survivor_s.mean, ci: s.time_to_first_survivor_s.ci95})],
  ['ch-energy', 'Energy per rescued survivor (Wh)', s => ({v: s.energy_efficiency_wh_per_rescued, ci: null})],
];
let lastBench = null;
function drawCharts(b) {
  lastBench = b;
  CHARTS.forEach(([id, title, get]) => {
    const vals = POL.map(([p]) => b.summary[p] && b.summary[p].rescue_rate_pct ? get(b.summary[p]) : {v: null, ci: null});
    const hasCi = vals.some(x => x.ci && x.ci[0] != null);
    Plotly.react(id, [{
      type: 'bar', x: POL.map(p => p[1]), y: vals.map(x => x.v), marker: {color: POL.map(p => p[2])},
      text: vals.map(x => fmt(x.v, 1)), textposition: 'outside', cliponaxis: false,
      error_y: hasCi ? {type: 'data', symmetric: false, array: vals.map(x => x.ci ? x.ci[1] - x.v : 0), arrayminus: vals.map(x => x.ci ? x.v - x.ci[0] : 0), color: '#c9d1d9', thickness: 1.2} : undefined,
    }], {title: {text: title, font: {size: 13}}, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: '#c9d1d9', size: 11},
      margin: {l: 40, r: 10, t: 36, b: 30}, height: 280, yaxis: {gridcolor: '#2a313c', rangemode: 'tozero'}}, {displaylogo: false, responsive: true});
  });
}
function resizeCharts() { CHARTS.forEach(([id]) => Plotly.Plots.resize($('#' + id))); }
function renderBench(b) {
  const m = b.meta;
  $('#bm-meta').innerHTML = `Source <b>${esc(b.source_file)}</b> · ${m.episodes_per_policy} episodes/policy × ${m.policies.length} policies · paired seeds from ${m.seed_base} · LLM mode <b>${esc(m.llm_mode)}</b> · ${fmt(m.duration_s, 0)} s on ${esc(m.workers)} workers · Python ${esc(m.python)} · bars show 95% CI`;
  drawCharts(b);
  const rows = Object.entries(b.comparison || {}).map(([k, c]) => {
    const red = c.robot_loss_reduction_pct;
    return `<div class="box"><div class="muted small">${esc(k.replace(/_/g, ' '))}</div>
      <div class="big">${red == null ? 'n/a' : fmt(red, 1) + '%'}</div><div class="small">robot-loss reduction
      <span class="badge ${c.meets_60pct_loss_reduction_target ? 'yes' : 'no'}">${c.meets_60pct_loss_reduction_target ? '>60% target met' : '>60% target not met'}</span></div>
      <div class="small muted">rescue rate ${c.rescue_rate_delta_pts >= 0 ? '+' : ''}${fmt(c.rescue_rate_delta_pts, 1)} pts · first-survivor ${fmt(c.time_to_first_survivor_delta_s, 1)} s · energy/rescued ${fmt(c.energy_per_rescued_delta_wh, 1)} Wh</div></div>`;
  }).join('');
  $('#bm-cmp').innerHTML = `<div class="cmp">${rows}</div>`;
  const pt = b.paired_tests || {};
  const pf = x => x && x.p_value != null ? (x.p_value < 0.001 ? '<0.001' : fmt(x.p_value, 3)) + (x.p_value < 0.05 ? ' ✓' : '') : 'n/a';
  $('#bm-tests').innerHTML = table(Object.entries(pt).map(([k, v]) => [esc(k.replace(/_/g, ' ')), v.paired_episodes, pf(v.rescue_rate_pct), pf(v.robot_loss)]),
    ['Comparison', 'Pairs', 'p (rescue rate)', 'p (robot loss)']) + '<div class="small muted">✓ = p &lt; 0.05</div>';
}
async function loadBenchmark(src) {
  try {
    renderBench(await api('/api/benchmark/result?source=' + src));
    $('#bm-status').textContent = '';
  } catch (e) { $('#bm-status').innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}
$('#bm-load').onclick = () => loadBenchmark('serverless');
let benchTimer = null;
$('#bm-run').onclick = async () => {
  try {
    await post('/api/benchmark/run', {episodes: +$('#bm-eps').value, live: $('#bm-live').checked});
  } catch (e) { $('#bm-status').innerHTML = `<span class="err">${esc(e.message)}</span>`; return; }
  $('#bm-run').disabled = true;
  benchTimer = setInterval(async () => {
    const s = await api('/api/benchmark/status');
    if (s.progress) $('#bm-prog').style.width = (100 * s.progress.done / s.progress.total) + '%';
    $('#bm-status').textContent = `${s.state} · ${s.elapsed_s}s` + (s.progress ? ` · ${s.progress.done}/${s.progress.total} episodes` : '');
    if (s.state !== 'running') {
      clearInterval(benchTimer); $('#bm-run').disabled = false;
      if (s.state === 'finished') { $('#bm-prog').style.width = '100%'; loadBenchmark('ui'); }
      else $('#bm-status').innerHTML = `<span class="err">failed: ${esc((s.log_tail || []).join(' | '))}</span>`;
    }
  }, 1500);
};

/* -------------------------------------------------------- TAB 4: inspector */
function kv(o) { return `<div class="kv">${Object.entries(o).map(([k, v]) => `<span>${esc(k)}</span><b>${esc(typeof v === 'object' ? JSON.stringify(v) : v)}</b>`).join('')}</div>`; }
async function loadInspector() {
  try {
    const d = await api('/api/inspector');
    $('#i-arch').innerHTML = kv({
      'Planner (strategic)': d.models.planner, 'Reflex (tactical)': d.models.reflex, Endpoint: d.models.base_url,
      Timeout: d.models.timeout_s + ' s', 'Max tokens': d.models.max_tokens, 'Nebius mode': d.mode.nebius, 'Tavily mode': d.mode.tavily,
      Calls: `${d.orchestrator.calls} (ok ${d.orchestrator.ok_calls}, errors ${d.orchestrator.errors})`, 'Mean latency': fmt(d.orchestrator.mean_latency_ms, 0) + ' ms'});
    const lat = Object.entries(d.latency);
    $('#i-lat').innerHTML = lat.length ? table(lat.map(([k, v]) => [esc(k), v.calls, fmt(v.mean_ms, 0), fmt(v.p95_ms, 0), fmt(v.max_ms, 0)]), ['Role|mode', 'Calls', 'Mean', 'p95', 'Max'])
      : '<span class="muted">No LLM calls logged yet. Run an adaptive mission (set NEBIUS_API_KEY for live calls).</span>';
    $('#i-dr').innerHTML = table(Object.entries(d.domain_randomization).map(([k, v]) => [esc(k), fmt(v.min, 2), fmt(v.mean, 2), fmt(v.max, 2)]), ['Parameter', 'Min', 'Mean', 'Max']);
    const c = d.cosmos_style_conditioning;
    $('#i-cosmos').innerHTML = kv({Implementation: c.implementation, 'Cosmos weights called': String(c.cosmos_weights_called), 'Smoke peak / max': `${c.smoke_peak} / ${c.smoke_max}`,
      'Smoke tau': c.smoke_tau_s + ' s', 'Pillar collapse p': c.pillar_collapse_prob, 'Pillar spacing': c.pillar_spacing_m + ' m'});
    $('#i-count').textContent = `${d.payload_count} logged`;
    $('#i-payloads').innerHTML = d.payloads.length ? d.payloads.slice().reverse().map((p, i) =>
      `<details ${i === 0 ? 'open' : ''}><summary>${esc(p.role || p.model || 'call')} · ${esc(p.mode || '')} · ${fmt(p.latency_ms, 0)} ms</summary><pre>${esc(JSON.stringify(p, null, 2))}</pre></details>`).join('')
      : '<span class="muted">No payloads yet.</span>';
  } catch (e) { $('#i-arch').innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}
$('#i-refresh').onclick = loadInspector;
$('#i-models').onclick = async () => {
  const box = $('#i-models-box');
  box.textContent = 'Querying Nebius model catalog...';
  const r = await api('/api/models');
  if (!r.online) { box.innerHTML = `<span class="err">${esc(r.error)}</span>`; return; }
  if (r.error) { box.innerHTML = `<span class="err">${esc(r.error)}</span>`; return; }
  box.innerHTML = `<div class="kv"><span>Planner ID</span><b>${esc(r.configured.planner)} — ${r.planner_found ? '✓ found' : '✗ not in catalog'}</b>
    <span>Reflex ID</span><b>${esc(r.configured.reflex)} — ${r.reflex_found ? '✓ found' : '✗ not in catalog'}</b></div>
    <details open><summary>NVIDIA / Nemotron models available (${r.nvidia_models.length} of ${r.total_models})</summary><pre>${esc(r.nvidia_models.join('\n') || r.models.join('\n'))}</pre></details>`;
};
$('#i-probe').onclick = async () => {
  const box = $('#i-probe-box');
  box.textContent = 'Probing reflex agent...';
  const r = await post('/api/inspector/probe');
  box.innerHTML = table(Object.entries(r).map(([k, v]) => [esc(k), esc(v.features.est_temp_c) + ' °C', `<b>${esc(v.verdict.label)}</b> (${fmt(v.verdict.confidence, 2)})`,
    esc(v.verdict.mode + '/' + v.verdict.model), esc(v.verdict.rationale), fmt(v.verdict.latency_ms, 0) + ' ms']), ['Case', 'Temp', 'Verdict', 'Engine', 'Rationale', 'Latency']);
  loadInspector();
};

/* -------------------------------------------------------------------- boot */
loadHealth();
api('/api/hazards/state').then(s => setHazPill(s.active)).catch(() => {});
connect();

/* --------------------------------------------------- TAB 1: policy comparison */
const CMP_LABEL = {greedy: 'Baseline Greedy', static: 'Static Conservative A*', adaptive: 'Nemotron Adaptive'};
$('#b-compare').onclick = async () => {
  const btn = $('#b-compare');
  btn.disabled = true;
  btn.textContent = '⏳ Comparing...';
  $('#cmp-card').classList.remove('hidden');
  $('#cmp-box').textContent = 'Running all three policies on this seed (offline planner, about 15 s)...';
  try {
    const seed = +$('#c-seed').value;
    const r = await post('/api/compare', {seed, survivors: +$('#c-surv').value, use_live_hazards: $('#use-live').checked});
    $('#cmp-sub').textContent = `seed ${r.seed} · offline planner${r.used_live_hazards ? ' · Tavily hazards' : ''}`;
    const rows = ['greedy', 'static', 'adaptive'].map(p => {
      const m = r.policies[p];
      if (!m || m.error) return [esc(CMP_LABEL[p]), `<span class='err'>${esc(m ? m.error : 'no result')}</span>`, '', '', '', ''];
      const lost = m.robot_loss ? [m.rover_lost ? 'rover (' + (m.rover_loss_reason || 'lost') + ')' : '', m.drone_lost ? 'drone' : ''].filter(Boolean).join(' + ') : 'none';
      return [`<b>${esc(CMP_LABEL[p])}</b>`, `${m.survivors_rescued}/${m.survivors_total}`, esc(lost), fmt(m.first_located_s, 0) + ' s', fmt(m.time_s, 0) + ' s', fmt(m.energy_wh_per_rescued, 1)];
    });
    $('#cmp-box').innerHTML = table(rows, ['Policy', 'Rescued', 'Robots lost', 'First located', 'Mission time', 'Wh/rescued']) +
      `<div class='muted small'>One seed is an anecdote, not a statistic: see Tab 3 for the 100-episode results.</div>`;
  } catch (e) {
    $('#cmp-box').innerHTML = `<span class='err'>${esc(e.message)}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '⚖ Compare policies (this seed)';
  }
};
