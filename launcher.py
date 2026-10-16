#!/usr/bin/env python3
"""
Lanceur de campagnes de simulations parallèles (SBNE) — panneau web.

Lance N simulations headless en parallèle (seeds / paramètres différents),
suit leur avancement en temps réel et permet de les arrêter.
Aucune dépendance hors bibliothèque standard de Python.

Usage :
    .venv/bin/python launcher.py [--port 8765] [--host 127.0.0.1]

Chaque run est un sous-processus :
    Environnement.py --headless --max-steps N --seed S --outdir <nom> [--param CLE=VALEUR ...]
et écrit dans simulations/<nom>/ (stats.json, evolution.db, parameters.json, run.log).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_SCRIPT = os.path.join(BASE_DIR, "Environnement.py")
SIMS_DIR = os.path.join(BASE_DIR, "simulations")
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Registre des campagnes
# ---------------------------------------------------------------------------
TERMINAL_STATES = ("done", "failed", "stopped", "skipped")
_lock = threading.Lock()
_campaigns = {}          # id -> dict
_server_started = time.time()


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _parse_param_lines(text):
    """'CLE=VALEUR\nCLE2=VALEUR2' -> {CLE: valeur, ...} (valeur gardée en str)."""
    params = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            params[key] = value
    return params


def create_campaign(label, n_runs, seed_base, seed_list, max_steps, max_parallel, params):
    """Crée une campagne et ses runs en attente. Retourne (campaign, erreur)."""
    if not isinstance(n_runs, int) or n_runs < 1 or n_runs > 256:
        return None, "n_runs doit être un entier entre 1 et 256"
    if not isinstance(max_steps, int) or max_steps < 0:
        return None, "max_steps doit être un entier >= 0"
    if isinstance(seed_list, list) and seed_list:
        seeds = [int(s) for s in seed_list[:n_runs]]
        while len(seeds) < n_runs:
            seeds.append(seeds[-1])
    else:
        seeds = [int(seed_base) + i for i in range(n_runs)]
    if max_parallel is None or not isinstance(max_parallel, int) or max_parallel < 1:
        max_parallel = min(n_runs, os.cpu_count() or 1)
    max_parallel = min(max_parallel, n_runs)

    cid = "c" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(int(time.time()) % 100000)
    folder_prefix = cid.replace("c", "")
    runs = {}
    for i, seed in enumerate(seeds):
        run_id = "r%02d" % i
        folder = "%s_seed%s_run%02d" % (folder_prefix, seed, i)
        runs[run_id] = {
            "id": run_id,
            "seed": seed,
            "folder": folder,
            "log": os.path.join(SIMS_DIR, folder, "run.log"),
            "status": "pending",
            "started": None,
            "ended": None,
            "exit_code": None,
            "summary": None,
            "_proc": None,
            "_logf": None,
        }
    campaign = {
        "id": cid,
        "label": (label or "").strip() or ("campagne " + str(len(_campaigns) + 1)),
        "created": _now_iso(),
        "status": "running",
        "n_runs": n_runs,
        "max_steps": max_steps,
        "max_parallel": max_parallel,
        "seeds": seeds,
        "params": params,
        "runs": runs,
    }
    with _lock:
        _campaigns[cid] = campaign
    return campaign, None


def _launch_run(campaign, run):
    """Démarrage d'un run (sous-processus headless)."""
    folder = os.path.join(SIMS_DIR, run["folder"])
    os.makedirs(folder, exist_ok=True)
    cmd = [PYTHON, SIM_SCRIPT, "--headless",
           "--seed", str(run["seed"]),
           "--outdir", run["folder"]]
    # 0 = illimité : on ne passe pas --max-steps pour que PARAMS["MAX_ITERATIONS"]
    # (surchargeable via le champ "paramètres") reste le paramètre de temps maximal
    if campaign["max_steps"] > 0:
        cmd += ["--max-steps", str(campaign["max_steps"])]
    for key, value in campaign["params"].items():
        cmd += ["--param", "%s=%s" % (key, value)]
    run["_logf"] = open(run["log"], "w", encoding="utf-8")
    run["_proc"] = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=run["_logf"],
                                    stderr=subprocess.STDOUT)
    run["status"] = "running"
    run["started"] = _now_iso()
    try:
        with open(os.path.join(folder, "campaign.json"), "w", encoding="utf-8") as f:
            json.dump({
                "campaign_id": campaign["id"],
                "campaign_label": campaign["label"],
                "run_id": run["id"],
                "seed": run["seed"],
                "max_steps": campaign["max_steps"],
                "params_override": campaign["params"],
                "launched": run["started"],
            }, f, indent=2)
    except OSError:
        pass


def _read_run_summary(run):
    """Résumé issues de stats.json (si le run est terminé proprement)."""
    path = os.path.join(SIMS_DIR, run["folder"], "stats.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        run["summary"] = {
            "total_steps": stats.get("total_steps"),
            "final_pop": stats.get("final_pop"),
            "final_foods": stats.get("final_foods"),
            "duration_s": round(time.time() - _started_epoch(run), 1) if run["started"] else None,
        }
    except (OSError, ValueError, TypeError):
        pass


def _started_epoch(run):
    if not run["started"]:
        return time.time()
    return datetime.fromisoformat(run["started"]).timestamp()


def _manager_tick():
    with _lock:
        campaigns = list(_campaigns.values())
    for campaign in campaigns:
        if campaign["status"] != "running":
            continue
        with _lock:
            runs = list(campaign["runs"].values())
        # 1) Récolte des runs terminés (+ escalation SIGKILL si SIGTERM ignoré :
        #    pygame/SDL intercepte le SIGTERM et le transforme en événement QUIT,
        #    ignoré en headless — le processus survit donc au SIGTERM)
        for run in runs:
            if run["status"] != "running":
                continue
            if run.get("_stopped"):
                proc = run.get("_proc")
                if (proc is not None and proc.poll() is None
                        and time.time() - run.get("_stop_requested_at", 0) > 3):
                    try:
                        proc.kill()
                    except OSError:
                        pass
            proc = run.get("_proc")
            if proc is not None and proc.poll() is not None:
                if run.get("_stopped"):
                    run["status"] = "stopped"
                else:
                    run["status"] = "done" if proc.returncode == 0 else "failed"
                run["ended"] = _now_iso()
                run["exit_code"] = proc.returncode
                if run["_logf"] is not None:
                    run["_logf"].close()
                    run["_logf"] = None
                _read_run_summary(run)
        # 2) État de la campagne
        with _lock:
            runs = list(campaign["runs"].values())
            if all(r["status"] in TERMINAL_STATES for r in runs):
                if any(r["status"] == "done" for r in runs):
                    campaign["status"] = "done"
                else:
                    campaign["status"] = "stopped"
                continue
            running = sum(1 for r in runs if r["status"] == "running")
            pending = [r for r in runs if r["status"] == "pending"]
        # 3) Lancement des runs en attente (respect de max_parallel)
        for run in pending:
            if running >= campaign["max_parallel"]:
                break
            try:
                _launch_run(campaign, run)
                running += 1
            except OSError as exc:
                run["status"] = "failed"
                run["ended"] = _now_iso()
                run["summary"] = {"error": str(exc)}


def _manager_loop():
    while True:
        try:
            _manager_tick()
        except Exception as exc:  # le manager ne doit jamais mourir
            sys.stderr.write("[launcher] erreur manager : %r\n" % (exc,))
            sys.stderr.flush()
        time.sleep(1)


def stop_campaign(cid):
    """Arrête une campagne : SIGTERM sur les runs actifs, skip des runs en attente."""
    with _lock:
        campaign = _campaigns.get(cid)
        if campaign is None or campaign["status"] != "running":
            return False
        for run in campaign["runs"].values():
            if run["status"] == "pending":
                run["status"] = "skipped"
                run["ended"] = _now_iso()
            elif run["status"] == "running":
                run["_stopped"] = True
                run["_stop_requested_at"] = time.time()
                proc = run.get("_proc")
                if proc is not None:
                    try:
                        proc.terminate()
                    except OSError:
                        pass
    return True


def _public_state():
    """État sérialisable pour /api/state (sans objets internes)."""
    with _lock:
        out = {}
        for cid in sorted(_campaigns, reverse=True):
            c = _campaigns[cid]
            runs = []
            for rid in sorted(c["runs"]):
                r = c["runs"][rid]
                runs.append({
                    "id": rid,
                    "seed": r["seed"],
                    "folder": r["folder"],
                    "status": r["status"],
                    "started": r["started"],
                    "ended": r["ended"],
                    "exit_code": r["exit_code"],
                    "summary": r["summary"],
                })
            out[cid] = {
                "id": cid,
                "label": c["label"],
                "created": c["created"],
                "status": c["status"],
                "n_runs": c["n_runs"],
                "max_steps": c["max_steps"],
                "max_parallel": c["max_parallel"],
                "params": c["params"],
                "runs": runs,
            }
    return out


def _run_log_tail(campaign_id, run_id, lines):
    with _lock:
        campaign = _campaigns.get(campaign_id)
        if campaign is None:
            return None
        run = campaign["runs"].get(run_id)
    if run is None:
        return None
    try:
        with open(run["log"], "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except OSError:
        return ""


def _run_curve(campaign_id, run_id):
    with _lock:
        campaign = _campaigns.get(campaign_id)
        if campaign is None:
            return None
        run = campaign["runs"].get(run_id)
    if run is None:
        return None
    path = os.path.join(SIMS_DIR, run["folder"], "stats.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return {"steps": stats.get("stats_steps", []), "pop": stats.get("stats_pop", [])}
    except (OSError, ValueError):
        return {"steps": [], "pop": []}


# ---------------------------------------------------------------------------
# Interface web (une seule page, aucune dépendance externe)
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>SBNE — Lanceur de campagnes</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
          --muted:#8b949e; --accent:#58a6ff; --ok:#3fb950; --warn:#d29922; --err:#f85149; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.45 -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { padding:14px 22px; border-bottom:1px solid var(--border);
           display:flex; align-items:baseline; gap:14px; }
  header h1 { font-size:17px; margin:0; }
  header .meta { color:var(--muted); font-size:12px; }
  main { padding:18px 22px; max-width:1200px; margin:0 auto; }
  .panel { background:var(--panel); border:1px solid var(--border);
           border-radius:8px; padding:16px; margin-bottom:18px; }
  .panel h2 { font-size:14px; margin:0 0 12px; color:var(--accent);
              text-transform:uppercase; letter-spacing:.05em; }
  label { display:block; color:var(--muted); font-size:12px; margin:10px 0 4px; }
  input[type=text], input[type=number], textarea {
    width:100%; background:var(--bg); color:var(--text); border:1px solid var(--border);
    border-radius:6px; padding:7px 9px; font:13px/1.4 ui-monospace, monospace; }
  textarea { resize:vertical; min-height:56px; }
  .row { display:flex; gap:14px; flex-wrap:wrap; }
  .row > div { flex:1; min-width:130px; }
  button { background:var(--accent); border:0; color:#04121f; font-weight:600;
           border-radius:6px; padding:8px 18px; cursor:pointer; font-size:13px; }
  button:hover { filter:brightness(1.1); }
  button.danger { background:var(--err); color:#fff; padding:4px 10px; font-size:12px; }
  button.ghost { background:transparent; color:var(--accent); border:1px solid var(--border);
                 padding:4px 10px; font-size:12px; font-weight:400; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { text-align:left; color:var(--muted); font-weight:500; font-size:11px;
       text-transform:uppercase; letter-spacing:.04em; padding:6px 8px;
       border-bottom:1px solid var(--border); }
  td { padding:6px 8px; border-bottom:1px solid #21262d; vertical-align:middle; }
  tr:last-child td { border-bottom:0; }
  .badge { display:inline-block; padding:2px 9px; border-radius:10px; font-size:11px;
           font-weight:600; }
  .b-pending  { background:#21262d; color:var(--muted); }
  .b-running  { background:#12261e; color:var(--ok); animation:pulse 1.2s infinite; }
  .b-done     { background:#12261e; color:var(--ok); }
  .b-failed   { background:#2d1214; color:var(--err); }
  .b-stopped  { background:#2a2113; color:var(--warn); }
  .b-skipped  { background:#21262d; color:var(--muted); }
  .b-crunning { background:#12261e; color:var(--ok); }
  .b-cdone    { background:#12261e; color:var(--ok); }
  .b-cstopped { background:#2a2113; color:var(--warn); }
  @keyframes pulse { 50% { opacity:.45; } }
  .bar { height:6px; background:#21262d; border-radius:3px; overflow:hidden; min-width:110px; }
  .bar > div { height:100%; background:var(--ok); transition:width .5s; }
  .camp { border:1px solid var(--border); border-radius:8px; margin-bottom:14px;
          background:var(--bg); }
  .camp-head { display:flex; align-items:center; gap:12px; padding:10px 14px;
               border-bottom:1px solid var(--border); flex-wrap:wrap; }
  .camp-head .title { font-weight:600; }
  .camp-head .sub { color:var(--muted); font-size:12px; }
  .camp-head .spacer { flex:1; }
  .camp-body { padding:12px 14px; }
  pre { background:var(--bg); border:1px solid var(--border); border-radius:6px;
        padding:10px; font-size:11.5px; overflow:auto; max-height:260px;
        white-space:pre-wrap; color:var(--muted); }
  .muted { color:var(--muted); }
  .detail { margin-top:10px; }
  canvas.curve { background:var(--bg); border:1px solid var(--border);
                 border-radius:6px; width:100%; height:90px; }
  .params-chip { font:11.5px ui-monospace, monospace; background:#21262d;
                 border-radius:4px; padding:1px 6px; color:var(--accent); }
  #notice { padding:8px 14px; background:#2a2113; color:var(--warn);
            border:1px solid var(--warn); border-radius:6px; margin-bottom:14px;
            display:none; }
</style>
</head>
<body>
<header>
  <h1>SBNE — Lanceur de campagnes</h1>
  <span class="meta" id="meta">chargement…</span>
</header>
<main>
  <div id="notice"></div>
  <div class="panel">
    <h2>Nouvelle campagne</h2>
    <div class="row">
      <div><label>Étiquette</label><input type="text" id="f-label" placeholder="exp. batch #12"></div>
      <div><label>Nb de runs</label><input type="number" id="f-nruns" value="4" min="1" max="256"></div>
      <div><label>Graine de base (run i = base + i)</label><input type="number" id="f-seed" value="1000"></div>
      <div><label>Pas max (0 = paramètre MAX_ITERATIONS)</label><input type="number" id="f-steps" value="2000" min="0"></div>
      <div><label>Parallélisme max (vide = auto)</label><input type="number" id="f-par" min="1" placeholder="auto"></div>
    </div>
    <label>Surcharge de paramètres (une CLE=VALEUR par ligne — laisser vide pour les PARAMS par défaut)</label>
    <textarea id="f-params" spellcheck="false" placeholder="NUM_AGENTS=300&#10;PROBA_INSERTION=0.6"></textarea>
    <div style="margin-top:14px">
      <button onclick="createCampaign()">Lancer la campagne</button>
    </div>
  </div>
  <div id="campaigns"><p class="muted">Aucune campagne pour l'instant.</p></div>
</main>
<script>
const $ = id => document.getElementById(id);
function notice(msg, ok) {
  const n = $("notice");
  n.textContent = msg;
  n.style.display = msg ? "block" : "none";
  n.style.background = ok ? "#12261e" : "#2a2113";
  n.style.color = ok ? "var(--ok)" : "var(--warn)";
  n.style.borderColor = ok ? "var(--ok)" : "var(--warn)";
  setTimeout(() => { if (n.textContent === msg) n.style.display = "none"; }, 6000);
}
async function createCampaign() {
  const body = {
    label: $("f-label").value,
    n_runs: parseInt($("f-nruns").value, 10),
    seed_base: parseInt($("f-seed").value, 10),
    max_steps: parseInt($("f-steps").value, 10),
    max_parallel: $("f-par").value ? parseInt($("f-par").value, 10) : null,
    params: $("f-params").value
  };
  const r = await fetch("/api/campaign", {method:"POST",
    headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
  const j = await r.json();
  if (!r.ok) { notice("Erreur : " + (j.error || r.status)); return; }
  notice("Campagne " + j.id + " lancée (" + j.n_runs + " runs).", true);
}
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function fmtDur(run) {
  if (!run.started) return "—";
  const end = run.ended ? new Date(run.ended).getTime() : Date.now();
  return ((end - new Date(run.started).getTime()) / 1000).toFixed(0) + " s";
}
function campaignCard(c) {
  const done = c.runs.filter(r => ["done","failed","stopped","skipped"].includes(r.status)).length;
  const pct = Math.round(100 * done / c.runs.length);
  const params = Object.entries(c.params).map(([k,v]) =>
    `<span class="params-chip">${esc(k)}=${esc(v)}</span>`).join(" ");
  const rows = c.runs.map(r => {
    const s = r.summary || {};
    return `<tr>
      <td class="muted">${esc(r.id)}</td>
      <td>${esc(r.seed)}</td>
      <td><span class="badge b-${r.status}">${r.status}</span></td>
      <td class="muted">${esc(r.started ? r.started.slice(11,19) : "—")}</td>
      <td class="muted">${fmtDur(r)}</td>
      <td>${s.total_steps ?? "—"}</td>
      <td>${s.final_pop ?? "—"}</td>
      <td class="muted" style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
          title="simulations/${esc(r.folder)}">simulations/${esc(r.folder)}</td>
      <td style="white-space:nowrap">
        <button class="ghost" onclick="showDetail('${c.id}','${r.id}')">détail</button>
      </td>
    </tr>`;
  }).join("");
  const stopBtn = c.status === "running"
    ? `<button class="danger" onclick="stopCampaign('${c.id}')">Arrêter</button>` : "";
  return `<div class="camp">
    <div class="camp-head">
      <span class="badge b-c${c.status === "running" ? "running" : c.status}">${c.status}</span>
      <span class="title">${esc(c.label)}</span>
      <span class="sub">${esc(c.id)} · ${done}/${c.runs.length} · ${c.max_steps} pas max
        · seeds ${esc(c.runs[0]?.seed)}…${esc(c.runs[c.runs.length-1]?.seed)} · parallélisme ${c.max_parallel}</span>
      ${params}
      <span class="bar" title="${pct}%"><div style="width:${pct}%"></div></span>
      <span class="spacer"></span>
      ${stopBtn}
    </div>
    <div class="camp-body">
      <table>
        <tr><th>Run</th><th>Seed</th><th>État</th><th>Démarrage</th><th>Durée</th>
            <th>Pas</th><th>Pop fin</th><th>Dossier</th><th></th></tr>
        ${rows}
      </table>
      <div id="detail-${c.id}"></div>
    </div>
  </div>`;
}
async function refresh() {
  try {
    const r = await fetch("/api/state");
    const j = await r.json();
    $("meta").textContent = `${j.server.cores} cores · serveur lancé ${j.server.started}
      · ${Object.keys(j.campaigns).length} campagne(s)`;
    const camps = Object.values(j.campaigns);
    $("campaigns").innerHTML = camps.length
      ? camps.map(campaignCard).join("")
      : '<p class="muted">Aucune campagne pour l\'instant.</p>';
    openDetails.forEach(d => drawDetail(d.c, d.r));
  } catch (e) { /* serveur momentanément indisponible */ }
}
const openDetails = new Map();
async function showDetail(cid, rid) {
  const key = cid + "/" + rid;
  if (openDetails.has(key) && openDetails.get(key).el) { openDetails.delete(key); renderDetailBox(cid, null); return; }
  const resp = await Promise.all([
    fetch(`/api/run/${cid}/${rid}/log?lines=40`),
    fetch(`/api/run/${cid}/${rid}/curve`)
  ]);
  const log = await resp[0].text();
  const curve = await resp[1].json();
  const detail = {c: cid, r: rid, el: true, log, curve};
  openDetails.set(key, detail);
  renderDetailBox(cid, detail);
}
function renderDetailBox(cid, detail) {
  const box = $("detail-" + cid);
  if (!box) return;
  if (!detail) { box.innerHTML = ""; return; }
  drawDetail(detail.c, detail.r);
}
function drawDetail(cid, rid) {
  const detail = openDetails.get(cid + "/" + rid);
  const box = $("detail-" + cid);
  if (!detail || !box) return;
  box.innerHTML = `<div class="detail">
    <label>Courbe population (run ${esc(rid)})</label>
    <canvas class="curve" id="curve-${esc(cid)}-${esc(rid)}" width="900" height="90"></canvas>
    <label>Fin du log</label>
    <pre>${esc(detail.log || "(vide)")}</pre>
  </div>`;
  const cv = $(`curve-${cid}-${rid}`);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const {steps, pop} = detail.curve || {};
  if (!steps || steps.length < 2) {
    ctx.fillStyle = "#8b949e"; ctx.font = "12px sans-serif";
    ctx.fillText("pas encore de courbe (run en cours ou terminé sans stats)", 10, 50);
    return;
  }
  const maxP = Math.max(...pop), minP = Math.min(...pop);
  const span = Math.max(1, maxP - minP);
  ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 1.5; ctx.beginPath();
  steps.forEach((s, i) => {
    const x = 8 + (cv.width - 16) * i / (steps.length - 1);
    const y = 8 + (cv.height - 20) * (1 - (pop[i] - minP) / span);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#8b949e"; ctx.font = "11px sans-serif";
  ctx.fillText(`pop ${minP} → ${maxP}`, 10, 14);
}
async function stopCampaign(cid) {
  if (!confirm("Arrêter la campagne " + cid + " ?")) return;
  await fetch(`/api/campaign/${cid}/stop`, {method:"POST"});
  notice("Arrêt demandé pour " + cid);
}
refresh();
setInterval(refresh, 2500);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Serveur HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "SBNE-Launcher/1.0"

    def log_message(self, fmt, *args):  # silences l'accès log par défaut
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(200, HTML_PAGE, "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(200, {
                "server": {
                    "cores": os.cpu_count() or 1,
                    "started": datetime.fromtimestamp(_server_started).strftime("%H:%M:%S"),
                    "uptime_s": int(time.time() - _server_started),
                },
                "campaigns": _public_state(),
            })
        elif re.fullmatch(r"/api/run/([^/]+)/([^/]+)/log", path):
            m = re.fullmatch(r"/api/run/([^/]+)/([^/]+)/log", path)
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            lines = 40
            for part in qs.split("&"):
                if part.startswith("lines="):
                    try:
                        lines = max(1, min(500, int(part[6:])))
                    except ValueError:
                        pass
            tail = _run_log_tail(m.group(1), m.group(2), lines)
            if tail is None:
                self._json(404, {"error": "run introuvable"})
            else:
                self._send(200, tail, "text/plain; charset=utf-8")
        elif re.fullmatch(r"/api/run/([^/]+)/([^/]+)/curve", path):
            m = re.fullmatch(r"/api/run/([^/]+)/([^/]+)/curve", path)
            curve = _run_curve(m.group(1), m.group(2))
            if curve is None:
                self._json(404, {"error": "run introuvable"})
            else:
                self._json(200, curve)
        else:
            self._json(404, {"error": "route inconnue"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except ValueError:
            self._json(400, {"error": "JSON invalide"})
            return
        if path == "/api/campaign":
            campaign, err = create_campaign(
                label=body.get("label"),
                n_runs=body.get("n_runs"),
                seed_base=body.get("seed_base", 0),
                seed_list=body.get("seed_list"),
                max_steps=body.get("max_steps", 0),
                max_parallel=body.get("max_parallel"),
                params=_parse_param_lines(body.get("params") or ""),
            )
            if err:
                self._json(400, {"error": err})
            else:
                self._json(201, {"id": campaign["id"], "n_runs": campaign["n_runs"]})
        elif re.fullmatch(r"/api/campaign/([^/]+)/stop", path):
            cid = re.fullmatch(r"/api/campaign/([^/]+)/stop", path).group(1)
            if stop_campaign(cid):
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "campagne inconnue ou déjà terminée"})
        else:
            self._json(404, {"error": "route inconnue"})


def main():
    parser = argparse.ArgumentParser(description="Lanceur de campagnes SBNE (panneau web)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Adresse d'écoute (défaut : 127.0.0.1, machine locale uniquement)")
    args = parser.parse_args()
    os.makedirs(SIMS_DIR, exist_ok=True)
    threading.Thread(target=_manager_loop, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"🚀 Lanceur SBNE : http://{args.host}:{args.port}  "
          f"({os.cpu_count()} cores détectés, Ctrl+C pour quitter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du lanceur.")


if __name__ == "__main__":
    main()
