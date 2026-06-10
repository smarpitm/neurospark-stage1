"""
Live NeuroSpark dashboard — streams agent episodes over Flask-SocketIO.

Run:  python -m dashboard.server
Open: http://localhost:5000
"""

import threading
import time

import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

from brain.network import FlyBrainSNN
from bridge.encoder import Encoder
from bridge.decoder import Decoder
from inference.generative_model import GenerativeModel
from run_episode import run_episode

app = Flask(__name__)
app.config['SECRET_KEY'] = 'neurospark-dashboard'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# Shared agent session state
_session_lock = threading.Lock()
_stop_flag = False
_running = False

_agent = {
    'snn': None,
    'encoder': None,
    'decoder': None,
    'gen_model': None,
}

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>NeuroSpark Dashboard</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
  <style>
    :root {
      --bg: #0f172a; --panel: #1e293b; --border: #334155;
      --text: #e2e8f0; --muted: #94a3b8;
      --accent: #38bdf8; --goal: #10b981; --agent: #818cf8; --warn: #fbbf24;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: 'Segoe UI', system-ui, sans-serif;
      background: var(--bg); color: var(--text); min-height: 100vh;
    }
    header {
      padding: 16px 24px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between;
    }
    header h1 { margin: 0; font-size: 20px; color: var(--accent); }
    .status { display: flex; align-items: center; gap: 8px; font-size: 14px; color: var(--muted); }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; }
    .dot.on { background: #10b981; box-shadow: 0 0 8px #10b981; }
    main {
      display: grid; grid-template-columns: 340px 1fr 320px; gap: 16px;
      padding: 16px 24px 24px; max-width: 1400px; margin: 0 auto;
    }
    .panel {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 16px;
    }
    .panel h2 { margin: 0 0 12px; font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
    canvas { display: block; width: 100%; border-radius: 8px; background: #0b1220; }
    .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .stat {
      background: #0b1220; border-radius: 8px; padding: 10px 12px;
      border: 1px solid var(--border);
    }
    .stat label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
    .stat span { font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums; }
    .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    button {
      border: none; border-radius: 8px; padding: 8px 14px; cursor: pointer;
      font-size: 13px; font-weight: 600; color: #0f172a; background: var(--accent);
    }
    button.secondary { background: #475569; color: var(--text); }
    button.danger { background: #ef4444; color: white; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .efe-bar { margin-bottom: 8px; }
    .efe-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
    .efe-track { height: 8px; background: #0b1220; border-radius: 4px; overflow: hidden; }
    .efe-fill { height: 100%; background: var(--accent); border-radius: 4px; transition: width .2s; }
    .efe-fill.selected { background: var(--goal); }
    #log {
      height: 280px; overflow-y: auto; font-family: ui-monospace, monospace;
      font-size: 12px; background: #0b1220; border-radius: 8px; padding: 10px;
      border: 1px solid var(--border);
    }
    .log-line { margin: 4px 0; color: var(--muted); }
    .log-line strong { color: var(--text); }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>NeuroSpark Live Dashboard</h1>
    <div class="status">
      <span id="conn-dot" class="dot"></span>
      <span id="conn-text">Connecting…</span>
    </div>
  </header>

  <main>
    <section class="panel">
      <h2>Grid World</h2>
      <canvas id="grid" width="300" height="300"></canvas>
      <div class="stats" style="margin-top:12px">
        <div class="stat"><label>Position</label><span id="pos">—</span></div>
        <div class="stat"><label>Goal</label><span id="goal">—</span></div>
        <div class="stat"><label>Step</label><span id="step">0</span></div>
        <div class="stat"><label>Action</label><span id="action">—</span></div>
      </div>
    </section>

    <section class="panel">
      <h2>Controls &amp; Metrics</h2>
      <div class="controls">
        <button id="btn-episode">Run Episode</button>
        <button id="btn-train" class="secondary">Train (5 ep)</button>
        <button id="btn-stop" class="danger" disabled>Stop</button>
      </div>
      <div class="stats">
        <div class="stat"><label>Reward</label><span id="reward">—</span></div>
        <div class="stat"><label>Free Energy</label><span id="fe">—</span></div>
        <div class="stat"><label>Motor Spikes</label><span id="motor">—</span></div>
        <div class="stat"><label>Firing %</label><span id="firing">—</span></div>
      </div>
      <h2 style="margin-top:16px">Activity Log</h2>
      <div id="log"></div>
    </section>

    <section class="panel">
      <h2>Planning (EFE)</h2>
      <div id="efe-bars"><p style="color:var(--muted);font-size:13px">Waiting for first step…</p></div>
      <h2 style="margin-top:16px">Session</h2>
      <div class="stats">
        <div class="stat"><label>Episode</label><span id="episode-num">0</span></div>
        <div class="stat"><label>Result</label><span id="result">Idle</span></div>
      </div>
    </section>
  </main>

  <script>
    const socket = io();
    const canvas = document.getElementById('grid');
    const ctx = canvas.getContext('2d');
    let gridSize = 10;
    let agentPos = [0, 0];
    let goalPos = [9, 9];
    let running = false;

    const $ = id => document.getElementById(id);
    const logEl = $('log');

    function setConnected(on) {
      $('conn-dot').className = 'dot' + (on ? ' on' : '');
      $('conn-text').textContent = on ? 'Connected' : 'Disconnected';
    }

    function log(msg) {
      const line = document.createElement('div');
      line.className = 'log-line';
      line.innerHTML = `<strong>[${new Date().toLocaleTimeString()}]</strong> ${msg}`;
      logEl.prepend(line);
      while (logEl.children.length > 80) logEl.removeChild(logEl.lastChild);
    }

    function drawGrid() {
      const n = gridSize;
      const pad = 12;
      const cell = (canvas.width - pad * 2) / n;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          const px = pad + x * cell;
          const py = pad + y * cell;
          ctx.fillStyle = (x + y) % 2 === 0 ? '#162032' : '#111827';
          ctx.fillRect(px, py, cell - 1, cell - 1);
        }
      }
      const gx = pad + goalPos[0] * cell + cell / 2;
      const gy = pad + goalPos[1] * cell + cell / 2;
      ctx.beginPath();
      ctx.fillStyle = '#10b981';
      ctx.arc(gx, gy, cell * 0.28, 0, Math.PI * 2);
      ctx.fill();

      const ax = pad + agentPos[0] * cell + cell / 2;
      const ay = pad + agentPos[1] * cell + cell / 2;
      ctx.beginPath();
      ctx.fillStyle = '#818cf8';
      ctx.arc(ax, ay, cell * 0.28, 0, Math.PI * 2);
      ctx.fill();
    }

    function renderEfe(plan) {
      if (!plan || !plan.action_names) return;
      const container = $('efe-bars');
      container.innerHTML = '';
      const maxEfe = Math.max(...plan.efe, 0.01);
      plan.action_names.forEach((name, i) => {
        const pct = Math.min(100, (plan.efe[i] / maxEfe) * 100);
        const selected = plan.selected_name === name;
        const div = document.createElement('div');
        div.className = 'efe-bar';
        div.innerHTML = `
          <div class="efe-label"><span>${name}${selected ? ' ◀' : ''}</span><span>${plan.efe[i].toFixed(3)}</span></div>
          <div class="efe-track"><div class="efe-fill${selected ? ' selected' : ''}" style="width:${pct}%"></div></div>`;
        container.appendChild(div);
      });
    }

    function setRunning(on) {
      running = on;
      $('btn-episode').disabled = on;
      $('btn-train').disabled = on;
      $('btn-stop').disabled = !on;
    }

    socket.on('connect', () => { setConnected(true); log('Connected to NeuroSpark backend.'); });
    socket.on('disconnect', () => { setConnected(false); setRunning(false); });

    socket.on('session_status', data => {
      setRunning(data.running);
      $('result').textContent = data.message || 'Idle';
    });

    socket.on('episode_start', data => {
      gridSize = data.grid_size;
      goalPos = data.goal;
      agentPos = data.start;
      $('goal').textContent = goalPos.join(', ');
      $('episode-num').textContent = data.episode || 1;
      $('result').textContent = 'Running…';
      drawGrid();
      log(`Episode started — ${data.grid_size}×${data.grid_size}, goal ${goalPos}`);
    });

    socket.on('episode_step', data => {
      agentPos = data.pos_after;
      $('pos').textContent = agentPos.join(', ');
      $('step').textContent = data.step;
      $('action').textContent = data.action_name;
      $('reward').textContent = data.reward.toFixed(2);
      $('fe').textContent = data.free_energy.toFixed(3);
      $('motor').textContent = data.motor_spikes;
      $('firing').textContent = data.motor_firing_pct.toFixed(1) + '%';
      renderEfe(data.plan);
      drawGrid();
      log(`Step ${data.step}: ${data.action_name} → [${agentPos}]  FE=${data.free_energy.toFixed(2)}`);
    });

    socket.on('episode_end', data => {
      setRunning(false);
      $('result').textContent = data.reached_goal ? `Goal in ${data.steps} steps` : `Stopped (${data.steps} steps)`;
      log(data.reached_goal ? `✓ Goal reached in ${data.steps} steps` : `Episode ended — goal not reached (${data.steps} steps)`);
    });

    socket.on('error_msg', data => log('Error: ' + data.message));

    $('btn-episode').onclick = () => socket.emit('start_episode', { grid_size: 10, max_steps: 100 });
    $('btn-train').onclick = () => socket.emit('start_training', { episodes: 5, max_steps: 80 });
    $('btn-stop').onclick = () => socket.emit('stop_run');

    drawGrid();
  </script>
</body>
</html>
"""


def _init_agent():
    with _session_lock:
        if _agent['snn'] is None:
            np.random.seed(int(time.time()) % 10000)
            _agent['snn'] = FlyBrainSNN(use_noise=True)
            _agent['encoder'] = Encoder()
            _agent['decoder'] = Decoder()
            _agent['gen_model'] = GenerativeModel()


def _set_running(running, message=''):
    global _running
    _running = running
    socketio.emit('session_status', {'running': running, 'message': message})


def _request_stop():
    global _stop_flag
    _stop_flag = True


def _clear_stop():
    global _stop_flag
    _stop_flag = False


def _should_stop():
    return _stop_flag


def _run_episodes(episodes, grid_size, goal_pos, max_steps, learning_rate=0.01):
    global _running
    _init_agent()
    _clear_stop()
    _set_running(True, 'Running')

    snn = _agent['snn']
    encoder = _agent['encoder']
    decoder = _agent['decoder']
    gen_model = _agent['gen_model']

    for ep in range(episodes):
        if _should_stop():
            break

        if ep > 0:
            snn.reset_monitors()

        socketio.emit('episode_start', {
            'episode': ep + 1,
            'grid_size': grid_size,
            'goal': list(goal_pos),
            'start': [0, 0],
        })

        def on_step(step_data):
            socketio.emit('episode_step', step_data)
            socketio.sleep(0.05)

        _, fe_history, reached = run_episode(
            snn=snn,
            encoder=encoder,
            decoder=decoder,
            gen_model=gen_model,
            max_steps=max_steps,
            learning_rate=learning_rate,
            grid_size=grid_size,
            start_pos=(0, 0),
            goal_pos=goal_pos,
            on_step=on_step,
            verbose=False,
            should_stop=_should_stop,
        )

        socketio.emit('episode_end', {
            'episode': ep + 1,
            'reached_goal': reached,
            'steps': len(fe_history),
        })

        if _should_stop():
            break

    _set_running(False, 'Idle')


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@socketio.on('connect')
def on_connect():
    emit('session_status', {'running': _running, 'message': 'Connected'})


@socketio.on('start_episode')
def on_start_episode(data):
    if _running:
        emit('error_msg', {'message': 'A run is already in progress.'})
        return

    grid_size = int(data.get('grid_size', 10))
    max_steps = int(data.get('max_steps', 100))
    goal = (grid_size - 1, grid_size - 1)

    def worker():
        try:
            _run_episodes(1, grid_size, goal, max_steps)
        except Exception as exc:
            socketio.emit('error_msg', {'message': str(exc)})
            _set_running(False, 'Error')

    threading.Thread(target=worker, daemon=True).start()


@socketio.on('start_training')
def on_start_training(data):
    if _running:
        emit('error_msg', {'message': 'A run is already in progress.'})
        return

    episodes = int(data.get('episodes', 5))
    max_steps = int(data.get('max_steps', 80))

    def worker():
        try:
            half = max(1, episodes // 2)
            for ep in range(episodes):
                if _should_stop():
                    break
                gs = 5 if ep < half else 10
                goal = (gs - 1, gs - 1)
                _run_episodes(1, gs, goal, max_steps)
        except Exception as exc:
            socketio.emit('error_msg', {'message': str(exc)})
            _set_running(False, 'Error')

    threading.Thread(target=worker, daemon=True).start()


@socketio.on('stop_run')
def on_stop():
    _request_stop()
    emit('session_status', {'running': True, 'message': 'Stopping…'})


if __name__ == '__main__':
    print('NeuroSpark dashboard → http://localhost:5000')
    socketio.run(app, host='127.0.0.1', port=5000, debug=False, allow_unsafe_werkzeug=True)
