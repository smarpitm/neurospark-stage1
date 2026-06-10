// dashboard/static/js/dashboard.js

// Initialize WebSocket connection
const socket = io();

// UI Elements
const socketStatusDot = document.getElementById('socket-status-dot');
const socketStatusText = document.getElementById('socket-status-text');
const runStatusIndicator = document.getElementById('run-status-indicator');

const btnStart = document.getElementById('btn-start');
const btnPause = document.getElementById('btn-pause');
const btnStop = document.getElementById('btn-stop');

const speedSlider = document.getElementById('speed-slider');
const speedVal = document.getElementById('speed-val');
const trainSwitch = document.getElementById('train-switch');

const btnReloadWeights = document.getElementById('btn-reload-weights');
const btnResetWeights = document.getElementById('btn-reset-weights');

// Telemetry Elements
const telStep = document.getElementById('telemetry-step');
const telPos = document.getElementById('telemetry-pos');
const telAction = document.getElementById('telemetry-action');
const telReward = document.getElementById('telemetry-reward');
const telFE = document.getElementById('telemetry-fe');
const telSpikes = document.getElementById('telemetry-spikes');

// Canvases
const gridCanvas = document.getElementById('grid-canvas');
const gridCtx = gridCanvas.getContext('2d');

const rasterCanvas = document.getElementById('raster-canvas');
const rasterCtx = rasterCanvas.getContext('2d');

// Make canvases HDPI responsive
function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return rect;
}

let gridRect = setupCanvas(gridCanvas);
let rasterRect = setupCanvas(rasterCanvas);

window.addEventListener('resize', () => {
    gridRect = setupCanvas(gridCanvas);
    rasterRect = setupCanvas(rasterCanvas);
    drawRaster();
});

// Raster state
let globalRasterSpikes = [];
let latestTime = 0;

// Socket connection handlers
socket.on('connect', () => {
    socketStatusDot.className = 'status-dot connected';
    socketStatusText.textContent = 'Connected';
    console.log('Connected to server via WebSocket.');
});

socket.on('disconnect', () => {
    socketStatusDot.className = 'status-dot';
    socketStatusText.textContent = 'Disconnected';
});

// ---------------------------------------------------------
// Chart.js Setup
// ---------------------------------------------------------

// 1. Free Energy Curve Chart
const feCtx = document.getElementById('fe-chart').getContext('2d');
const feChart = new Chart(feCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Variational Free Energy',
            data: [],
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.05)',
            borderWidth: 2,
            tension: 0.35,
            fill: true,
            pointRadius: 2,
            pointHoverRadius: 4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }
        },
        scales: {
            x: {
                grid: { color: 'rgba(255, 255, 255, 0.03)' },
                ticks: { color: '#64748b', font: { family: 'Outfit' } }
            },
            y: {
                grid: { color: 'rgba(255, 255, 255, 0.03)' },
                ticks: { color: '#64748b', font: { family: 'Outfit' } },
                suggestedMin: 0
            }
        }
    }
});

// 2. Prediction Accuracy Bar Chart (Actual vs Predicted)
const accCtx = document.getElementById('accuracy-chart').getContext('2d');
const accLabels = [
    'FOV-NW', 'FOV-N', 'FOV-NE', 
    'FOV-W',  'FOV-C', 'FOV-E', 
    'FOV-SW', 'FOV-S', 'FOV-SE',
    'Goal-DX', 'Goal-DY'
];
const accChart = new Chart(accCtx, {
    type: 'bar',
    data: {
        labels: accLabels,
        datasets: [
            {
                label: 'Actual state',
                data: Array(11).fill(0),
                backgroundColor: 'rgba(16, 185, 129, 0.65)',
                borderColor: '#10b981',
                borderWidth: 1,
                borderRadius: 4
            },
            {
                label: 'Predicted state',
                data: Array(11).fill(0),
                backgroundColor: 'rgba(168, 85, 247, 0.65)',
                borderColor: '#a855f7',
                borderWidth: 1,
                borderRadius: 4
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                labels: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } }
            }
        },
        scales: {
            x: {
                grid: { display: false },
                ticks: { color: '#64748b', font: { family: 'Outfit', size: 10 } }
            },
            y: {
                grid: { color: 'rgba(255, 255, 255, 0.03)' },
                ticks: { color: '#64748b', font: { family: 'Outfit' } },
                min: -1.5,
                max: 1.5
            }
        }
    }
});

// ---------------------------------------------------------
// Visualizers drawing
// ---------------------------------------------------------

// Draw SNN Spike Raster
function drawRasterAll() {
    const dpr = window.devicePixelRatio || 1;
    const w = rasterCanvas.width / dpr;
    const h = rasterCanvas.height / dpr;
    
    rasterCtx.clearRect(0, 0, w, h);
    
    // Draw background grids
    rasterCtx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    rasterCtx.lineWidth = 0.5;
    for (let i = 0; i <= 1000; i += 100) {
        let y = (i / 1000) * h;
        rasterCtx.beginPath();
        rasterCtx.moveTo(0, y);
        rasterCtx.lineTo(w, y);
        rasterCtx.stroke();
    }
    
    if (globalRasterSpikes.length === 0) return;
    
    const timeWindow = 500; // 500ms
    const startTime = Math.max(0, latestTime - timeWindow);
    
    for (const spike of globalRasterSpikes) {
        const x = ((spike.t - startTime) / timeWindow) * w;
        if (x < 0 || x > w) continue;
        
        const y = h - ((spike.i / 1000) * h);
        
        if (spike.layer === 'sensory') rasterCtx.fillStyle = '#38bdf8';
        else if (spike.layer === 'recurrent') rasterCtx.fillStyle = '#c084fc';
        else if (spike.layer === 'motor') rasterCtx.fillStyle = '#10b981';
        
        rasterCtx.beginPath();
        rasterCtx.arc(x, y, 2, 0, Math.PI * 2);
        rasterCtx.fill();
    }
}

// Draw Grid World View
function drawGridWorld(agentPos, goalPos, trajectory, actState) {
    const dpr = window.devicePixelRatio || 1;
    const w = gridCanvas.width / dpr;
    const h = gridCanvas.height / dpr;
    
    gridCtx.clearRect(0, 0, w, h);
    
    const gridSize = 10;
    const padding = 20;
    const innerW = w - padding * 2;
    const innerH = h - padding * 2;
    const cellSize = innerW / gridSize;
    
    // Draw outer grid frame
    gridCtx.fillStyle = '#0f121d';
    gridCtx.fillRect(padding, padding, innerW, innerH);
    gridCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    gridCtx.lineWidth = 2;
    gridCtx.strokeRect(padding, padding, innerW, innerH);
    
    // Draw Grid lines
    gridCtx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    gridCtx.lineWidth = 0.5;
    for (let i = 0; i <= gridSize; i++) {
        const offset = padding + i * cellSize;
        // Vertical
        gridCtx.beginPath();
        gridCtx.moveTo(offset, padding);
        gridCtx.lineTo(offset, padding + innerH);
        gridCtx.stroke();
        // Horizontal
        gridCtx.beginPath();
        gridCtx.moveTo(padding, offset);
        gridCtx.lineTo(padding + innerW, offset);
        gridCtx.stroke();
    }
    
    // Draw Trajectory Trail
    if (trajectory && trajectory.length > 1) {
        gridCtx.strokeStyle = 'rgba(56, 189, 248, 0.25)';
        gridCtx.lineWidth = 2.5;
        gridCtx.lineCap = 'round';
        gridCtx.lineJoin = 'round';
        gridCtx.beginPath();
        trajectory.forEach((pos, idx) => {
            const tx = padding + pos[0] * cellSize + cellSize / 2;
            const ty = padding + pos[1] * cellSize + cellSize / 2;
            if (idx === 0) gridCtx.moveTo(tx, ty);
            else gridCtx.lineTo(tx, ty);
        });
        gridCtx.stroke();
        
        // Draw small markers on trail nodes
        gridCtx.fillStyle = 'rgba(56, 189, 248, 0.4)';
        trajectory.forEach(pos => {
            const tx = padding + pos[0] * cellSize + cellSize / 2;
            const ty = padding + pos[1] * cellSize + cellSize / 2;
            gridCtx.beginPath();
            gridCtx.arc(tx, ty, 2, 0, Math.PI * 2);
            gridCtx.fill();
        });
    }
    
    // Draw Goal position (9, 9)
    if (goalPos) {
        const gx = padding + goalPos[0] * cellSize + cellSize / 2;
        const gy = padding + goalPos[1] * cellSize + cellSize / 2;
        
        // Glow effect
        const pulse = 1 + 0.15 * Math.sin(Date.now() / 150);
        const glowGrad = gridCtx.createRadialGradient(gx, gy, 2, gx, gy, cellSize * 0.7 * pulse);
        glowGrad.addColorStop(0, 'rgba(16, 185, 129, 0.4)');
        glowGrad.addColorStop(1, 'rgba(16, 185, 129, 0)');
        gridCtx.fillStyle = glowGrad;
        gridCtx.beginPath();
        gridCtx.arc(gx, gy, cellSize * 0.7 * pulse, 0, Math.PI * 2);
        gridCtx.fill();
        
        // Core
        gridCtx.fillStyle = '#10b981';
        gridCtx.beginPath();
        gridCtx.arc(gx, gy, cellSize * 0.24, 0, Math.PI * 2);
        gridCtx.fill();
        gridCtx.strokeStyle = '#ffffff';
        gridCtx.lineWidth = 1.5;
        gridCtx.stroke();
    }
    
    // Draw 3x3 FOV Outline around the agent position
    if (agentPos && actState) {
        const ax = agentPos[0];
        const ay = agentPos[1];
        
        const fovSize = cellSize * 3;
        const fx = padding + (ax - 1) * cellSize;
        const fy = padding + (ay - 1) * cellSize;
        
        gridCtx.strokeStyle = 'rgba(251, 146, 60, 0.4)';
        gridCtx.lineWidth = 1.5;
        gridCtx.strokeRect(fx, fy, fovSize, fovSize);
        
        // Shading wall regions in 3x3 FOV
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                const cellVal = actState[r * 3 + c];
                if (cellVal === 1) { // wall
                    const wx = fx + c * cellSize;
                    const wy = fy + r * cellSize;
                    gridCtx.fillStyle = 'rgba(244, 63, 94, 0.25)';
                    gridCtx.fillRect(wx, wy, cellSize, cellSize);
                    gridCtx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
                    gridCtx.lineWidth = 0.5;
                    gridCtx.strokeRect(wx, wy, cellSize, cellSize);
                }
            }
        }
    }
    
    // Draw Agent Position
    if (agentPos) {
        const ax = padding + agentPos[0] * cellSize + cellSize / 2;
        const ay = padding + agentPos[1] * cellSize + cellSize / 2;
        
        // Glow effect
        const pulse = 1 + 0.1 * Math.sin(Date.now() / 150 + 1);
        const glowGrad = gridCtx.createRadialGradient(ax, ay, 2, ax, ay, cellSize * 0.6 * pulse);
        glowGrad.addColorStop(0, 'rgba(56, 189, 248, 0.45)');
        glowGrad.addColorStop(1, 'rgba(56, 189, 248, 0)');
        gridCtx.fillStyle = glowGrad;
        gridCtx.beginPath();
        gridCtx.arc(ax, ay, cellSize * 0.6 * pulse, 0, Math.PI * 2);
        gridCtx.fill();
        
        // Core
        gridCtx.fillStyle = '#38bdf8';
        gridCtx.beginPath();
        gridCtx.arc(ax, ay, cellSize * 0.24, 0, Math.PI * 2);
        gridCtx.fill();
        gridCtx.strokeStyle = '#ffffff';
        gridCtx.lineWidth = 1.5;
        gridCtx.stroke();
    }
}

// Draw initial state of Grid World on load
drawGridWorld([0, 0], [9, 9], [], Array(11).fill(0));

// ---------------------------------------------------------
// WebSocket events reception
// ---------------------------------------------------------

socket.on('sim_step', (data) => {
    // 1. Update Telemetry HUD
    telStep.textContent = data.step;
    telPos.textContent = `(${data.agent_pos[0]}, ${data.agent_pos[1]})`;
    telAction.textContent = data.action;
    telReward.textContent = data.reward.toFixed(2);
    telFE.textContent = data.free_energy.toFixed(3);
    
    const activeMotorSpikes = data.spikes.reduce((a, b) => a + b, 0);
    telSpikes.textContent = activeMotorSpikes;
    
    // 2. Update Free Energy line chart
    feChart.data.labels.push(data.step);
    feChart.data.datasets[0].data.push(data.free_energy);
    feChart.update('none'); // Update without full animation for performance
    
    // 3. Update Bar chart datasets (predicted next state vs actual next state)
    accChart.data.datasets[0].data = data.act_state;
    accChart.data.datasets[1].data = data.pred_state;
    accChart.update('none');
    
    // 4. Removed old raster buffer
    // Updated by spike_raster event
    
    // 5. Update Grid World canvas
    drawGridWorld(data.agent_pos, data.goal_pos, data.trajectory, data.act_state);
});

socket.on('sim_finished', (data) => {
    console.log('Simulation finished: ', data.reason);
    runStatusIndicator.textContent = 'FINISHED';
    runStatusIndicator.style.color = 'var(--accent-green)';
    setButtonStates(false);
});

socket.on('status_update', (data) => {
    if (data.running) {
        runStatusIndicator.textContent = data.paused ? 'PAUSED' : 'RUNNING';
        runStatusIndicator.style.color = data.paused ? 'var(--accent-orange)' : 'var(--accent-blue)';
        setButtonStates(true, data.paused);
    } else {
        runStatusIndicator.textContent = 'IDLE';
        runStatusIndicator.style.color = 'var(--text-muted)';
        setButtonStates(false);
    }
    
    // Update toggles and values from server
    if (data.delay !== undefined) {
        speedSlider.value = data.delay * 1000;
        speedVal.textContent = `${Math.round(data.delay * 1000)}ms`;
    }
    if (data.train_mode !== undefined) {
        trainSwitch.checked = data.train_mode;
    }
});

socket.on('weights_reset', (data) => {
    alert('SNN weights and decoder mapping reset to random initialization!');
});

socket.on('weights_reloaded', (data) => {
    if (data.status === 'success') {
        alert('Trained weights loaded successfully from data/trained_weights.npz!');
    } else {
        alert('Failed to load weights. Make sure data/trained_weights.npz exists!');
    }
});

// Helper to set UI control button states
function setButtonStates(running, paused = false) {
    if (running) {
        btnStart.disabled = !paused; // Disabled if running and unpaused
        btnStart.innerHTML = paused ? 
            `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> Resume` :
            `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> Running`;
        btnPause.disabled = paused;
        btnStop.disabled = false;
    } else {
        btnStart.disabled = false;
        btnStart.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg> Run Episode`;
        btnPause.disabled = true;
        btnStop.disabled = true;
    }
}

// ---------------------------------------------------------
// UI Controls Event Listeners
// ---------------------------------------------------------

btnStart.addEventListener('click', () => {
    // If starting a fresh run, clear charts
    if (runStatusIndicator.textContent === 'IDLE' || runStatusIndicator.textContent === 'FINISHED') {
        feChart.data.labels = [];
        feChart.data.datasets[0].data = [];
        feChart.update();
        globalRasterSpikes = [];
        drawRasterAll();
    }
    socket.emit('start_sim');
});

btnPause.addEventListener('click', () => {
    socket.emit('pause_sim');
});

btnStop.addEventListener('click', () => {
    socket.emit('stop_sim');
});

speedSlider.addEventListener('input', (e) => {
    const msValue = parseInt(e.target.value);
    speedVal.textContent = `${msValue}ms`;
    socket.emit('set_speed', { delay: msValue / 1000 });
});

trainSwitch.addEventListener('change', (e) => {
    socket.emit('toggle_train', { train_mode: e.target.checked });
});

btnReloadWeights.addEventListener('click', () => {
    socket.emit('reload_trained');
});

btnResetWeights.addEventListener('click', () => {
    if (confirm('Are you sure you want to reset the SNN network synapses and decoder maps? All training progress will be lost.')) {
        socket.emit('reset_weights');
    }
});



socket.on('spike_raster', (layers) => {
    layers.forEach(layer => {
        for (let idx = 0; idx < layer.t.length; idx++) {
            let time = layer.t[idx];
            latestTime = Math.max(latestTime, time);
            globalRasterSpikes.push({
                t: time,
                i: layer.i[idx],
                layer: layer.layer
            });
        }
    });
    
    // Keep only last 500ms
    globalRasterSpikes = globalRasterSpikes.filter(s => s.t >= latestTime - 500);
    drawRasterAll();
});

// ---------------------------------------------------------
// UI Tab Navigation
// ---------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const view = btn.getAttribute('data-view');
        
        document.querySelectorAll('.card').forEach(card => {
            if (card.className.includes('view-')) {
                card.style.display = card.className.includes('view-' + view) ? '' : 'none';
            }
        });

        if (view === 'weights') {
            fetchWeightDist();
        }
        if (view === 'feedback') {
            fetchFeedback();
        }
    });
});

// ---------------------------------------------------------
// Weight Distribution Charts
// ---------------------------------------------------------
const weightCharts = {};

function initWeightChart(canvasId, label, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Mean', 'Max', '% Saturated (x100)'],
            datasets: [{
                label: label,
                data: [0, 0, 0],
                backgroundColor: color,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.03)' } },
                x: { grid: { display: false } }
            }
        }
    });
}

weightCharts['S_in'] = initWeightChart('w-in-chart', 'Sensory → Recurrent', 'rgba(56, 189, 248, 0.7)');
weightCharts['S_rec'] = initWeightChart('w-rec-chart', 'Recurrent → Recurrent', 'rgba(168, 85, 247, 0.7)');
weightCharts['S_out'] = initWeightChart('w-out-chart', 'Recurrent → Motor', 'rgba(16, 185, 129, 0.7)');
weightCharts['S_direct'] = initWeightChart('w-direct-chart', 'Sensory → Motor', 'rgba(251, 146, 60, 0.7)');

function fetchWeightDist() {
    fetch('/api/weight_distributions')
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                console.error('Error fetching weights:', data.error);
                return;
            }
            for (const layer in data) {
                if (weightCharts[layer]) {
                    const stats = data[layer];
                    weightCharts[layer].data.datasets[0].data = [
                        stats.mean,
                        stats.max,
                        stats.pct_saturated * 100
                    ];
                    weightCharts[layer].update();
                }
            }
        })
        .catch(err => console.error('Error fetching weights:', err));
}

// ---------------------------------------------------------
// Feedback Table
// ---------------------------------------------------------
function fetchFeedback() {
    fetch('/api/feedback')
        .then(res => res.json())
        .then(data => {
            const tbody = document.getElementById('feedback-tbody');
            tbody.innerHTML = '';
            if (data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="padding: 10px; text-align: center; color: var(--text-muted);">No feedback data found yet.</td></tr>';
                return;
            }
            
            data.forEach(fb => {
                const tr = document.createElement('tr');
                tr.style.borderBottom = '1px solid rgba(255,255,255,0.03)';
                
                const statusColor = fb.goal_reached ? 'var(--accent-green)' : 'var(--accent-orange)';
                const statusText = fb.goal_reached ? 'Goal Reached' : 'Failed';
                
                tr.innerHTML = `
                    <td style="padding: 10px; color: var(--text-muted);">#${fb.episode}</td>
                    <td style="padding: 10px; font-weight: 500;">${fb.steps}</td>
                    <td style="padding: 10px; font-family: monospace;">${fb.total_reward.toFixed(2)}</td>
                    <td style="padding: 10px; font-family: monospace;">${fb.avg_free_energy.toFixed(3)}</td>
                    <td style="padding: 10px; color: ${statusColor}; font-weight: 500;">${statusText}</td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(err => {
            console.error('Error fetching feedback:', err);
            const tbody = document.getElementById('feedback-tbody');
            tbody.innerHTML = `<tr><td colspan="5" style="padding: 10px; text-align: center; color: var(--accent-orange);">Error loading feedback.</td></tr>`;
        });
}

// ---------------------------------------------------------
// Episode Replay
// ---------------------------------------------------------
let replayInterval = null;

document.getElementById('btn-replay-play')?.addEventListener('click', () => {
    const epId = document.getElementById('replay-ep-id').value;
    const statusEl = document.getElementById('replay-status');
    
    if (replayInterval) clearInterval(replayInterval);
    
    statusEl.textContent = 'Loading...';
    
    fetch(`/api/replay/${epId}`)
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                statusEl.textContent = 'Error: ' + data.error;
                return;
            }
            statusEl.textContent = `Playing ${data.length} steps...`;
            
            let stepIdx = 0;
            const trajectorySoFar = [];
            
            replayInterval = setInterval(() => {
                if (stepIdx >= data.length) {
                    clearInterval(replayInterval);
                    statusEl.textContent = 'Replay finished.';
                    return;
                }
                
                const stepData = data[stepIdx];
                // Render the position on the grid canvas
                trajectorySoFar.push(stepData.pos);
                drawGridWorld(stepData.pos, [9, 9], trajectorySoFar, Array(11).fill(0));
                
                stepIdx++;
            }, 200);
        })
        .catch(err => {
            statusEl.textContent = 'Error: ' + err.message;
        });
});
