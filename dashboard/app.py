# dashboard/app.py
import os
import sys
import threading
import time
import numpy as np

# Ensure parent directory is in path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from brian2 import ms
from environment.grid_world import GridWorld
from brain.network import FlyBrainSNN
from bridge.encoder import Encoder
from bridge.decoder import Decoder
from inference.generative_model import GenerativeModel
from inference.active_inference import select_action
from inference.free_energy import compute_free_energy

app = Flask(__name__)
app.config['SECRET_KEY'] = 'neurospark_dashboard_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables to control the active simulation thread
sim_thread = None
sim_running = False
sim_paused = False
sim_delay = 0.2  # delay in seconds between steps
train_mode = False  # training enabled/disabled in dashboard

# SNN components (will be lazy-loaded or reset as needed)
snn = None
encoder = None
decoder = None
gen_model = None

def load_or_init_components():
    global snn, encoder, decoder, gen_model
    
    # 1. Initialize models
    snn = FlyBrainSNN(use_noise=True)
    encoder = Encoder()
    decoder = Decoder()
    gen_model = GenerativeModel()
    
    # 2. Try to load weights from file
    weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'trained_weights.npz'))
    if os.path.exists(weights_path):
        try:
            print(f"Loading trained weights from: {weights_path}")
            data = np.load(weights_path)
            snn.S_in.w = data['S_in_w']
            snn.S_rec.w = data['S_rec_w']
            snn.S_out.w = data['S_out_w']
            decoder.W_out = data['decoder_W_out']
            decoder.W_reward = data['decoder_W_reward']
            decoder.W_action = data['decoder_W_action']
            encoder.W = data['encoder_W']
            encoder.W_intention = data['encoder_W_intention']
            print("Successfully loaded trained weights!")
            return True
        except Exception as e:
            print(f"Failed to load weights: {e}. Falling back to random initialization.")
    else:
        print("No pre-trained weights found. Starting with random weights.")
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    global sim_running, sim_paused, train_mode, sim_delay
    return jsonify({
        'running': sim_running,
        'paused': sim_paused,
        'train_mode': train_mode,
        'delay': sim_delay,
        'weights_loaded': os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'trained_weights.npz')))
    })

def simulation_worker():
    global sim_running, sim_paused, sim_delay, train_mode
    global snn, encoder, decoder, gen_model
    
    print("Simulation worker thread started.")
    
    # Load/initialize SNN components
    if snn is None:
        load_or_init_components()
        
    env = GridWorld()
    state = env.reset()
    
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
    step = 0
    max_steps = 150
    trajectory = [list(env.agent_pos)]
    
    # Emit initial state
    socketio.emit('sim_step', {
        'step': 0,
        'agent_pos': list(env.agent_pos),
        'goal_pos': list(env.goal_pos),
        'action': 'START',
        'reward': 0.0,
        'free_energy': 0.0,
        'spikes': [0] * 100,
        'pred_state': [0.0] * 11,
        'act_state': list(state),
        'trajectory': trajectory,
        'done': False
    })
    
    while sim_running and step < max_steps:
        if sim_paused:
            time.sleep(0.1)
            continue
            
        step += 1
        pos_before = list(env.agent_pos)
        
        # 1. Action selection via Active Inference
        # Uses Softmax over expected free energy (EFE)
        action = select_action(snn, state, env.get_actions(), encoder, decoder, gen_model)
        
        # 2. Execute step
        next_state, reward, done = env.step(action)
        trajectory.append(list(env.agent_pos))
        
        # 3. Simulate SNN response to the chosen intention
        intention = encoder.encode_intention(state, action)
        snn.inject_sensory(intention)
        
        t_start = snn.net.t
        snn.run(50 * ms)
        t_end = snn.net.t
        
        # 4. Extract motor spikes and decode predictions
        motor_spikes = snn.get_motor_spikes_in_window(t_start, t_end)
        predicted_next, predicted_reward = decoder.decode(motor_spikes)
        
        # 5. Compute Prediction Error (Free Energy)
        fe = compute_free_energy(predicted_next, next_state, predicted_reward, reward)
        
        # 6. Apply learning updates only if train_mode is True
        if train_mode:
            snn.update_weights(fe, t_start, t_end, learning_rate=0.01)
            decoder.update(motor_spikes, next_state, reward, learning_rate=0.01)
            decoder.update_action_weights(motor_spikes, action, fe, learning_rate=0.01)
            
        # Send data to dashboard frontend
        socketio.emit('sim_step', {
            'step': step,
            'agent_pos': list(env.agent_pos),
            'goal_pos': list(env.goal_pos),
            'action': action_names[action],
            'reward': reward,
            'free_energy': float(fe),
            'spikes': motor_spikes.tolist(),
            'pred_state': predicted_next.tolist(),
            'act_state': next_state.tolist(),
            'trajectory': trajectory,
            'done': done
        })
        
        state = next_state
        if done:
            print(f"Goal reached in {step} steps!")
            break
            
        time.sleep(sim_delay)
        
    sim_running = False
    socketio.emit('sim_finished', {'reason': 'goal_reached' if step < max_steps else 'max_steps'})
    print("Simulation worker thread finished.")

@socketio.on('start_sim')
def handle_start_sim(data=None):
    global sim_running, sim_thread, sim_paused
    
    if sim_running:
        # If running, we just unpause if paused
        if sim_paused:
            sim_paused = False
            socketio.emit('status_update', {'running': True, 'paused': False})
        return
        
    sim_running = True
    sim_paused = False
    socketio.emit('status_update', {'running': True, 'paused': False})
    
    sim_thread = threading.Thread(target=simulation_worker)
    sim_thread.daemon = True
    sim_thread.start()

@socketio.on('pause_sim')
def handle_pause_sim():
    global sim_paused
    sim_paused = True
    socketio.emit('status_update', {'running': True, 'paused': True})

@socketio.on('stop_sim')
def handle_stop_sim():
    global sim_running, sim_paused
    sim_running = False
    sim_paused = False
    socketio.emit('status_update', {'running': False, 'paused': False})

@socketio.on('set_speed')
def handle_set_speed(data):
    global sim_delay
    # data has 'delay' in seconds (e.g. 0.05 to 1.0)
    sim_delay = float(data.get('delay', 0.2))
    print(f"Delay updated to {sim_delay}s")

@socketio.on('toggle_train')
def handle_toggle_train(data):
    global train_mode
    train_mode = bool(data.get('train_mode', False))
    print(f"Train mode set to {train_mode}")
    socketio.emit('status_update', {'train_mode': train_mode})

@socketio.on('reset_weights')
def handle_reset_weights():
    global snn, encoder, decoder, gen_model
    # Re-initialize to random weights
    snn = FlyBrainSNN(use_noise=True)
    encoder = Encoder()
    decoder = Decoder()
    gen_model = GenerativeModel()
    print("SNN weights and decoder reset to random initialization.")
    socketio.emit('weights_reset', {'status': 'success'})

@socketio.on('reload_trained')
def handle_reload_trained():
    loaded = load_or_init_components()
    socketio.emit('weights_reloaded', {'status': 'success' if loaded else 'failed'})

@socketio.on('connect')
def handle_connect():
    print('Dashboard client connected')
    emit('status_update', {
        'running': sim_running,
        'paused': sim_paused,
        'train_mode': train_mode,
        'delay': sim_delay
    })

if __name__ == '__main__':
    # Initialize components on startup
    load_or_init_components()
    
    port = 5000
    print(f"Starting NeuroSpark SNN Dashboard on http://localhost:{port}")
    socketio.run(app, host='127.0.0.1', port=port, debug=True, allow_unsafe_werkzeug=True)
