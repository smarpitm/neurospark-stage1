from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Flask-SocketIO Hello World Test</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: radial-gradient(circle at top right, #1a1e29, #0d0f14);
            color: #e2e8f0;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            width: 100%;
            max-width: 500px;
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.08);
            text-align: center;
        }
        h1 {
            color: #38bdf8;
            font-size: 24px;
            margin-bottom: 8px;
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        p {
            color: #94a3b8;
            margin-bottom: 24px;
        }
        .status-container {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(15, 23, 42, 0.6);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 20px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #ef4444;
            display: inline-block;
            box-shadow: 0 0 8px #ef4444;
            transition: all 0.3s ease;
        }
        .status-dot.connected {
            background-color: #10b981;
            box-shadow: 0 0 8px #10b981;
        }
        #messages {
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 15px;
            height: 200px;
            overflow-y: auto;
            background: rgba(15, 23, 42, 0.4);
            border-radius: 10px;
            text-align: left;
        }
        .message {
            margin: 8px 0;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 6px;
            border-left: 3px solid #38bdf8;
            font-family: monospace;
            font-size: 13px;
            animation: fadeIn 0.3s ease-out;
        }
        .time-tag {
            color: #64748b;
            margin-right: 8px;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>NeuroSpark Gateway</h1>
        <p>Flask + WebSockets Verification (Stage 1)</p>
        
        <div class="status-container">
            <span id="status-dot" class="status-dot"></span>
            <span id="status-text">Connecting...</span>
        </div>
        
        <div id="messages"></div>
    </div>

    <script>
        const socket = io();
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        const messagesDiv = document.getElementById('messages');

        socket.on('connect', () => {
            statusDot.className = 'status-dot connected';
            statusText.textContent = 'Connected';
            console.log('Connected to server');
        });

        socket.on('disconnect', () => {
            statusDot.className = 'status-dot';
            statusText.textContent = 'Disconnected';
        });

        socket.on('broadcast_msg', (data) => {
            console.log('Received broadcast:', data);
            const msgElement = document.createElement('div');
            msgElement.className = 'message';
            msgElement.innerHTML = `<span class="time-tag">[${data.time}]</span><span>${data.message}</span>`;
            messagesDiv.appendChild(msgElement);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('broadcast_msg', {'message': 'Connection established with backend.', 'time': time.strftime('%H:%M:%S')})

def background_thread():
    count = 0
    while True:
        time.sleep(2)
        count += 1
        # Broadcast "hello world" to all clients
        socketio.emit('broadcast_msg', {
            'message': f'Hello World! Broadcast #{count}',
            'time': time.strftime('%H:%M:%S')
        })
        print(f"Broadcasted Hello World #{count}")

if __name__ == '__main__':
    # Start background thread to broadcast messages
    t = threading.Thread(target=background_thread)
    t.daemon = True
    t.start()
    
    print("Starting Flask-SocketIO server on http://localhost:5000")
    socketio.run(app, host='127.0.0.1', port=5000, debug=True, allow_unsafe_werkzeug=True)
