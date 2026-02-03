import threading
from flask import Flask, render_template
from flask_socketio import SocketIO
from src.video import process_video

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
video_path = "video/match.mp4"


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print("Client connected")
    threading.Thread(target=process_video, args=(socketio, video_path), daemon=True).start()


if __name__ == '__main__':
    print("Starting Football Analytics server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
