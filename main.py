import cv2
from ultralytics import YOLO
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
import requests
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
import time
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
import requests
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Initialize models and resources
model = None
knn_model = None
knn_scaler = None
video_running = False
# LINE Notify token
token = 'SOZLBqxtLUQ8X60AKpkRq5D2jmiFz2NUW2uJX4fPbQZ'

# Server endpoint URL
server_url = 'http://192.168.24.94:8081/api/pictures'
# Load models
def load_models():
    global model, knn_model, knn_scaler
    model = YOLO("yolov8n-pose.pt")
    knn_model = joblib.load('knn_model.pkl')
    knn_scaler = joblib.load('knn_scaler.pkl')

# Process YOLO results
def process_image(results):
    r = results[0]
    combined_results = []
    for i in range(len(r.boxes)):
        box_results = []
        for j in range(17):
            x, y = r.keypoints.xyn[i][j].numpy()
            box_results.extend([x, y])
        combined_results.append(box_results)
    return combined_results

# Send fall event to server and upload image
def send_fall_event_to_server(timestamp, kp, image_path=None):
    data = {
        'createdAt': timestamp,
        'points': kp.tolist()
    }
    files = None
    if image_path:
        files = {'image': open(image_path, 'rb')}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(server_url,
                             data=json.dumps(data),
                             headers=headers,
                             files=files)
    if response.status_code == 200:
        print(f'事件成功发送: {timestamp}')
    else:
        print(f'事件发送失败: {response.status_code}, {response.text}')
# Send LINE notification
def send_line_notify(token, image_path=None):
    message = '有人跌倒了'
    headers = {"Authorization": "Bearer " + token}
    data = {'message': message}
    if image_path:
        with open(image_path, 'rb') as image_file:
            image_data = {'imageFile': image_file}
            response = requests.post("https://notify-api.line.me/api/notify", headers=headers, data=data, files=image_data)
    else:
        response = requests.post("https://notify-api.line.me/api/notify", headers=headers, data=data)
# Video streaming function
def video_stream():
    notice_time = datetime(1970, 1, 1) 
    load_models()
    cap = cv2.VideoCapture("./fall.mp4")  # Change to your video file path
    while video_running:
        success, frame = cap.read()
        if not success:
            break
        else:
            current_time = datetime.now()
            results = model.track(frame, conf=0.7)
            r = results[0]
            if len(results[0].boxes) >= 1:
                data = np.array(process_image(results))
                data = knn_scaler.transform(data)
                y_pred = knn_model.predict(data)
                kp = data.reshape(-1, 2)
                if y_pred[0] == 2:
                    fall_current_time = current_time
                    folder_name = 'Fall_img'
                    os.makedirs(folder_name, exist_ok=True)
                    time_text = current_time.strftime("%Y-%m-%dT%H:%M:%S")
                    image_filename = current_time.strftime("%Y-%m-%d_%H%M%S_fall.jpg")
                    cv2.imwrite(os.path.join(folder_name, image_filename), frame)
                    send_fall_event_to_server(time_text, kp)
                    #通知line，並將圖片送出
                    time_difference = notice_time - current_time
                    if notice_time == datetime(1970, 1, 1) or time_difference <= timedelta(seconds=1):
                        image_path = "./Fall_img/"+fall_current_time.strftime("%Y-%m-%d_%H%M%S_fall.jpg")
                        send_line_notify(token, image_path=image_path)
                        notice_time = current_time + timedelta(seconds=1)
                        print("sent2line")
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Emit frame to clients via socketio
            socketio.emit('video_frame', {'image': frame_bytes})

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_video():
    global video_running
    if not video_running:
        video_running = True
        socketio.start_background_task(target=video_stream)
    return 'Video stream started'

@app.route('/stop')
def stop_video():
    global video_running
    video_running = False
    return 'Video stream stopped'

if __name__ == '__main__':
    socketio.run(app, host='192.168.24.152', port=5000)
