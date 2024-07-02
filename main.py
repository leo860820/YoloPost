import cv2
from ultralytics import YOLO
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
import requests
import json
from sklearn.preprocessing import StandardScaler
from flask import Flask, Response, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import time
import base64
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
CORS(app)

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
    print("Loading YOLO model")
    model = YOLO("yolov8n-pose.pt")
    model_dir = os.path.dirname(__file__)
    knn_model = joblib.load(os.path.join(model_dir, 'rf_model.pkl'))
    knn_scaler = joblib.load(os.path.join(model_dir, 'rf_scaler.pkl'))
    print("Models loaded successfully")

# Process YOLO results
def main_process(results):
    r = results[0]
    for i in range(len(r.keypoints.xyn)):
        combined_results = []
        notice_time = datetime(1970, 1, 1)
        for j in range(17):  # Assuming there are 17 keypoints
            x, y = r.keypoints.xyn[i][j].numpy()
            combined_results.append((x, y))
        box = results[0].boxes[i].xywhn.numpy().reshape(-1, 2)
        list_of_tuples = [tuple(row) for row in box]
        combined_results.extend(list_of_tuples)
        data = np.array(combined_results).reshape(1,-1)
        print(data)
        data = my_scaler.transform(data)
        y_pred = my_model.predict(data)
        kp = data.reshape(-1, 2)
        if y_pred[0] == 1:
            fall_current_time = current_time
            folder_name = f'Fall_img_{m}'
            os.makedirs(folder_name, exist_ok=True)
            time_text = fall_current_time.strftime("%Y-%m-%dT%H:%M:%S")
            image_filename = fall_current_time.strftime("%Y-%m-%d_%H%M%S_fall.jpg")
            cv2.imwrite(os.path.join(folder_name, image_filename), frame)
            print("output save")
            time_difference = notice_time - current_time
            if notice_time == datetime(1970, 1, 1) or time_difference <= timedelta(seconds=1):
                image_path = f"./Fall_img_{m}/"+fall_current_time.strftime("%Y-%m-%d_%H%M%S_fall.jpg")
                send_line_notify(token, image_path=image_path)
                # send_fall_event_to_server(time_text, kp, image_path)
                notice_time = current_time + timedelta(seconds=11)
                print("sent2line")
                break

# Send fall event to server and upload image
def send_fall_event_to_server(timestamp, kp, image_path):
    data = {
        'createdAt': timestamp,
        'points': kp.tolist(),
    }
    try:
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')    
        data['image'] = encoded_image   
        json_data = json.dumps(data)
        headers = {'Content-Type': 'application/json'}
        response = requests.post(server_url, data=json_data, headers=headers)
        if response.status_code == 200:
            print(f'資料成功發送: {timestamp}')
        else:
            print(f'資料發送失敗: {response.status_code}, {response.text}')
    except requests.exceptions.RequestException as e:
        print(f'資料發送失敗: {e}')
# Send LINE notification
def send_line_notify(token, image_path=None):
    message = '有人跌倒了'
    headers = {"Authorization": "Bearer " + token}
    payload = {'message': message}
    files = {'imageFile': open(image_path, 'rb')} if image_path else None
    r = requests.post("https://notify-api.line.me/api/notify", headers=headers, params=payload, files=files)
    return r.status_code


# Video stream processing
def generate_video_feed():
    global video_running
    cap = cv2.VideoCapture("./fall2.mp4")  # 改為您的視頻源
    load_models()
    notice_time = datetime(1970, 1, 1)
    last_heartbeat = time.time()
    
    while video_running:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # 水平翻轉影像
            
            # Process frame with YOLO
            results = model(frame)
            current_time = datetime.now()
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # Emit frame to clients via socketio
            # socketio.emit('video_frame', {'image': frame_bytes})

            # 心跳檢查
            current_time = time.time()
            if current_time - last_heartbeat > 10:  # 每10秒檢查一次
                print("Heartbeat check")
                last_heartbeat = current_time

        except Exception as e:
            print(f"Error occurred: {e}")
           
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
    return '相機已開啟'

@app.route('/stop')
def stop_video():
    global video_running
    video_running = False
    return '相機已關閉'

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)