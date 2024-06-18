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
model = YOLO("yolov8n-pose.pt")
my_model = load_model('knn_model.pk1')
my_scaler = joblib.load('knn_scaler.pkl')
path = "./fall.mp4"
cap = cv2.VideoCapture(path)
# LINE Notify token
token = 'UU7Ig7LUNtddRq0q3GROs2X0X72957PEVVw1qacyfEd'
notice_time = datetime(1970, 1, 1) 
# URL of the server endpoint
url = "https://196.168.24.94:8081/api/pictures"
# 傳送資料至伺服器
def send_image_with_data(url, image_path, current_time, data):
    keypoints = data.flatten()
    new_list = []
    current_date = current_time.isoformat()
    for i in range(0, len(keypoints) - 1, 2):
        new_list.append(keypoints[i:i + 2].tolist())
    data = {
            "current_date": current_date , # Add the current date to the data
            "keypoints": new_list
        }
    json_data = json.dumps(data)
    headers = {
            "Content-Type": "multipart/form-data"
        }
    with open(image_path, 'rb') as image_file:
            # Create the payload with the image and JSON data
            files = {
                'image': image_file,
                'data': ('data', json_data, 'application/json'),
            }
            response = requests.post(url, files=files, headers=headers)
# 加工資料取得關鍵點座標
def process_image(results):
    r = results[0]
    combined_results = []
    for i in range(len(r.boxes)):
        box_results = []
        for  j  in range(17):
            x, y = r.keypoints.xyn[i][j].numpy()
            box_results.extend([x, y])
            # confidence = r.keypoints.conf[i][j].item()  # Convert tensor to float
        combined_results.append(box_results)
    return combined_results
# line傳送訊息
def send_line_notify(token, image_path=None):
    # Message to send
    message = '有人跌倒了'

    # HTTP headers and data
    headers = { "Authorization": "Bearer " + token }
    data = { 'message': message }
    
    # Send the POST request
    if image_path:
        # Open the image file in binary mode
        with open(image_path, 'rb') as image_file:
            image_data = {'imageFile': image_file}
            response = requests.post("https://notify-api.line.me/api/notify", headers=headers, data=data, files=image_data)
    else:
        response = requests.post("https://notify-api.line.me/api/notify", headers=headers, data=data)

    # Print the response
    print(response.status_code)
    print(response.text)
# 開始影片偵測
while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_time = datetime.now()
    # Perform pose detection
    results = model.track(frame, conf = 0.7) #偵測影片中人體關節點.track 可以將偵測到的人加上id
    annotated_frame = results[0].plot() #.plot 將關鍵點畫到影片上
    r = results[0] # r = 輸出結果 
    # #印出 所有關鍵點的座標位置
    # print(r.keypoints.xyn.numpy().reshape(-1, 2)) 
    # #印出 所有關鍵點的信心值
    # print(r.keypoints.conf.numpy()) 
    #將所有關鍵點的位置與信心值丟入knn/svm/random forest/mlp 模型進行預測(0:沒跌倒/1:失衡/2:跌倒)
    if len(results[0].boxes) >= 1:
        data = np.array(process_image(results)) #  將資料轉成2d numpy array
        data = my_scaler.transform(data)
        y_pred = my_model.predict(data)
        print(y_pred)
        predicted_class = np.argmax(y_pred[0])
        if predicted_class == 2:
            fall_current_time = current_time
            folder_name = 'Fall_img'
            os.makedirs(folder_name, exist_ok=True)
            time_text = fall_current_time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.imwrite(os.path.join(folder_name, fall_current_time.strftime("%Y-%m-%d_%H%M%S_fall.jpg")), frame)
            print("output save")
            # 將資料傳送至伺服器
            # image_path = "./Fall_img/"+fall_current_time.strftime("%Y-%m-%d_%H%M%S_fall.jpg")
            # send_image_with_data(url, image_path, current_time, data)
            #通知line，並將圖片送出
            time_difference = notice_time - current_time
            if notice_time == datetime(1970, 1, 1) or time_difference <= timedelta(seconds=1):
                image_path = "./Fall_img/"+fall_current_time.strftime("%Y-%m-%d_%H%M%S_fall.jpg")
                send_line_notify(token, image_path=image_path)
                notice_time = current_time + timedelta(seconds=1)


    # 存進database



    
    cv2.imshow('YOLOv8 Pose Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()