from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import math
from ultralytics import YOLO
import base64
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize YOLO model and laptop camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the laptop camera.")
    exit()

model = YOLO('best.pt')
classnames = ['Fire', 'Smoke']
stop_flag = False  # New flag to stop the stream

# Email configuration
sender_email = os.environ.get("EMAIL_1")
sender_password = os.environ.get("KEY")
receiver_email = os.environ.get("EMAIL_2")

def send_email():
    # Create message container
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Fire Detected"

    # Email body
    body = "Fire has been detected. Please take necessary actions."
    msg.attach(MIMEText(body, 'plain'))

    # Create SMTP session
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    
    # Send email
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('stop_stream')
def handle_stop_stream():
    global stop_flag
    stop_flag = True  # Set the flag to stop the stream
    print('Stopping stream...')

def object_detection():
    global stop_flag
    stop_flag = False  # Reset stop flag when starting a new stream
    while True:
        if stop_flag:
            break  # Exit the loop when stop flag is set

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        # Getting bbox, confidence, and class names information to work with
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    cv2.putText(frame, f'{classnames[Class]} {confidence}%', (x1 + 8, y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                    
                    # Check if 'Fire' is detected and send email if true
                    if classnames[Class] == 'Fire':
                        send_email()

        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        yield jpg_as_text

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image_request')
def send_image():
    for jpg_as_text in object_detection():
        emit('image', jpg_as_text)

if __name__ == '__main__':
    socketio.run(app, debug=True)
