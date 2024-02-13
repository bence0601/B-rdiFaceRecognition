import cv2
from deepface import DeepFace
from flask import Flask, render_template, Response
import numpy as np

app = Flask(__name__)

cap = cv2.VideoCapture(0)

reference_images = {
    'Apa': cv2.imread("Apa.jpg"),
    'Bence': cv2.imread("Bence.jpg"),
    'Anya': cv2.imread("Anya.jpg"),
    'Jasi': cv2.imread("Jasi.jpg")
}

# Downsampling factor for frames
DOWNSAMPLE_FACTOR = 2


def check_face(frame):
    for name, reference_img in reference_images.items():
        try:
            if DeepFace.verify(frame, reference_img.copy())['verified']:
                return name
        except ValueError:
            pass
    return None


def gen_frames():
    frame_num = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_num += 1
        if frame_num % DOWNSAMPLE_FACTOR != 0:
            continue

        # Downsample the frame
        frame = cv2.resize(frame, (0, 0), fx=1 / DOWNSAMPLE_FACTOR, fy=1 / DOWNSAMPLE_FACTOR)

        name = check_face(frame.copy())
        if name:
            cv2.putText(frame, f"MATCH: {name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO MATCH", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
