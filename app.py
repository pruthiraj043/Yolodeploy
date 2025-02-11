import onnxruntime as ort
import cv2
import numpy as np
from flask import Flask, render_template
from waitress import serve
from PIL import Image
import threading

app = Flask(__name__)

# YOLOv8 model
model = ort.InferenceSession("yolov8m.onnx", providers=['CPUExecutionProvider'])

yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def detect_objects_on_image(frame):
    input, img_width, img_height = prepare_input(frame)
    output = run_model(input)
    return process_output(output, img_width, img_height)

def prepare_input(frame):
    img = Image.fromarray(frame)
    img_width, img_height = img.size
    img = img.resize((640, 640)).convert("RGB")
    input = np.array(img) / 255.0
    input = input.transpose(2, 0, 1).reshape(1, 3, 640, 640)
    return input.astype(np.float32), img_width, img_height

def run_model(input):
    outputs = model.run(["output0"], {"images": input})
    return outputs[0]

def process_output(output, img_width, img_height):
    output = output[0].astype(float).transpose()
    boxes = []

    for row in output:
        prob = row[4:].max()
        if prob < 0.5:
            continue
        class_id = row[4:].argmax()
        label = yolo_classes[class_id]
        xc, yc, w, h = row[:4]
        x1 = (xc - w / 2) / 640 * img_width
        y1 = (yc - h / 2) / 640 * img_height
        x2 = (xc + w / 2) / 640 * img_width
        y2 = (yc + h / 2) / 640 * img_height
        boxes.append([x1, y1, x2, y2, label, prob])

    boxes.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(boxes) > 0:
        result.append(boxes[0])
        boxes = [box for box in boxes if iou(box, boxes[0]) < 0.5]

    return result

def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)

def union(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)

def intersection(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return max(0, x2 - x1) * max(0, y2 - y1)

@app.route('/')
def index():
    return render_template('index.html')

def process_video():
    WINDOW_NAME1 = "My window 1"
    cv2.namedWindow(WINDOW_NAME1, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        boxes = detect_objects_on_image(frame)

        for box in boxes:
            x1, y1, x2, y2, label, prob = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {prob:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME1, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Run OpenCV in main thread
    video_thread = threading.Thread(target=process_video)
    video_thread.start()

    # Run Flask in another thread to avoid blocking
    flask_thread = threading.Thread(target=lambda: serve(app, host='0.0.0.0', port=8080))
    flask_thread.start()
