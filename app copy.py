import onnxruntime as ort
import cv2
import numpy as np
from flask import Flask, jsonify, render_template
from waitress import serve
from PIL import Image

app = Flask(__name__)

# YOLOv8 model load once at the start to optimize performance
model = ort.InferenceSession("yolov8m.onnx", providers=['CPUExecutionProvider'])

# Array of YOLOv8 class labels
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
    """
    Function receives an image (frame),
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param frame: Input video frame
    :return: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    input, img_width, img_height = prepare_input(frame)
    output = run_model(input)
    return process_output(output, img_width, img_height)


def prepare_input(frame):
    """
    Function used to convert input image to tensor,
    required as an input to YOLOv8 object detection
    network.
    :param frame: Input image/frame from webcam
    :return: Numpy array in a shape (3,width,height) where 3 is number of color channels
    """
    img = Image.fromarray(frame)
    img_width, img_height = img.size
    img = img.resize((640, 640))
    img = img.convert("RGB")
    input = np.array(img) / 255.0
    input = input.transpose(2, 0, 1)
    input = input.reshape(1, 3, 640, 640)
    return input.astype(np.float32), img_width, img_height


def run_model(input):
    """
    Function used to pass provided input tensor to
    YOLOv8 neural network and return result
    :param input: Numpy array in a shape (3,width,height)
    :return: Raw output of YOLOv8 network as an array of shape (1,84,8400)
    """
    outputs = model.run(["output0"], {"images": input})
    return outputs[0]


def process_output(output, img_width, img_height):
    """
    Function used to convert RAW output from YOLOv8 to an array
    of detected objects. Each object contains the bounding box of
    this object, the type of object, and the probability
    :param output: Raw output of YOLOv8 network which is an array of shape (1,84,8400)
    :param img_width: The width of the original image
    :param img_height: The height of the original image
    :return: Array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
    """
    output = output[0].astype(float)
    output = output.transpose()

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
    """
    Function calculates "Intersection-over-union" coefficient for specified two boxes
    https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
    :param box1: First box in format: [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format: [x1,y1,x2,y2,object_class,probability]
    :return: Intersection over union ratio as a float number
    """
    return intersection(box1, box2) / union(box1, box2)


def union(box1, box2):
    """
    Function calculates union area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of the boxes union as a float number
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)


def intersection(box1, box2):
    """
    Function calculates intersection area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of intersection of the boxes as a float number
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2 - x1) * (y2 - y1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    # Receive the frame from the webcam through POST
    # We'll be processing frames live in real-time instead of through uploads
    pass


def process_video():
    """
    Function to capture video from webcam and run object detection on it
    """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_objects_on_image(frame)

        # Draw the boxes on the frame
        for box in boxes:
            x1, y1, x2, y2, label, prob = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {prob:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Live Video Feed', frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Start video processing in a separate thread
    import threading
    video_thread = threading.Thread(target=process_video)
    video_thread.start()

    # Run the Flask web server
    serve(app, host='0.0.0.0', port=8080)
