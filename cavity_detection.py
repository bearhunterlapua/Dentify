import os
import sys
import torch
import cv2
import numpy as np

# If YOLOv5 is cloned at /Users/eric/Desktop/cavity_detection_app/yolov5:
yolov5_path = "/Users/eric/Desktop/cavity_detection_app/yolov5"
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# from models.common import DetectMultiBackend
from common import DetectMultiBackend, letterbox, non_max_suppression, scale_boxes

# from utils.augmentations import letterbox
# from utils.general import non_max_suppression, scale_boxes

###############################################################################
# GLOBAL DATA (class names, alveolar detection, etc.)
###############################################################################

class_names = {
    0: "Implant",
    1: "Filling",
    2: "Impacted",
    3: "Cavity",
    4: "Bridges",
    5: "Braces",
    6: "Root Canal",
    7: "Porcelain",
    8: "Alveolar Nerve"   # NEW CLASS
}

# (BGR) colors, if you need them for any reason
class_colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 128),
    7: (50, 50, 200),
    8: (200, 50, 50)
}

###############################################################################
# HELPER: ALVEOLAR NERVE DETECTION
###############################################################################
def detect_alveolar_nerve(image, device, alveolar_model_path='runs/train/alveolar_nerve/best.pt'):
    """
    Runs alveolar nerve detection on the given image using a separate model.
    Returns a list of alveolar nerve bounding boxes:
       [ (x1, y1, x2, y2, 8), ... ]
    Class index = 8 for alveolar nerve.
    """
    alveolar_detections = []

    if not os.path.isfile(alveolar_model_path):
        print(f"Alveolar nerve model not found at: {alveolar_model_path}")
        return alveolar_detections  # empty if not found

    # Load alveolar model
    try:
        alveolar_model = DetectMultiBackend(alveolar_model_path, device=device)
        alveolar_model.eval()
    except Exception as e:
        print(f"Error loading alveolar nerve model: {e}")
        return alveolar_detections

    # Preprocess
    img_size = 640
    im_letterbox = letterbox(image, img_size, stride=32, auto=True)[0]
    im_letterbox = im_letterbox.transpose((2, 0, 1))
    im_letterbox = np.ascontiguousarray(im_letterbox)
    alveolar_tensor = torch.from_numpy(im_letterbox).to(device)
    alveolar_tensor = alveolar_tensor.float() / 255.0
    if alveolar_tensor.ndimension() == 3:
        alveolar_tensor = alveolar_tensor.unsqueeze(0)

    # Inference
    try:
        with torch.no_grad():
            alveolar_pred = alveolar_model(alveolar_tensor)
    except Exception as e:
        print(f"Error during alveolar nerve model inference: {e}")
        return alveolar_detections

    # NMS
    alveolar_pred = non_max_suppression(alveolar_pred, conf_thres=0.3, iou_thres=0.45)
    for det in alveolar_pred:
        if len(det):
            # Scale boxes back to original image
            try:
                det[:, :4] = scale_boxes(alveolar_tensor.shape[2:], det[:, :4], image.shape).round()
            except Exception as e:
                print(f"Error scaling alveolar nerve boxes: {e}")
                continue

            for *xyxy, conf, cls in det:
                try:
                    x1, y1, x2, y2 = map(int, xyxy)
                    alveolar_detections.append((x1, y1, x2, y2, 8))
                except Exception as e:
                    print(f"Error processing alveolar detection: {e}")
                    continue

    return alveolar_detections

###############################################################################
# HELPER: LOAD IMAGE
###############################################################################
def load_image(path, img_size=640):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Using np.fromfile() helps handle special filenames on Windows,
    # but normal open() + np.frombuffer() is usually fine too.
    img_data = np.fromfile(path, dtype=np.uint8)
    if img_data.size == 0:
        raise ValueError(f"File is empty: {path}")

    img0 = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img0 is None:
        raise ValueError(f"Failed to decode image: {path}")

    # YOLO letterbox
    img = letterbox(img0, img_size, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.ascontiguousarray(img)

    return img, img0

###############################################################################
# MAIN FUNCTION TO CALL FOR DETECTION
###############################################################################
def run_cavity_detection(img_path, model_path='/Users/eric/Desktop/cavity_detection_app/yolov5/runs/train/2experiment_data3_adjusted/weights/best.pt'):
    """
    1) Loads the image from img_path
    2) Runs the main detection model (cavities, impacted, etc.)
    3) If impacted teeth (class=2) found, also run alveolar detection
    4) Returns a list of bounding boxes in the format:
       [ [x1, y1, x2, y2, class_id], ... ]
    """

    # 1) Load the image
    img, img0 = load_image(img_path)

    # 2) Prepare the tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # 3) Load the main detection model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = DetectMultiBackend(model_path, device=device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    # 4) Run inference
    with torch.no_grad():
        try:
            pred = model(img_tensor)
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {e}")

    # 5) Non-maximum Suppression
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45)

    # 6) Process detections
    final_detections = []  # will hold [x1, y1, x2, y2, class_id]
    for det in pred:
        if len(det):
            # Scale boxes to original image
            try:
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            except Exception as e:
                print(f"Error scaling boxes: {e}")
                continue

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                cls = int(cls)
                final_detections.append([x1, y1, x2, y2, cls])

    # 7) If "Impacted" (class=2) found, run alveolar detection
    impacted_found = any(d[4] == 2 for d in final_detections)
    if impacted_found:
        alveolar_boxes = detect_alveolar_nerve(img0, device=device)
        if alveolar_boxes:
            # alveolar_boxes is [(x1, y1, x2, y2, 8), ...]
            for x1, y1, x2, y2, alveolar_cls in alveolar_boxes:
                final_detections.append([x1, y1, x2, y2, alveolar_cls])

    return final_detections

# End of file