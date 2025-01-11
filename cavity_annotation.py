import os
import torch
import cv2
import numpy as np
import re
import sys 

# Example: if YOLOv5 is cloned at /Users/eric/yolov5
yolov5_path = "/Users/eric/Desktop/cavity_detection_app/yolov5"
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)


from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# Define class names based on the indices
# Add "Alveolar Nerve" at index 8
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

# Define colors for classes (BGR format)
class_colors = {
    0: (255, 0, 0),       # Blue for Implant
    1: (0, 255, 0),       # Green for Filling
    2: (0, 0, 255),       # Red for Impacted
    3: (255, 255, 0),     # Cyan for Cavity
    4: (255, 0, 255),     # Magenta for Bridges
    5: (0, 255, 255),     # Yellow for Braces
    6: (128, 0, 128),     # Purple for Root Canal
    7: (50, 50, 200),     # Custom color for Porcelain
    8: (200, 50, 50)      # NEW COLOR for Alveolar Nerve
}

# Initialize global variables
current_class = 3  # Default to "Cavity"
detections = []    
manual_boxes = []
img_display = None
img_original = None
start_point = None
end_point = None
drawing = False
flipped = False
original_image_backup = None
original_detections_backup = None
original_manual_boxes_backup = None

###############################################################################
# 1) HELPER FUNCTION: LOAD ALVEOLAR MODEL (ONCE) AND RUN INFERENCE
###############################################################################
def detect_alveolar_nerve(image, device):
    """
    Runs alveolar nerve detection on the given image using a separate model.
    Returns a list of alveolar nerve bounding boxes in the format:
    [(x1, y1, x2, y2, class_index), ...]
    where class_index = 8 for alveolar nerve.
    """
    alveolar_detections = []

    # TODO: Replace with your actual alveolar nerve model path:
    alveolar_model_path = 'runs/train/alveolar_nerve/best.pt'  
    if not os.path.isfile(alveolar_model_path):
        print(f"Alveolar nerve model not found at: {alveolar_model_path}")
        return alveolar_detections  # Return empty if no alveolar model
    
    try:
        alveolar_model = DetectMultiBackend(alveolar_model_path, device=device)
        alveolar_model.eval()
    except Exception as e:
        print(f"Error loading alveolar nerve model: {e}")
        return alveolar_detections

    # Preprocess image similarly
    img_size = 640
    im_letterbox = letterbox(image, img_size, stride=32, auto=True)[0]
    im_letterbox = im_letterbox.transpose((2, 0, 1))
    im_letterbox = np.ascontiguousarray(im_letterbox)
    alveolar_tensor = torch.from_numpy(im_letterbox).to(device)
    alveolar_tensor = alveolar_tensor.float() / 255.0
    if alveolar_tensor.ndimension() == 3:
        alveolar_tensor = alveolar_tensor.unsqueeze(0)

    with torch.no_grad():
        try:
            alveolar_pred = alveolar_model(alveolar_tensor)
        except Exception as e:
            print(f"Error during alveolar nerve model inference: {e}")
            return alveolar_detections

    # NMS, thresholding
    alveolar_pred = non_max_suppression(alveolar_pred, conf_thres=0.3, iou_thres=0.45)
    for det in alveolar_pred:
        if len(det):
            try:
                det[:, :4] = scale_boxes(alveolar_tensor.shape[2:], det[:, :4], image.shape).round()
            except Exception as e:
                print(f"Error scaling alveolar nerve boxes: {e}")
                continue

            for *xyxy, conf, cls in det:
                try:
                    x1, y1, x2, y2 = map(int, xyxy)
                    # Force alveolar nerve class index = 8
                    alveolar_detections.append((x1, y1, x2, y2, 8))
                except Exception as e:
                    print(f"Error processing alveolar detection: {e}")
                    continue

    return alveolar_detections


###############################################################################
# 2) HELPER FUNCTION: LOAD IMAGE
###############################################################################
def load_image(path, img_size=640):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    img_data = np.fromfile(path, dtype=np.uint8)
    if img_data.size == 0:
        raise ValueError(f"File is empty: {path}")

    img0 = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img0 is None:
        raise ValueError(f"Failed to decode image: {path}")

    img = letterbox(img0, img_size, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))  # Convert HWC to CHW
    img = np.ascontiguousarray(img)
    return img, img0


def draw_current_class(img, class_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Current Class: {class_name}"
    position = (10, 30)  # Position at the top-left corner
    font_scale = 0.8
    color = (0, 0, 255)  
    thickness = 2
    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)


###############################################################################
# 3) MOUSE EVENT CALLBACK: DRAW OR REMOVE BOXES
#    - If the user draws an "Impacted" box => run alveolar detection.
###############################################################################
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, manual_boxes, current_class, detections, img_original

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        width = abs(end_point[0] - start_point[0])
        height = abs(end_point[1] - start_point[1])
        min_size = 10  # pixels

        if width > min_size and height > min_size:
            # Add the new bounding box to manual_boxes
            manual_boxes.append((start_point, end_point, current_class))
            print(f"Added new bounding box: {start_point} to {end_point}, class {current_class}")

            # -----------------------------------------------------------------
            # If the new box is class = 2 (Impacted), run alveolar detection
            # -----------------------------------------------------------------
            if current_class == 2:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                alveolar_boxes = detect_alveolar_nerve(img_original, device=device)
                if alveolar_boxes:
                    # Append alveolar nerve detections
                    detections.extend(alveolar_boxes)
                    print(f"Detected alveolar nerve in region of impacted tooth. Found {len(alveolar_boxes)} boxes.")

        else:
            # A click without a large drag => remove a box if clicked inside
            point = (x, y)
            box_removed = False
            # Check manual_boxes
            for idx in range(len(manual_boxes)-1, -1, -1):
                (bx1, by1), (bx2, by2), bcls = manual_boxes[idx]
                x_min, x_max = min(bx1, bx2), max(bx1, bx2)
                y_min, y_max = min(by1, by2), max(by1, by2)
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    del manual_boxes[idx]
                    box_removed = True
                    print("Removed manual annotation.")
                    break
            # Check detections if none removed
            if not box_removed:
                for idx in range(len(detections)-1, -1, -1):
                    x1, y1, x2, y2, cls = detections[idx]
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        del detections[idx]
                        box_removed = True
                        print(f"Removed automated detection at index {idx}.")
                        break
                if not box_removed:
                    print("Clicked outside of any bounding box.")


###############################################################################
# 4) FLIP IMAGE AND BOXES (IF NEEDED)
###############################################################################
def flip_image_and_boxes():
    global img_original, detections, manual_boxes, flipped

    # Flip image horizontally
    img_original = cv2.flip(img_original, 1)
    height, width = img_original.shape[:2]

    # Flip detections
    flipped_detections = []
    for (x1, y1, x2, y2, cls) in detections:
        nx1 = width - x2 - 1
        nx2 = width - x1 - 1
        flipped_detections.append((nx1, y1, nx2, y2, cls))
    detections[:] = flipped_detections

    # Flip manual_boxes
    flipped_manual_boxes = []
    for (sx, sy), (ex, ey), ccls in manual_boxes:
        nx1 = width - ex - 1
        nx2 = width - sx - 1
        flipped_manual_boxes.append(((nx1, sy), (nx2, ey), ccls))
    manual_boxes[:] = flipped_manual_boxes

    flipped = not flipped
    state = "Flipped" if flipped else "Original"
    print(f"Image state: {state}")


###############################################################################
# 5) MAIN DETECTION + DRAW FUNCTION
#    - If the model finds an "Impacted" box => run alveolar detection.
###############################################################################
def detect_and_draw(img_path, output_directory):
    global img_display, img_original, current_class
    global detections, manual_boxes, drawing, start_point, end_point, flipped
    global original_image_backup, original_detections_backup, original_manual_boxes_backup

    text_content = []

    # Load image
    try:
        img, img0 = load_image(img_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Load main detection model (for cavities, impacted, etc.)
    model_path = 'runs/train/2experiment_data3_adjusted/weights/best.pt'
    if not os.path.isfile(model_path):
        print(f"Model file not found: {model_path}")
        return False

    try:
        model = DetectMultiBackend(model_path, device=device)
        print("Model classes:", model.names)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    model.eval()
    with torch.no_grad():
        try:
            pred = model(img_tensor)
        except Exception as e:
            print(f"Error during model inference: {e}")
            return False

    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45)

    # Clear old detections
    detections.clear()

    # Scale and store new detections
    for det in pred:
        if len(det):
            try:
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            except Exception as e:
                print(f"Error scaling boxes: {e}")
                continue

            for *xyxy, conf, cls in det:
                try:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls = int(cls)
                    detections.append((x1, y1, x2, y2, cls))
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue

    # -------------------------------------------------------------------------
    #  If we have "Impacted" boxes (class=2) from the model, run alveolar detection
    # -------------------------------------------------------------------------
    impacted_found = any(d[4] == 2 for d in detections)
    if impacted_found:
        alveolar_boxes = detect_alveolar_nerve(img0, device=device)
        if alveolar_boxes:
            detections.extend(alveolar_boxes)
            print(f"Detected alveolar nerve for impacted tooth. Found {len(alveolar_boxes)} alveolar box(es).")

    # Store backups
    img_original = img0.copy()
    img_display = img0.copy()
    original_image_backup = img_original.copy()
    original_detections_backup = detections[:]
    original_manual_boxes_backup = manual_boxes[:]
    flipped = False

    # Set up window / callback
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_rectangle)

    while True:
        # Refresh display
        img_display = img_original.copy()

        # Draw automated detections
        for det in detections:
            x1, y1, x2, y2, cls = det
            color = class_colors.get(cls, (0, 255, 255))
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names.get(cls, 'Unknown')}"
            cv2.putText(img_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

        # Draw manual boxes
        for box in manual_boxes:
            (bx1, by1), (bx2, by2), bcls = box
            color = class_colors.get(bcls, (255, 0, 255))
            cv2.rectangle(img_display, (bx1, by1), (bx2, by2), color, 2)
            label = f"{class_names.get(bcls, 'Unknown')}"
            cv2.putText(img_display, label, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        # If drawing in progress, show rectangle
        if drawing:
            cv2.rectangle(img_display, start_point, end_point, (255, 0, 0), 2)  # Blue
            width = end_point[0] - start_point[0]
            height = end_point[1] - start_point[1]
            cv2.putText(img_display, f"{width}x{height}", (end_point[0], end_point[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Show current class
        class_name = class_names.get(current_class, "Unknown")
        draw_current_class(img_display, class_name)

        cv2.imshow('Image', img_display)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('s') or key == ord('S'):
            print("Save key 's' pressed.")
            if detections or manual_boxes:
                # Determine filename suffix
                suffix = "_flipped" if flipped else ""
                for box in manual_boxes:
                    (x1, y1), (x2, y2), cls = box
                    norm_x_center = ((x1 + x2) / 2) / img_original.shape[1]
                    norm_y_center = ((y1 + y2) / 2) / img_original.shape[0]
                    norm_width = abs(x2 - x1) / img_original.shape[1]
                    norm_height = abs(y2 - y1) / img_original.shape[0]
                    text_content.append(
                        f"{cls} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                    )
                for det in detections:
                    x1, y1, x2, y2, cls = det
                    norm_x_center = ((x1 + x2) / 2) / img_original.shape[1]
                    norm_y_center = ((y1 + y2) / 2) / img_original.shape[0]
                    norm_width = abs(x2 - x1) / img_original.shape[1]
                    norm_height = abs(y2 - y1) / img_original.shape[0]
                    text_content.append(
                        f"{cls} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                    )

                base, ext = os.path.splitext(img_path)
                text_output_path = os.path.join(
                    output_directory,
                    f"{os.path.splitext(os.path.basename(img_path))[0]}{suffix}.txt"
                )
                print(f"Saving annotations to: {text_output_path}")
                try:
                    with open(text_output_path, 'w') as f:
                        f.write('\n'.join(text_content))
                    print(f"Detections saved to {text_output_path}")
                except Exception as e:
                    print(f"Failed to write detections to file: {e}")

                annotated_image_path = os.path.join(
                    output_directory,
                    f"{os.path.splitext(os.path.basename(img_path))[0]}{suffix}{ext}"
                )
                cv2.imwrite(annotated_image_path, img_display)
                print(f"Annotated image saved to {annotated_image_path}")
            else:
                print("No detections or annotations to save.")
            break

        elif key == ord('q') or key == ord('Q'):
            print("Skipping saving annotations for this image.")
            break

        # Change current_class
        elif key == ord('0'):
            current_class = 0
        elif key == ord('1'):
            current_class = 1
        elif key == ord('2'):
            current_class = 2
        elif key == ord('3'):
            current_class = 3
        elif key == ord('4'):
            current_class = 4
        elif key == ord('5'):
            current_class = 5
        elif key == ord('6'):
            current_class = 6
        elif key == ord('7'):
            current_class = 7
        elif key == ord('8'):
            current_class = 8  # Alveolar Nerve if you want to annotate manually

        # Flip image option
        elif key == ord('f') or key == ord('F'):
            if flipped:
                # Restore from backup
                img_original = original_image_backup.copy()
                detections.clear()
                detections.extend(original_detections_backup)
                manual_boxes.clear()
                manual_boxes.extend(original_manual_boxes_backup)
                flipped = False
                print("Image state: Original")
            else:
                # Flip and update
                original_image_backup = img_original.copy()
                original_detections_backup = detections[:]
                original_manual_boxes_backup = manual_boxes[:]
                flip_image_and_boxes()

    cv2.destroyAllWindows()

    if detections or manual_boxes:
        return True
    else:
        print('No detections or annotations')
        return False


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def main():
    directory = "/Users/eric/Desktop/cavity_detection_app/train2"
    output_directory = "/Users/eric/Desktop/cavity_detection_app/annotations"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        files = sorted(files, key=natural_sort_key)
    except Exception as e:
        print(f"Failed to list files in directory {directory}: {e}")
        files = []

    if not files:
        print("No image files found in the directory.")
        return  # Exit if no files are found

    # Display available images
    print("Available images:")
    for idx, file in enumerate(files, start=1):
        print(f"{idx}: {file}")

    # Prompt the user to select a starting index
    while True:
        try:
            start_index_input = input("Enter the index of the image to start with (default is 1): ").strip()
            if start_index_input == '':
                start_index = 1
            else:
                start_index = int(start_index_input)
            if 1 <= start_index <= len(files):
                break
            else:
                print(f"Please enter a number between 1 and {len(files)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    total_files = len(files)
    for idx, file in enumerate(files[start_index - 1:], start=start_index):
        img_path = os.path.join(directory, file)
        print(f"\nProcessing {idx}/{total_files}: {file}...")
        global detections, manual_boxes, current_class
        detections.clear()
        manual_boxes.clear()
        current_class = 3
        saved = detect_and_draw(img_path, output_directory)
        if saved:
            print(f"Processed {file} and saved annotations.")
        else:
            print(f"Processed {file} without annotations.")
    print("\nFinished processing all files.")


if __name__ == "__main__":
    main()