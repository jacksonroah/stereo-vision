from inference import get_model
import supervision as sv
import cv2
from ultralytics import YOLO

# define the image url to use for inference
image_file = "/Users/jacksonroah/Desktop/balltest.png"
image = cv2.imread(image_file)

def detect_ball_yolo(image, model, conf_threshold=0.5):
    """Detect ball in image using YOLOv8 model."""
    try:
        # Run inference
        results = model(image, conf=conf_threshold, verbose=False)[0]
        
        # Extract detections
        for det in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            
            # Check if it's a sports ball (class 32) or any detection if using ball-specific model
            # COCO class 32 is 'sports ball'
            if conf > conf_threshold and (cls == 32 or True):
                # Calculate center of bounding box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # Approximate radius (average of half-width and half-height)
                radius = ((x2 - x1) + (y2 - y1)) / 4
                
                return (cx, cy, radius, conf)
        
        return None
    except Exception as e:
        print(f"YOLO detection error: {e}")
        return None

# load a pre-trained yolov8n model
model = YOLO('yolov8m.pt')

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model(image)

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results)

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)