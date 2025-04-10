from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import io
from PIL import Image
import os
from dotenv import load_dotenv
from ultralytics import YOLO
import logging
import socket
import torch
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

app = Flask(__name__)
# Configure CORS to allow all origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Load environment variables
load_dotenv()

# Get local IP address
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

local_ip = get_local_ip()
logger.info(f"Server will be accessible at: http://{local_ip}:5000")

# Load YOLOv8 model with CUDA if available
logger.info("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # Load model first

# Optimize model for inference
if device == 'cuda':
    # First fuse the model in float32
    model.fuse()
    # Then move to CUDA and convert to half precision
    model = model.to(device)
    model.half()
    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

logger.info("Model loaded successfully")

def preprocess_image(image, orientation=1):
    logger.debug("Preprocessing image...")
    try:
        # Convert base64 to image
        image_data = base64.b64decode(image)
        image = Image.open(io.BytesIO(image_data))
        
        # Apply orientation correction if needed
        if orientation in [3, 6, 8]:
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
        
        # Resize image to YOLOv8's expected size (divisible by 32)
        target_size = 640
        ratio = min(target_size / image.width, target_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        # Ensure dimensions are divisible by 32
        new_size = (new_size[0] - (new_size[0] % 32), new_size[1] - (new_size[1] % 32))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image = np.array(image)
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Convert to BCHW format
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.expand_dims(image, 0)  # Add batch dimension
        
        logger.debug(f"Processed image shape: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def get_detailed_description(results):
    logger.debug("Generating description from results...")
    try:
        descriptions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Format the description
                description = f"A {class_name} (confidence: {confidence:.2f}) at position ({int(x1)}, {int(y1)})"
                descriptions.append(description)
        
        return " ".join(descriptions) if descriptions else "No objects detected"
    except Exception as e:
        logger.error(f"Error in get_detailed_description: {str(e)}")
        raise

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        logger.info("Received analyze request")
        data = request.json
        if 'image' not in data:
            logger.error("No image provided in request")
            return jsonify({'error': 'No image provided'}), 400

        # Get orientation from request, default to 1 (normal)
        orientation = data.get('orientation', 1)
        
        # Preprocess the image with orientation
        image = preprocess_image(data['image'], orientation)
        
        # Convert to tensor and move to device
        image = torch.from_numpy(image)
        if device == 'cuda':
            image = image.half().to(device)
        else:
            image = image.to(device)
        
        # Run YOLOv8 inference with CUDA and optimized settings
        logger.debug("Running YOLOv8 inference...")
        start_time = time.time()
        
        with torch.no_grad():  # Disable gradient calculation for inference
            results = model(
                image,
                conf=0.25,  # Confidence threshold
                iou=0.45,   # IoU threshold
                agnostic_nms=True,  # Class-agnostic NMS
                max_det=100,  # Maximum number of detections
            )
        
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        logger.debug(f"Inference completed in {inference_time:.2f}ms")
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
        logger.debug("Image processing completed")
        
        return jsonify({
            'annotated_image': annotated_image_base64,
            'inference_time_ms': inference_time
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Allow external connections
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Server will be accessible at: http://{local_ip}:5000")
    app.run(host=host, port=port, debug=True) 