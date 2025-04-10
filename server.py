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

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

# Load YOLOv8 model
logger.info("Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8n model
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
        
        # Convert to numpy array
        image = np.array(image)
        logger.debug(f"Image shape: {image.shape}")
        
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
        
        # Run YOLOv8 inference
        logger.debug("Running YOLOv8 inference...")
        results = model(image)
        logger.debug("Inference completed")
        
        # Get detailed description of detected objects
        description = get_detailed_description(results)
        logger.debug(f"Generated description: {description}")
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
        logger.debug("Image processing completed")
        
        return jsonify({
            'description': description,
            'annotated_image': annotated_image_base64
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