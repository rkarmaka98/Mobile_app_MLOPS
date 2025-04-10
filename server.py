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
import paddleocr
import requests
from urllib.parse import quote
from contextlib import nullcontext
from bs4 import BeautifulSoup
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# Enable mixed precision for PyTorch
if device == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("TF32 enabled for better performance")

app = Flask(__name__)
# Configure CORS to allow all origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Load environment variables
load_dotenv()

# Initialize PaddleOCR with CUDA support
logger.info("Initializing PaddleOCR...")
try:
    ocr = paddleocr.PaddleOCR(
        use_angle_cls=False,  # Disable angle classification for speed
        lang='en',
        use_gpu=device=='cuda',  # Enable GPU if available
        show_log=False,
        enable_mkldnn=True,  # Enable MKL-DNN acceleration
        cpu_threads=4  # Optimize CPU threads if needed
    )
    logger.info("PaddleOCR initialized successfully with GPU support")
except Exception as e:
    logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
    raise

# Load YOLOv8 model with CUDA and FP16
logger.info("Loading YOLOv8 model...")
try:
    model = YOLO('yolov8n.pt')  # Using nano model for speed

    # Optimize model for inference
    if device == 'cuda':
        model.fuse()
        model = model.to(device)
        # Enable FP16 for faster inference
        model = model.half()  # Convert model to FP16
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logger.info("Model optimized with FP16 precision")

    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {str(e)}")
    raise

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
        new_size = (new_size[0] - (new_size[0] % 32), new_size[1] - (new_size[1] % 32))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        
        # Convert to BCHW format
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, 0)
        
        logger.debug(f"Processed image shape: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def detect_text(image):
    """Detect text in the image using PaddleOCR"""
    try:
        # Convert image to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Perform OCR
        results = ocr.ocr(image, cls=False)  # Disable classification for speed
        if results and results[0]:
            return [text[1][0] for text in results[0]]  # Extract text only
        return []
    except Exception as e:
        logger.error(f"Error in detect_text: {str(e)}")
        return []

def get_google_search_url(object_class, text=""):
    """Generate a Google search URL for shopping based on object class and detected text"""
    # Create a shopping-specific query
    query = f"{object_class}"
    if text:
        query += f" {text}"
    
    # Add shopping-related terms
    query += " shopping list category products"
    
    # Add site-specific search for major shopping platforms
    site_filters = " site:amazon.com OR site:walmart.com OR site:target.com OR site:bestbuy.com"
    query += site_filters
    
    # URL encode the query
    encoded_query = quote(query)
    
    # Add shopping-specific parameters
    return f"https://www.google.com/search?q={encoded_query}&tbm=shop"

def get_shopping_results(search_url):
    """Fetch and parse shopping results from Google"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find shopping results
        results = []
        for item in soup.select('.sh-dgr__content'):
            title = item.select_one('.tAxDx')
            price = item.select_one('.a8Pemb')
            store = item.select_one('.aULzUe')
            
            if title and price:
                result = {
                    'title': title.text.strip(),
                    'price': price.text.strip(),
                    'store': store.text.strip() if store else 'Unknown Store'
                }
                results.append(result)
        
        return results
    except Exception as e:
        logger.error(f"Error fetching shopping results: {str(e)}")
        return []

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        logger.info("Received analyze request")
        data = request.json
        if 'image' not in data:
            logger.error("No image provided in request")
            return jsonify({'error': 'No image provided'}), 400

        # Get orientation from request
        orientation = data.get('orientation', 1)
        
        # Preprocess the image
        image = preprocess_image(data['image'], orientation)
        
        # Convert to tensor and move to device with FP16 if CUDA is available
        image_tensor = torch.from_numpy(image)
        if device == 'cuda':
            image_tensor = image_tensor.to(device).half()  # Convert to FP16
        else:
            image_tensor = image_tensor.to(device)
        
        # Run YOLOv8 inference
        logger.debug("Running YOLOv8 inference...")
        start_time = time.time()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast() if device == 'cuda' else nullcontext():
                results = model(
                    image_tensor,
                    conf=0.25,
                    iou=0.45,
                    agnostic_nms=True,
                    max_det=100,
                )
        
        inference_time = (time.time() - start_time) * 1000
        logger.debug(f"Inference completed in {inference_time:.2f}ms")
        
        # Get the original image for OCR
        original_image = cv2.imdecode(np.frombuffer(base64.b64decode(data['image']), np.uint8), cv2.IMREAD_COLOR)
        
        # Process each detected object
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get object information
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                
                # Extract region of interest for text detection
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                roi = original_image[int(y1):int(y2), int(x1):int(x2)]
                
                # Detect text in the region
                detected_texts = detect_text(roi)
                detected_text = " ".join(detected_texts)
                
                # Generate Google search URL
                search_url = get_google_search_url(class_name, detected_text)
                
                # Fetch and display shopping results
                shopping_results = get_shopping_results(search_url)
                logger.info("\n=== Shopping Results ===")
                logger.info(f"Search Query: {class_name} {detected_text}")
                logger.info("Found Products:")
                for idx, item in enumerate(shopping_results[:5], 1):
                    logger.info(f"{idx}. {item['title']}")
                    logger.info(f"   Price: {item['price']}")
                    logger.info(f"   Store: {item['store']}")
                    logger.info("   " + "-"*50)
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'text': detected_text,
                    'search_url': search_url,
                    'shopping_results': shopping_results[:5]  # Include top 5 results in response
                })
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'annotated_image': annotated_image_base64,
            'detections': detections,
            'inference_time_ms': inference_time
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Allow external connections
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=True) 