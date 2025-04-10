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
import easyocr
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

# Initialize EasyOCR with optimized settings
logger.info("Initializing EasyOCR...")
try:
    ocr = easyocr.Reader(
        ['en'],
        gpu=True if device == 'cuda' else False,
        model_storage_directory='./model',
        download_enabled=True
    )
    logger.info("EasyOCR initialized successfully")
    # Test OCR with a simple image to verify it's working
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_results = ocr.readtext(test_image)
    logger.info("EasyOCR test completed successfully")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {str(e)}")
    raise

# Load YOLOv8 model with optimized settings
logger.info("Loading YOLOv8 model...")
try:
    model = YOLO('yolov8n.pt')  # Using nano model for speed

    # Optimize model for inference
    if device == 'cuda':
        try:
            model.fuse()
            model = model.to(device)
            # Enable FP16 for faster inference
            model = model.half()  # Convert model to FP16
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False  # Disable deterministic mode for speed
            logger.info("Model optimized with FP16 precision")
        except Exception as cuda_error:
            logger.warning(f"CUDA optimization failed: {str(cuda_error)}")
            logger.info("Falling back to CPU mode for YOLO...")
            device = 'cpu'
            model = model.to(device)

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
        
        # Resize image to a smaller size for faster processing
        target_size = 480  # Reduced from 640
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
    """Detect text in the image using EasyOCR"""
    try:
        # Log initial image properties
        logger.debug(f"Input image shape: {image.shape}, dtype: {image.dtype}, min: {np.min(image)}, max: {np.max(image)}")
        
        # Convert image to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            logger.debug("Converted grayscale to RGB")
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            logger.debug("Converted RGBA to RGB")
        
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            logger.debug("Converted image to uint8")
        
        # Save original image for debugging
        cv2.imwrite('debug_original.jpg', image)
        logger.debug("Saved original image for debugging")
        
        # Apply image preprocessing for better text detection
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.imwrite('debug_gray.jpg', gray)
        logger.debug("Saved grayscale image for debugging")
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        cv2.imwrite('debug_thresh.jpg', thresh)
        logger.debug("Saved thresholded image for debugging")
        
        # Apply dilation to connect text components
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        cv2.imwrite('debug_dilated.jpg', dilated)
        logger.debug("Saved dilated image for debugging")
        
        # Convert back to RGB for OCR
        processed_image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
        
        # Perform OCR with error handling and retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"OCR attempt {attempt + 1}")
                
                # Try with both original and processed image
                results = ocr.readtext(processed_image)
                logger.debug(f"Processed image OCR results: {results}")
                
                if not results:
                    # If no results with processed image, try original
                    results = ocr.readtext(image)
                    logger.debug(f"Original image OCR results: {results}")
                
                if results:
                    # Extract text from EasyOCR results
                    detected_texts = []
                    for detection in results:
                        text = str(detection[1])  # Get the recognized text
                        confidence = float(detection[2])  # Get the confidence score
                        logger.debug(f"Detected text: '{text}' with confidence: {confidence}")
                        if confidence > 0.3:  # Lower confidence threshold
                            detected_texts.append(text)
                    
                    if detected_texts:
                        logger.info("=== Detected Text ===")
                        for idx, text in enumerate(detected_texts, 1):
                            logger.info(f"Text {idx}: {text}")
                        logger.info("=" * 20)
                        return detected_texts
                    else:
                        logger.info("No high confidence text detected in the image")
                        return []
                logger.info("No text detected in the image")
                return []
            except Exception as ocr_error:
                if attempt == max_retries - 1:
                    logger.warning(f"OCR processing failed after {max_retries} attempts: {str(ocr_error)}")
                    return []
                logger.warning(f"OCR attempt {attempt + 1} failed, retrying...")
                time.sleep(0.1)  # Small delay before retry
        
        return []
    except Exception as e:
        logger.error(f"Error in detect_text: {str(e)}")
        return []

def get_google_search_url(object_class, text=""):
    """Generate a Google search URL for shopping based on object class and detected text"""
    # Create a simple query combining object class and text
    query = f"{object_class}"
    if text:
        query += f" {text}"
    
    # URL encode the query
    encoded_query = quote(query)
    
    # Return simple shopping search URL
    return f"https://www.google.com/search?q={encoded_query}&tbm=shop"

def get_shopping_results(search_url):
    """Fetch and parse shopping results from Google"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find shopping results
        results = []
        for item in soup.select('.sh-dgr__content, .sh-dgr__grid-result'):
            try:
                title = item.select_one('.tAxDx, .sh-dgr__title')
                price = item.select_one('.a8Pemb, .sh-dgr__price')
                link = item.select_one('a')
                
                if title and price:
                    result = {
                        'title': title.text.strip(),
                        'price': price.text.strip(),
                        'link': link['href'] if link else search_url
                    }
                    results.append(result)
            except Exception as e:
                logger.warning(f"Error parsing shopping result: {str(e)}")
                continue
        
        # If no results found in shopping section, try general search
        if not results:
            for item in soup.select('.g'):
                try:
                    title = item.select_one('h3')
                    price = item.select_one('.a8Pemb, .sh-dgr__price')
                    link = item.select_one('a')
                    
                    if title and (price or 'price' in title.text.lower()):
                        result = {
                            'title': title.text.strip(),
                            'price': price.text.strip() if price else 'Price not available',
                            'link': link['href'] if link else search_url
                        }
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Error parsing general result: {str(e)}")
                    continue
        
        return results[:10]  # Return top 10 results
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
        
        # Run YOLOv8 inference with optimized parameters
        logger.debug("Running YOLOv8 inference...")
        start_time = time.time()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast() if device == 'cuda' else nullcontext():
                results = model(
                    image_tensor,
                    conf=0.3,  # Increased confidence threshold
                    iou=0.5,  # Increased IOU threshold
                    agnostic_nms=True,
                    max_det=50,  # Reduced max detections
                    verbose=False  # Disable verbose output
                )
        
        inference_time = (time.time() - start_time) * 1000
        logger.debug(f"Inference completed in {inference_time:.2f}ms")
        
        # Get the original image for OCR
        original_image = cv2.imdecode(np.frombuffer(base64.b64decode(data['image']), np.uint8), cv2.IMREAD_COLOR)
        if original_image is None:
            logger.error("Failed to decode the input image")
            return jsonify({'error': 'Failed to decode the input image'}), 400
            
        # Process detections
        detections = []
        if results and results[0].boxes:
            # Get the detection with highest confidence
            boxes = results[0].boxes
            confidences = boxes.conf
            max_conf_idx = torch.argmax(confidences).item()
            
            # Get information for the highest confidence detection
            box = boxes[max_conf_idx]
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            confidence = float(box.conf[0])
            
            logger.info(f"\n=== Processing {class_name} (Confidence: {confidence:.2f}) ===")
            
            # Extract region of interest for text detection with padding
            # Get coordinates from YOLO (already in pixels)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Log raw coordinates for debugging
            logger.debug(f"Raw YOLO coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Get image dimensions
            height, width = original_image.shape[:2]
            logger.debug(f"Image dimensions: width={width}, height={height}")
            
            # Convert to integers and ensure they're within image bounds
            x1 = max(0, min(width - 1, int(x1)))
            y1 = max(0, min(height - 1, int(y1)))
            x2 = max(0, min(width - 1, int(x2)))
            y2 = max(0, min(height - 1, int(y2)))
            
            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1
            
            # Log converted coordinates for debugging
            logger.debug(f"Converted pixel coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Add padding to the ROI
            padding = 20  # Increased padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width - 1, x2 + padding)
            y2 = min(height - 1, y2 + padding)
            
            # Log final coordinates for debugging
            logger.debug(f"Final ROI coordinates with padding: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # Validate ROI coordinates
            if x2 <= x1 or y2 <= y1:
                logger.error(f"Invalid ROI coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                return jsonify({'error': 'Invalid object detection coordinates'}), 400
            
            # Extract and save the ROI for debugging
            try:
                roi = original_image[y1:y2, x1:x2]
                if roi.size == 0:
                    logger.error("Extracted ROI is empty")
                    return jsonify({'error': 'Failed to extract object region'}), 400
                
                # Draw rectangle on original image for debugging
                debug_image = original_image.copy()
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite('debug_roi.jpg', debug_image)
                cv2.imwrite('debug_cropped_roi.jpg', roi)
                
                logger.debug(f"ROI coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                logger.debug(f"ROI shape: {roi.shape}")
            except Exception as e:
                logger.error(f"Error extracting ROI: {str(e)}")
                return jsonify({'error': f'Error extracting ROI: {str(e)}'}), 400
            
            # Perform OCR on the entire image
            logger.info("Performing OCR on the entire image...")
            detected_texts = detect_text(original_image)
            detected_text = " ".join(detected_texts)
            if detected_text:
                logger.info(f"Detected text in image: {detected_text}")
            else:
                logger.info("No text detected in image")
            
            # Generate Google search URL
            search_url = get_google_search_url(class_name, detected_text)
            
            # Fetch shopping results
            shopping_results = get_shopping_results(search_url)
            
            # Log shopping results
            logger.info("\n=== Shopping Results ===")
            logger.info(f"Search Query: {class_name} {detected_text}")
            logger.info("Found Products:")
            for idx, item in enumerate(shopping_results, 1):
                logger.info(f"{idx}. {item['title']}")
                logger.info(f"   Price: {item['price']}")
                logger.info(f"   Link: {item['link']}")
                logger.info("   " + "-"*50)
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'text': detected_text,
                'detected_texts': detected_texts,
                'bbox': [x1, y1, x2, y2],
                'search_url': search_url,
                'shopping_results': shopping_results
            })
        
        # Get annotated image
        annotated_image = results[0].plot()
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'annotated_image': annotated_image_base64,
            'detections': detections,
            'inference_time_ms': inference_time,
            'success': True,
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'success': False,
            'message': 'Analysis failed'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Allow external connections
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=True) 