# Object Detection App

A React Native Expo app that uses the device's camera to detect objects and provide descriptions.

## Features

- Camera access for capturing images
- Object detection and description
- Real-time analysis
- Clean and intuitive UI

## Prerequisites

- Node.js (v14 or newer)
- npm or yarn
- Expo CLI (`npm install -g expo-cli`)
- iOS Simulator (for iOS development)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

4. Press 'i' to open in iOS simulator or scan the QR code with your iOS device using the Expo Go app.

## Configuration

Before running the app, you need to:

1. Replace `YOUR_SERVER_ENDPOINT` in `App.js` with your actual server endpoint that handles:
   - Receiving the image
   - Processing it for object detection
   - Returning a description

## Server Requirements

Your server should:
- Accept POST requests with base64-encoded images
- Process the image using an object detection model
- Return a JSON response with a `description` field containing the object description

Example server response:
```json
{
  "description": "This is a description of the detected object..."
}
```

## Troubleshooting

- If you encounter camera permission issues, make sure to:
  - Grant camera permissions in your device settings
  - Check that your device supports the required features
- For iOS development, ensure you have the latest Xcode installed 