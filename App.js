import React, { useRef, useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ActivityIndicator, Image, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import axios from 'axios';
import { StatusBar } from 'expo-status-bar';

// Update this to your computer's IP address
const SERVER_URL = 'http://10.65.243.154:5000'; // Replace with your actual IP address

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [objectDescription, setObjectDescription] = useState('');
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [facing, setFacing] = useState('back');

  if (!permission) {
    return null;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>We need your permission to use the camera</Text>
        <TouchableOpacity 
          style={styles.button} 
          onPress={requestPermission}
        >
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const analyzeImage = async () => {
    if (!cameraRef.current) return;
    
    setIsAnalyzing(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.5,
        base64: true,
        exif: true,
        skipProcessing: true,
      });

      console.log('Sending image to server...');
      const response = await axios.post(`${SERVER_URL}/analyze`, {
        image: photo.base64,
        orientation: photo.exif?.Orientation || 1,
      });

      console.log('Received response:', response.data);
      setObjectDescription(response.data.description);
      setAnnotatedImage(`data:image/jpeg;base64,${response.data.annotated_image}`);
    } catch (error) {
      console.error('Error analyzing image:', error);
      Alert.alert(
        'Error',
        'Failed to analyze image. Please check your connection to the server.',
        [
          {
            text: 'OK',
            onPress: () => console.log('OK Pressed'),
          },
        ]
      );
      setObjectDescription('Error analyzing image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const toggleFacing = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  return (
    <View style={styles.container}>
      <StatusBar style="auto" />
      {annotatedImage ? (
        <View style={styles.resultContainer}>
          <View style={styles.imageWrapper}>
            <Image 
              source={{ uri: annotatedImage }} 
              style={styles.annotatedImage}
            />
            <View style={styles.overlayContainer}>
              <View style={styles.bottomOverlay}>
                <TouchableOpacity
                  style={styles.backButton}
                  onPress={() => setAnnotatedImage(null)}
                >
                  <Text style={styles.backButtonText}>Back to Camera</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </View>
      ) : (
        <CameraView 
          style={styles.camera} 
          ref={cameraRef}
          facing={facing}
        >
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={styles.button}
              onPress={toggleFacing}
            >
              <Text style={styles.buttonText}>Flip</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.button}
              onPress={analyzeImage}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <ActivityIndicator color="white" />
              ) : (
                <Text style={styles.buttonText}>Analyze</Text>
              )}
            </TouchableOpacity>
          </View>
        </CameraView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  resultContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  imageWrapper: {
    flex: 1,
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  annotatedImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  overlayContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'flex-end',
  },
  bottomOverlay: {
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 15,
    marginBottom: 40,
    marginHorizontal: 20,
    borderRadius: 10,
    alignItems: 'center',
  },
  backButton: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    padding: 12,
    borderRadius: 8,
    minWidth: 150,
    alignItems: 'center',
  },
  backButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  camera: {
    flex: 1,
    width: '100%',
  },
  buttonContainer: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 20,
  },
  button: {
    backgroundColor: 'rgba(0,0,0,0.8)',
    padding: 15,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  text: {
    fontSize: 18,
    marginBottom: 20,
    textAlign: 'center',
    color: 'white',
  },
}); 