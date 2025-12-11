import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

class EyeStateClassifier:
    def __init__(self, model_path=None):
        """
        CNN model for classifying eye state (open/closed)
        Input: 64x64 grayscale eye image
        Output: Binary classification (0=closed, 1=open)
        """
        self.img_size = (64, 64)
        self.model = None
        
        if model_path:
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build CNN architecture"""
        model = models.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(64, 64, 1), padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Fourth Conv Block (optional for better accuracy)
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(),
                    keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def get_model_summary(self):
        """Print model architecture"""
        if self.model:
            return self.model.summary()
        return "Model not built yet"
    
    def preprocess_eye_image(self, eye_image):
        """
        Preprocess eye image for model input
        Args:
            eye_image: BGR image of eye region
        Returns:
            Preprocessed image ready for prediction
        """
        # Convert to grayscale if needed
        if len(eye_image.shape) == 3:
            gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_image
        
        # Resize to model input size
        resized = cv2.resize(gray, self.img_size)
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        # Reshape for model input (add batch and channel dimensions)
        processed = normalized.reshape(1, 64, 64, 1)
        
        return processed
    
    def predict(self, eye_image):
        """
        Predict eye state
        Args:
            eye_image: Eye region image
        Returns:
            tuple: (prediction_score, is_open)
                   prediction_score: float between 0 and 1
                   is_open: boolean (True if eye is open)
        """
        if self.model is None:
            raise ValueError("Model not loaded or built")
        
        # Preprocess image
        processed = self.preprocess_eye_image(eye_image)
        
        # Get prediction
        prediction = self.model.predict(processed, verbose=0)[0][0]
        
        # Threshold at 0.5 (can be adjusted)
        is_open = prediction > 0.5
        
        return float(prediction), bool(is_open)
    
    def train(self, train_dir, val_dir, epochs=30, batch_size=32):
        """
        Train the model
        Args:
            train_dir: Directory with training data (should have 'open' and 'closed' subdirs)
            val_dir: Directory with validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            color_mode='grayscale'
        )
        
        # Load validation data
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            color_mode='grayscale'
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_eye_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        return history
    
    def save_model(self, filepath='eye_state_model.h5'):
        """Save trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save")
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")

class HybridDrowsinessDecision:
    """
    Hybrid decision engine combining EAR, MAR, and CNN predictions
    """
    def __init__(self, cnn_model_path=None):
        self.cnn_classifier = None
        if cnn_model_path:
            self.cnn_classifier = EyeStateClassifier(cnn_model_path)
        
        # Thresholds
        self.EAR_THRESH = 0.21
        self.CNN_THRESH = 0.5
        self.HYBRID_WEIGHT_EAR = 0.4
        self.HYBRID_WEIGHT_CNN = 0.6
        
    def make_decision(self, ear, left_eye_crop, right_eye_crop):
        """
        Make drowsiness decision using hybrid approach
        Args:
            ear: Eye Aspect Ratio
            left_eye_crop: Left eye image
            right_eye_crop: Right eye image
        Returns:
            dict with decision details
        """
        decision = {
            'drowsy': False,
            'confidence': 0.0,
            'ear_drowsy': False,
            'cnn_drowsy': False,
            'method': 'ear_only'
        }
        
        # EAR-based decision
        ear_drowsy = ear < self.EAR_THRESH
        decision['ear_drowsy'] = ear_drowsy
        
        # CNN-based decision (if model available)
        if self.cnn_classifier and left_eye_crop is not None and right_eye_crop is not None:
            try:
                left_score, left_open = self.cnn_classifier.predict(left_eye_crop)
                right_score, right_open = self.cnn_classifier.predict(right_eye_crop)
                
                # Average of both eyes
                avg_cnn_score = (left_score + right_score) / 2.0
                cnn_drowsy = not (left_open and right_open)
                
                decision['cnn_drowsy'] = cnn_drowsy
                decision['cnn_score'] = avg_cnn_score
                
                # Hybrid decision: weighted combination
                # Lower EAR = more drowsy, Higher CNN score = more open
                ear_normalized = 1 - (ear / 0.3)  # Normalize EAR (assuming max ~0.3)
                cnn_normalized = 1 - avg_cnn_score  # Invert (lower = more closed)
                
                hybrid_score = (self.HYBRID_WEIGHT_EAR * ear_normalized + 
                              self.HYBRID_WEIGHT_CNN * cnn_normalized)
                
                decision['drowsy'] = hybrid_score > 0.5
                decision['confidence'] = hybrid_score
                decision['method'] = 'hybrid'
                
            except Exception as e:
                print(f"CNN prediction error: {e}")
                decision['drowsy'] = ear_drowsy
                decision['confidence'] = 1.0 if ear_drowsy else 0.0
                decision['method'] = 'ear_fallback'
        else:
            # Fallback to EAR only
            decision['drowsy'] = ear_drowsy
            decision['confidence'] = 1.0 if ear_drowsy else 0.0
            decision['method'] = 'ear_only'
        
        return decision

# Example usage
if __name__ == "__main__":
    # Build and inspect model
    classifier = EyeStateClassifier()
    print("Model Architecture:")
    classifier.get_model_summary()
    
    # Example: Train model (uncomment when you have dataset)
    # classifier.train(
    #     train_dir='data/train',
    #     val_dir='data/val',
    #     epochs=30
    # )
    # classifier.save_model('models/eye_state_cnn.h5')
    
    # Example: Load and use trained model
    # classifier = EyeStateClassifier('models/eye_state_cnn.h5')
    # eye_image = cv2.imread('test_eye.jpg')
    # score, is_open = classifier.predict(eye_image)
    # print(f"Eye Open Probability: {score:.2f}, Is Open: {is_open}")