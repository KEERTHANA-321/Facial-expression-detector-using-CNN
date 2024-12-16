Facial Expression Recognition Using CNN

Problem Statement
The objective of this project is to develop a deep learning model that can classify facial expressions into 
different categories of emotions such as happy,sad,angry,fear etc.using images.
This could help in improving feedback mechanisms in campus
activities


Approach
1.Data Preprocessing
-Dataset:Used the FER 2013 dataset consisting of grayscale facial images resized to 48x48 pixels.
-Data Augmentation:Applied rescaling to normalize pixel values between 0 and 1.

2.Model Architecture
-Convolutional Layers: 
  - Used multiple Conv2D layers for feature extraction.
  - Employed ReLU activation for non-linearity.
  - Incorporated MaxPooling to downsample feature maps and reduce dimensionality.
- Fully Connected Layers:
  - Flattened extracted features into a dense representation.
  - Added Dropout to reduce overfitting.
  - Output layer with a `softmax` activation to classify into 7 emotion categories.
  
3.Training
- Loss Function:Categorical Crossentropy.
- Optimizer:Adam Optimizer with a learning rate of 0.0001.
- Evaluation Metric:Accuracy and f1 scores.
- Training Epochs:Trained for 50 epochs on the FER 2013 dataset.

4.Performance Evaluation
- Computed metrics like Accuracy and F1-score.
- Visualized confusion matrix to assess class-wise performance.


Results
- Validation Accuracy:Achieved 0.67
- F1-Score:0.60

Visualizations
Confusion Matrix:




3.Observations
- The model performed well on classes like sad and Neutral but struggles with some features.
- Misclassifications often occurred in images with poor lighting.

Challenges
1.Overfitting
  Model works almost accurately with the test data but fails often for new data
 Introduced Dropout layers and reduced the learning rate to improve generalization.
