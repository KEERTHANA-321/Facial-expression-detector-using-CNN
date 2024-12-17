Facial Expression Recognition Using CNN
web app link:https://facial-expression-detector-using-cnn-ug2drjaobs7zvfneiiaytg.streamlit.app/

Problem Statement
The objective of this project is to develop a deep learning model that can classify facial expressions into 
different categories of emotions such as happy,sad,angry,fear etc.using images.
This could help in improving feedback mechanisms in campus
activities


Approach
1.Data Preprocessing
-Dataset:https://www.kaggle.com/datasets/msambare/fer2013
Used the FER 2013 dataset consisting of grayscale facial images resized to 48x48 pixels.
-Applied rescaling to normalize pixel values between 0 and 1.

2.Model Architecture
-Convolutional Layers: 
  - Used multiple Conv2D layers for feature extraction.
  - ReLU activation for non-linearity.
  - Maxpooling to reduce dimensions and overfitting.
- Fully Connected Layers:
  - Flattened extracted features into a dense representation.
  - Added Dropout to reduce overfitting.
  - Output layer  to classify into 7 emotion categories.
  
3.Training
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
https://github.com/KEERTHANA-321/Facial-expression-detector-using-CNN/blob/main/confusionmatrix.png



3.Observations
- The model performed well on classes like sad and happy but struggles with some features like disgust and fear.
- Misclassifications often occurred in images with poor lighting.

Challenges
1.Overfitting
  Model works almost accurately with the test data but fails often for new data
 Introduced Dropout layers and reduced the learning rate to improve generalization.
