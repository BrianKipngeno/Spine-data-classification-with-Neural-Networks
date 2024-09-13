# Spine-data-classification-with-Neural-Networks

This project aims to classify a person's spine condition as either Normal or Abnormal based on various biomechanical features extracted from the dataset. Using a neural network model, we predict the condition of a spine and apply this model to classify observations from random spine data.

**Dataset**

The dataset used for this project can be found  http://bit.ly/spine_data. It includes features such as pelvic tilt, sacral slope, and lumbar lordosis angle, which are critical for predicting spinal abnormalities.

**Project Overview**

The project involves the following steps:

- Data Preparation: Preprocessing the dataset, including handling missing values, encoding target labels, and standardizing features.
- Building the Model: Developing a neural network model with PyTorch to classify spine conditions.
- Model Training: Training the model with training data using cross-entropy loss and stochastic gradient descent (SGD).
- Model Evaluation: Testing the model's accuracy and applying it to new data for predictions.
- Prediction: Using the trained model to classify a new observation of spine data.
  
**Dataset Information**

- Number of features: 12
- Classes: Normal (1), Abnormal (0)

  
**Steps Involved**

**1. Data Preparation**

The data preparation step includes:

- Loading the dataset from a CSV file.
- Dropping any unnecessary columns (e.g., unnamed columns).
- Encoding the target variable, where Abnormal is mapped to 0 and Normal to 1.
- Splitting the data into training and testing sets (70% training, 30% testing).
- Standardizing the features to ensure consistent input values for the neural network.
  
**2. Model Architecture**

The neural network model is built using the following structure:

- Input Layer: 12 input features.
- Hidden Layer: One hidden layer with 24 neurons and ReLU activation.
- Output Layer: A binary output for classification (0 = Abnormal, 1 = Normal).
  
**3. Model Training**

The model is trained using the cross-entropy loss function and stochastic gradient descent (SGD) optimizer with a learning rate of 0.01 and momentum of 0.9. The training process runs for 1000 epochs to ensure optimal performance.

**4. Model Evaluation**

The model is evaluated on the test dataset, and the accuracy is computed. The trained model achieves an accuracy of 81.72% on the test data, indicating reliable performance in classifying spinal conditions.

**5. Prediction**

The trained model is also applied to a new observation with the following features
[56.12492019, 23.64048856, 38.32974793, 26.47443163, 121.456011, 1.653204816, 
0.85687869, 13.6686, 12.57, 17.12951, -16.630363, 27.1902]

The model predicts this new data point as Abnormal.

**Results**

- Final Model Accuracy: 81.72%
- Predicted Condition for New Data: Abnormal
  

**Installation**

To replicate this project, the following Python libraries are required:

- NumPy
- Pandas
- PyTorch
- Scikit-learn
- Seaborn
- Matplotlib
These can be installed using pip or any other Python package manager.

**Conclusion**

This project demonstrates the use of PyTorch to develop a neural network for classifying spine conditions. The model performs well with an accuracy of 81.72% on the test data. Future improvements can include trying different architectures or hyperparameter tuning to further enhance accuracy.

**License**

This project is licensed under the MIT License.
