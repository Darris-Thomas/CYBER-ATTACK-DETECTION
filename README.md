# CYBER ATTACK DETECTION USING CNN-LSTM MODEL
## OVERVIEW
This project implements a cyber attack detection model using Convolutional Neural Networks (CNN) - Long Short-Term Memory (LSTM). The main objective is to enhance the accuracy and precision of detection compared to traditional methods. This model demonstrates outstanding performance with an accuracy rate of 98%.
## METHODOLOGY
The CNN-LSTM model for cyberattack detection is divided into seven modules.
- 1. Pre-processing: This module deals with removing null rows, columns and duplicate values. The dataset is splitted into training and testing.
- 2. Data balancing: SMOTE is a machine learning technique used to balance the imbalanced dataset. The balanced dataset is then saved for further processing.
- 3. Feature selection: chi2 is algorithm used to select the best features from the dataset. Then the features and labels are splitted into training and testing dataset.
- 4. Training Deep Belief Network (DBN): PEO algorithm is used for DBN optimization. The model is trained using training set features and labels. The DBN model is then saved and accuracy is calculated. 
- 5. Prediction using DBN Model: The trained DBN model and test data is loaded for prediction using DBN model. The prediction is done and output is displayed.
- 6. Training CNN -LSTM Model: CNN-LSTM model is initialized and training parameter are established. The model is then trained using training set features and labels. The trained DBN model is saved for finding accuracy.
- 7. Prediction Using CNN-LSTM Model: The trained CNN-LSTM model and test data are loaded for prediction using CNN-LSTM model. The prediction is done and and output is displayed.
## SOFTWARE SPECIFICATION
- Tool: Python IDLE
- Python: version3
- Operating System: Windows 7 or later
- Front End: Python
