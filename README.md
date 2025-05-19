# neural-networks

<p></p>
This two-part project explores the benefits of building LSTM neural network models with optimization techniques.
<p></p>
<b>gyroscope.ipynb</b> is a multiclassification project analyzing biometric data in a time series and predicts human activity using hybrid neural network model construction. Application for wearable device sensors.
<p></p>
<b>social-media.ipynb</b> is a binary classification project predicting positivity/negativity of yelp reviews based on language content. Application for market analysis.  

---  
---  

## 📊 Tri-Axial Gyroscope Data Classification / Hybrid Neural Network Modeling
- gyroscope.ipynb
- data/UCI HAR Dataset.zip
- data/Inertial Signals.7z + data/subject_train.7z should decompress to 'UCI HAR Dataset/train'

**Purpose:** The purpose of this assignment is to test concepts on building & training LSTM models.  

**Data:** We were provided with a folder UCI HAR Dataset from the publicly-available data Human Activity Recognition Using Smartphones, available [here](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).  
The data consists of 6 activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING. Each activity is recorded for a series of 128 time points. During each time point, 9 features are measured elated to Triaxial acceleration from the accelerometer and Triaxial Angular velocity
from the gyroscope. 9 features at each time point are:  
- x, y, and z acceleration signals from smartphone accelerometer.  
- x, y, and z body acceleration signals.  
- x, y, and z angular velocities measured by the gyroscope.  
We collected the files according to their provided pre-divided train and test data sets as follows:  
- 'train/Inertial Signals/total_acc_x_train.txt': The acceleration signal from the smartphone accelerometer X axis in standard gravity units ‘g’. Every row shows a 128 element vector. The same description applies for the ‘total_acc_x_train.txt’ and ‘total_acc_z_train.txt’ files for the Y and Z axis.
- 'train/Inertial Signals/body_acc_x_train.txt': The body acceleration signal obtained by subtracting the gravity from the total acceleration.
- 'train/Inertial Signals/body_gyro_x_train.txt': The angular velocity vector measured by the gyroscope for each window sample. The units are radians/second.

**Approach:** We will build & train four (4) hybrid-architecture models:
- a single-layer LSTM Model  
- a dual-layer LSTM Model  
- a combination CNN-LSTM Model, and  
- a bidirectional LSTM Model  
Using early stopping criterion, we will save and report the accuracies of the best model.

**Results:**  
- a single-layer LSTM Model: Accuracy 91.01%, Loss 34.73%
- a dual-layer LSTM Model: Accuracy 91.11%, Loss 20.51%
- a combination CNN-LSTM Model: Accuracy 92.03%, Loss 26.34%
- a bidirectional LSTM Model: Accuracy 92.67%, Loss 21.55%
Best Model for probabilistic calibration :: dual-layer LSTM due to its lowest loss & competitive accuracy
Best Model for multi-class classification :: bidirectional LSTM Model due to its highest accuracy & competitive loss
The tie-breaker could be determined by analyzing distinguishing characteristics between classes & ranking importance.

---  

## ⭐ Social Media Sentiment Analysis / NLP + Hyperparameter Optimization
- social-media.ipynb
- data/yelp_review_polarity_csv.zip
- use the [link](https://nlp.stanford.edu/projects/glove/) in the .ipynb file to download 'glove.6B' for NLP tokenization

**Purpose:** The purpose of this section is to test concepts of hyperparameter optimization.  

**Data:** We were provided with a folder _yelp_review_polarity_csv_, consisting of files
_train_small.csv_ and _test_small.csv_ containing pre-split training/testing groups of data. First
column is class label (“1” and “2” in this example), second column is text. Note that entries are
within double quotes (“). These files will be loaded using python package ‘pandas’ and stored in
pandas dataframes (like ‘data frame’ in R). Each column in pandas data frame is a dictionary,
column name being the key. Additionally, we were provided with a folder _glove.6B_ containing a
tokenizer dictionary _glove.6B.100d.txt_, which is publicly-available [here](https://nlp.stanford.edu/projects/glove/). The dictionary includes
a vocabulary of 400k words in 100 dimensions.  

**Approach:** We will perform hyperparameter optimization using Python’s *HyperOpt* library on a dual-layer LSTM Model, optimizing at least 2 tuning parameters.

**Results**: Our best model has an accuracy of 85.75% and a loss of 30.70%. Its max_eval = 10 and its parameters are:  
• ‘activation_function’: [‘tanh’]  
• ‘batch_size’: [112.0]  
• ‘dropout’: [0.3274283406727577]  
• ‘epochs’: [54.0]  
• ‘kernel_size’: [3.0]  
• ‘learning_rate’: [0.013220910276634638]  
• ‘num_kernel’: [80.0]  
• ‘optimizer’: [‘adam’]  
• ‘patience’: [4.0]  
• ‘size_pooling’: [2.0]  
• ‘strides’: [2.0]}  

