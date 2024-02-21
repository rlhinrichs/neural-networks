# neural-networks
Part A is a multiclassification project analyzing biometric data in a time series and predicts human activity using hybrid neural network model construction. <br>
Part B is a binary classification project predicting positivity/negativity of yelp reviews based on language content.<br>
Both projects require vectorizing the data into numerical representation using NumPy.
<p><center><b>Part A</b></center>
Purpose: The purpose of this assignment is to test concepts on building & training LSTM models.
</p><p>
Data: We were provided with a folder UCI HAR Dataset from the publicly-available data Human Activity Recognition Using Smartphones, available here. The data consists of 6 activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING. Each activity is recorded for a series of 128 time points. During each time point, 9 features are measured elated to Triaxial acceleration from the accelerometer and Triaxial Angular velocity from the gyroscope.
</p><p>
9 features at each time point are:
</p><p>
- x, y, and z acceleration signals from smartphone accelerometer.
- x, y, and z body acceleration signals.
- x, y, and z angular velocities measured by the gyroscope.
<br>We collected the files according to their provided pre-divided train and test data sets as follows:
<p>
'train/Inertial Signals/total_acc_x_train.txt': The acceleration signal from the smartphone accelerometer X axis in standard gravity units 'g'. Every row shows a 128 element vector. The same description applies for the 'total_acc_x_train.txt' and 'total_acc_z_train.txt' files for the Y and Z axis.
</p><p>
'train/Inertial Signals/body_acc_x_train.txt': The body acceleration signal obtained by subtracting the gravity from the total acceleration.
</p><p>
'train/Inertial Signals/body_gyro_x_train.txt': The angular velocity vector measured by the gyroscope for each window sample. The units are radians/second.
</p><p>
Approach: We will build & train four (4) models matching the architecture we were provided in Dr. Soibam's TextClassification_LSTM.ipynb Jupyter Notebook. They will consist of the following:
</p><p>
- a single-layer LSTM Model
- a dual-layer LSTM Model
- a combination CNN-LSTM Model, and
- a bidirectional LSTM Model
<br>Using early stopping criterion, we will save and report the accuracies of the best model.
<p><center><b>Part B</b></center>
