# neural-networks
<p></p>
Part A is a multiclassification project analyzing biometric data in a time series and predicts human activity using hybrid neural network model construction.
<p></p>
Part B is a binary classification project predicting positivity/negativity of yelp reviews based on language content.
<p></p>
Output file can be viewed <a href="https://www.dropbox.com/scl/fi/cogcngc7udpo6ugxe2gw9/neural-networks.html?rlkey=17na23lj8d8iee8xf7h4401k5&dl=0">here</a>.<br>
<p></p>
<center><line></line></center>
<p></p>
<center><h1><b>Part A</b></h1></center>
<p></p>
<h3>Purpose:</h3> The purpose of this assignment is to test concepts on building & training LSTM models.
</p><p>
<h3>Data:</h3> We were provided with a folder UCI HAR Dataset from the publicly-available data Human Activity Recognition Using Smartphones, available here. The data consists of 6 activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING. Each activity is recorded for a series of 128 time points. During each time point, 9 features are measured elated to Triaxial acceleration from the accelerometer and Triaxial Angular velocity from the gyroscope.
</p><p>
9 features at each time point are:
</p><p>
- x, y, and z acceleration signals from smartphone accelerometer.<br>
- x, y, and z body acceleration signals.<br>
- x, y, and z angular velocities measured by the gyroscope.<br>
We collected the files according to their provided pre-divided train and test data sets as follows:<br>
<p></p>
'train/Inertial Signals/total_acc_x_train.txt': The acceleration signal from the smartphone accelerometer X axis in standard gravity units 'g'. Every row shows a 128 element vector. The same description applies for the 'total_acc_x_train.txt' and 'total_acc_z_train.txt' files for the Y and Z axis.
<p></p>
'train/Inertial Signals/body_acc_x_train.txt': The body acceleration signal obtained by subtracting the gravity from the total acceleration.
<p></p>
'train/Inertial Signals/body_gyro_x_train.txt': The angular velocity vector measured by the gyroscope for each window sample. The units are radians/second.
<p></p>
<h3>Approach:</h3> We will build & train four (4) models matching the architecture we were provided in Dr. Soibam's TextClassification_LSTM.ipynb Jupyter Notebook. They will consist of the following:
<p></p>
- a single-layer LSTM Model<br>
- a dual-layer LSTM Model<br>
- a combination CNN-LSTM Model, and<br>
- a bidirectional LSTM Model<br>
Using early stopping criterion, we will save and report the accuracies of the best model.<br>
<p></p>
<center><line></line></center>
<p></p>
<center><h1><b>Part B</b></h1></center>
<p></p>
<h3>Purpose:</h3> The purpose of this section is to test concepts of hyperparameter optimization.
<p></p>
<h3>Data:</h3> We were provided with a folder yelp_review_polarity_csv, consisting of files
train_small.csv and test_small.csv containing pre-split training/testing groups of data. First
column is class label (“1” and “2” in this example), second column is text. Note that entries are
within double quotes (“). These files will be loaded using python package ‘pandas’ and stored in
pandas dataframes (like ‘data frame’ in R). Each column in pandas data frame is a dictionary,
column name being the key. Additionally, we were provided with a folder glove.6B containing a
tokenizer dictionary glove.6B.100d.txt, which is publicly-available here. The dictionary includes
a vocabulary of 400k words in 100 dimensions.
<p></p>
<h3>Approach:</h3> We will perform hyperparameter optimization using Python’s HyperOpt library on a
dual-layer LSTM Model, optimizing at least 2 tuning parameters.
<p></p>
<center><line></line></center>
<p></p>
