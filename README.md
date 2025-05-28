# neural-networks

This two-part project explores the capabilities of LSTM-based neural networks and examines how different architectural and optimization strategies impact model performance.

## 📊 Tri-Axial Gyroscope Data Classification // Hybrid Neural Network Modeling

* `gyroscope.ipynb`
* `data/UCI HAR Dataset.zip`
* `data/Inertial Signals.7z` + `data/subject_train.7z` should decompress to `UCI HAR Dataset/train`

**Overview:**
This study investigates a multiclass classification problem using biometric time-series data. The task is to predict human physical activity based on signals from wearable device sensors. The goal was to gain a better understanding of how various LSTM and hybrid architectures perform on sensor data.

**Dataset:**
We used the UCI HAR Dataset, which records six human activities (WALKING, WALKING\_UPSTAIRS, WALKING\_DOWNSTAIRS, SITTING, STANDING, LAYING) across 128 time steps using 9 features. These features represent:

* Raw accelerometer data (x, y, z)
* Body-acceleration (x, y, z)
* Angular velocity from the gyroscope (x, y, z)
  More about the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).

**Approach:**
To compare learning patterns, we implemented and evaluated four different architectures:

* Single-layer LSTM
* Dual-layer LSTM
* CNN-LSTM hybrid
* Bidirectional LSTM

We used early stopping and validation accuracy to select the best-performing models.

**Results:**

* Single-layer LSTM: 91.01% accuracy, 34.73% loss
* Dual-layer LSTM: 91.11% accuracy, 20.51% loss
* CNN-LSTM: 92.03% accuracy, 26.34% loss
* Bidirectional LSTM: 92.67% accuracy, 21.55% loss

**Reflection:**

* The dual-layer LSTM showed strong probabilistic calibration.
* The bidirectional LSTM was most effective for classification.
* Future iterations could examine class-specific confusion matrices and feature importance to fine-tune architecture selection.

---

## ⭐ Social Media Sentiment Analysis // NLP + Hyperparameter Optimization

* `social-media.ipynb`
* `data/yelp_review_polarity_csv.zip`
* Download `glove.6B` embeddings from the [official source](https://nlp.stanford.edu/projects/glove/)

**Overview:**
This project applies natural language processing to classify Yelp reviews as positive or negative. The emphasis was on experimenting with hyperparameter tuning to improve performance.

**Dataset:**
We used pre-split training and testing sets (`train_small.csv`, `test_small.csv`) from the Yelp Review Polarity dataset. Each record includes a sentiment label and a text review. Word embeddings were sourced from GloVe’s `glove.6B.100d.txt` (100-dimensional vectors, 400k vocabulary).

**Approach:**
Using a dual-layer LSTM model, we applied the HyperOpt library to tune a set of hyperparameters, including learning rate, batch size, dropout rate, and architecture-specific options like kernel size and optimizer type.

**Results:**
Our best model achieved:

* Accuracy: **85.75%**
* Loss: **30.70%**

Tuned parameters included:

```python
{
  'activation_function': 'tanh',
  'batch_size': 112,
  'dropout': 0.327,
  'epochs': 54,
  'kernel_size': 3,
  'learning_rate': 0.0132,
  'num_kernel': 80,
  'optimizer': 'adam',
  'patience': 4,
  'size_pooling': 2,
  'strides': 2
}
```

**Reflection:**
This project helped me understand the importance of tuning even minor parameters and how these changes affect convergence and generalization. Future directions include using attention layers or experimenting with transformer-based architectures.  

---  

© Rebecca Leigh Hinrichs. All Rights Reserved.
