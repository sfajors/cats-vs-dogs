# Cats vs. Dogs classifier
## Overview
This notebook is designed to classify images into two categories: cats and dogs. It demonstrates how to preprocess image data, design a a convolutional neural network (CNN), and evaluate its perfromance using TensorFlow and Keras
## How it Works
1. **Data Preparation:** The notebook starts by unzipping the dataset sourced from Kaggle's "Dogs vs. Cats" competition. The dataset contains labeled images of cats and dogs which are used for training and testing the model.
2. **Data Exploration:** Various techniques are used to visualize and understand the distribution and structure of the images.
3. **Data Preprocessing:** The images are resized and normalized to be fed into the neural network.
4. **Model Building:** A CNN is built using TensorFlow and Keras. The CNN contains multiple layers, including convolutional layers, pooling layers, and dense layers.
5. **Training:** The model is trained on the dataset using appropriate optimizers and loss functions.
6. **Evaluation:** After training, the model's performance is evaluated using various metrics such as accuracy, precision, recall, etc. Confusion matrices and other visualizations might also be used for a deeper understanding of the model's performance
## Requirements 
```
  PIL
  cv2
  itertools
  matplotlib
  numpy
  os
  pandas
  pathlib
  seaborn
  shutil
  sklearn
  tensorflow
  warnings
```
