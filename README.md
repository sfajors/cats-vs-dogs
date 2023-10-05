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
Pillow==8.1.2
opencv-python==4.5.1.48
matplotlib==3.3.4
numpy==1.19.5
pandas==1.2.3
seaborn==0.11.1
scikit-learn==0.24.1
tensorflow==2.4.1

```
# Using the Cats vs. Dogs Classifier for Predictions:
## 1. Set Up:
1. Clone the repository
  ``` bash
    git clone https://github.com/sfajors/cats-vs-dogs.git
  ```
2. Navigatte to the Project Directory
   ``` bash
     cd path_to_cloned_repo
3. Install the Necessary Dependencies
   ```
    pip install -r requirements.txt
## 2. Open the Notebook:
```
  jupyter notebook
```
## 3. Run Initialization Cells 
Run the cells at the beginning of the notebook that contain import statements and any other initialization code. This will ensure all necessary libraries are loaded and any predefined functions or variables are set.
## 4. Load the Trained Model:
If you've saved the trained model to a file (e.g., `model.h5`), there should be a cell in the notebook that loads this model. Run this cell to load the model into memory.
## 5. Make a Prediction:
1. **Insert a New Cell in the Notebook:** This will be where users input the path to their image and get the prediction
2. **Add Prediction Code:** In the new cell, users should add code similar to the following:
   ```
     def predict_image_using_model(model, image_path, img_size=(224, 224)):
    # Preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = cv2.resize(img, img_size)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match the model's input shape
    
    # Make a prediction
    predictions = model.predict(img)
    predicted_class = int(predictions > 0.5)  # Convert probability to class label (0 or 1)
    
    # Interpret the prediction
    if predicted_class == 0:
        label = "cat"
    else:
        label = "dog"
    
    confidence = predictions[0][0] * 100 if label == "dog" else (1 - predictions[0][0]) * 100
    print(f"The image is predicted to be a {label} with {confidence:.2f}% confidence.")

    image_path = "/path_to_image"
    predict_image_using_model(model, image_path)
   
4. **Run the cell**: After adding the code, run the cell to see the prediction
## 6. Interpret the Results:
The notebook will output the predicted class for the provided image.
