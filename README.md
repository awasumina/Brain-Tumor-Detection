# Brain Tumor Detection using VGG and CNN

This project uses Convolutional Neural Networks (CNN) with a VGG backbone to detect brain tumors from MRI images. The dataset contains MRI images classified into two categories: "tumor" and "no tumor." The model is fine-tuned on top of the pre-trained VGG model to improve accuracy and speed up the training process.

## Dataset

The dataset used for this project is the **Brain MRI Images for Brain Tumor Detection** available on Kaggle:
- [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

The dataset consists of:
- **Images with Tumor (Yes)**
- **Images without Tumor (No)**

Each image is resized to a shape of `(224, 224, 3)` to match the input size required by the VGG model.

## Project Structure

- **Preprocessing**: Images are loaded and resized to `(224, 224)`. They are split into training and testing sets in the ratio 67:33.
- **Model Architecture**: The VGG pre-trained model on ImageNet is used as the base model, with the top layers removed and replaced by custom layers.
- **Training**: The final model is trained with frozen VGG layers and custom fully connected layers added on top. The model is compiled using the `adam` optimizer and `categorical_crossentropy` loss function.
- **Evaluation**: The model is evaluated using accuracy, loss, and confusion matrix visualizations.


## Model Architecture

The architecture is built using VGG for feature extraction followed by a custom head for binary classification. Here’s an overview of the layers added on top of the VGG model:
- Global Average Pooling
- Dense (1024 units) with ReLU activation
- Dense (512 units) with ReLU activation
- Dense (2 units) with Softmax activation (for binary classification)

## Results

- The model was trained for 5 epochs with the following performance metrics:
  - **Training Accuracy**: >90%
  - **Validation Accuracy**: >85%
  - **Confusion Matrix**: Visualizes model performance on the test set, showing correct and incorrect classifications.

## Visualizations

1. **Training and Validation Accuracy and Loss**: Plots for accuracy and loss over epochs.
2. **Confusion Matrix**: Heatmap visualizing the performance of the model.
3. **Sample Predictions**: Visual representation of predicted labels vs actual labels for sample test images.
