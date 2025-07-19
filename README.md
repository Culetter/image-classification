# Intel Image Classifier

This project is an image classifier based on EfficientNet-B0, built with PyTorch. It is trained on the Intel Image Classification dataset and classifies images into six categories: buildings, forest, glacier, mountain, sea and street.

# Project Structure
```
├── /data
│   ├── seg_train/              # Training images (organized by class)
│   ├── seg_test/               # Test images (organized by class)
│   └── seg_pred/               # Images to classify (not organized by class)
├── /notebooks
│   ├── data-analysis.ipynb     # Exploratory data analysis and dataset overview
│   └── model-training.ipynb    # Model training experiments and results visualization
├── /results
│   ├── model/                  # Saved model
│   ├── plots/                  # Saved plots
│   └── predicted_img/          # Classified images copied into folders by predicted class
├── /src
│   └── train_model.py          # Main training, evaluation, and prediction script
```

# Installation
1. Clone the repository:
```bash
git clone https://github.com/Culetter/image-classification.git
cd image-classification
```
2. Install the dependencies
```
pip install -r requirements.txt
```
# Usage
To train the model and run predictions:
```
python src/train_model.py
```
# Model
* Backbone: EfficientNet-B0
* Final layer: (1280 → 6)
* Loss function: CrossEntropyLoss
* Optimizer: Adam (learning rate 0.001)
* Epochs: 6
# Predictions
After training, the script automatically:
* Classifies all images in ```.../data/seg_pred/```
* Copies them into folders named after the predicted class in ```../results/predicted_img/```
# Author
@Culetter
# License
The dataset used in this project is the "Intel Image Classification" dataset, available on Kaggle:  
https://www.kaggle.com/datasets/puneet6060/intel-image-classification

The dataset does not have a specific license listed, so it is used here only for educational and non-commercial purposes.  
All rights to the original dataset remain with the original author.
