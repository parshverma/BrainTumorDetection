
# Tumor Detection Model

This project implements a brain tumor detection model using Convolutional Neural Networks (CNNs) with TensorFlow/Keras. The model is trained to classify images into four categories: `No Tumor`, `Pituitary Tumor`, `Meningioma Tumor`, and `Glioma Tumor`.


## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/parshverma/BrainTumorDetection.git
   cd BrainTumorDetection
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

- The dataset for training and testing consists of MRI images categorized into four tumor types (`no_tumor`, `glioma_tumor`, `meningioma_tumor`, and `pituitary_tumor`).
- You can replace the images in the `Training/` and `Testing/` folders with your own dataset.

## Running the Model

To run the model, execute the following command:
```bash
python main.py
```

This script will load the data, train the CNN model, evaluate its performance, and output the results, including accuracy and loss plots.

## Model Architecture

- The model uses a pre-trained DenseNet201 base model for feature extraction, followed by custom dense layers for classification.
- Batch normalization, data augmentation, and early stopping are used to enhance model performance.

## Results

- The trained model produces a classification report and plots for training vs. validation accuracy and loss.

## Future Improvements

- Fine-tuning the pre-trained DenseNet layers.
- Experimenting with additional data augmentation techniques.
- Hyperparameter optimization for improved accuracy.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
