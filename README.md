# Syook AI Intern Assignment - Pooja Talele

## Repository Structure

### 1. Scripts

- **`inference.py`**:
  - **Purpose**: This script runs inference using the trained person detection and PPE detection models. It takes the following arguments: input directory, output directory, person detection model, and PPE detection model.
  - **Usage**: Use this script to process images, detect persons, and identify PPE items on the detected persons.
  
- **`crop_and_adjust_ppe_labels.py`**:
  - **Purpose**: This script crops person images based on bounding boxes from the person detection model and adjusts the PPE labels accordingly.
  - **Usage**: Use this script to prepare datasets by cropping images and adjusting the labels for training the PPE detection model.
  
- **`pascalVOC_to_yolo.py`**:
  - **Purpose**: Converts Pascal VOC formatted annotations to YOLO format.
  - **Usage**: Run this script to convert XML annotations in Pascal VOC format to YOLO text files.

- **`separate_labels.py`**:
  - **Purpose**: This script separates the person class labels into a separate directory and adjusts the other class IDs to maintain 0-based indexing.
  - **Usage**: Use this script to preprocess labels by separating person labels and modifying class IDs for training.

### 2. Weights

- **`person_detection.pt`**: Trained model for person detection.
- **`ppe_detection.pt`**: Trained model for PPE detection on cropped person images.

## Detailed Documentation

For a detailed explanation of the project, including the approaches, learning, and challenges faced, please refer to the [Google Document](https://docs.google.com/document/d/1VX0LaWa_wYX51xaU4TQxtizhnoSXIKgC5-fIR08hyVA/edit?usp=sharing).

## How to Use

1. **Prepare the Dataset**: Use the `pascalVOC_to_yolo.py` script to convert your dataset annotations to YOLO format.
2. **Separate Labels**: Use the `separate_labels.py` script to organize your labels by separating the person class labels.
3. **Crop and Adjust Labels**: Run the `crop_and_adjust_ppe_labels.py` script to crop person images and adjust the corresponding PPE labels.
4. **Inference**: Finally, use the `inference.py` script to run inference on new images using the trained models.

