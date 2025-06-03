
# Corpus Callosum Segmentation from T2 MRI Scans

This project focuses on the **automatic segmentation of the corpus callosum** from T2-weighted MRI images using data from the **ABIDE** and **OASIS** public datasets. A convolutional neural network (CNN) based on the **U-Net architecture** was developed, trained, and evaluated for this purpose.

---

## Project Overview

The **corpus callosum** is a critical brain structure involved in inter-hemispheric communication. Accurate segmentation can support neurodevelopmental and neurodegenerative studies, particularly in research on autism (ABIDE) and aging (OASIS).

This project applies deep learning to segment the corpus callosum from MRI images with a focus on:
- High specificity
- Generalizability across multiple datasets
- Efficient preprocessing and augmentation pipelines

---

## Notebooks

Two main notebooks were used to prepare the dataset and train the model:

### 1. `generate_dataset.ipynb`
This notebook is responsible for:
- Performing **data augmentation** (translations and rotation).
- Splitting the dataset into **training**, **validation**, and **test** sets.
- Saving the preprocessed and augmented data as `.pkl` files for efficient loading during training.

### 2. `train.ipynb`
This notebook handles:
- **Model training** using the data from the `.pkl` files.
- **Evaluation** of the model on the test set.
- **Visualization** of predicted segmentation masks against ground truth.

##  Data Sources

### 1. [ABIDE Dataset](http://fcon_1000.projects.nitrc.org/indi/abide/)
- Contains T2-weighted MR images collected from 17 independent sites.
- Includes ~1100 images with expert-segmented corpus callosum masks.
- Loaded in `.nii` format.

### 2. [OASIS Dataset](https://www.oasis-brains.org/)
- Contains MRI scans from 416 subjects aged 18–96.
- Selected images include ~903 `.tiff` scans and corresponding expert masks.

---

##  Preprocessing

To ensure consistency between sources:
- All images are **rescaled to 128×128** pixels.
- **Standardization** is applied:
  - Mean and standard deviation are computed from the training set.
  - These parameters are used to normalize the train, validation, and test sets.


##  Data Augmentation

To expand the dataset and improve generalization, the following augmentations were applied:
- Translations: (0, -1), (-1, 0), (-1, -1)
- Rotation: 1° clockwise

These transformations increased the training set to **11,216 images**.

---

##  Model Architecture

The model is based on a **U-Net** architecture with:
- Convolutional and max-pooling layers
- **Batch Normalization** for gradient flow and training stability
- **Dropout (p = 0.5)** for regularization and overfitting prevention

Total parameters: **87,157,889**  
Trainable parameters: **87,153,921**

---

## ⚙ Training Details

- **Loss function**: Dice Loss  
  Chosen for its robustness in class-imbalanced segmentation tasks.

  ```
  Dice Loss = 1 - (2 * intersection) / (|y| + |ŷ|)
  ```

- **Batch size**: 64  
- **Optimizer**: Mini-batch Gradient Descent  
- **Learning rate**: Exponentially decaying, starting from `1e-5`

---

## Results

- **High Dice coefficient**: 0.874, demonstrating strong overlap with expert masks.
- **High specificity**: 0.999, meaning the model accurately excludes non-corpus callosum areas.
- **Slightly lower sensitivity**: 0.779, indicating occasional under-segmentation.

Compared to a reference study with 7,000 more training samples and a hybrid loss function (Dice + BCE), this model performs competitively, though future improvements are possible.

---

## Limitations & Future Work

- Limited data augmentation due to compute constraints.
- Dice-only loss function; future work could explore hybrid losses.
- Integration of **metadata (e.g., patient diagnosis)** could enable per-group performance analysis.

---

## Dependencies

- Python 3.8+
- TensorFlow 
- NumPy, Matplotlib, nibabel, Pillow

---

## References
Chandra, Anjali, Verma, Shrish, Raghuvanshi, Ajay, & Bodhey, Narendra. (2022). CCsNeT: Automated Corpus Callosum segmentation using fully convolutional network based on U-Net. Biocybernetics and Biomedical Engineering, 42. https://doi.org/10.1016/j.bbe.2021.12.008

