# HSI Classification with 3D CNN using SS-DARTS

This repository contains code for hyperspectral image (HSI) classification using a 3D Convolutional Neural Network (CNN) implemented with the SS-DARTS (Single-Stage Differentiable Architecture Search) algorithm. The SS-DARTS algorithm is used to automatically search for an optimal architecture for HSI classification.

## Folders

- **SS-DARTS:** Contains the implementation of the SS-DARTS algorithm for architecture search.
- **images:** Contains images used in the repository, such as diagrams, plots, or visualizations.
- **preprocessing:** Includes scripts or notebooks for preprocessing HSI data, such as data cleaning, normalization, or dimensionality reduction.
- **reference_papers:** Contains relevant research papers or articles related to HSI classification, 3D CNNs, and architecture search.
- **results:** Stores the results, evaluation metrics, or performance analysis obtained from the experiments.
- **3D_CNN_HSI_classification.ipynb:** Jupyter Notebook with the implementation of the 3D CNN for HSI classification.
- **SS-DARTS.ipynb:** Jupyter Notebook demonstrating the implementation of the SS-DARTS algorithm for architecture search.

## Usage

1. Clone the repository:
   ```shell
   git clone https://github.com/your-username/HSI-classification-with-3D-CNN-using-SS-DARTS.git
2. Set up the necessary dependencies and environment.
3. Preprocess the HSI data using the scripts or notebooks in the preprocessing folder.
4. Run the .py notebooks in SS-DARTS to train and evaluate the SS-DARTS model for HSI classification using
   ```shell
   !python train_HSI.py
5. To evaluate the architecture obtained, replace the searched genotype in genotype.py test the model
   ```shell
   !python test_HSI.py

This repo contains implementations for four Hyperspectral Image Datasets namely:
- Indian Pines
- Salinas Valley
- Kennedy Space Centre
- Botswana


### License
This project is licensed under the MIT License.
