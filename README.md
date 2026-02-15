## NN and DL: SOM Clustering and SLFN Classification

This project explores the fundamental differences between unsupervised learning (clustering) and supervised learning (classification) using neural networks. The implementation includes:


1. Self-Organizing Map (SOM) for clustering and classifying the Digits dataset with different grid configurations (4×4 and 20×20)


2. Single Layer Feedforward Network (SLFN) implemented from scratch for binary classification on the Titanic dataset

## Tasks Implemented
### Task 1: Clustering Digits Dataset with SOM
###### Implemented SOM clustering using MiniSom library
#### Trained SOMs with two different grid configurations:
###### 4×4 grid (16 neurons)
###### 20×20 grid (400 neurons)
#### Visualized clustering results including:
###### Weight vectors for each neuron
###### Hit maps showing data distribution across neurons
###### Sample images mapped to representative neurons
#### Calculated quantization error to evaluate representation quality
#### Identified and analyzed dead neurons (neurons with no mapped samples)
#### Compared clustering performance between grid sizes


### Task 2: Classification of Digits Dataset Using SOM
###### Assigned class labels to neurons using majority voting from training data
#### Classified test samples by mapping to Best Matching Units (BMUs)
#### Evaluated classification performance:
###### Accuracy metrics for both grid configurations
###### Confusion matrices
###### Precision, recall, and F1-scores per class
#### Visualized neuron class maps with dead neurons marked distinctly


### Task 3: Classification with Single Layer Feedforward Network (SLFN)
###### Implemented SLFN from scratch using only NumPy (no deep learning frameworks)
#### Architecture:
###### Input layer: 7 features (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
###### Hidden layer: 128 neurons with ReLU activation
###### Output layer: 1 neuron with sigmoid activation for binary classification

#### Implemented key components:
###### Forward propagation with ReLU and sigmoid activations
###### Binary cross-entropy loss computation
###### Backpropagation algorithm for gradient calculation
###### Gradient descent for weight updates

#### Preprocessed Titanic dataset:
###### Handled missing values
###### Encoded categorical variables
###### Normalized numerical features
###### 80/20 train-test split

#### Evaluated model performance:
###### Accuracy, precision, recall, F1-score
###### Confusion matrix visualization
###### ROC curve and AUC score
###### Training loss curve

## Datasets Used:

Digits Dataset: 
```
    from sklearn.datasets import load_digits
    digits = load_digits()
```
Titanic Dataset: [download](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)


## Project Structure:
```
neural-network/
├── README.md
├── code.ipynb             
├── data/
│   └── titanic.csv         # Titanic dataset (downloaded automatically)
└── images/                 # Generated visualizations
    ├── som_weight_vectors_4x4.png
    ├── som_weight_vectors_20x20.png
    ├── hit_maps.png
    ├── confusion_matrices.png
    ├── roc_curve.png
    └── training_loss.png
```

## Requirements
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
minisom>=2.3.0
```

## References
1. Kohonen, T. (1990). The self-organizing map. Proceedings of the IEEE, 78(9), 1464-1480.


2. Haykin, S. (2009). Neural Networks and Learning Machines (3rd ed.). Prentice Hall.


3. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830.


4. Vesanto, J., & Alhoniemi, E. (2000). Clustering of the Self-Organizing Map. IEEE Transactions on Neural Networks, 11(3), 586-600.


