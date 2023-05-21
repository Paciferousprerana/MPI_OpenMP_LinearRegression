# Implementation of Multiple Linear Regression using MPI and OpenMP

## Problem Statement

Leverage parallel computing techniques to accelerate the training and prediction process of linear regression models.

## Objectives

* Load the data and perform data pre-processing
* Identify the features, target and split the data into train and test
* Implement multiple Linear Regression by estimating the coefficients on the given data
* Use MPI package to distribute the data and implement `communicator`
* Define functions for each objective and make a script (.py) file to execute using MPI command
* Use OpenMP component to predict the data and calculate the error on the predicted data
* Implement the Linear Regression from `sklearn` and compare the results

## Dataset

The dataset chosen for this project is [Combined Cycle Power Plant](https://archive.ics.uci.edu/ml/datasets/combined+cycle+power+plant). The dataset is made up of 9568 records and 5 columns. Each record contains the values for Ambient Temperature, Exhaust Vaccum, Ambient Pressure, Relative Humidity and Energy Output. The data was collected over a six year period (2006-11).

## Project Setup
Follow the steps below to set up the project:

1. Clone the project repository or download the project files to your local machine.

2. Open ipynb file in Jupyter Notebook or Google Colab.

3. Download the dataset from the link provided above

4. Ensure that you have installed mpi4py for MPI and pymp-pypi for OpenMP.

## Usage
1. Open the linear regression implementation file (ipynb file).

2. Load the dataset into the code using the appropriate file reading function (e.g., pd.read_csv() in the case of a CSV file).

3. Preprocess the dataset by performing any necessary data cleaning, scaling, or feature engineering steps.

4. Implement the linear regression algorithm using MPI and OpenMP. Distribute the workload across multiple processors or cores using MPI, and leverage parallelism within each processor or core using OpenMP directives.

5. Train the linear regression model using the parallel implementation and evaluate its performance using suitable evaluation metrics (e.g., mean squared error).

6. Make predictions using the trained model and assess the accuracy of the predictions.

7. Implement sklearn LinearRegression module and compare the results.

## Troubleshooting
1. Ensure that you have properly installed MPI and OpenMP on your machine and that they are set up correctly.

2. Check for any missing libraries or dependencies and install them using the pip install command if necessary.
