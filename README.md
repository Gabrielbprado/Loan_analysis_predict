# HMEQ Data Processing and Machine Learning Project

## Overview

This project focuses on processing the HMEQ dataset and training a machine learning model to predict a certain outcome. The project follows a structured approach involving data extraction, transformation, and loading (ETL), exploratory data analysis (EDA), and model training and prediction.

## Project Structure

The project is organized into the following directories and files:

-   **`dataset/`**: Contains the raw HMEQ dataset.
    -   `hmeq.csv`: The original HMEQ dataset file.
      ## Medalion Architecture 🥉🥈🥇
-   **`data/`**: Stores the processed data at different stages.
    -   **`bronze/`**: Contains the raw data loaded into the system.
        -   `hmeq_bronze.csv`: The raw data of HMEQ after loading.
    -   **`silver/`**: Contains the data after cleaning and basic transformations.
        -   `hmeq_silver.csv`: The cleaned data.
    -   **`gold/`**: Contains the data prepared for model training.
        -   `X_train.csv`: Training features.
        -   `X_test.csv`: Testing features.
        -   `y_train.csv`: Training target variable.
        -   `y_test.csv`: Testing target variable.
        - `y_pred.csv`: Predicted target variable.
         # Scripts
        
-   **`etl/`**: Contains the scripts for the ETL process.
    -   `etl_script.py`: Main script for the ETL pipeline.
    -   `prepare-gold.py`: Script to prepare data for the model.
-   **`models/`**: Contains the trained model and related scripts.
    -   `logistic_model.pkl`: The trained logistic regression model.
    -   `train_model.py`: Script to train the model.
    -   `model_prevision.py`: Script to predict with the model.
-   **`notebooks/`**: Contains Jupyter notebooks for exploratory data analysis.
    -   `eda_hmeq.ipynb`: Notebook with the exploratory data analysis.

## Workflow

1.  **Data Extraction**: The `hmeq.csv` dataset is the initial data.
2.  **Data Transformation and Loading (ETL)**:
    -   The `etl_script.py` is used to load the raw data into the bronze folder.
    -   The data is then cleaned and transformed and moved to the silver folder.
    -   The data is transformed again and loaded into the gold folder to be used by the model.
3.  **Exploratory Data Analysis (EDA)**:
    -   The `eda_hmeq.ipynb` notebook is used to understand the data, its patterns, and its characteristics.
4.  **Model Training**:
    -   The `train_model.py` script uses the data from the gold folder to train a logistic regression model.
    -   The trained model is saved as `logistic_model.pkl`.
5.  **Model Prediction**:
    -   The `model_prevision.py` script can load the trained model and predict the results with the testing set.

## Technologies

-   Python
-   Pandas
-   Scikit-learn
-   Jupyter Notebook
-   VS Code

  ## Natural Language
  This project also contains a natural language script that uses ollama, a dataframe and a sample of 100 lines of the file predictions_vs_real.csv in gold folder.
  This script use a question and answer approach and it can answare questions about the dataset.
  The file containing the natural language script is in pln/pln_script.py


## How to Run

1.  Ensure you have all the dependencies installed.
2.  Run the ETL scripts in the `etl/` folder.
3.  Explore the data using the `eda_hmeq.ipynb` notebook.
4.  Train the model using `train_model.py`.
5. Predict with the model `model_prevision.py`.

# Images of ML Predict

![image](https://github.com/user-attachments/assets/bace3b41-4674-4f51-b103-abd3d361dc44)

![image](https://github.com/user-attachments/assets/c6575705-ce17-43e8-8ebb-ea1438d74d38)

![image](https://github.com/user-attachments/assets/3dd5eaab-c2dd-458f-bee3-b2c3d30a6878)

