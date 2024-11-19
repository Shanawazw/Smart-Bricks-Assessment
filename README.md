# Real Estate Price Prediction Model

This repository contains a machine learning pipeline for predicting real estate prices based on various features, including property size, building age, and transaction type. The pipeline utilizes multiple models and techniques such as feature selection, ensemble methods, and hyperparameter optimization.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Requirements](#data-requirements)
- [How to Train the Model](#how-to-train-the-model)
- [How to Make Predictions](#how-to-make-predictions)
- [Testing the Pipeline](#testing-the-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [File Structure](#file-structure)

---

## Environment Setup

To run the model, you'll need to set up your environment with the necessary dependencies. Follow the steps below:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/real-estate-price-prediction.git
    cd real-estate-price-prediction
    ```

2. **Create and activate a virtual environment**:

    For Python 3.x:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Linux/macOS
    .\venv\Scripts\activate   # For Windows
    ```

3. **Install dependencies**:

    Install the required libraries using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

---

## Data Requirements

The model requires a dataset containing real estate transaction details. Below is an example of the expected dataset format:

| property_size_sqm | building_age | rooms_en | rooms_ar | transaction_type_en | property_usage_en | is_freehold_text |
|-------------------|--------------|----------|----------|----------------------|-------------------|------------------|
| 100               | 10           | 3        | 3        | Sale                 | Residential       | Yes              |
| 150               | 15           | 4        | 4        | Lease                | Commercial        | No               |
| ...               | ...          | ...      | ...      | ...                  | ...               | ...              |

The dataset should include:
- **Numerical features**: `property_size_sqm`, `building_age`, `rooms_en`, etc.
- **Categorical features**: `transaction_type_en`, `property_usage_en`, etc.
- **Target variable**: `property_size_sqm` (price or other relevant feature)

You can modify the data loading script (`data_loader.py`) to adapt to your dataset.

---

## How to Train the Model

1. **Load the dataset**:
    The data can be loaded using the `load_data()` function from `data_loader.py`. Ensure that your data is preprocessed (missing values filled, categorical features encoded).

2. **Train the model**:
    After preprocessing the data, you can train the model by running the following command:
    
    ```bash
    python main.py
    ```

    This will execute the pipeline in `main.py`, which will:
    - Load and preprocess the data
    - Perform feature selection
    - Train multiple models (e.g., Random Forest, XGBoost, Linear Regression, etc.)
    - Select the best model and make predictions

3. **Hyperparameter Tuning**:
    The model leverages `Bayesian Optimization` for hyperparameter tuning. The optimization can be adjusted in the `config.py` file by changing the parameter grid and optimization settings.

---

## How to Make Predictions

Once the model is trained, predictions can be made by using the trained model on new data.

1. **Load the trained model**:
    The trained model is saved as a `.pkl` file in the `models/` directory. You can load the model as follows:

    ```python
    import pickle

    with open('models/best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    ```

2. **Make predictions**:
    You can use the model to make predictions on new data:

    ```python
    predictions = model.predict(new_data)
    ```

---

## Testing the Pipeline

The code includes unit tests to ensure that each component of the pipeline functions as expected. To run the tests, use:

```bash
python -m unittest test.py

.
├── config.py             # Configuration file with parameters
├── data_loader.py        # Data loading and preprocessing
├── evaluation.py         # Evaluation metrics (RMSE, R², MAE)
├── main.py               # Main script to run the entire pipeline
├── models/               # Directory where trained models are saved
├── test.py               # Unit tests for the pipeline
├── requirements.txt      # List of Python dependencies
└── README.md             # This README file
