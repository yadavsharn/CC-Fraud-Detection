# Credit Card Fraud Detection

## Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The aim is to build a reliable model that can accurately classify transactions as legitimate or fraudulent, helping banks and financial institutions reduce losses due to fraud.

## Dataset
The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by European cardholders in September 2013, with features transformed using PCA (v1 to v28), along with `Time`, `Amount`, and a target label `Class` (0 for legitimate, 1 for fraud).

## Features
- **Time**: Seconds elapsed between each transaction and the first transaction.
- **V1 to V28**: Principal components obtained using PCA (for confidentiality reasons).
- **Amount**: Transaction amount.
- **Class**: Target variable (0 = legitimate, 1 = fraud).

## Project Structure
- `app.py`: Flask web app for real-time fraud prediction.
- `model.pkl`: Trained machine learning model saved using pickle.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.
- `data/`: Contains the dataset CSV files.
- `notebooks/`: Jupyter notebooks with exploratory data analysis and model training.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yadavsharn/CC-Fraud-Detection.git
   cd credit-card-fraud-detection

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run
   ```bash
   streamlit run webapp/app.py 
