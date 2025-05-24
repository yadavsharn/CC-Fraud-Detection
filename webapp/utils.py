# webapp/utils.py
import pandas as pd

def preprocess(df):
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    if 'Amount' in df.columns:
        df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    return df
