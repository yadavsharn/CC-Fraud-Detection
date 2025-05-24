# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load data
df = pd.read_csv('data/creditcard.csv')
df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
df.drop(['Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# Handle imbalance
X_res, y_res = SMOTE().fit_resample(X, y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/fraud_model.pkl')
