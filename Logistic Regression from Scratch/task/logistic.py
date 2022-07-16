import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            I = np.ones(row.shape[0])
            r = np.c_[I, row]
        t = np.dot(r, coef_)
        return self.sigmoid(t)


coef = [0.77001597, -2.12842434, -2.39305793]

# df = pd.read_csv('example1_stage1.txt')
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame
X = df[['worst concave points', 'worst perimeter']]
y = df['target']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
X = X_test[:10]

clr = CustomLogisticRegression()
p = clr.predict_proba(X, coef)
print('['+', '.join('{:0.5f}'.format(i) for i in p)+']')
