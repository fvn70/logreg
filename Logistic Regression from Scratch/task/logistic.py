import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            t = coef_[0] + np.dot(row, coef_[1:])
        else:
            t = np.dot(row, coef_)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        N = X_train.shape[0]
        self.coef_ = [0. for _ in range(X_train.shape[1])]
        if self.fit_intercept:
            self.coef_ = [0.] + self.coef_
            k = 1
        else:
            k = 0
        for l in range(self.n_epoch):
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                delta = self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
                self.coef_[0] = self.coef_[0] - delta
                for j in range(len(row)):
                    self.coef_[j + k] = self.coef_[j + k] - delta * row[j]
                if l in [0, self.n_epoch - 1]:
                    y_hat = self.predict_proba(row, self.coef_)
                    mse_err = (y_hat - y_train[i])**2 / N
                    if l == 0:
                        mse_error_first.append(mse_err)
                    else:
                        mse_error_last.append(mse_err)

    def fit_log_los(self, X_train, y_train):
        N = X_train.shape[0]
        self.coef_ = [0. for _ in range(X_train.shape[1])]
        if self.fit_intercept:
            self.coef_ = [0.] + self.coef_
            k = 1
        else:
            k = 0

        for l in range(self.n_epoch):
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                delta = self.l_rate * (y_hat - y_train[i]) / N
                self.coef_[0] = self.coef_[0] - delta
                for j in range(len(row)):
                    self.coef_[j + k] = self.coef_[j + k] - delta * row[j]
                if l in [0, self.n_epoch - 1]:
                    y_hat = self.predict_proba(row, self.coef_)
                    log_err = -(y_train[i] * np.log(y_hat) + (1 - y_train[i]) * np.log(1 - y_hat)) / N
                    if l == 0:
                        log_loss_error_first.append(log_err)
                    else:
                        log_loss_error_last.append(log_err)


    def predict(self, X_test, cut_off=0.5):
        return [int(self.predict_proba(row, self.coef_) >= cut_off) for row in X_test]


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame
X = df[['worst concave points', 'worst perimeter', 'worst radius']]
y = df['target']

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)

lr = LogisticRegression(fit_intercept=True)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

acc_skl = accuracy_score(y_test, y_pred)

clr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)

mse_error_first = []
mse_error_last = []
clr.fit_mse(X_train, y_train.to_numpy())
y_pred = clr.predict(X_test)
acc_mse = accuracy_score(y_test, y_pred)

log_loss_error_first = []
log_loss_error_last = []
clr.fit_log_los(X_train, y_train.to_numpy())
y_pred = clr.predict(X_test)
acc_log = accuracy_score(y_test, y_pred)

answer_dict = {'mse_accuracy': acc_mse,
               'logloss_accuracy': acc_log,
               'sklearn_accuracy': acc_skl,
               'mse_error_first': mse_error_first,
               'mse_error_last': mse_error_last,
               'logloss_error_first': log_loss_error_first,
                'logloss_error_last': log_loss_error_last}

print(answer_dict)
print('''Answers to the questions:
1) 0.00003
2) 0.00000
3) 0.00153
4) 0.00576
5) expanded
6) expanded''')
