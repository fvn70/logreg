import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    def predict(self, X_test, cut_off=0.5):
        return [int(self.predict_proba(row, self.coef_) >= cut_off) for row in X_test]


from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame
X = df[['worst concave points', 'worst perimeter', 'worst radius']]
y = df['target']

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

clr = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)

# df = pd.read_csv('example_stage2-3.txt')
# X_train = df[['worst concave points', 'worst perimeter', 'worst radius']].to_numpy()
# y_train = df['y'].to_numpy()

# clr.fit_mse(X_train, y_train.to_numpy())
clr.fit_log_los(X_train, y_train.to_numpy())
y_pred = clr.predict(X_test)
acc = accuracy_score(y_test, y_pred)

answer_dict = {'coef_': clr.coef_, 'accuracy': acc}
print(answer_dict)
