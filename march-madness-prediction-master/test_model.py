from sklearn.model_selection import train_test_split
from sklearn import metrics
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

# create sample dataset
X, y = sklearn.datasets.make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,
    random_state=1
)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# simple plot
plt.hist(y_pred)
plt.title("Prediction Distribution")
plt.show()