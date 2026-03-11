"""Utils to perform error checking, CV, and hyperparameter tuning."""

from sklearn.model_selection import train_test_split
from sklearn import metrics
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def do_split_data(X, y, k=10):
    """Splits data into k portions for k-fold CV."""
    return np.array_split(X, k), np.array_split(y, k)


def cross_validate(classifier, X, y, k=10):
    """Performs cross validation."""
    X_split, y_split = do_split_data(X, y, k)

    training_errors, testing_errors = [], []

    for i in range(k):
        print(f"Using {i+1} split for validation")

        X_test, y_test = X_split[i], y_split[i]

        X_train = np.concatenate(
            [X_split[j] for j in range(len(X_split)) if j != i]
        )

        y_train = np.concatenate(
            [y_split[j] for j in range(len(y_split)) if j != i]
        )

        train_error, test_error = get_errors_already_split(
            classifier,
            X_train,
            y_train,
            X_test,
            y_test,
            num_iterations=1
        )

        training_errors.append(train_error)
        testing_errors.append(test_error)

    mean_train_error = np.mean(np.array(training_errors), axis=0)
    mean_test_error = np.mean(np.array(testing_errors), axis=0)

    return mean_train_error, mean_test_error


def get_errors_already_split(
    classifier,
    X_train,
    y_train,
    X_test,
    y_test,
    num_iterations=100
):

    train_error, test_error = 0.0, 0.0

    for i in range(num_iterations):

        print("Training classifier...")

        classifier.fit(X_train, y_train)

        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)

        train_error += 1 - metrics.accuracy_score(y_train, y_train_pred)
        test_error += 1 - metrics.accuracy_score(y_test, y_test_pred)

    train_error /= num_iterations
    test_error /= num_iterations

    return train_error, test_error


def get_train_test_error(classifier, X, y, num_iterations=100, split=0.2):

    train_error, test_error = 0.0, 0.0

    for i in range(num_iterations):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=split,
            random_state=i
        )

        classifier.fit(X_train, y_train)

        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)

        train_error += 1 - metrics.accuracy_score(y_train, y_train_pred)
        test_error += 1 - metrics.accuracy_score(y_test, y_test_pred)

    train_error /= num_iterations
    test_error /= num_iterations

    return train_error, test_error


def split_data(X, y, random=False, train_proportion=0.8):

    assert X.shape[0] == y.shape[0]

    if not random:

        split_index = int(train_proportion * X.shape[0])

        X_train = X[:split_index]
        y_train = y[:split_index]

        X_test = X[split_index:]
        y_test = y[split_index:]

    else:

        X_train, y_train, X_test, y_test = [], [], [], []

        for i in range(X.shape[0]):

            if np.random.random() < train_proportion:
                X_train.append(X[i])
                y_train.append(y[i])
            else:
                X_test.append(X[i])
                y_test.append(y[i])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def get_best_depth(X, y, k=10, depths=None):

    if depths is None:
        depths = [None]

    depth_to_err = {}
    depth_to_train_err = {}

    for depth in depths:

        test_errors, train_errors = [], []

        X_split, y_split = do_split_data(X, y, k)

        for i in range(k):

            X_test, y_test = X_split[i], y_split[i]

            X_train = np.concatenate(
                [X_split[j] for j in range(len(X_split)) if j != i]
            )

            y_train = np.concatenate(
                [y_split[j] for j in range(len(y_split)) if j != i]
            )

            dclf = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=depth
            )

            dclf.fit(X_train, y_train)

            y_test_pred = dclf.predict(X_test)
            y_train_pred = dclf.predict(X_train)

            test_error = 1 - metrics.accuracy_score(y_test, y_test_pred)
            train_error = 1 - metrics.accuracy_score(y_train, y_train_pred)

            test_errors.append(test_error)
            train_errors.append(train_error)

        depth_to_err[depth] = np.mean(test_errors)
        depth_to_train_err[depth] = np.mean(train_errors)

    print(depth_to_err)
    print(depth_to_train_err)

    plt.plot(list(depth_to_train_err.keys()), list(depth_to_train_err.values()))
    plt.title("Training Error vs Depth")
    plt.show()

    plt.figure()
    plt.plot(list(depth_to_err.keys()), list(depth_to_err.values()))
    plt.title("Testing Error vs Depth")
    plt.show()

    return min(depth_to_err.items(), key=lambda x: x[1])


if __name__ == "__main__":

    print("Running tests with decision tree")

    print("Creating dataset")

    X, y = sklearn.datasets.make_classification(
        n_samples=1000,
        n_features=10,
        n_redundant=6,
        n_informative=4,
        random_state=1,
        n_clusters_per_class=2,
        n_classes=7
    )

    X, y = np.array(X), np.array(y)

    d_tree = DecisionTreeClassifier(criterion="entropy")

    print("Training & evaluating decision tree")

    train_err, test_err = get_train_test_error(d_tree, X, y, split=0.7)

    print("Training error:", train_err)
    print("Testing error:", test_err)

    print("Getting cross validation errors")

    train_error_cv, test_error_cv = cross_validate(d_tree, X, y, k=10)

    print("Training CV error:", train_error_cv)
    print("Testing CV error:", test_error_cv)

    print("Finding best depth...")

    depths = np.arange(1, 15)

    best_depth, best_test_err = get_best_depth(X, y, depths=depths)

    print("Best depth:", best_depth)
    print("Testing error:", best_test_err)