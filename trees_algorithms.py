from names import SplitPartNames

from time import perf_counter
from id3 import Id3Estimator
from sklearn import tree, metrics


def measures_of_id3(subsets):
    clf = Id3Estimator()

    start_time = perf_counter()
    clf.fit(subsets[SplitPartNames['X_train']], subsets[SplitPartNames['y_train']])
    end_time = perf_counter()
    learning_time = end_time - start_time

    start_time = perf_counter()
    prediction = clf.predict(subsets[SplitPartNames['X_test']])
    end_time = perf_counter()
    prediction_time = end_time - start_time
    accuracy = metrics.accuracy_score(subsets[SplitPartNames['y_test']], prediction)
    return round(learning_time, 4), round(prediction_time, 4), round(accuracy, 2)


def measures_of_cart(subsets):
    clf = tree.DecisionTreeClassifier()

    start_time = perf_counter()
    clf.fit(subsets[SplitPartNames['X_train']], subsets[SplitPartNames['y_train']])
    end_time = perf_counter()
    learning_time = end_time - start_time

    start_time = perf_counter()
    prediction = clf.predict(subsets[SplitPartNames['X_test']])
    end_time = perf_counter()
    prediction_time = end_time - start_time
    accuracy = metrics.accuracy_score(subsets[SplitPartNames['y_test']], prediction)
    return round(learning_time, 4), round(prediction_time, 4), round(accuracy, 2)
