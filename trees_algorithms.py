from names import SplitPartNames

from time import perf_counter
from id3 import Id3Estimator
from sklearn import tree


def measure_times_of_id3(sets):
    clf = Id3Estimator()

    start_time = perf_counter()
    clf.fit(sets[SplitPartNames['X_train']], sets[SplitPartNames['y_train']])
    end_time = perf_counter()
    learning_time = end_time - start_time

    start_time = perf_counter()
    clf.predict(sets[SplitPartNames['X_test']])
    end_time = perf_counter()
    prediction_time = end_time - start_time
    return learning_time, prediction_time


def measure_times_of_cart(sets):
    clf = tree.DecisionTreeClassifier()

    start_time = perf_counter()
    clf.fit(sets[SplitPartNames['X_train']], sets[SplitPartNames['y_train']])
    end_time = perf_counter()
    learning_time = end_time - start_time

    start_time = perf_counter()
    clf.predict(sets[SplitPartNames['X_test']])
    end_time = perf_counter()
    prediction_time = end_time - start_time
    return learning_time, prediction_time
