from names import SplitPartNames

from time import perf_counter
from id3 import Id3Estimator
from sklearn import tree
import sys
sys.path.append('C4.5/c45')
from c45 import C45

def measure_times_of_id3(sets):
    clf = Id3Estimator()

    start_time = perf_counter()
    clf.fit(sets[SplitPartNames['X_train']], sets[SplitPartNames['y_train']])
    end_time = perf_counter()
    print("time of tree building {}".format(end_time - start_time))

    start_time = perf_counter()
    clf.predict(sets[SplitPartNames['X_test']])
    end_time = perf_counter()
    print("time of labels prediction {}".format(end_time - start_time))


def measure_times_of_cart(sets):
    clf = tree.DecisionTreeClassifier()

    start_time = perf_counter()
    clf.fit(sets[SplitPartNames['X_train']], sets[SplitPartNames['y_train']])
    end_time = perf_counter()
    print("time of tree building {}".format(end_time - start_time))

    start_time = perf_counter()
    clf.predict(sets[SplitPartNames['X_test']])
    end_time = perf_counter()
    print("time of labels prediction {}".format(end_time - start_time))

def measure_times_of_c45(dataset_name):
    clf = C45('datasets/{0}/{0}.data'.format(dataset_name), 'datasets/{0}/{0}.names'.format(dataset_name))
    clf.fetchData()
    clf.preprocessData()
    #clf.generateTree()
    #clf.printTree()

