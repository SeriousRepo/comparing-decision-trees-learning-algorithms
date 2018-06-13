import utils
from names import SplitPartNames
import trees_algorithms

from sklearn import model_selection

datasets = utils.get_datasets()

Xs = []
ys = []

utils.create_directory('datasets')

for dataset_index in range(0, len(datasets)):
    X, y = utils.split_dataset(datasets[dataset_index])
    Xs.append(X), ys.append(y)

    utils.create_dataset_files(datasets[dataset_index], dataset_index)

cv = model_selection.StratifiedKFold(n_splits=30)

for dataset_index in range(0, len(datasets)):
    fold = 0
    for train_indexes, test_indexes in cv.split(Xs[dataset_index], ys[dataset_index]):
        #print("Zbiór {}, Fold {} ({} w TS, {} w VS)".format(utils.get_dataset_name(dataset_index), fold, len(train_indexes), len(test_indexes)))
        fold_sets = [[], [], [], []]

        fold_sets[SplitPartNames['X_train']] = Xs[dataset_index][train_indexes]
        fold_sets[SplitPartNames['y_train']] = ys[dataset_index][train_indexes]

        fold_sets[SplitPartNames['X_test']] = Xs[dataset_index][train_indexes]
        fold_sets[SplitPartNames['y_test']] = ys[dataset_index][train_indexes]


        #trees_algorithms.measure_times_of_cart(fold_sets)
        trees_algorithms.measure_times_of_c45(utils.get_dataset_name(dataset_index))



        """
        print("predicted labels for {} dataset".format(getDatasetName(dataset_index)))
        print(estimator.predict(fold_sets[SplitPartNames['X_test']]))
        print("true labels for {} dataset".format(getDatasetName(dataset_index)))
        print(fold_sets[SplitPartNames['y_test']])
        """
        fold += 1

utils.remove_directory('datasets')
