import utils
from names import SplitPartNames, DatasetNames, set_names
import trees_algorithms

from sklearn import model_selection

print('downloading datasets...')
datasets = utils.get_datasets()
set_names()

Xs = []
ys = []

for dataset_index in range(0, len(datasets)):
    X, y = utils.split_dataset(datasets[dataset_index])
    Xs.append(X), ys.append(y)

k = 30
cv = model_selection.StratifiedKFold(n_splits=k)

id3_measures = []
cart_measures = []

for dataset_index in range(0, len(datasets)):
    fold = 0
    id3_fold_measures = []
    cart_fold_measures = []
    for train_indexes, test_indexes in cv.split(Xs[dataset_index], ys[dataset_index]):
        print('processing {} fold of {} algorithm...'.format(fold, utils.get_dataset_name(dataset_index)))
        fold_sets = [[], [], [], []]

        fold_sets[SplitPartNames['X_train']] = Xs[dataset_index][train_indexes]
        fold_sets[SplitPartNames['y_train']] = ys[dataset_index][train_indexes]

        fold_sets[SplitPartNames['X_test']] = Xs[dataset_index][train_indexes]
        fold_sets[SplitPartNames['y_test']] = ys[dataset_index][train_indexes]

        id3_fold_measures.append(trees_algorithms.measures_of_id3(fold_sets))
        cart_fold_measures.append(trees_algorithms.measures_of_cart(fold_sets))

        fold += 1

    id3_measures.append(id3_fold_measures)
    cart_measures.append(cart_fold_measures)

print('Algorithm\tFold\tLearning time\tPrediction time\t\tAccuracy\tDataset')
for dataset_index in range(0, len(DatasetNames)):
    for fold_index in range(0, k):
        learning, prediction, accuracy = id3_measures[dataset_index][fold_index]
        print('ID3\t\t{}\t{}\t\t{}\t\t\t{}\t\t{}'.format(fold_index, learning, prediction,
                                                        accuracy, utils.get_dataset_name(dataset_index)))
        learning, prediction, accuracy = cart_measures[dataset_index][fold_index]
        print('CART\t\t{}\t{}\t\t{}\t\t\t{}\t\t{}'.format(fold_index, learning, prediction,
                                                         accuracy, utils.get_dataset_name(dataset_index)))

print('\nAlgorithm\tAvg learning time\tAvg prediction time\tAvg accuracy\tDataset')
for dataset_index in range(0, len(DatasetNames)):
    learning, prediction, accuracy = utils.get_averages(id3_measures[dataset_index])
    print('ID3\t\t{}\t\t\t{}\t\t\t{}\t\t{}'
          .format(learning, prediction, accuracy, utils.get_dataset_name(dataset_index)))
    learning, prediction, accuracy = utils.get_averages(cart_measures[dataset_index])
    print('CART\t\t{}\t\t\t{}\t\t\t{}\t\t{}'
          .format(learning, prediction, accuracy, utils.get_dataset_name(dataset_index)))
