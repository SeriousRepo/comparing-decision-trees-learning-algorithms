import utils
from names import SplitPartNames, DatasetNames
import trees_algorithms

from sklearn import model_selection

datasets = utils.get_datasets()

Xs = []
ys = []

for dataset_index in range(0, len(datasets)):
    X, y = utils.split_dataset(datasets[dataset_index])
    Xs.append(X), ys.append(y)

k = 30
cv = model_selection.StratifiedKFold(n_splits=k)

id3_times = []
cart_times = []

for dataset_index in range(0, len(datasets)):
    fold = 0
    id3_fold_times = []
    cart_fold_times = []
    for train_indexes, test_indexes in cv.split(Xs[dataset_index], ys[dataset_index]):
        print(fold)
        fold_sets = [[], [], [], []]

        fold_sets[SplitPartNames['X_train']] = Xs[dataset_index][train_indexes]
        fold_sets[SplitPartNames['y_train']] = ys[dataset_index][train_indexes]

        fold_sets[SplitPartNames['X_test']] = Xs[dataset_index][train_indexes]
        fold_sets[SplitPartNames['y_test']] = ys[dataset_index][train_indexes]

        id3_fold_times.append(trees_algorithms.measure_times_of_id3(fold_sets))
        cart_fold_times.append(trees_algorithms.measure_times_of_cart(fold_sets))

        fold += 1

    id3_times.append(id3_fold_times)
    cart_times.append(cart_fold_times)

print('Algorithm\tFold\tDataset\tLearning time\tPrediction time')
for algorithm_index in range(0, len(DatasetNames)):
    for fold_index in range(0, k):
        learning, prediction = id3_times[algorithm_index][fold_index]
        print('ID3\t{}\t{}\t{}\t{}'.format(fold_index, utils.get_dataset_name(algorithm_index), learning, prediction))
        learning, prediction = cart_times[algorithm_index][fold_index]
        print('CART\t{}\t{}\t{}\t{}'.format(fold_index, utils.get_dataset_name(algorithm_index), learning, prediction))
