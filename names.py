DatasetUrls = ['https://raw.githubusercontent.com/w4k2/data/master/datasets/iris.csv',
               'https://raw.githubusercontent.com/w4k2/data/master/datasets/diabetes.csv',
               'https://raw.githubusercontent.com/w4k2/data/master/datasets/ionosphere.csv',
               'https://raw.githubusercontent.com/w4k2/data/master/datasets/digit.csv',
               'https://raw.githubusercontent.com/w4k2/data/master/datasets/waveform.csv'
               ]

DatasetNames = {}


def set_names():
    i = 0
    for url in DatasetUrls:
        name = url.split('/')[-1].split('.')[0]
        DatasetNames[name] = i
        i += 1


SplitPartNames = {'X_train': 0,
                  'X_test': 1,
                  'y_train': 2,
                  'y_test': 3
                  }
