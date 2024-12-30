COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'
]

CATEGORICAL_COLUMNS = [
    'workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 
    'education', 
]

TRAIN_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
TEST_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

BACKUP_URL = 'https://archive.ics.uci.edu/static/public/2/adult.zip'
