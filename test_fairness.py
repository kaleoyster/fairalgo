import pandas as pd
from pandas import read_csv
from parity.pair import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

# load the dataset

def load_dataset(path, column_names):
    dataframes = read_csv(path, header=None, na_values='?')
    dataframes = dataframes.dropna()
    last_ix = len(dataframes.columns) - 1
    X, y = dataframes.drop(last_ix, axis=1), dataframes[last_ix]
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    #y = LabelEncoder().fit_transform(y)
    return X, y, cat_ix, num_ix

# Load up the test dataset
train_csv_path = 'dataset/adult.data'
#test_csv_path = 'dataset/adult.test'

column_names = [
        'age',#
        'workclass',
        'fnlwgt', #
        'education',
        'education-num',#
        'martial-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain', #
        'capital-loss', #
        'hours-per-week',#
        'native-country',
        'salary'
]

column_names = [
        'age',
        'fnlwgt',
        'education-num',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'salary'
]
X, y, cat_ix, num_ix = load_dataset(train_csv_path, column_names)
#X_test, y_test, cat_ix_test, num_ix_test  = load_dataset(test_csv_path)
X = X.drop(cat_ix, axis=1)

#X_test = X_test.drop(cat_ix, axis=1)
#print(X)
#print(cat_ix)
#print(num_ix)

clf = DecisionTreeClassifier()
clf.fit(X, y)
#
#train_data = pd.read_csv('dataset/adult.data', names=column_names,
#             sep=' *, *', na_values='?')
#test_data  = pd.read_csv('dataset/adult.test', names=column_names,
#             sep=' *, *', skiprows=1, na_values='?')
#
#test_df = pd.read_csv(test_csv_path, names=column_names, skipinitialspace=True, skiprows=1)
#
## define classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

X['salary'] = y
num_datapoints = 2000
tool_height_in_px = 1000

label_column = 'salary'
make_label_column_numeric(X, label_column, lambda val: val == '>50K.')

X.columns = column_names
test_examples = df_to_examples(X[0:num_datapoints])
labels = ['Under 50K', 'Over 50K']
#
## Setup the tool with the test examples and the trained classifier
config_builder = config(test_examples, clf, labels, column_names)
#
#### Visualization only appears through jupyter notebook (not jupyter lab)
WitWidget(config_builder, height=tool_height_in_px)
