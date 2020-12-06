#
#from parity.fairness_metrics import show_bias
#
#priv_category = 'Race-White'
#priv_value = 'True'
#target_label = 'high pay'
#unencoded_target_label = 'True'
#cols_to_drop = ''
#show_bias(data, priv_category, priv_value, target_label, unencoded_target_label, cols_to_dop)


from parity.pair import *
import pandas as pd

### Insert more code here

# Load up the test dataset
test_csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
test_df = pd.read_csv(test_csv_path, skipinitialspace=True,
  skiprows=1)

num_datapoints = 2000
tool_height_in_px = 1000

make_label_column_numeric(test_df, label_column, lambda val: val == '>50K.')
test_examples = df_to_examples(test_df[0:num_datapoints])
labels = ['Under 50K', 'Over 50K']

# Setup the tool with the test examples and the trained classifier
config_builder = config(test_examples, classifier, labels, feature_spec)

## Visualization only appears through jupyter notebook (not jupyter lab)
WitWidget(config_builder, height=tool_height_in_px)
