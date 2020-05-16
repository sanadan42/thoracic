import pandas as pd
import numpy as np

from scipy.io import arff

from . import project_constants as constants

data = arff.loadarff(constants.INPUT_PATH + constants.INPUT_FILE)
df = pd.DataFrame(data[0])

thoracic = df.copy()

# convert byte columns to boolean values 
bytes_cols = ['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'Risk1Yr']
thoracic.loc[:, bytes_cols] = thoracic.loc[:, bytes_cols].apply(lambda x: x.str.decode('UTF-8').replace({'F': 0, 'T': 1}))

# convert categorical columns
cats_cols = ['DGN', 'PRE6', 'PRE14'] 
thoracic.loc[:, cats_cols] = thoracic.loc[:, cats_cols].apply(lambda x: x.str.decode('UTF-8').astype('category'))

# rename columns to provide additional meaning
col_names = {
    'DGN': 'Diagnosis',
    'PRE4': 'FVC',
    'PRE5': 'FEV1',
    'PRE6': 'Performance',
    'PRE7': 'Pain',
    'PRE8': 'Haemoptysis',
    'PRE9': 'Dyspnoea',
    'PRE10': 'Cough',
    'PRE11': 'Weakness',
    'PRE14': 'Tumour_Size',
    'PRE17': 'Type2Diabetes',
    'PRE19': 'MI',
    'PRE25': 'PAD',
    'PRE30': 'Smoking',
    'PRE32': 'Asthma',
    'AGE': 'Age',
    'Risk1Y': 'Risk1Y'
}
thoracic = thoracic.rename(columns = col_names)

thoracic.to_csv( constants.PROCESSED_PATH + constants.PROCESSED_FILE)