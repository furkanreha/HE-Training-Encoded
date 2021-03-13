from xgboost import XGBClassifier
from sklearn.utils import shuffle
import numpy as np
import json
import pathlib
import pandas

pandas.set_option('display.max_columns', None)


def one_hot_encoding_data(data):
    new_data = []
    for index_i in range(len(data)):
        new_data.append([])
        for index_k in range(len(data[index_i])):
            value = data[index_i][index_k]
            if float(value) < 0:
                encoded_string = "b001"
            elif float(value) == 0:
                encoded_string = "b010"
            else:
                encoded_string = "b100"

            new_data[index_i].append(encoded_string)
    return new_data


def all_values_to_ones(data):
    for index_i in range(len(data)):
        for index_k in range(len(data[index_i])):
            if abs(data[index_i][index_k]) != 1:
                data[index_i][index_k] *= 2


tree = 100
depth = 2

currentDirectory = str(pathlib.Path().absolute())

X = np.load(currentDirectory + '/ConcatData/xScaledConcat.npy')
y = np.load(currentDirectory + '/ConcatData/yConcat.npy')

X, y = shuffle(X, y, random_state=5)

testX = X[45:90]
all_values_to_ones(testX)
testY = y[45:90]

X = X[0:45]  # Smaller size for test run
all_values_to_ones(X)
y = y[0:45]  # Smaller size for test run

if len(set(y)) == 11:
    xgbModel = XGBClassifier(n_estimators=tree, max_depth=depth, objective='multi:softprob', eval_metric='merror',
                             use_label_encoder=False)
    model = xgbModel.fit(X, y)
    predictions = model.predict(testX)

    testX_encoded = one_hot_encoding_data(testX)

    with open('testX.txt', 'w') as file_handle:
        json.dump(testX_encoded, file_handle)

    with open('predictions.npy', 'wb') as f:
        np.save(f, predictions)

    df_trees = model.get_booster().trees_to_dataframe()
    df_trees["Split-Encoded"] = ["None"] * df_trees.shape[0]

    for i, row in df_trees.iterrows():
        nodeValue = row["Split"]
        if nodeValue == 0.5:
            encoding = "b100"
        elif nodeValue == 0.0:
            encoding = "b010"
        elif nodeValue == -0.5:
            encoding = "b001"
        else:
            encoding = nodeValue
        df_trees.at[i, "Split-Encoded"] = encoding

    df_trees.to_csv('model.csv')
