import numpy as np
import pathlib
import json


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


currentDirectory = str(pathlib.Path().absolute())

X = np.load(currentDirectory + '/ConcatData/xScaledConcat.npy')
y = np.load(currentDirectory + '/ConcatData/yConcat.npy')

x_encoded = one_hot_encoding_data(X)
with open('xScaledConcatEncoded.txt', 'w') as file_handle:
    json.dump(x_encoded, file_handle)




