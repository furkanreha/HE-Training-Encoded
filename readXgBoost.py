import json
import numpy as np
import pandas
import time

pandas.set_option('display.max_columns', None)


def bitwise_is_smaller(attribute_value, tree_node_value):  # 100 --> 0.5 001 --> -0.5
    if tree_node_value[1] == "1":
        return attribute_value[1] == "0"
    else:
        return attribute_value[3] == "1"


def score_calc_class(class_no, sample):
    i = class_no
    gain = 0

    for s in range(i, numberOfClasses * numberOfTrees, numberOfClasses):
        my_id = str(s) + "-0"
        curr = df_mdl[df_mdl['ID'] == my_id].values[0]
        while curr[3] != 'Leaf':
            feat = int(curr[3][1:])
            if bitwise_is_smaller(test_X[sample][feat], curr[10]):
                my_id = curr[5]
            else:
                my_id = curr[6]
            curr = df_mdl[df_mdl['ID'] == my_id].values[0]
        gain = gain + curr[-3]
    return gain


def predict_samples():
    final_predictions_calculated = []
    for i in range(len(test_X)):
        sample = i
        score_list = []
        for k in range(numberOfClasses):
            score_list.append(score_calc_class(k, sample))
        final_predictions_calculated.append(score_list.index(max(score_list)))
    return final_predictions_calculated


with open("testX.txt", "r") as f:
    test_X = json.load(f)

df_mdl = pandas.read_csv('model.csv', index_col=0)

predictions = np.load('predictions.npy')

print("Non-leaf nodes in XG Boost Trees don't have the value 010(0)")
print("Encoding: (001 means <0, 100 means >0)")
print(50 * "*")
numberOfClasses = 11
numberOfTrees = 100

start_time = time.time()
predict_samples()
print("--- %s seconds ---" % (time.time() - start_time))



