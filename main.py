
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from scipy import interpolate

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import time
from xgboost import XGBClassifier
from numpy import mean, std
import random
import json
import csv

def label2dict():
    myDict = {
        'Bronchusandlung': 0,

        'Bladder': 1,

        'Colon': 2,

        'Skin': 3,

        'Stomach': 4,

        'Corpusuteri': 5,

        'Liverandintrahepaticbileducts': 6,

        'Ovary': 7,

        'Kidney': 8,

        'Cervixuteri': 9,

        'Breast': 10}

    return myDict


# def draw(y_test, y_score, labels, fold, drawingtitle):
#     # Compute ROC curve and ROC area for each class
#     n_classes = 11
#
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     # Compute micro-average ROC curve and ROC area
#     fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#     roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#     lw = 2
#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += interpolate(all_fpr, fpr[i], tpr[i])
#
#     # Finally average it and compute AUC
#     mean_tpr /= n_classes
#
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    # plt.rc('font', family='serif')
    # plt.rc('xtick', labelsize='large')
    # plt.rc('ytick', labelsize='large')
    # plt.figure(figsize=(10, 12))
    #
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)
    #
    # colors = cycle(
    #     ['green', 'gold', 'chartreuse', 'mediumvioletred', 'rebeccapurple', 'darkorange', 'cadetblue', 'sienna', 'red',
    #      'cyan', 'blue'])
    #
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(labels[i], roc_auc[i]))
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate', fontsize='large')
    # plt.ylabel('True Positive Rate', fontsize='large')
    # plt.title('Some extension of Receiver operating characteristic to multi-class', fontsize='large')
    # plt.legend(loc="lower right")
    #
    # plt.savefig('reports/' + drawingtitle + '_f_' + str(fold) + '.png', dpi=200)

    # return roc_auc["micro"]
    # plt.show()


# def flat(Y):
#     Y = np.asarray(Y, dtype='int8')
#     newArray = np.zeros((Y.size, Y.max() + 1))
#
#     newArray[np.arange(Y.size), Y] = 1
#
#     return newArray


# def convertLabels(labelDict, y_true, y_pred):
#     y_Actual = []
#     y_Predicted = []
#     labelValues = list(labelDict.values())
#     labelKeys = list(labelDict.keys())
#     for i in range(len(y_true)):
#         trueItem = y_true[i]
#         predictedItem = y_pred[i]
#         trueIndex = labelValues.index(trueItem)
#         predictedIndex = labelValues.index(predictedItem)
#
#         trueLabel = labelKeys[trueIndex]
#         predictedLabel = labelKeys[predictedIndex]
#
#         y_Actual.append(trueLabel)
#         y_Predicted.append(predictedLabel)
#
#     return y_Actual, y_Predicted
#
#
# def confusionMatrix(labelDict, y_true, y_pred, fold):
#     y_Actual, y_Predicted = convertLabels(labelDict, y_true, y_pred)
#
#     data = {'y_Actual': y_Actual,
#             'y_Predicted': y_Predicted
#             }
#     plt.figure(figsize=(10, 10))
#     df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
#     confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
#     sn.heatmap(confusion_matrix, annot=True)
#     plt.savefig('cm/ConcatenateOuterfold' + str(fold) + '.png', dpi=200)
#     # plt.show()

def one_hot_encoding_data(data):
    new_data0 = []
    new_data2 = []
    for index_i in range(len(data)):
        new_data0.append([])
        new_data2.append([])
        for index_k in range(len(data[index_i])):
            value = data[index_i][index_k]
            if float(value) < 0:
                x0_ = "1"
                x2_ = "0"
            elif float(value) == 0:
                x0_ = "0"
                x2_ = "0"
            else:
                x0_ = "0"
                x2_ = "1"
            new_data0[index_i].append(x0_)
            new_data2[index_i].append(x2_)
    return new_data0, new_data2


def all_values_to_ones(X):
    X[X==2] = 1
    X[X==-2] = -1

def treeRuleEncoder(val):
    if val > 0:
        return 1
    elif val < 0:
        return 0
    else:
        return ""

def modelFeatures(df_mdl):
    numberOfClasses = 11
    numberOfTrees = 100

    featureTree = []

    for classIndex in range(numberOfClasses):
        for treeIndex in range(classIndex, numberOfClasses * numberOfTrees, numberOfClasses):
            toBeAdded = []

            my_id = str(treeIndex) + "-0"
            curr = df_mdl[df_mdl['ID'] == my_id].values[0]

            toBeAdded.append(int(curr[3][1:]))
            # first branching
            firstLeftChild = df_mdl[df_mdl['ID'] == curr[5]].values[0]
            firstRightChild = df_mdl[df_mdl['ID'] == curr[6]].values[0]

            if firstLeftChild[3] == "Leaf":
                randomRule = 25428
                toBeAdded.append(randomRule)
            else:
                toBeAdded.append(int(firstLeftChild[3][1:]))

            if firstRightChild[3] == "Leaf":
                randomRule = 25428
                toBeAdded.append(randomRule)
            else:
                toBeAdded.append(int(firstRightChild[3][1:]))
            featureTree.append(toBeAdded)

    with open('featureTree.txt', 'w') as fileHandle:
        json.dump(featureTree, fileHandle)
    return featureTree

def encodeModel(df_mdl):
    numberOfClasses = 11
    numberOfTrees = 100

    encodedTree = []

    for classIndex in range(numberOfClasses):
        for treeIndex in range(classIndex, numberOfClasses * numberOfTrees, numberOfClasses):
            toBeAdded = []

            my_id = str(treeIndex) + "-0"
            curr = df_mdl[df_mdl['ID'] == my_id].values[0]

            toBeAdded.append(treeRuleEncoder(curr[4]))
            # first branching
            firstLeftChild = df_mdl[df_mdl['ID'] == curr[5]].values[0]
            firstRightChild = df_mdl[df_mdl['ID'] == curr[6]].values[0]
            # second branching
            if firstLeftChild[3] == "Leaf":
                secondLeftLeftChild = firstLeftChild[8]
                secondLeftRightChild = firstLeftChild[8]

                randomRule = random.randint(0, 1)
                toBeAdded.append(randomRule)
            else:
                secondLeftLeftChild = df_mdl[df_mdl['ID'] == firstLeftChild[5]].values[0][8]
                secondLeftRightChild = df_mdl[df_mdl['ID'] == firstLeftChild[6]].values[0][8]

                toBeAdded.append(treeRuleEncoder(firstLeftChild[4]))

            if firstRightChild[3] == "Leaf":
                secondRightLeftChild = firstRightChild[8]
                secondRightRightChild = firstRightChild[8]

                randomRule = random.randint(0, 1)
                toBeAdded.append(randomRule)

            else:
                secondRightLeftChild = df_mdl[df_mdl['ID'] == firstRightChild[5]].values[0][8]
                secondRightRightChild = df_mdl[df_mdl['ID'] == firstRightChild[6]].values[0][8]

                toBeAdded.append(treeRuleEncoder(firstRightChild[4]))

            toBeAdded.append(secondLeftLeftChild)
            toBeAdded.append(secondLeftRightChild)
            toBeAdded.append(secondRightLeftChild)
            toBeAdded.append(secondRightRightChild)
            encodedTree.append(toBeAdded)

    with open('encodedTree.txt', 'w') as fileHandle:
        json.dump(encodedTree, fileHandle)

X = np.load('concatData/xConcat.npy')  #loads your saved array into variable a.
y = np.load('concatData/yConcat.npy')  #loads your saved array into variable a.

labelDict = label2dict()

print("Feature selection is starting...")
print("We have that number of features : ", X.shape[1])

labels = ['Bladder',
          'Breast',
          'Bronchusandlung',
          'Cervixuteri',
          'Colon',
          'Corpusuteri',
          'Kidney',
          'Liverandintrahepaticbileducts',
          'Ovary',
          'Skin',
          'Stomach']


tree = 100
depth = 2

print("===========Start XGBOOST Classifier Section===========")
counter = 0
testTime = 0
testTimes = []
microAucs = []
auc = 0
description = ""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgbModel = XGBClassifier(n_estimators=tree, max_depth=depth, objective='multi:softprob')


#df_model = pd.read_csv('model.csv')
model = xgbModel.fit(X_train, y_train)
df_model = model.get_booster().trees_to_dataframe()
df_model.to_csv('model.csv', index=False)
y_true, y_pred = y_test, xgbModel.predict(X_test)
print(classification_report(y_true, y_pred))
yy_true, yy_pred = y_test, xgbModel.predict_proba(X_test)

with open('predictions.npy', 'wb') as f:
    np.save(f, y_pred)

with open('labels.npy', 'wb') as f:
    np.save(f, yy_true)
#Encode Model
#encodeModel(df_model)

#Feature Tree of the Model
featureTree = modelFeatures(df_model)


#Encode Test Data
x0, x2 = one_hot_encoding_data(X_test)
with open('testDatax2.txt', 'w', newline="") as f:
    json.dump(x2,f)
with open('testDatax0.txt', 'w', newline="") as f:
    json.dump(x0,f)

x0_ordered = []
x2_ordered = []
for i in range(len(X_test)):
    sample0 = []
    sample2 = []
    for tree in featureTree:
        temp0 = []
        temp2 = []
        for feat in tree:
            temp0.append(x0[i][feat])
            temp2.append(x2[i][feat])
        sample0.append(temp0)
        sample2.append(temp2)
    x0_ordered.append(sample0)
    x2_ordered.append(sample2)

with open('testDatax2_ordered.txt', 'w') as f:
    for sample in x2_ordered:
        json.dump(sample ,f)
        f.write("\n")

with open('testDatax0_ordered.txt', 'w') as f:
    for sample in x0_ordered:
        json.dump(sample ,f)
        f.write("\n")
