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
from numpy import mean, std, interp
import random
import json
import csv


def draw(y_test, y_score, labels, fold, drawingtitle):
    # Compute ROC curve and ROC area for each class
    n_classes = 11

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    lw = 2
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc["micro"]

def flat(Y):
    Y = np.asarray(Y, dtype='int8')
    newArray = np.zeros((Y.size, Y.max() + 1))

    newArray[np.arange(Y.size), Y] = 1

    return newArray


def convertLabels(labelDict, y_true, y_pred):
    y_Actual = []
    y_Predicted = []
    labelValues = list(labelDict.values())
    labelKeys = list(labelDict.keys())
    for i in range(len(y_true)):
        trueItem = y_true[i]
        predictedItem = y_pred[i]
        trueIndex = labelValues.index(trueItem)
        predictedIndex = labelValues.index(predictedItem)

        trueLabel = labelKeys[trueIndex]
        predictedLabel = labelKeys[predictedIndex]

        y_Actual.append(trueLabel)
        y_Predicted.append(predictedLabel)

    return y_Actual, y_Predicted


def confusionMatrix(labelDict, y_true, y_pred, fold):
    y_Actual, y_Predicted = convertLabels(labelDict, y_true, y_pred)

    data = {'y_Actual': y_Actual,
            'y_Predicted': y_Predicted
            }
    plt.figure(figsize=(10, 10))
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.savefig('cm/ConcatenateOuterfold' + str(fold) + '.png', dpi=200)
    # plt.show()
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

def one_hot_encoding_data(data):
    new_data0 = []
    new_data2 = []
    for index_i in range(len(data)):
        new_data0.append([])
        new_data2.append([])
        for index_k in range(len(data[index_i])):
            value = data[index_i][index_k]
            if float(value) < 0:
                x0_ = 1
                x2_ = 0
            elif float(value) == 0:
                x0_ = 0
                x2_ = 0
            else:
                x0_ = 0
                x2_ = 1
            new_data0[index_i].append(x0_)
            new_data2[index_i].append(x2_)
    return new_data0, new_data2


def all_values_to_ones(X):
    X[X==2] = 1
    X[X==-2] = -1

def treeRuleEncoder(val):
    if val > 0:
        return 0
    elif val < 0:
        return 1
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
    return encodedTree

    with open('encodedTree.txt', 'w') as fileHandle:
        json.dump(encodedTree, fileHandle)

X = np.load('concatData/xConcat.npy')  #loads your saved array into variable a.
y = np.load('concatData/yConcat.npy')  #loads your saved array into variable a.
all_values_to_ones(X)


labelDict = label2dict()


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

def compare(X0, X2, Y):
    return ((1 - int(X0)) * (int(X2) * (int(Y) - 1) - int(Y))) + 1

tree = 100
depth = 2

print("===========Start XGBOOST Classifier Section===========")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgbModel = XGBClassifier(n_estimators=tree, max_depth=depth, objective='multi:softprob')


#df_model = pd.read_csv('model.csv')
model = xgbModel.fit(X_train, y_train)
df_model = model.get_booster().trees_to_dataframe()
df_model.to_csv('model.csv', index=False)
y_true, y_pred = y_test, xgbModel.predict(X_test)
# print(classification_report(y_true, y_pred))
yy_true, yy_pred = y_test, xgbModel.predict_proba(X_test)

with open('predictions.npy', 'wb') as f:
    np.save(f, y_pred)

with open('labels.npy', 'wb') as f:
    np.save(f, yy_true)

#Encode Test Data
x0, x2 = one_hot_encoding_data(X_test)
with open('testDatax2.txt', 'w', newline="") as f:
    json.dump(x2,f)
with open('testDatax0.txt', 'w', newline="") as f:
    json.dump(x0,f)

#Feature Tree of the Model
featureTree = modelFeatures(df_model)


#Encode Model
counter = 0
encodedModel = encodeModel(df_model)


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

maxx = []
for x0_, x2_, y in zip(x0_ordered,x2_ordered, y_test):
    scores = [0]*11
    for i in range(0, 11):
        for j in range(i*100, (i+1)*100):
            z1 = compare(x0_[j][0], x2_[j][0], encodedModel[j][0])
            z2 = compare(x0_[j][1], x2_[j][1], encodedModel[j][1])
            z3 = compare(x0_[j][2], x2_[j][2], encodedModel[j][2])
            f1 = z1*z2*encodedModel[j][3]
            f2 = z1*(1-z2)*encodedModel[j][4]
            f3 = (1-z1)*z3*encodedModel[j][5]
            f4 = (1-z1)*(1-z3)*encodedModel[j][6]
            scores[i] = scores[i] + f2 + f1 + f3 + f4

    if (scores.index(max(scores)) == y): counter +=1
    maxx.append(scores)

drawingtitle = "tree_" + str(tree) + "_depth_" + str(depth)
auc = draw(flat(y_test), np.asarray(maxx), labels, counter, drawingtitle)
print("microauc" 
