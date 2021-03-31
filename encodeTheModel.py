import pandas
import random
import json


def treeRuleEncoder(val):
    if val > 0:
        return "b1"
    elif val < 0:
        return "b0"
    else:
        return ""


randomRuleArray = ["b0", "b1"]

df_mdl = pandas.read_csv('model.csv', index_col=0)

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

            randomRule = randomRuleArray[random.randint(0, 1)]
            toBeAdded.append(randomRule)
        else:
            secondLeftLeftChild = df_mdl[df_mdl['ID'] == firstLeftChild[5]].values[0][8]
            secondLeftRightChild = df_mdl[df_mdl['ID'] == firstLeftChild[6]].values[0][8]

            toBeAdded.append(treeRuleEncoder(firstLeftChild[4]))

        if firstRightChild[3] == "Leaf":
            secondRightLeftChild = firstRightChild[8]
            secondRightRightChild = firstRightChild[8]

            randomRule = randomRuleArray[random.randint(0, 1)]
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