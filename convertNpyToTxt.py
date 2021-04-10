
import numpy as np
import pathlib

currentDirectory = str(pathlib.Path().absolute())
labels = np.load(currentDirectory + '/labels.npy')

string = ""
for label in labels:
    string += str(label) + " "

string = string[0:-1]
text_file = open("labels.txt", "w")
text_file.write(string)
text_file.close()