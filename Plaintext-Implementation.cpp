#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <sstream>
#include <vector>
using namespace std;


void cleanInput(string & input) {
	input.erase(remove(input.begin(), input.end(), '['), input.end());
	input.erase(remove(input.begin(), input.end(), ']'), input.end());
	input.erase(remove(input.begin(), input.end(), '\"'), input.end());
	input.erase(remove(input.begin(), input.end(), ','), input.end());
}

void readTrees(vector<vector<double>> & trees, string nameOfTheFile) {
	ifstream file(nameOfTheFile);
	if (file.is_open())
	{
		string input;
		int totalNodes = 0, currentTreeIndex = 0;

		while (file >> input)
		{
			if (totalNodes % 7 == 0) {
				trees.push_back(vector<double>());
			}
			currentTreeIndex = int(totalNodes / 7);
			cleanInput(input);
			trees[currentTreeIndex].push_back(stof(input));
			totalNodes += 1;
		}
	}
}

template <typename T>
void printVectors(vector<vector<T>> & trees, string name) {
	string value;
	for (int i = 0; i < trees.size(); i++) {
		cout << name + ":" << i << endl;
		value = "";
		for (int j = 0; j < trees[i].size(); j++) {
			value += to_string(trees[i][j]) + " ";
		}
		cout << value.substr(0, value.length() - 1) << endl;
	}
}
 
template <typename T>
void printVector(vector<T> & arr, string name) {
	cout << name + ":" << i << endl;
	for (int i = 0; i < arr.size(); i++) {
		cout << arr[i] << " ";
	}
	cout << endl;
}

void readRowTestData(vector<vector<int>> & data, string nameOfTheFile) {
	ifstream file(nameOfTheFile);
	if (file.is_open()) {
		string line;
		int readerInt;
		while (getline(file, line)) {
			vector<int> currentTestData;

			istringstream iss(line);
			while (iss >> readerInt) {
				currentTestData.push_back(readerInt);
			}

			data.push_back(currentTestData);
		}
	}
}

void readLabels(vector<int> & labes, string nameOfTheFile) {
	ifstream file(nameOfTheFile);
	if (file.is_open()) {
		
		string line;
		int readerInt;
		
		getline(file, line);
		
		istringstream iss(line);
		
		while (iss >> readerInt) {
			labes.push_back(readerInt);
		}
	}
}

double firstWay(double x0, double x2, double y) {
	return (1 - x2) * (y * (x0 - 1)) + 1;
}

double secondWay(double x0, double x2, double y) {
	return (1 - y - x0) * (x2 * (y - 1) - y) + 1;
} 

double thirdWay(double x0, double x2, double y) {
	return (1 - x0) * (x2 * (y - 1) - y) + 1;
}


double calculateOneTree(double z1, double z2, double z3, double c1, double c2, double c3, double c4) {
	double f1 = z1 * z2 * c1;
	double f2 = z1 * (1 - z2) * c2;
	double f3 = (1 - z1) * z3 * c3;
	double f4 = (1 - z1) * (1 - z3) * c4;
	return f1 + f2 + f3 + f4;
}

int findMax(vector<double> & scores, int size) {
	double currentMax = scores[0], currentScore;
	int currentMaxIndex = 0;
	for (int i = 1; i < size; i++) {
		currentScore = scores[i];
		if (currentScore > currentMax) {
			currentMax = currentScore;
			currentMaxIndex = i;
		}
	}
	return currentMaxIndex;
}

int calculateOneSample(vector<vector<double>> & trees, int treeSizePerClass, int numberOfClasses, vector<int> & testDataX0, vector<int> & testDataX2) {
	vector<double> scores;
	double z1, z2, z3, c1, c2, c3, c4, x0, x2, y, classScore, treeScore;
	int currentTreeIndex = 0, currentLabelIndex = 0;
	for (int i = 0; i < numberOfClasses; i++) {
		classScore = 0;
		for (int k = 0; k < treeSizePerClass; k++) {

			x0 = testDataX0[currentLabelIndex], x2 = testDataX2[currentLabelIndex], y = trees[currentTreeIndex][0];
			z1 = thirdWay(x0, x2, y); currentLabelIndex++;

			x0 = testDataX0[currentLabelIndex], x2 = testDataX2[currentLabelIndex], y = trees[currentTreeIndex][1];
			z2 = thirdWay(x0, x2, y); currentLabelIndex++;

			x0 = testDataX0[currentLabelIndex], x2 = testDataX2[currentLabelIndex], y = trees[currentTreeIndex][2];
			z3 = thirdWay(x0, x2, y); currentLabelIndex++;

			c1 = trees[currentTreeIndex][3], c2 = trees[currentTreeIndex][4], c3 = trees[currentTreeIndex][5], c4 = trees[currentTreeIndex][6];
			
			treeScore = calculateOneTree(z1, z2, z3, c1, c2, c3, c4);
			classScore += treeScore;

			currentTreeIndex++;
		}
		scores.push_back(classScore);
	}

	return findMax(scores, numberOfClasses);
}

double evaluateAllSamples(vector<vector<double>> & trees, int treeSizePerClass, int numberOfClasses, vector<vector<int>> & testDataX0, vector<vector<int>> & testDataX2, vector<int> & labels) {
	int predictedClass, correctGuess = 0, sampleSize = labels.size(); 

	for (int labelIndex = 0; labelIndex < sampleSize; labelIndex++) {
		predictedClass = calculateOneSample(trees, treeSizePerClass, numberOfClasses, testDataX0[labelIndex], testDataX2[labelIndex]);
		if (predictedClass == (labels[labelIndex])) {
			correctGuess++;
		}
	}
	return double(correctGuess) / double(sampleSize);
}



void print(string s) {
	cout << s << endl;
}




int main() {
	vector<vector<double>> trees;
	readTrees(trees, "encodedTree.txt");

	vector<vector<int>> testDataX0;
	readRowTestData(testDataX0, "testDatax0_ordered_raw.txt");

	vector<vector<int>> testDataX2;
	readRowTestData(testDataX2, "testDatax2_ordered_raw.txt");

	vector<int> labels;
	readLabels(labels, "labels.txt");

	print("Tree and Test Data are ready to go...");

	//cout << "Correctness:" << evaluateAllSamples(trees, 100, 11, testDataX0, testDataX2, labels) << endl;


	system("pause");
	return 0;
}