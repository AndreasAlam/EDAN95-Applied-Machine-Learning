import ToyData as td
import ID3
import numpy as np
from sklearn import tree, metrics, datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def main():
    # # First
    # attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
    # id3 = ID3.ID3DecisionTreeClassifier(attributes)

    # myTree = id3.fit(data, target, attributes, classes)
    # plot = id3.make_dot_data()
    # plot.render("testTree")
    # predicted = id3.predict(data2, myTree, attributes)
    # print(predicted)

    # Second
    digits = datasets.load_digits()
    num_examples = len(digits.data)
    num_split = int(0.7*num_examples)
    train_features = digits.data[:num_split]
    train_labels = digits.target[:num_split]
    test_features = digits.data[num_split:]
    test_labels = digits.target[num_split:]
    classes = list(range(10)) # 0-9
    attributes = {str(i):list(range(17)) for i in range(64)} #64 pixels with blackness 0-16
    id3 = ID3.ID3DecisionTreeClassifier(attributes)

    myTree = id3.fit(train_features, list(train_labels), attributes, classes)
    plot = id3.make_dot_data()
    plot.render("testTree")
    predicted = id3.predict(test_features, myTree)
    print("Classification report for classifier %s:\n%s\n"
      % (myTree, metrics.classification_report(test_labels, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted))

    # Third
    def comp(vec):
        vec_new = []
        if len(vec) == 1: vec = [vec]
        for x in vec:
            if x<=5: vec_new.append(0)
            if x<=10: vec_new.append(1)
            else: vec_new.append(2)
        return vec_new

    train_features_comp = [comp(i) for i in train_features]
    train_labels_comp = list(train_labels)
    test_features_comp = [comp(i) for i in test_features]
    test_labels_comp = list(test_labels)
    classes = list(range(10)) # 0-9
    attributes = {str(i):list(range(3)) for i in range(64)} #64 pixels with blackness 0-16

    id3_dig_comp = ID3.ID3DecisionTreeClassifier(attributes)
    myTree_dig_comp = id3_dig_comp.fit(train_features_comp, train_labels_comp, attributes, classes)
    predicted = id3_dig_comp.predict(test_features_comp, myTree_dig_comp)

    print("Classification report for classifier %s:\n%s\n"
      % (myTree_dig_comp, metrics.classification_report(test_labels_comp, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels_comp, predicted))


if __name__ == "__main__": main()
