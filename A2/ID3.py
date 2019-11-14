from collections import Counter
from graphviz import Digraph
from scipy.stats import entropy
import numpy as np
from sklearn import tree, metrics, datasets
import operator



class ID3DecisionTreeClassifier :


    def __init__(self, attributes, minSamplesLeaf = 1, minSamplesSplit = 2) :
        self.attributes = attributes

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit


    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'label': None, 'attribute': None, 'entropy': None, 'samples': None,
                         'classCounts': None, 'nodes': []}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))


    # make the visualisation available
    def make_dot_data(self) :
        return self.__dot

    def firstI(self, target):
        I_s = 0
        for i in set(target):
            s = target.count(i)/len(target)
            if (len(target) != 0):
                I_s += s*np.log2(s)
        return -I_s

    def classCount(self, target):
        count = {}
        for i in set(target):
            c = target.count(i)
            count[i] = c
        return count


    # Function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, data, target, attributes):
        I = []
        for i, ai in enumerate(attributes.keys()):
            temp = 0
            lst1 = list(elm[i] for elm in data)
            lst2 = list(zip(lst1, target))
            for att_val in attributes[ai]:
                I_initial = 0
                tot_state = lst1.count(att_val)
                for j in set(target):
                    s = lst2.count(tuple([att_val, j]))
                    if (tot_state != 0):
                        s1 = s/tot_state
                        I_initial += s1*np.log2(s1)
                I_initial *= tot_state/len(data)
                temp += I_initial
            I.append(-temp)

        index = np.argmin(I)
        attribute = None
        entropy = self.firstI(target)

        for i, ai in enumerate(attributes.keys()):
            if i == index:
                attribute = ai

        return attribute, entropy #returns key and corresponding entroopy

    # The entry point for the recursive ID3-algorithm
    def fit(self, data, target, attributes, classes):
        target_attribute = None
        root = self.ID3(data, target_attribute, attributes, target)
        self.add_node_to_graph(root)

        return root

    # ID3 algorithm
    def ID3(self, samples, target_attribute, attributes2, target_modified):
        att_temp = attributes2.copy()
        node = self.new_ID3_node()
        c = self.classCount(target_modified)

        if len(set(target_modified)) == 1:
            node['label'] = list(set(target_modified))[0]
            node['samples'] = len(samples)
            node['classCounts'] = c
            return node

        if not attributes2:
            c = self.classCount(list(target_modified))
            node["label"] = max(c.items(), key = operator.itemgetter(1))[0]
            node["entropy"] = self.firstI(target_modified)
            node["samples"] = len(samples)
            node['classCounts'] = c
            return node
        else:
            A, entropy = self.find_split_attr(samples, target_modified, attributes2)
            att_temp.pop(A, None)
            node["attribute"] = A
            node["entropy"] = entropy
            node["samples"] = len(samples)
            node['classCounts'] = c

            for v in (self.attributes[A]):
                sample_vi = [samples[i] for i, x in enumerate(samples) if (v == x[list(self.attributes.keys()).index(A)])]
                target_sample = [target_modified[i] for i, x in enumerate(samples) if (v == x[list(self.attributes.keys()).index(A)])]

                if len(sample_vi) == 0:
                    sub_node = self.new_ID3_node()
                    c = self.classCount(target_modified)
                    sub_node["label"] = max(c.items(), key = operator.itemgetter(1))[0]
                    sub_node["samples"] = 0

                else:
                    sub_node = self.ID3(sample_vi, A, att_temp, target_sample)
                self.add_node_to_graph(sub_node, node['id'])
                node['nodes'].append(sub_node)

        return node

    # Predicter
    def predict(self, data, tree):
        predicted = list()
        for dat in data:
            next_node = tree
            while next_node['label'] == None:
                att = next_node['attribute']
                att_index = list(self.attributes.keys()).index(att)
                value_index = self.attributes[att].index(dat[att_index])
                next_node = next_node['nodes'][value_index]
            predicted.append(next_node['label'])

        return predicted
