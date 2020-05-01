from __future__ import print_function 
import numpy as np 
import pandas as pd
from flask_restful import reqparse, Resource 
from werkzeug.datastructures import FileStorage
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import pickle
from sklearn import tree

class ID3(Resource):
    def post(self):
        # parser = reqparse.RequestParser()
        # parser.add_argument('outlook',
        #             type=str,
        #             required=True,
        #             help='CSV file')
        # parser.add_argument('temperature',
        #             type=str,
        #             required=True,
        #             help='CSV file')
        # parser.add_argument('humidity',
        #                  type=str,
        #                  required=True,
        #                  help='CSV file')
        # parser.add_argument('wind',
        #                  type=str,
        #                  required=True,
        #                  help='CSV file')
        # data = parser.parse_args()
        print("1")
        test = pd.DataFrame({'C1': 0, 
                             'C2': 4,
                             'C3': 1,
                             'C4': 1,
                             'C5': 2,
                             'C6': 1,
                             'C7': 5,
                             'C8': 1,
                             'C9': 1,
                             'C10': 4,
                             'C11': 1,
                             'C12': 3,
                             'C13': 1, 
                             'C14': 0, 
                             },
                    index=[1])
        print("2")
        filename = "id3.pkl"

        try: 
            tree = pickle.load(open(filename, 'rb'))
            print("3")
            predict = tree.predict(test)[0],
            print("pre: ", predict)
        except Exception as e:
            return {
                "msg": "bad request"
            }, 400
        return {
                "msg": str(predict),
            }, 200

class TrainID3(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('resource_csv',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='CSV file')
        
        data = parser.parse_args()
        try:
             dataToTrain = pd.read_csv(data["resource_csv"])
        except: 
            return {
                "msg": "Bad request"
            }, 400
        X = dataToTrain.iloc[:, :-1]
        
        y = dataToTrain.iloc[:, -1]
        decisionTree = tree.DecisionTreeClassifier(max_depth=None, criterion='entropy', class_weight=None)
        decisionTree = decisionTree.fit(X,y)

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        y_pred = decisionTree.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        pkl_filename = "id3.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(tree, file)
        
        return {
                "msg": "okke",
                "score": score
            }, 200



class TrainID3Test(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('resource_csv',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='CSV file')
        print("1")
        data = parser.parse_args()
        try:
             dataToTrain = pd.read_csv(data["resource_csv"])
        except: 
            return {
                "msg": "Bad request"
            }, 400
        X = dataToTrain.iloc[:, 1:-1]
        print("2")
        
        y = dataToTrain.iloc[:, -1]
        tree = DecisionTreeID3(max_depth = 3, min_samples_split = 2)
        print("3")
        tree.fit(X, y)
        print("4")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02)
        print("5")
        y_pred = tree.predict(X_test)
        print("6")
        score = accuracy_score(y_test, y_pred)
        print("7")
        pkl_filename = "id3.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(tree, file)
        print("8")
        return {
                "msg": "okke",
                "score": score
            }, 200


class TreeNode(object): 
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        self.ids = ids           # index of data in this node
        self.entropy = entropy   # entropy, will fill later
        self.depth = depth       # distance to root node
        self.split_attribute = None # which attribute is chosen, it non-leaf
        self.children = children # list of its child nodes
        self.order = None       # order of values of split_attribute in children
        self.label = None       # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    # remove prob 0 
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0*np.log(prob_0))

class DecisionTreeID3(object):
    def __init__(self, max_depth= 10, min_samples_split = 2, min_gain = 1e-4):
        self.root = None
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split 
        self.Ntrain = 0
        self.min_gain = min_gain
    
    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data 
        self.attributes = list(data)
        self.target = target 
        self.labels = target.unique()
        
        ids = range(self.Ntrain)
        self.root = TreeNode(ids = ids, entropy = self._entropy(ids), depth = 0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)
                if not node.children: #leaf node
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)
                
    def _entropy(self, ids):
        # calculate entropy of a node with index ids
        if len(ids) == 0: return 0
        ids = [i+1 for i in ids] # panda series index starts from 1
        freq = np.array(self.target[ids].value_counts())
        return entropy(freq)

    def _set_label(self, node):
        # find label for a node if it is a leaf
        # simply chose by major voting 
        target_ids = [i + 1 for i in node.ids]  # target is a series variable
        node.set_label(self.target[target_ids].mode()[0]) # most frequent label
    
    def _split(self, node):
        ids = node.ids 
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1: continue # entropy = 0
            splits = []
            for val in values: 
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id-1 for sub_id in sub_ids])
            # don't split if a node has too small number of points
            if min(map(len, splits)) < self.min_samples_split: continue
            # information gain
            HxS= 0
            for split in splits:
                HxS += len(split)*self._entropy(split)/len(ids)
            gain = node.entropy - HxS 
            if gain < self.min_gain: continue # stop if small gain 
            if gain > best_gain:
                best_gain = gain 
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids = split,
                     entropy = self._entropy(split), depth = node.depth + 1) for split in best_splits]
        return child_nodes

    def predict(self, new_data):
        """
        :param new_data: a new dataframe, each row is a datapoint
        :return: predicted labels for each row
        """
        npoints = new_data.count()[0]
        print("npoints: ", npoints)
        labels = [None]*npoints
        for n in range(npoints):
            try:
                x = new_data.iloc[n, :] # one point 
                # start from root and recursively travel if not meet a leaf 
                node = self.root
                while node.children: 
                    print("node: ", node.children)
                    node = node.children[node.order.index(x[node.split_attribute])]
                labels[n] = node.label
            except Exception as e:
                print("n: ", n)
                print("error in here: ", x)
                print("err: ", e)
                break
            
        return labels

