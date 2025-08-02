import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, feature=None, threshold=None, left_node=None, right_node=None, leaf=None, entropy=0, info_gain=0):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.leaf = leaf
        self.entropy = entropy
        self.info_gain = info_gain

class DecisionTreeClassifier:
    def __init__(self, max_depth : int = None, min_samples_split : int = None, min_samples_leaf : int = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def __entropy(self, y):

        if len(y) == 0:
            return 0
        
        proportion = pd.Series(y).value_counts(normalize=True)
        total_data = len(y)

        entropy = -np.sum([pi / total_data * np.log2(pi) for pi in proportion if pi > 0])
        return entropy
    
    def __information_gain(self, X, y, feature, threshold):
        entropy_before = self.__entropy(y)

        if pd.api.types.is_numeric_dtype(X[feature]):
            left_mask = X[feature] <= threshold
            right_mask = X[feature] > threshold
        else:
            left_mask = X[feature] == threshold
            right_mask = X[feature] != threshold
        
        y_left = y[left_mask]
        y_right = y[right_mask]
        total_y_left = len(y_left)
        total_y_right = len(y_right)
        total_data = len(y)

        left_entropy = self.__entropy(y_left)
        right_entropy = self.__entropy(y_right)

        weighted_entropy = ((total_y_left / total_data) * left_entropy) + ((total_y_right / total_data) * right_entropy)
        info_gain = entropy_before - weighted_entropy

        return info_gain
    
    def __find_best_splitter(self, X, y, feature):
        best_info_gain = -1
        best_threshold = None
        
        unique_datas = np.unique(X[feature])

        for threshold in unique_datas:
            info_gain = self.__information_gain(X, y, feature, threshold)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_threshold = threshold
        return best_info_gain, best_threshold

    def __find_best_split(self, X, y):
        best_info_gain = -1
        best_threshold = None
        best_feature = None

        for feature in pd.DataFrame.column(X):
            info_gain, threshold = self.__find_best_splitter(X, y, feature)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_threshold = threshold
                best_feature = feature
        return best_info_gain, best_threshold, best_feature
    
    def __build_tree(self, X, y, depth=0):
        if self.__entropy(y) == 0 or len(y) < self.min_samples_split or len(y) < self.min_samples_leaf or depth >= self.max_depth:
            count_label = np.bincount(y)
            majority_label = np.argmax(count_label)
            leaf_node = majority_label
            return leaf_node
        else:
            feature, threshold = self.__find_best_split(X, y)

            if pd.api.types.is_numeric_dtype(X[feature]):
                left_mask = X[feature] <= threshold
                right_mask = X[feature] > threshold
            else:
                left_mask = X[feature] == threshold
                right_mask = X[feature] != threshold
            
            X_left, y_left = X[left_mask], y[left_mask]
            X_right, y_right = X[right_mask], y[right_mask]

            left_node = self.__build_tree(X_left, y_left, depth + 1)
            right_node = self.__build_tree(X_right, y_right, depth + 1)

        return TreeNode(feature, threshold, left_node, right_node)

    def fit(self, X, y):
        self.root = self.__build_tree(X, y, depth=0)

    def __predict_one(self, x):
        node = self.root
        if node.leaf is None:
            if x[node.feature] <= node.threshold:
                node.left_node
            else:
                node.right_node
        return node.leaf

    def predict(self, X):
        return [self.__predict_one(x) for x in X]

# DONE
