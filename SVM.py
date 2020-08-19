from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier


def svc(train_set,label_set,test_set,ground_truth):
    svm_clf = LinearSVC(max_iter=10000)
    svm_clf.fit(train_set, label_set)
    y = svm_clf.predict(test_set)
    p = precision_score(y,ground_truth,average= None)
    r = recall_score(y,ground_truth,average=None)
    f = f1_score(y,ground_truth,average=None)

    return p,r,f

def decision_tree(train_set,label_set,test_set,ground_truth):
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(train_set,label_set)
    y = tree_clf.predict(test_set)
    p = precision_score(y, ground_truth, average=None)
    r = recall_score(y, ground_truth, average=None)
    f = f1_score(y, ground_truth, average=None)

    return p,r,f