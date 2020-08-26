import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import SVM


def Load_data():
    filepath = os.path.join(os.getcwd(),'research_1.csv')
    data = pd.read_csv(filepath,index_col = 'Id')
    data = data.dropna(how='all',axis=1)

    for index,col in data.iteritems(): #判断空值占比
        null_proportion = cal_null(col)

        if null_proportion >= 0.4:
            data = data.drop(index, axis=1)
    data = data.dropna()

    for index, col in data.iteritems():#判断是不是数字
        types = type_(col)
        if types == 'catogerical':
            if index == 'StageName':
                labels_corres = pd.factorize(data[index])[1]
                data[index] = pd.factorize(data[index])[0].astype(np.uint16)
            else:
                data[index] = pd.factorize(data[index])[0].astype(np.uint16)

    x = data
    Labels = x['StageName']
    Trains = x.drop('StageName',axis=1)
    Trains = Trains.drop('Status_Reason__c',axis = 1)
    names = Trains.columns


    return Trains,Labels,names


def cal_null(col):
    is_null = col[col.isnull()]
    percentage = len(is_null)/len(col)
    return percentage

def type_(col):
    not_null = col[col.notnull()]
    element = str(not_null[0])
    if element.isdigit():
        return 'number'
    else:
        return 'catogerical'

def random_forest(train_set,label_set,test_set,ground_truth):
    forest_clf = RandomForestRegressor()
    forest_clf.fit(train_set,label_set)
    y = forest_clf.predict(test_set)
    print(sorted(zip(map(lambda x: round(x, 4), forest_clf.feature_importances_),names),
             reverse=True))


Trains,Labels,names = Load_data()
X_train,X_test,y_train,y_test = train_test_split(Trains,Labels,train_size=0.8,random_state=1)
svm_precision,svm_recall,svm_f1 = SVM.svc(X_train,y_train,X_test,y_test)
print('svm_precision for each labels:',svm_precision,'\n')
print('svm_recall for each labels:',svm_recall,'\n')
print('svm_f1 for each labels:',svm_f1,'\n')
tree_precision,tree_recall,tree_f1 = SVM.decision_tree(X_train,y_train,X_test,y_test)
print('tree_precision for each labels:',tree_precision,'\n')
print('tree_recall for each labels:',tree_recall,'\n')
print('tree_f1 for each labels:',tree_f1,'\n')
random_forest(X_train,y_train,X_test,y_test)
