from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#import Data
#import Missing_value_test as missing
#import KNN
#import Kmeans_missing as km
#import Missing_real as mr
import Origin_randomforset as orf


def logit(train,label_train,test):
    clf = LogisticRegression(max_iter=10000)
    clf.fit(train,label_train)
    y = clf.predict(test)
    return y

def Svm(train,label_train,test):
    clf = svm.LinearSVC(max_iter=10000)# it always can not converge
    clf.fit(train, label_train)
    y = clf.predict(test)
    return y

def rf(train,label_train,test):
    clf = RandomForestClassifier()
    clf.fit(train, label_train)
    y = clf.predict(test)
    return y

def scoring(predictions,label_test):
    print('Precision:'+'\t')
    print(precision_score(predictions,label_test))
    print('Recall:'+'\t')
    print(recall_score(predictions,label_test))
    print('F1 score:'+'\t')
    print(f1_score(predictions,label_test))


if __name__ == '__main__':
    ## Do not remove any features
    #y_logit = logit(Data.X_train,Data.y_train,Data.X_test)
    #y_svm = Svm(Data.X_train,Data.y_train,Data.X_test)
    #y_rf = rf(Data.X_train,Data.y_train,Data.X_test)

    ## Remove low score featires
    #y_logit = logit(Data.processed_train,Data.y_train,Data.processed_test)
    #y_svm = Svm(Data.processed_train,Data.y_train,Data.processed_test)
    #y_rf = rf(Data.processed_train,Data.y_train,Data.processed_test)

    ## Missing estimate set(small misiing)
    #y_logit = logit(missing.X_train, missing.y_train, missing.X_test)
    #y_svm = Svm(missing.X_train, missing.y_train, missing.X_test)
    #y_rf = rf(missing.X_train, missing.y_train, missing.X_test)

    ## KNN missing value estimate(large missing)
    #y_logit = logit(km.X_train, km.y_train, km.X_test)
    #y_svm = Svm(km.X_train, km.y_train, km.X_test)
    #y_rf = rf(km.X_train, km.y_train, km.X_test)

    ## Leads missing value
    #y_logit = logit(mr.X_train, mr.y_train, mr.X_test)
    #y_svm = Svm(mr.X_train, mr.y_train, mr.X_test)
    #y_rf = rf(mr.X_train, mr.y_train, mr.X_test)

    ## Leads missing rf
    y_logit = logit(orf.X_train, orf.y_train, orf.X_test)
    y_svm = Svm(orf.X_train, orf.y_train, orf.X_test)
    y_rf = rf(orf.X_train, orf.y_train, orf.X_test)

    print('Logisticregression:')
    scoring(y_logit,orf.y_test)
    print('---------------------------------------')
    print('SVM:')
    scoring(y_svm,orf.y_test)
    print('---------------------------------------')
    print('RandomForest:')
    scoring(y_rf,orf.y_test)
