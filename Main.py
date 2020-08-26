import SVM as training
import imputation as imp
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    raw_data = imp.load_data('research_joint_data.csv')
    no_missing, missing_set, index_no_missing, index_missing, labels, names = imp.deal_data(raw_data)
    X_set, y = imp.impute(no_missing, missing_set, index_missing, labels, names, raw_data)
    X_train, X_test, y_train, y_test = train_test_split(X_set, y, train_size=0.8, random_state=1)
    p,r,f = training.decision_tree(X_train,y_train,X_test,y_test)
    print(p,r,f)
