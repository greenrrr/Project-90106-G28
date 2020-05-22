from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

Numeric_features = ['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Converted']


def set_missing_time(filename):
    data = pd.read_csv(filename)
    x = data.drop("Prospect ID",axis=1) # 36 features 9240 rows
    x = x.drop('Lead Number',axis=1)
    x = x.set_index('ID')
    features = list(x.columns)
    index = list(x.index)

    no_missing = x.dropna(how="any",axis=0) # remove rows which have missing values 1943 rows
    no_missing_index = list(no_missing.index) # 没有任何缺失值的行索引

    missing = x
    for i in index:
        if i in no_missing_index:
            missing = missing.drop(axis=0,index=i) # 7297 rows 含有缺失值的行
    missing_index = list(missing.index) #含有缺失值的行索引

    # remove the features which have missing values--leaft 19 features
    missing_1 = missing.dropna(how='any',axis = 1)
    no_missing_features = list(missing_1.columns)

    missing_features = [i for i in features if i not in no_missing_features]
    no_missing_1 = no_missing.drop(missing_features,axis=1)

    return no_missing_1,missing_1,missing_features,features,no_missing_index,missing_index,no_missing,missing,x

def estimate_kmenas(no_missing,missing,missing_features,raw_set):
    clf = KMeans()
    clf.fit(no_missing)
    predicted = clf.predict(missing)
    labels = clf.labels_
    sums = defaultdict(int)
    count = defaultdict(int)
    means = defaultdict(int)
    for feature in features:
        temp = defaultdict(int)
        temp_count =defaultdict(int)
        i = 0
        for label in labels:
            index = list(raw_set.index)
            temp[label] += raw_set.loc[index[i],feature]
            temp_count[label] += 1
            i += 1
        sums[feature] = temp # 每一类中含缺失值的features的没有缺失的值的总合
        count[feature] = temp_count
    for feature in missing_features:
        temp_mean = defaultdict(int)
        s = sums[feature]
        c = count[feature]
        for key in s.keys():
            temp_mean[key] = s[key]/c[key]
        means[feature] = temp_mean
    return means,predicted

def encoding(data,missing_features,features):
    if len(data.columns) < 19:
        for feature in features:
            if feature not in missing_features:
                if feature not in Numeric_features:
                    data[feature] = pd.factorize(data[feature])[0].astype(np.uint16)
    else:
        for feature in features:
            if feature not in Numeric_features:
                data[feature] = pd.factorize(data[feature])[0].astype(np.uint16)
    return data

def sub_data(raw_set,predictions,means,missing_index):
    missing_features_index = []
    for i in range(len(missing_index)):
        row = raw_missing.iloc[[i]]
        temp = row.isnull().any()
        temp_list = temp[temp.values == True]
        index_list = list(temp_list.index)
        missing_features_index.append(index_list) # 记录了每一行有哪些features是缺失的

    for i in range(len(missing_index)):
        f = missing_features_index[i]
        predict_value = predictions[i]
        indexs = missing_index[i]
        for item in f:
            means_set = means[item]
            estimate_value = means_set[predict_value]
            raw_set.loc[indexs, item] = estimate_value
    y = raw_set["Converted"]
    raw_set = raw_set.drop("Converted",axis=1)
    return raw_set,y



no_missing,missing,missing_features,features,no_missing_index,missing_index,raw_no_mssing,raw_missing,raw_set = set_missing_time('Leads.csv')
no_missing_1 = encoding(no_missing,missing_features,features)
missing_1 = encoding(missing,missing_features,features)
raw_no_mssing_1 = encoding(raw_no_mssing,missing_features,features)

means_dict,predicted = estimate_kmenas(no_missing_1,missing_1,missing_features,raw_no_mssing_1)

X,y_label = sub_data(raw_set,predicted,means_dict,missing_index)
X_data = encoding(X,missing_features,features)
X_train,X_test,y_train,y_test = train_test_split(X_data,y_label,train_size=0.8,random_state=1)