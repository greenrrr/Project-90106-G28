from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

Numeric_features = ['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Converted']


def set_missing_time(filename):
    data = pd.read_csv(filename)
    x = data.drop("Prospect ID", axis=1)  # 36 features 9240 rows
    x = x.drop('Lead Number', axis=1)
    x = x.set_index('ID')
    index = list(x.index)
    features = list(x.columns)

    raw_no_missing = x.dropna(how="any", axis=0)  # remove rows which have missing values 1943 rows
    no_missing_index = list(raw_no_missing.index)  # 没有任何缺失值的行索引

    raw_missing = x
    for i in index:
        if i in no_missing_index:
            raw_missing = raw_missing.drop(axis=0, index=i)  # 7297 rows 含有缺失值的行
    missing_index = list(raw_missing.index)  # 含有缺失值的行索引

    # remove the features which have missing values--leaft 19 features
    missing = raw_missing.dropna(how='any', axis=1)
    no_missing_features = list(missing.columns)

    missing_features = [i for i in features if i not in no_missing_features]
    no_missing = raw_no_missing.drop(missing_features, axis=1)

    return raw_no_missing,raw_missing,no_missing,missing,missing_features,x,features,missing_index

def each_missing_feature(raw_missing,feature):
    missing = raw_missing[feature]
    temp = missing.isnull()
    temp_list = temp[temp.values == True]
    index = list(temp_list.index)
    return index

def estimate(no_missing,y_no_missing,missing,missing_features,raw_missing,raw_set):
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    i = 0
    for label in y_no_missing:
        c = 0
        feature = missing_features[i]
        index = each_missing_feature(raw_missing, feature)
        each_missing = missing.loc[index]
        rfr.fit(no_missing, label)
        predicted = rfr.predict(each_missing)
        i += 1
        for pre in predicted:
            indexs = index[c]
            raw_set.loc[indexs, feature] = pre
            c += 1
    y = raw_set['Converted']
    raw_set = raw_set.drop("Converted",axis=1)
    return raw_set,y

def encoding(data):
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

def sub_data(raw_data,predictions,missing_index):
    c = 0
    for item in missing_index:
        raw_data.loc[item, 'Total Time Spent on Website'] = predictions[c]
        c += 1
    raw_data = encoding(raw_data)
    y = raw_data['Converted']
    raw_data = raw_data.drop("Converted",axis=1)
    return raw_data,y

def get_label(features,raw_no_missing):
    y = raw_no_missing[features]
    return y

raw_no_missing,raw_missing,no_missing,missing,missing_features,raw_set,features,missing_index = set_missing_time("Leads.csv")
raw_no_missing_encode = encoding(raw_no_missing)
no_missing_encode = encoding(no_missing)
missing_encode = encoding(missing)

labels_list = [get_label(feature,raw_no_missing_encode) for feature in missing_features]
x,y = estimate(no_missing_encode,labels_list,missing_encode,missing_features,raw_missing,raw_set)
X_data = encoding(x)
X_train,X_test,y_train,y_test = train_test_split(X_data,y,train_size=0.8,random_state=1)

