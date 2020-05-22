from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def set_missing_time(filename,raw_file):
    data = pd.read_csv(filename)
    raw = pd.read_csv(raw_file)
    raw = raw.drop("row_number",axis=1)
    raw = raw.set_index("Prospect ID")
    x = data.drop("row_number", axis=1)
    x = x.set_index("Prospect ID")
    no_missing = x
    missing = x
    no_missing = no_missing.dropna(subset=["Total Time Spent on Website"],how="any",axis=0)
    missing = missing[missing.isnull().values]
    y_no_missing = no_missing["Total Time Spent on Website"]

    no_missing = no_missing.drop("Total Time Spent on Website",axis=1)
    missing = missing.drop("Total Time Spent on Website",axis = 1)

    return no_missing,missing,y_no_missing,x,raw

def estimate(no_missing,lab_no_missing,missing):
    clf = KNeighborsClassifier()
    clf.fit(no_missing,lab_no_missing)
    predictedtime = clf.predict(missing)
    #print(clf.fit(no_missing,lab_no_missing))
    #print(clf.kneighbors(missing))
    return predictedtime

def encoding(data):
    data['Lead Origin'] = pd.factorize(data['Lead Origin'])[0].astype(np.uint16)
    data['Lead Source'] = pd.factorize(data['Lead Source'])[0].astype(np.uint16)
    data['Do Not Email'] = pd.factorize(data['Do Not Email'])[0].astype(np.uint16)
    data['Do Not Call'] = pd.factorize(data['Do Not Call'])[0].astype(np.uint16)
    data['Last Activity'] = pd.factorize(data['Last Activity'])[0].astype(np.uint16)
    data['Country'] = pd.factorize(data['Country'])[0].astype(np.uint16)
    data['Specialization'] = pd.factorize(data['Specialization'])[0].astype(np.uint16)
    data['What is your current occupation'] = pd.factorize(data['What is your current occupation'])[0].astype(np.uint16)
    data['What matters most to you in choosing a course'] = pd.factorize(data['What matters most to you in choosing a course'])[0].astype(np.uint16)
    data['Search'] = pd.factorize(data['Search'])[0].astype(np.uint16)
    data['Magazine'] = pd.factorize(data['Magazine'])[0].astype(np.uint16)
    data['Newspaper Article'] = pd.factorize(data['Newspaper Article'])[0].astype(np.uint16)
    data['X Education Forums'] = pd.factorize(data['X Education Forums'])[0].astype(np.uint16)
    data['Newspaper'] = pd.factorize(data['Newspaper'])[0].astype(np.uint16)
    data['Digital Advertisement'] = pd.factorize(data['Digital Advertisement'])[0].astype(np.uint16)
    data['Through Recommendations'] = pd.factorize(data['Through Recommendations'])[0].astype(np.uint16)
    data['Receive More Updates About Our Courses'] = pd.factorize(data['Receive More Updates About Our Courses'])[
        0].astype(np.uint16)
    data['Tags'] = pd.factorize(data['Tags'])[0].astype(np.uint16)
    data['Lead Quality'] = pd.factorize(data['Lead Quality'])[0].astype(np.uint16)
    data['Update me on Supply Chain Content'] = pd.factorize(data['Update me on Supply Chain Content'])[0].astype(
        np.uint16)
    data['Get updates on DM Content'] = pd.factorize(data['Get updates on DM Content'])[0].astype(np.uint16)
    data['City'] = pd.factorize(data['City'])[0].astype(np.uint16)
    data['I agree to pay the amount through cheque'] = pd.factorize(data['I agree to pay the amount through cheque'])[
        0].astype(np.uint16)
    data['A free copy of Mastering The Interview'] = pd.factorize(data['A free copy of Mastering The Interview'])[
        0].astype(np.uint16)
    data['Last Notable Activity'] = pd.factorize(data['Last Notable Activity'])[0].astype(np.uint16)
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

def raw_label(data,missing_index):
    #no_missing_label = {}
    no_missing_label = []
    for item in missing_index:
        #no_missing_label[item] = data.loc[item,'Total Time Spent on Website']
        no_missing_label.append(data.loc[item,'Total Time Spent on Website'])
    #no_missing_label = pd.DataFrame.from_dict(no_missing_label,orient='index',columns=['Total Time Spent on Website'])
    return no_missing_label

no_missing,missing,y_no_missing,raw_set,Raw = set_missing_time("Missing1.csv","public_data1.csv")
X_no_missing = encoding(no_missing)
X_missing = encoding(missing)
missing_index = list(missing.index)
no_missing_label = raw_label(Raw,missing_index)
result = estimate(X_no_missing,y_no_missing,X_missing)
Estimate_set,label = sub_data(raw_set,result,missing_index)
X_train,X_test,y_train,y_test = train_test_split(Estimate_set,label,train_size=0.8,random_state=1)


