import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os


def random_forest(X_train,y_train):
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
             reverse=True))
    result = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
             reverse=True)
    return result

def load_data(filename):
    data = pd.read_csv(filename)
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
    data['Receive More Updates About Our Courses'] = pd.factorize(data['Receive More Updates About Our Courses'])[0].astype(np.uint16)
    data['Tags'] = pd.factorize(data['Tags'])[0].astype(np.uint16)
    data['Lead Quality'] = pd.factorize(data['Lead Quality'])[0].astype(np.uint16)
    data['Update me on Supply Chain Content'] = pd.factorize(data['Update me on Supply Chain Content'])[0].astype(np.uint16)
    data['Get updates on DM Content'] = pd.factorize(data['Get updates on DM Content'])[0].astype(np.uint16)
    data['City'] = pd.factorize(data['City'])[0].astype(np.uint16)
    data['I agree to pay the amount through cheque'] = pd.factorize(data['I agree to pay the amount through cheque'])[0].astype(np.uint16)
    data['A free copy of Mastering The Interview'] = pd.factorize(data['A free copy of Mastering The Interview'])[0].astype(np.uint16)
    data['Last Notable Activity'] = pd.factorize(data['Last Notable Activity'])[0].astype(np.uint16)

    x = data.drop('Converted',axis=1)
    #x = x.drop("row_number",axis=1)
    x = x.set_index("Prospect ID")
    y = data['Converted']
    return x,y,x.columns

def remove(remove_list,data):
    for item in remove_list:
        if item[0] <= 0.0001:
            data = data.drop(item[1],axis=1)
    return data


data,label,names = load_data('public_data1.csv')
X_train,X_test,y_train,y_test = train_test_split(data,label,train_size=0.8,random_state=1)
result = random_forest(X_train,y_train)
processed_train = remove(result,X_train)
processed_test = remove(result,X_test)
#with open(os.path.join(os.getcwd(),'Feature-score.txt'),'w') as file:
#    file.write(str(result))


