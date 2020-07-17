from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
import numpy as np
from model.header import Header


def Processing_data_knn(data, columns):
    le = LabelEncoder()
    mm_scaler  = preprocessing.MinMaxScaler()
    long_column = []
    for column in range(0,columns):
        
        value = data.iloc[:, column].values
        feauture = value.tolist()
        if checktype(obj=feauture, type=str) is True:
            changed = le.fit_transform(feauture)
            data.iloc[:, column] = changed
        else:
            print(column)
            long_column.append(column)
    try:
        maxx = []
        minn = []
        for column in range(0,columns):
            value = data.iloc[:, column].values
            maxx.append(int(np.amax(value)))
            minn.append(int(np.amin(value)))

        value = data.iloc[:, long_column].values
        
        changed = mm_scaler.fit_transform(value)
        data.iloc[:, long_column] = changed
        print("data: ", data)
    except Exception as e:
        print("err: ", e)
        return 0

    return data, maxx, minn

def checktype(obj, type):
    return bool(obj) and all(isinstance(elem, type) for elem in obj)


def Processing_data_naive(data, columns):
    le = LabelEncoder()
    long_column = []
    for column in range(0,columns):
        
        value = data.iloc[:, column].values
        feauture = value.tolist()
        if checktype(obj=feauture, type=str) is True:
            le.fit(feauture)
            value_original = list(le.classes_)
            value_changed = le.transform(value_original)
            data_saved = {}
            for index, value in enumerate(value_changed):
                data_saved[str(value)] = value_original[index]
            print("data_saved: ", data_saved)
            headers_old = Header.find_one(query={
                "identify": "train_naive_sample"
            })
            if headers_old == -1:
                Header.insert(data={"identify": "train_naive_sample",
                                    "header":data_saved})
            else:
                Header.update_one(query={"identify": "train_naive_sample"},
                                change={"header": data_saved})
            changed = le.fit_transform(feauture)
            print("aaaaa: ", changed)
            data.iloc[:, column] = changed
        else:
            continue

    return data