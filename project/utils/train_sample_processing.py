from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing

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
        value = data.iloc[:, long_column].values
        changed = mm_scaler.fit_transform(value)
        data.iloc[:, long_column] = changed
    except Exception as e:
        print("err: ", e)
        return 0

    return data

def checktype(obj, type):
    return bool(obj) and all(isinstance(elem, type) for elem in obj)


def Processing_data_naive(data, columns):
    le = LabelEncoder()
    long_column = []
    for column in range(0,columns):
        
        value = data.iloc[:, column].values
        feauture = value.tolist()
        if checktype(obj=feauture, type=str) is True:
            print("phai thay doi: ", column)
            changed = le.fit_transform(feauture)
            data.iloc[:, column] = changed
        else:
            continue

    return data