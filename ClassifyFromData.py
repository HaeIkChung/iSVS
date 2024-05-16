import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

train=pd.read_excel(io="BFI 상관관계 비교 ver3.xlsx")

# 라벨링을 숫자로 변환 하는 코드
feature="label"
le=LabelEncoder()
train[feature] = le.fit_transform(train[feature])
train[feature].head(23)

#train, test split
X = train.drop(columns=['label', 'T3 left upper', 'T3 right upper', 'T3 right lower'])  # Features
X_includeBFI = train.drop(columns=['label'])  # Features
X_T3leftupper = train.drop(columns=['label', 'T3 right upper', 'T3 right lower'])
X_T3rightupper = train.drop(columns=['label', 'T3 left upper', 'T3 right lower'])
X_T3rightlower = train.drop(columns=['label', 'T3 left upper','T3 right upper'])
y = train['label']  # Target variable
y_includeBFI=y
y_T3leftupper=y
y_T3rightupper=y
y_T3rightlower=y

X, y = X.to_numpy() ,np.transpose(y.to_numpy())
X_T3leftupper, y_T3leftupper = X_T3leftupper.to_numpy() ,y_T3leftupper.to_numpy()
X_T3rightupper, y_T3rightupper = X_T3rightupper.to_numpy() ,y_T3rightupper.to_numpy()
X_T3rightlower, y_T3rightlower = X_T3rightlower.to_numpy() ,y_T3rightlower.to_numpy()
X_includeBFI, y_includeBFI = X_includeBFI.to_numpy() , y_includeBFI.to_numpy()
#array로 변환

'''
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train_includeBFI, X_test_includeBFI, y_train_includeBFI, y_test_includeBFI = train_test_split(X_includeBFI, y, random_state=0)
X_train_T3leftupper, X_test_T3leftupper, y_train_T3leftupper, y_test_T3leftupper = train_test_split(X_T3leftupper, y, random_state=0)
X_train_T3rightupper, X_test_T3rightupper, y_train_T3rightupper, y_test_T3rightupper = train_test_split(X_T3rightupper, y, random_state=0)
X_train_T3rightlower, X_test_T3rightlower, y_train_T3rightlower, y_test_T3rightlower = train_test_split(X_T3rightlower, y, random_state=0)
'''
def MakeRandom(X_T3leftupper, y_T3leftupper,X_T3rightupper, y_T3rightupper,X_T3rightlower, y_T3rightlower,X_includeBFI, y_includeBFI):
    temp_X = np.empty((0, X_T3leftupper.shape[1]), dtype=float)
    temp_y = np.empty((0,), dtype=float)
    random = [3, 0, 4, 2, 21, 11, 10, 13, 18, 15, 20, 8, 6, 17, 19, 14, 5, 12, 16, 9, 1, 7]

    for i in random:
        temp_X = np.concatenate((temp_X, X_T3leftupper[i].reshape(1, -1)), axis=0)
        temp_y = np.concatenate((temp_y, [y_T3leftupper[i]]), axis=0)

    X_T3leftupper = temp_X
    y_T3leftupper = temp_y

    temp_X = np.empty((0, X_T3rightupper.shape[1]), dtype=float)
    temp_y = np.empty((0,), dtype=float)

    for i in random:
        temp_X = np.concatenate((temp_X, X_T3rightupper[i].reshape(1, -1)), axis=0)
        temp_y = np.concatenate((temp_y, [y_T3rightupper[i]]), axis=0)

    X_T3rightupper = temp_X
    y_T3rightupper = temp_y

    temp_X = np.empty((0, X_T3rightlower.shape[1]), dtype=float)
    temp_y = np.empty((0,), dtype=float)

    for i in random:
        temp_X = np.concatenate((temp_X, X_T3rightlower[i].reshape(1, -1)), axis=0)
        temp_y = np.concatenate((temp_y, [y_T3rightlower[i]]), axis=0)

    X_T3rightlower = temp_X
    y_T3rightlower = temp_y

    temp_X = np.empty((0, X_includeBFI.shape[1]), dtype=float)
    temp_y = np.empty((0,), dtype=float)

    for i in random:
        temp_X = np.concatenate((temp_X, X_includeBFI[i].reshape(1, -1)), axis=0)
        temp_y = np.concatenate((temp_y, [y_includeBFI[i]]), axis=0)

    X_includeBFI = temp_X
    y_includeBFI = temp_y

    return [X_T3leftupper,y_T3leftupper,X_T3rightupper,y_T3rightupper,X_T3rightlower,y_T3rightlower,X_includeBFI,y_includeBFI]

#train

X_T3leftupper, y_T3leftupper,X_T3rightupper, y_T3rightupper,X_T3rightlower, y_T3rightlower,X_includeBFI, y_includeBFI=MakeRandom(X_T3leftupper, y_T3leftupper,X_T3rightupper, y_T3rightupper,X_T3rightlower, y_T3rightlower,X_includeBFI, y_includeBFI)
model = SVC(kernel='linear',decision_function_shape="ovr")
model_KNN = KNeighborsClassifier()
model_DT = DecisionTreeClassifier()

# Create an ECOC model with logistic regression as the base classifier

ecoc_model = OutputCodeClassifier(model, code_size=2, random_state=0)

def ValidModel(model,name,X_T3leftupper, y_T3leftupper,X_T3rightupper, y_T3rightupper,X_T3rightlower, y_T3rightlower,X_includeBFI, y_includeBFI):

    scores = cross_val_score(model, X, y, cv=4)
    scores_T3leftupper = cross_val_score(model, X_T3leftupper, y_T3leftupper, cv=4)
    scores_T3rightupper = cross_val_score(model, X_T3rightupper, y_T3rightupper, cv=4)
    scores_T3rightlower = cross_val_score(model, X_T3rightlower, y_T3rightlower, cv=4)
    scores_includeBFI = cross_val_score(model, X_includeBFI, y_includeBFI, cv=4)

    #validation

    print('{1}-not include BFI, score: {0}'.format(scores.mean(),name))
    print('{1}-T3 left upper BFI, score: {0}'.format(scores_T3leftupper.mean(),name))
    print('{1}-T3 right upper BFI, score: {0}'.format(scores_T3rightupper.mean(),name))
    print('{1}-T3 right lower BFI, score: {0}'.format(scores_T3rightlower.mean(),name))
    print('{1}-include all BFI, score: {0}'.format(scores_includeBFI.mean(),name))

ValidModel(ecoc_model,'ecoc',X_T3leftupper, y_T3leftupper,X_T3rightupper, y_T3rightupper,X_T3rightlower, y_T3rightlower,X_includeBFI, y_includeBFI)
ValidModel(model,'svc',X_T3leftupper, y_T3leftupper,X_T3rightupper, y_T3rightupper,X_T3rightlower, y_T3rightlower,X_includeBFI, y_includeBFI)
ValidModel(model_KNN,'knn',X_T3leftupper, y_T3leftupper,X_T3rightupper, y_T3rightupper,X_T3rightlower, y_T3rightlower,X_includeBFI, y_includeBFI)
ValidModel(model_DT,'dt',X_T3leftupper, y_T3leftupper,X_T3rightupper, y_T3rightupper,X_T3rightlower, y_T3rightlower,X_includeBFI, y_includeBFI)