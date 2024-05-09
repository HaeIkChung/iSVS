import pandas as pd 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import decomposition
import numpy as np

train=pd.read_excel(io="BFI 상관관계 비교 ver2.xlsx")

# 라벨링을 숫자로 변환 하는 코드
feature="label"
le=LabelEncoder()
train[feature] = le.fit_transform(train[feature])
train[feature].head(23)


#train, test split
X = train.drop(columns=['label', 'T3 left upper', 'T3 right upper', 'T3 right lower'])  # Features
X_includeBFI = train.drop(columns=['label'])  # Features
y = train['label']  # Target variable

#array로 변환
X_includeBFI, y = X_includeBFI.to_numpy(), y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_includeBFI, y, random_state=0)

pca = decomposition.PCA(n_components=2).fit(X_train)
reduced_X = pca.transform(X_train)
reduced_test_X = pca.transform(X_test)

#train
model1 = SVC(gamma=0.01,kernel='rbf').fit(reduced_X, y_train)
model2 = KNeighborsClassifier().fit(reduced_X, y_train)
model3 = SVC(gamma=0.01,kernel='rbf').fit(X_train, y_train)
model4 = KNeighborsClassifier().fit(X_train, y_train)


#validation

print("SVC+PCA")
print("훈련 세트 정확도: {:.2f}".format(model1.score(reduced_X, y_train)))
print("테스트 세트 정확도: {:.2f}".format(model1.score(reduced_test_X, y_test)))

print("KNN+PCA")
print("훈련 세트 정확도: {:.2f}".format(model2.score(reduced_X, y_train)))
print("테스트 세트 정확도: {:.2f}".format(model2.score(reduced_test_X, y_test)))

print("SVC")
print("훈련 세트 정확도: {:.2f}".format(model3.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(model3.score(X_test, y_test)))

print("KNN")
print("훈련 세트 정확도: {:.2f}".format(model4.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(model4.score(X_test, y_test)))



from matplotlib.colors import ListedColormap

# Define meshgrid for visualization
X_set, y_set = reduced_X, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# Plot decision boundaries for SVM (model1)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(X1, X2, model1.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM Classifier (PCA)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()

# Plot decision boundaries for KNN (model2)
plt.subplot(1, 2, 2)
plt.contourf(X1, X2, model2.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('KNN Classifier (PCA)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()

plt.tight_layout()
plt.show()

