from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import mutual_info_score

def viz_img(y_pred,title):
    n = 100
    fig = plt.figure(title)
    box_index = 1
    for cluster in range(10):
        result = np.where(y_pred == cluster)
        for i in np.random.choice(result[0].tolist(), n):
            ax = fig.add_subplot(10, n, box_index)
            plt.imshow(sampled_images[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            box_index += 1
    plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

class_indices = [np.where(y_train == i)[0] for i in range(10)]

# Randomly sample 100 images from each class
num_samples_per_class = 100
sampled_images = []
for class_idx in class_indices:
    sampled_indices = np.random.choice(class_idx, size=num_samples_per_class, replace=False)
    sampled_images.extend(x_train[sampled_indices])
    
    

# Visualize the sampled images
plt.figure(figsize=(20, 10))
for i in range(10 * num_samples_per_class):
    plt.subplot(10, num_samples_per_class, i+1)
    plt.imshow(sampled_images[i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.show()

model1 = KMeans(init="k-means++", n_clusters=10)
model1.fit(sampled_images)
y_pred1 = model1.labels_
#viz_img(y_pred,'kmeans')

model2 = AgglomerativeClustering(n_clusters=10)
model2.fit(sampled_images)
y_pred2 = model2.labels_
#viz_img(y_pred,'agglomerative')

model3 = SpectralClustering(n_clusters=10)
model3.fit(sampled_images)
y_pred3 = model3.labels_
#viz_img(y_pred,'spectral')

model4 = GaussianMixture(n_components=10)
model4.fit(sampled_images)
y_pred4 = model4.predict(sampled_images)
#viz_img(y_pred,'GMM')

mutual_info_score(y_train,y_pred1)
