from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import json
import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D
from tensorflow.keras.layers import  MaxPooling2D, Dropout
from tensorflow.keras import  models, layers



def unpickle(file):
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, root_dir, transform, mode, noise_file=''):

        self.transform = transform
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise for cifar10
        # generate asymmetric noise for cifar100
        self.transition_cifar100 = {}
        nb_superclasses = 20
        nb_subclasses = 5
        base = [1, 2, 3, 4, 0]
        for i in range(nb_superclasses * nb_subclasses):
            self.transition_cifar100[i] = int(base[i % 5] + 5 * int(i / 5))

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                    #print(train_label)
                    #print(len(train_label))
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                print(train_label)
                print(len(train_label))
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            noise_label = json.load(open(noise_file, "r"))

            if self.mode == 'train':
                self.train_data = train_data
                self.noise_label = noise_label
                self.clean_label = train_label

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, batch_size, num_workers, root_dir, noise_file=''):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def run(self, mode):
        if mode == 'train':
            train_dataset = cifar_dataset(dataset=self.dataset,
                                          root_dir=self.root_dir, transform=self.transform_train, mode="train",
                                          noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return np.asarray(train_dataset.train_data), np.asarray(train_dataset.noise_label), np.asarray(train_dataset.clean_label)

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return np.asarray(test_dataset.test_data), np.asarray(test_dataset.test_label)


# test the custom loaders for CIFAR
dataset = 'cifar10'  # either cifar10 or cifar100
data_path = 'E:/CAS 771/cifar-10-python/cifar-10-batches-py/'  # path to the data file (don't forget to download the feature data and also put the noisy label file under this folder)

loader = cifar_dataloader(dataset, batch_size=128,
                          num_workers=4,
                          root_dir=data_path,
                          noise_file='%s/cifar10_noisy_labels_task1.json' % (data_path))

x_train, y_train, label = loader.run('train')
print(x_train.shape)
print(y_train.shape)
print(label.shape)
x_test,y_test = loader.run('test')
print(x_test.shape)
print(y_test.shape)

# todo: Code your own algorithm


x_train1 = x_train
x_train = x_train.reshape(len(x_train),-2)
x_test = x_test.reshape(len(x_test),-2)
y_train = y_train.reshape(len(y_train),1)

print("SHAPE X_TRAIN Y_TRAIN",x_train.shape,y_train.shape)

x_train = x_train.astype('float32') 
x_test = x_test.astype('float32')

# Normalization
x_train = x_train/255.0
x_test = x_test/255.0

print("SHAPE X_TRAIN Y_TRAIN",x_train.shape,y_train.shape)

df = pd.DataFrame(list(zip(x_train,y_train)), columns =['identified_cluster','label'])
pca = PCA(0.9)
x_train = pca.fit_transform(x_train)
x_train = np.append(x_train,y_train,axis = 1)

x_train = np.append(x_train,y_train,axis = 1)

print("SHAPE X_TRAIN Y_TRAIN",x_train.shape,y_train.shape)


plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='viridis')
plt.show()
km = KMeans(n_clusters=10,n_init = 100, max_iter = 1000)
#identified_clusters = KMeans(n_clusters = 10).fit_predict(x_train)
#identified_clusters = AgglomerativeClustering(n_clusters=10).fit_predict(x_train)
import pickle
#pickle.dump(identified_clusters, open("model.pkl", "wb"))
# load the model
#identified_clusters = pickle.load(open("model.pkl", "rb"))

input_img = Input(shape=(len(x_train[0]),))

encoded = Dense(500, activation='relu')(input_img)
encoded = Dense(500, activation='relu')(encoded)
encoded = Dense(500, activation='relu')(encoded)
encoded = Dense(len(x_train[0]), activation='relu')(encoded)

## bottleneck layer
#n_bottleneck = len(x_train)
## defining it with a name to extract it later
#bottleneck_layer = "bottleneck_layer"
# can also be defined with an activation function, relu for instance
#bottleneck = Dense(n_bottleneck-1, activation='relu', kernel_initializer = 'glorot_uniform')(encoded)

# "decoded" is the lossy reconstruction of the input

decoded = Dense(500, activation='relu')(encoded)
decoded = Dense(500, activation='relu')(decoded)
decoded = Dense(500, activation='relu')(decoded)
decoded = Dense(len(x_train[0]), activation = 'relu')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

print(autoencoder.summary())
encoder = Model(input_img, encoded)
autoencoder.compile(optimizer='adam', loss='mse')
train_history = autoencoder.fit(x_train, x_train, epochs=50, batch_size = 128,  verbose = 1)
print(autoencoder.summary())
autoencoder.save_weights('autoencoder.h5')
print(train_history)
pred = encoder.predict(x_train)

print("PRED SHAPE ",pred.shape)


'''model = Sequential()

model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())     # 32x32x32
model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))      # 16x16x32
model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 16x16x32
model.add(BatchNormalization())     # 16x16x32
model.add(UpSampling2D())
model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32
model.add(BatchNormalization())
model.add(Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid'))   # 32x32x3

model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')'''



# We want to add different noise vectors for each epoch
#pred =  model.fit(x_train, x_train, epochs=1, batch_size=100)
km.fit(pred)
identified_clusters = km.predict(pred)


pickle.dump(identified_clusters, open("model1.pkl", "wb"))
#plt.scatter(pred[:, 0], pred[:, 1], c=identified_clusters, cmap='viridis')
#plt.show()

identified_clusters = pickle.load(open("model1.pkl", "rb"))
clusters = np.unique(identified_clusters)
print("Y",y_train.shape)
y_train = y_train.flatten()

df = pd.DataFrame(list(zip(x_train1, identified_clusters,y_train)), columns =['Image', 'identified_cluster','label'])
df_test = (df['label'])
df_test['correct'] = [1 if x == z else 0 for x, z in zip(df['label'], label)]
accuracy = 100.0 * float(sum(df_test['correct'])) / float(df_test.shape[0])
print("ERROR = ",accuracy)

print(df_test.head(20))

new_df = df.head(1)

for cluster in clusters:
    temp = df[df.identified_cluster == cluster]
    y = temp['label']
    #print("Mode", y.mode())
    m = y.mode()
    #print(" m = ",int(m[0]))
    item_counts = (temp["label"].value_counts())
    #print(item_counts)
    temp_dict = dict()
    for i,j in item_counts.iteritems():
        print("i,j = ",i,j)
        if(j > 200 and j < 800):
            temp_dict[i] = j
    lis = list(temp_dict.keys())
    #print(lis)
    #print(dict)
    temp1 = df.query("(identified_cluster==@cluster) and (label not in @lis)")
    temp = df.query("(identified_cluster==@cluster) and (label in @lis)")
    #print(temp.head(20))
    temp['label'] = int(m)
    #print(temp.head(20))
    new_df = new_df.append(temp1)
    new_df = new_df.append(temp) 

print(new_df.info())
print(new_df.head(10))
df_test = (new_df['label'])
df_test['correct'] = [1 if x == z else 0 for x, z in zip(new_df['label'], label)]
accuracy = 100.0 * float(sum(df_test['correct'])) / float(df_test.shape[0])
print("ERROR = ",accuracy)

new_df = new_df.iloc[1: , :]
y_train = (new_df['label'])
print(y_train.head(20))
y_train = y_train.to_numpy()
print("SHAPES")
print(x_train.shape)
print(y_train.shape)

#new_df['correct'] = [1 if x == z else 0 for x, z in zip(df_test['predicted'], df_test_label)]
#accuracy = 100.0 * float(sum(df_test['correct'])) / float(df_test.shape[0])

#pca = PCA(0.9)
#x_train = pca.fit_transform(x_train)

#print("New x_train,y_train",y_train1.shape)

model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

print(model.summary())

model.add(layers.Flatten())

model.add(layers.Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())

import tensorflow as tf

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train1, y_train, epochs=50)
'''import torch.nn as nn
model = nn.Linear(3072, 10)
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
class YourFavoriteModel(BaseEstimator): # Inherits sklearn base classifier
    def __init__(self, ):
        pass
    def fit(self, X, y, sample_weight=None):
        return autoencoder.fit(X, y, epochs=1)
    def predict(self, X):
        return autoencoder.predict(X)
    def predict_proba(self, X):
        pass
    def score(self, X, y, sample_weight=None):
        pass


# Now you can use your model with `cleanlab`. Here's one example:
from cleanlab.classification import LearningWithNoisyLabels
lnl = LearningWithNoisyLabels(clf=LogisticRegression(verbose=True, n_jobs=-1))
lnl.fit(x_train, y_train)
predicted_label = lnl.predict(x_test)'''

#plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
#plt.legend(loc='lower right')

predicted_label = model.predict(x_test)
pickle.dump(predicted_label, open("classification.pkl", "wb"))

predicted_label = pickle.load(open("classification.pkl", "rb"))
test_loss, test_acc = model.evaluate(x_test,  y_test)

print(predicted_label)
print(test_loss)
print(test_acc)

