# -*- coding=UTF-8 -*-  
'''
参考：https://www.kaggle.com/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1?tdsourcetag=s_pcqq_aiomsg
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten # core layers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

epochs = 5
batch_size = 64

## 数据预处理
train=pd.read_csv("./CNN for MINST dataset/train.csv")
test=pd.read_csv("./CNN for MINST dataset/test.csv")
sub=pd.read_csv("./CNN for MINST dataset/sample_submission.csv")#submisson file
print("Data are Ready!!")

# feed features and labels
X = train.drop(['label'], 1).values
y = train['label'].values
test_x = test.values

# Normalization
X = X / 255.0
test_x = test_x / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , chanal = 1)
X = X.reshape(-1,28,28,1)
test_x = test_x.reshape(-1,28,28,1)

# One-Hot Encoding
y = to_categorical(y)
print(f"Label size {y.shape}")

# Split training and valdiation set
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.1, random_state=0)
	
## Data Visualization
X_train__ = X_train.reshape(X_train.shape[0], 28, 28)

fig, axis = plt.subplots(1, 4, figsize=(20, 10))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_train__[i], cmap='binary')
    digit = y_train[i].argmax()
    ax.set(title = f"Real Number is {digit}");	

## simple cnn model 
model=Sequential()
# 卷积层
model.add(Conv2D(filters=64,
	kernel_size=(3,3),activation="relu",input_shape=(28,28,1)))
	
model.add(Conv2D(filters=64,
	kernel_size=(3,3),activation="relu"))	
	
model.add(MaxPooling2D(pool_size=(2,2)))	
model.add(BatchNormalization())

model.add(Conv2D(filters=128,
	kernel_size=(3,3),activation="relu"))
	
model.add(Conv2D(filters=128,
	kernel_size=(3,3),activation="relu"))
	
model.add(MaxPooling2D(pool_size=(2,2)))	
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))  
model.add(MaxPooling2D(pool_size=(2,2)))
#全连接层
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512,activation="relu"))
model.add(Dense(10,activation="softmax"))

model.compile(loss="categorical_crossentropy",
	optimizer="adam", metrics=["accuracy"])
	
## With data augmentation to prevent overfitting
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

#datagen.fit(X_train)
train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)
test_gen = datagen.flow(X_test, y_test, batch_size=batch_size)	

## Model training
#fit
history = model.fit_generator(train_gen, 
                              epochs = epochs, 
                              steps_per_epoch = X_train.shape[0] // batch_size,
                              validation_data = test_gen,
                              validation_steps = X_test.shape[0] // batch_size)

# # Plot CNN model	
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	
## Evaluate the model
# Training and validation curves
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1, figsize=(18, 10))
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)
ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

# Confusion matrix
fig = plt.figure(figsize=(10, 10)) # Set Figure

y_pred = model.predict(X_test) # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

Y_pred = np.argmax(y_pred, 1) # Decode Predicted labels
Y_test = np.argmax(y_test, 1) # Decode labels

mat = confusion_matrix(Y_test, Y_pred) # Confusion matrix

# Plot Confusion matrix
sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
plt.xlabel('Predicted Values')
plt.ylabel('True Values');
plt.show();

## Prediction and submition
# Prediction validation results
y_pred = model.predict(X_test)
X_test__ = X_test.reshape(X_test.shape[0], 28, 28)

fig, axis = plt.subplots(4, 4, figsize=(12, 14))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_test__[i], cmap='binary')
    ax.set(title = f"Real Number is {y_test[i].argmax()}\nPredict Number is {y_pred[i].argmax()}");
	
# Prediciting the Outputs
pred = model.predict_classes(test_x, verbose=1)
sub['Label'] = pred
sub.to_csv("./CNN for MINST dataset/CNN_keras_sub.csv", index=False)
sub.head()	