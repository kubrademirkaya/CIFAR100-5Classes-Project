import numpy as np
import os
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras.preprocessing import image

#veriseti yükleniyor
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

#camel, cocroach, lawn_mower, man, streetcar verileriyle yeni bir veriseti oluşturuluyor
index = np.where((y_train == 15) | (y_train == 24) | (y_train == 41) | (y_train == 46) | (y_train == 81))
x_train = x_train[index[0]]
y_train = y_train[index[0]]

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
#print(x_train.shape[0], 'train samples')

index = np.where((y_test == 15) | (y_test == 24) | (y_test == 41) | (y_test == 46) | (y_test == 81))
x_test = x_test[index[0]]
y_test = y_test[index[0]]

print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)


#train ve test verileri için yeniden numaralandırma yapılıyor
for n, i in enumerate(y_train):
    if i == 15:
        y_train[n] = 0
    elif i==24:
        y_train[n]= 1
    elif i==41:
        y_train[n]= 2
    elif i==46:
        y_train[n]= 3
    else:
        y_train[n]= 4
        

for n, i in enumerate(y_test):
    if i == 15:
        y_test[n] = 0
    elif i==24:
        y_test[n]= 1
    elif i==41:
        y_test[n]= 2
    elif i==46:
        y_test[n]= 3
    else:
        y_test[n]= 4
    
#yeni oluşturulan versetindeki train ve test verilerinin sayısı gösteriliyor    
sns.set(style='white', context='notebook', palette='icefire')
fig, axs = plt.subplots(1,2,figsize=(15,5)) 

sns.countplot(y_train.ravel(), ax=axs[0])
axs[0].set_title('Training data')
axs[0].set_xlabel('Classes')

sns.countplot(y_test.ravel(), ax=axs[1])
axs[1].set_title('Testing data')
axs[1].set_xlabel('Classes')
plt.show()


os.mkdir('dataset')
os.mkdir('dataset\\train')
os.mkdir('dataset\\test')

for i in range(5):
    path=os.path.join('dataset\\train',str(i))
    os.mkdir(path)
    path=os.path.join('dataset\\test',str(i))
    os.mkdir(path)
    
for i in range(2500):
    path='dataset/train/'+str(int(y_train[i]))+'/'+str(i)+'.png'  
    plt.imsave(path,x_train[i])

for i in range(500):
    path='dataset/test/'+str(int(y_test[i]))+'/'+str(i)+'.png'  
    plt.imsave(path,x_test[i])
    

#veri setinden örnek veriler
labels = ['camel', 'cockroach', 'lawn mower', 'man', 'streetcar']
W_grid = 10
L_grid = 5
fig, axes = plt.subplots(L_grid, W_grid, figsize = (25, 25))
axes = axes.ravel()
for i in range(0,L_grid*W_grid):
    index = np.random.randint(0,len(x_train))
    axes[i].imshow(x_train[index])
    index = y_train[index]
    axes[i].set_title(labels[int(index)])
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.1 )
   
 
num_classes = 5
batch_size = 128
epochs = 30


#normalizasyon
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255 
x_test /= 255

#One Hot Encoding işlemi
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
   
#CNN yapısı
def myModel():
    model=Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3),
                         activation = "relu",
                         input_shape = (32, 32, 3)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(32, kernel_size = (3, 3),
                         activation = "relu"))
    model.add(Conv2D(64, kernel_size = (3, 3),
                         activation = "relu"))
    model.add(Conv2D(128, kernel_size = (3, 3),
                         activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))    
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,
                    activation = "softmax"))

    print("\n Oluşturulan CNN modeli ayrıntıları \n")
    model.summary()

    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

model = myModel()


#oluşturulan modelin şeması
keras.utils.plot_model(model,
                       to_file="modelGraph.png",
                       show_shapes=False,
                       show_dtype=False,
                       show_layer_names=True,
                       rankdir="TB",
                       expand_nested=False,
                       dpi=96,)

#oluşturulan modelin eğitimi
history = model.fit(x_train, y_train, 
          batch_size = batch_size, 
          epochs = epochs, 
          verbose = 1,
          validation_data = (x_test,y_test))


#tahmin işlemi yapılıyor
y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred, axis=1)
y_pred = keras.utils.to_categorical(y_pred, num_classes)


#accuracy grafiği
plt.figure(figsize=(25, 8))
plt.plot(history.history['accuracy'], color='purple')
plt.plot(history.history['val_accuracy'], color='blue')
plt.title('Model Accuracy', fontsize=25, fontweight='bold')
plt.ylabel('accuracy', fontsize=15, fontweight='bold')
plt.xlabel('epoch', fontsize=15, fontweight='bold')
plt.legend(['train', 'validation'], loc='best', fontsize=15, shadow=True)
plt.show()

#loss grafiği
plt.figure(figsize=(25, 8))
plt.plot(history.history['loss'], color='purple')
plt.plot(history.history['val_loss'], color='blue')
plt.title('Model Loss', fontsize=25, fontweight='bold')
plt.ylabel('loss', fontsize=15, fontweight='bold')
plt.xlabel('epoch', fontsize=15, fontweight='bold')
plt.legend(['train', 'validation'], loc='best', fontsize=15, shadow=True)
plt.show()


#classification report, accuracy ve loss değerleri
print("\n CLASSIFICATION REPORT \n")
print(classification_report(y_test, y_pred))
print("\n")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
#confusion matrix yazdırılıyor
cm=confusion_matrix(y_test,y_pred)
print(cm)

#confusion matrix grafiği
names = ['camel', 'cockroach', 'lawn mower', 'man', 'streetcar']
plt.subplots(figsize=(10,10))
plt.title('Confusion Matrix', fontsize=25, fontweight='bold')
sns.heatmap(cm,xticklabels=names,yticklabels=names,annot=True,fmt='d',  cmap=plt.cm.viridis)
plt.show()

#eğitilen model kaydediliyor
model.save("cifar100model.h5")

#modelin kullanılmak üzere projeye yüklenmesi
model =load_model("cifar100model.h5")






