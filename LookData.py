import numpy as np
import matplotlib.pyplot as plt

def showImgs(imgs, nbEx, nbCl):
    for i in range(nbCl):
        for j in range(nbEx):
            plt.subplot(nbEx, nbCl, i + 1)
            plt.imshow(imgs[i].astype('uint8'))
            plt.axis('off')

    plt.show()

nbShowimg = 5
index = 0
file_name = 'D:/DataSet/N64Game/NewBeetleAdventure/TrainingData/training_data-1.npy'
train_data = np.load(file_name)
print(train_data.shape)
print(train_data.shape[0])

imgs = train_data[index:index+nbShowimg,0]
imgs = np.stack(imgs)
labels = train_data[index:index+nbShowimg,1]
labels = np.stack(labels)

print(type(train_data))
print(type(imgs))
print(type(labels))
print(type(labels[0]))
print(imgs.shape[0])
print(labels.shape[0])

print(labels)

showImgs(imgs, 1, nbShowimg)


