from Utils.NetUtils import *
from Utils.Utils import *
import time

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name = "x")
x_norm = tf.divide(tf.subtract(x, tf.to_float(128)), tf.to_float(128), name="imgNorm")
y = tf.placeholder(tf.float32, shape=[None,10], name = "y")

#-------------------------------------------------------------

h_conv = descriptorNet(x_norm)

description = tf.identity(h_conv, name="descriptor")

h_avPool = tf.layers.average_pooling2d(description, [7, 7], [7, 7], name="Avg_Pooling")
inputSize = 1 * 1 * 512
h_avPool_flat = tf.reshape(h_avPool, [-1, inputSize], name = "Flatening")

y_pred = fullyLayerNoRelu("1", h_avPool_flat, 512, 10)

prediction = tf.identity(y_pred, name="prediction")

learning_rate = tf.placeholder(tf.float32, shape=[], name = "Learning_Rate")

with tf.name_scope("TrainStep"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction), name="CrossEntropy")
    train_step = tf.train.AdamOptimizer(learning_rate, name="Training").minimize(cross_entropy)

tf.summary.scalar('Cross_Entropy',cross_entropy)

with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="MeanAccuracy")

tf.summary.scalar('Train_Accuracy', accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./result", sess.graph)

init = tf.global_variables_initializer()
tvars = tf.trainable_variables()
saver = tf.train.Saver(tvars)
sess.run(init)

lr = 1e-4

trainingImgPath = "D:/DataSet/N64Game/NewBeetleAdventure/TrainingData/training_data-"
counter = 1
fileLenth = 2000
nbFile = 13
batchSize = 50
epoch = 60
print("Lets Begin!!")

for k in range(epoch):

    if (k == 45):
        lr = 1e-5

    print("==================================================")
    print("Doing epoch no "+str(k))
    print("Learning rate : "+str(lr))
    print("==================================================")

    for i in range(1, nbFile + 1, 1):

        localStartTime = time.time()
        fileSaveName = trainingImgPath + str(i) + ".npy"
        batchData = np.load(fileSaveName)
        print("Epoch no "+str(k))
        print("Processing file no "+str(i))

        for j in range(0, fileLenth - 1, batchSize):

            data = batchData[j:j+batchSize,0]
            label = batchData[j:j+batchSize,1]
            data = np.stack(data)
            label = np.stack(label)

            if (j % 1000 == 0):
                summary,trainAccuracy,loss = sess.run([merged,accuracy,cross_entropy],feed_dict={x:data,y:label})
                counter += 1
                writer.add_summary(summary,counter)
                print("Loss : "+str(loss))
                print("Train accuracy : "+str(trainAccuracy))

            sess.run([train_step], feed_dict={x: data, y: label, learning_rate:lr})

        localEndTime = time.time()
        localElapsedTime = localEndTime-localStartTime
        print("Local elapsed time (sec) : "+str(localElapsedTime))
        print("--------------------------------------------------")


save_path = saver.save(sess, "./model/model.ckpt")
print("Model saved in file : " + save_path)