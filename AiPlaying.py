import numpy as np
from Utils.grabscreen import grab_screen
import cv2
import time
from Utils.directkeys import PressKey, ReleaseKey, W, A, S, D, Q
from Utils.getkeys import key_check
import tensorflow as tf

previousKeys = []

vectToKey = {
0:[W],
1:[W,A],
2:[W,D],
3:[S],
4:[S,A],
5:[S,D],
6:[A],
7:[D],
8:[Q],
9:[]
}

sess = tf.Session()

saver = tf.train.import_meta_graph("./model/model.ckpt.meta")
saver.restore(sess, "./model/model.ckpt")

graph = tf.get_default_graph()
out = graph.get_tensor_by_name("prediction:0")
pred = tf.nn.softmax(out)

def pressNewKeys(keys):

    global previousKeys

    for key in previousKeys:
        ReleaseKey(key)

    for key in keys:
        PressKey(key)

    previousKeys = keys


for i in range(5,0,-1):

    print(str(i))
    time.sleep(1)

paused = False

while(True):

    keys = key_check()

    # p pauses game and can get annoying.
    if ('T' in keys):
        if paused:
            paused = False
            print("Unpaused")
            time.sleep(1)
        else:
            paused = True
            print("Paused")
            pressNewKeys([])
            time.sleep(1)

    elif ('X' in keys):

        break

    if (not paused):

        screen = grab_screen(region=(40, 80, 590, 490))
        screen = cv2.resize(screen, (224, 224))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        screen = np.array([screen])

        result = sess.run(pred, feed_dict={"x:0":screen})
        result = result[0]
        moveIndex = np.argmax(result)
        keyList = vectToKey[moveIndex]

        pressNewKeys(keyList)
        print("-----------------")
        print(moveIndex)
        time.sleep(0.01)

    else:
        print("currently paused")

pressNewKeys([])

