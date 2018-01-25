import numpy as np
from Utils.grabscreen import grab_screen
import cv2
import time
from Utils.getkeys import key_check
import os

w = np.array([1,0,0,0,0,0,0,0,0,0])
wa = np.array([0,1,0,0,0,0,0,0,0,0])
wd = np.array([0,0,1,0,0,0,0,0,0,0])
s = np.array([0,0,0,1,0,0,0,0,0,0])
sa = np.array([0,0,0,0,1,0,0,0,0,0])
sd = np.array([0,0,0,0,0,1,0,0,0,0])
a = np.array([0,0,0,0,0,0,1,0,0,0])
d = np.array([0,0,0,0,0,0,0,1,0,0])
q = np.array([0,0,0,0,0,0,0,0,1,0])
nk = np.array([0,0,0,0,0,0,0,0,0,1])


def keys_to_output(keys):

    output = nk

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys and 'A' in keys:
        output = sa
    elif 'S' in keys and 'D' in keys:
        output = sd
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    elif 'Q' in keys:
        output = q

    return output


def main():

    training_data = []
    starting_value = 1

    while True:

        checkFile = 'D:/DataSet/N64Game/NewBeetleAdventure/TrainingData/training_data-' + str(starting_value) + '.npy'
        if os.path.isfile(checkFile):
            print('File exists, moving along', starting_value)
            starting_value += 1
        else:
            print('File does not exist, starting fresh!', starting_value)
            break

    file_name = 'D:/DataSet/N64Game/NewBeetleAdventure/TrainingData/training_data-' + str(starting_value) + '.npy'


    for i in list(range(3))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    print('STARTING!!!')

    while (True):

        time.sleep(0.01)

        if not paused:
            screen = grab_screen(region=(40, 80, 590, 490))
            screen = cv2.resize(screen, (224, 224))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            keys = key_check()
            output = keys_to_output(keys)
            training_data.append(np.array([screen,output]))


            if len(training_data) % 500 == 0:
                print(len(training_data))

                if len(training_data) == 2000:
                    training_data = np.array(training_data)
                    np.save(file_name, training_data)
                    print('SAVED : ' + str(starting_value))
                    training_data = []
                    starting_value += 1
                    file_name = 'D:/DataSet/N64Game/NewBeetleAdventure/TrainingData/training_data-' + str(starting_value) + '.npy'

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

main()