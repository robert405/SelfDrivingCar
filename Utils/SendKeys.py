from directkeys import PressKey, ReleaseKey, W, A, S, D
import time

previousKeys = []

def pressNewKeys(keys):

    global previousKeys

    for key in keys:
        PressKey(key)

    for key in previousKeys:
        ReleaseKey(key)

    previousKeys = keys


for i in range(5,0,-1):

    print(str(i))
    time.sleep(1)

pressNewKeys([W])

time.sleep(1)

pressNewKeys([W, A])

time.sleep(1)

pressNewKeys([S, D])

time.sleep(1)

pressNewKeys([S])

time.sleep(1)

pressNewKeys([])