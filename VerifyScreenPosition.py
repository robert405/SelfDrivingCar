from Utils.grabscreen import grab_screen
import cv2

while (True):

    screen = grab_screen(region=(40, 80, 590, 490))
    screen = cv2.resize(screen, (224, 224))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    cv2.imshow('window',screen)

    if cv2.waitKey(25)&0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


