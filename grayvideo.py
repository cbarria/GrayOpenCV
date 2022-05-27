import cv2
import numpy as np
import matplotlib.pyplot as plt
from drawnow import drawnow
from datetime import datetime

def make_fig():
    plt.plot(x, y,)  # I think you meant this
    plt.xticks(rotation=90)
    
plt.ion()  # enable interactivity
fig = plt.figure() # make a figure


x = list()
y = list()

lower = [120,120, 120] 
upper = [148, 148, 148]

 # create NumPy arrays from the boundaries
lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")

cap = cv2.VideoCapture(1)  # udp://239.0.0.1:1234'  'opencv/assets/video.mp4'
#fps = cap.get(cv2.CAP_PROP_FPS) # 30 fps!

img_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("falló al abrir...")
        break

    halfF = cv2.resize(frame,(0,0) ,fx=0.5,fy=0.5)
    
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float ancho
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float alto

    mask = cv2.inRange(halfF, lower, upper)
    output = cv2.bitwise_and(halfF, halfF, mask = mask)

    cv2.imshow('Prueba3', np.hstack([halfF, output]))

    pxlsNoZero = cv2.countNonZero(mask) #cv2.cvtColor(output, cv2.COLOR_BGR2GRAY) sin transformar a gris.
    
    percent = ( pxlsNoZero / ( (width/2)*(height/2) ) *100 )
    
    if percent > 40:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        img_counter +=1
    print('Porcentaje Gris: ', round(percent, 2))

    x.append(datetime.now()) #.strftime('%H:%M:%S.%f')
    y.append(percent)  
    drawnow(make_fig)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
