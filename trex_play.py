from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

mon = {"top":300, "left":770, "width":250, "height":100}
sct = mss()

width = 125
height = 50

# model yükle
model = model_from_json(open("model.json","r").read())
model.load_weights("trex_weight.h5")

# down = 0, right = 1, up = 2
labels = ["Down", "Right", "Up"]

framerate_time = time.time()
counter = 0
i = 0
delay = 0.4 #bir komuttan sonra beklenmesi için verilen süre
key_down_pressed = False
while True:
    
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im2 = np.array(im.convert("L").resize((width, height)))
    im2 = im2 / 255 #normalizzasyn uygula
    
    X =np.array([im2])
    X = X.reshape(X.shape[0], width, height, 1)
    r = model.predict(X)
    
    result = np.argmax(r)
    
    
    if result == 0: # down = 0
        
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True
         
    elif result == 2:    # up = 2
        
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN) #önceden kalan butonu bırakmak için
        time.sleep(delay)
        keyboard.press(keyboard.KEY_UP)
        
        if i < 1500: #1500. frameden sonra oyun hızlanmaya başlar bu yüzde up iken 
            time.sleep(0.3)
        elif 1500 < i and i < 5000:
            time.sleep(0.2)
        else:
            time.sleep(0.17)
            
        keyboard.press(keyboard.KEY_DOWN) #yukarıdayken aşağı basmalı
        keyboard.release(keyboard.KEY_DOWN) #tuşu bırakmalı bırakmaz ise basılı kalır.
    
    counter += 1
    
    if (time.time() - framerate_time) > 1: #şu an ki zaman - oyun başlama zamanı
        
        counter = 0
        framerate_time = time.time()
        if i <= 1500: #1500. frame ise
            delay -= 0.003 #geikmeyi düşür çükü oyun hızlanır bu yüzden gcikme süreside azalmalıdır.
        else:
            delay -= 0.005
        if delay < 0:
            delay = 0
            
        print("---------------------")
        print("Down: {} \nRight:{} \nUp: {} \n".format(r[0][0],r[0][1],r[0][2]))
        i += 1
        




























