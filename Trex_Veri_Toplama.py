import keyboard  #tuşları  kullanmak için gerekli olan kütüphane
import uuid #ekrandan kayıt alınır
import time
from PIL import Image
from mss import mss

mon = {"top": 300, "left": 770, "width": 250, "height": 100}
sct =mss() #pikseller doğrultusunda ilgili alanı kesip frame haline dönüştüren kütüphane

i = 0

def record_screen(record_id,key):
    global i
    i+=1
    print("{} : {}".format(key, i))  #key = klavyede asıln tuş, i= kaç kez bastığımız
    img = sct.grab(mon) #ekranı al belirlediğim mon doğrultusunda
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save("./img/{}_{}_{}.png".format(key, record_id,i))
    
    
is_exit = False #veri dplama işleminden çıkmak için

def exit():
    global  is_exit
    is_exit = True
    
keyboard.add_hotkey("esc", exit)
record_id = uuid.uuid4()
while True:
    if is_exit: break
    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id, "up")
            time.sleep(0.1) #her tuşa basmadan sonra 0.1 saniye bekle
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id, "down")
            time.sleep(0.1)
        elif  keyboard.is_pressed("right"):
              record_screen(record_id, "right")
              time.sleep(0.1)
    except RuntimeError: continue





















