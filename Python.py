#%%tuple 
"""
Değiştirilemez ve ıralı bir veri tipidir.
"""

tuple = (1,2,3,3,4,5)
print(tuple[0])
print(tuple[1:3])
print(tuple.count(3))

tuple_xyz = (7,8,9)
x,y,z = tuple_xyz
print(x,y,z)

#%% Deque

from collections import deque

dq =deque(maxlen=3)

dq.append(3)
print(dq)

dq.append(2)
dq.append(3)

print(dq)

dq.append(5)
print(dq) #elemanlar 2,3,5 olur çünkü maxlen = 3 'tür

dq.appendleft((6)) #♠soldan ekle yani ilk sıraya ekle 

print(dq)

#%% Dictionary

"""
Karma tablo türüdür.
anahtar ve değer çiftlerinden oluşurlar.
{"anahtar" : "value"}

"""

d1 = {"istanbul" : 34,
      "bursa":16,
      "izmit":41}
print(d1)

print(d1["istanbul"])
print(d1.keys())
print(d1.values())

#%% Koşullu İfadeler: if-else

"""
Bi bool ifadesine göre doğru yada yanlış değerlenirilmesine bağlı olarak farklı hesaplamalar
veya eylemler gerçekleştiren ifadelerdir.

"""

sayi1 = 5
sayi2 = 6 

if sayi1<sayi2:
    print("sayi1 küçüktür sayi2")
elif sayi1 >sayi2:
    print("sayi1 büyüktür sayi2")
else:
    print("sayılar eşit")


liste  = [1,2,3,4,5]
deger =32 

if deger in liste:
    print("{} listein içerisindedir.".format(deger))
    
else:
    print("{} listenin içerisinde değildir.".format(deger) )
    
    
sehir = "konya"
keys = d1.keys()

if sehir in keys:
    print("{} sözlükte bulunur".format(sehir))    
else :
        print("{} sözlükte yok".format(sehir))   
        
    
#%% Döngüler

"""
Bir dizi üzerinde yineleme yapmak için kullanılan yapılardır.
diziler : liste,tuple,dictionary,string
"""
for i in range(1,11):
    print(i)
    
    
tuple = ((1,2,3),(4,5,6))

for x,y,z in tuple:
    print(x,y,z)
    
    
i = 0
while i<4:
    print(i)
    i += 1
    
    
    
list = [1,2,3,4,5,6,7,8,9,12,4,15]
sinir = len(list)

her =0
toplam =0

while her<sinir:
    toplam +=list[her]
    her += 1
print(toplam)
    
    
#%% Fonksiyonlar

#user defined

def daireAlanı(r):
    """
    Parameters
    ----------
    r : int - yarıçap

    Returns
    -------
    dairealani : float - dairenin alanı 

    """
    pi =3.14
    dairealani = pi*(r**2)
    return dairealani

daire_alanı = daireAlanı(8)
print(daire_alanı)
    
    
    
def daireCevre(r, pi =3.14):
    """
    

    Parameters
    ----------
    r : int - yarıçap
    pi : float

    Returns
    -------
    

    """
    daire_cevre = 2*pi*r
    return daire_cevre

daire_cevre = daireCevre(3)
print(daire_cevre)
#  daire_cevre = daireCevre(3,5) denilerek pi =5 olarakte verilebilir.


katsayi =5
def katsayiCarpim():
    global katsayi #fonksiyon dışında global olarak aldı.
    carpım = katsayi*katsayi
    print(carpım)
katsayiCarpim()

#boş fonksiyon
def bos():
    pass

#built-in fuctions
#hazır fonksiyonlar

liste = [1,2,3,4,5]
liste2= liste.copy()
print(liste2)

print(liste)

print(max(liste2))
print(min(liste))

#Lambda Functions
"""
ileri seviyeli
küçük ve anonim
"""

carpim = lambda x,y,z : x*y*z
carpim(2,3,4)

#%% Yield

#iterasyon 
list = [1,2,3,4]
for i in list:
    print(i)
    
    
"""
#generator yineleyici 
generator değerleri bellekte saklamazlar gerektiğinde anında üretirler.
"""

generator = (x for x in range(1,4))
 for i in generator:
     print(i)


""" 
#yield
fonksiyon eğer bir generator döndürecek ise "return" yerine "yield" döner.
"""

def create_generator():
    liste = range(1,4)
    fo i in liste:
        yield i
        
generator = create_generator()
print(generator)

#%% NumPy Kütüphanesi

"""
Matrisler için hesaplama kolaylığı sağlar.

"""
import numpy as np
dizi = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print(dizi)
print(dizi.shape)    

dizi2 = dizi.reshape((3,5))    
print(dizi2)     #şekil
print(dizi2.ndim) #boyut
print(dizi2.dtype) #veritipi
print(dizi2.size) #boy

#array type
print("type : ", type(dizi2))

#2 boyutlu array

dizi2D = np.array([[1,2,3,4],[5,6,7,8],[8,9,5,6]])
print(dizi2D)

sifir_diz = np.zeros((3,4))
print(sifir_diz)

bir_array  = np.ones((3,4))
print(bir_array)

bos_dizi = np.empty((3,4))
print(bos_dizi)


dizi_aralik = np.arange(10,50,5)
print(dizi_aralik)

dizi_line = np.linspace(10, 20,5)
print(dizi_line)

float_array = np.float32([[1,2],[3,4]])
print(float_array)

#matematikel işlemler

a = np.array([1,2,3])
b= np.array([4,5,6])

print(a+b)
print(a-b)
print(a**2)

print(np.sum(a))
print(np.max(a))

print(np.mean(a))
print(np.median(a))


rastgele_dizi = np.random.random((3,3)) #0-1 arası rastgele sayilar üretmek
print(rastgele_dizi)

dizi = np.array([1,2,3,4,5,6,7])
print(dizi[0])
print(dizi[0:4])

print(dizi[::-1]) #dizinin tersi

dizi2D = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(dizi2D[0,0])

print(dizi2D[:,1])
print(dizi2D[1,0:4])
print([-1,:])

vektor = dizi2D.ravel() #vektör haline çevirme
print(vektor)

max_sayinin_indeksi = vektor.argmax()
print(max_sayinin_indeksi)

#%% Pandas Kütüphanesi

"""
hızlı, güçlü ve esnek 
"""
import pandas as pd
dictionary = { "isim" : ["ali", "veli", "ayse","hilal"],
               "yas" : [15,16,20,25],
               "maas": [100,155,122,300]}

veri = pd.DataFrame(dictionary)"

print(veri.head())
print(veri.columns)
print(veri.info())
print(veri.describe())

print(veri["yas"])

veri["sehir"]  = ["ankara","bursa","ist", "gebze"]

print(veri.loc[:3, "yas"])

print(veri.loc[:3, "yas":"sehir"])
print(veri.iloc[:,1])

filtre = veri.yas > 22 
filtrelenmis_veri = veri [filtre]
print(filtrelenmis_veri)

ortalama_yas = veri.yas.mean()

veri["yas_grubu"] = ["kucuk" if ortalama_yas >i else "buyuk" for i in veri.yas]
print(veri)

#%% Matplotlib Kütüphanesi

"""
Göselleştirme
"""
import matplotlib.pyplot as plt
import numpy as np

a = np.array([1,2,3,4])
b = np.array([4,3,2,1])


plt.figure()
plt.plot(a,b, color ="red", alpha =0.7,  label= "line")
plt.scatter(a,b, color ="blue", alpha=0.4, label ="scatter")
plt.title ("matplotlib")
plt.xlabel("a")
plt.ylabel("b")
plt.grid(True)
plt.xticks([0,1,2,3,4,5])
plt.legend()
plt.show()

fig,axes = plt.subplot((2,1), figsize= (9,7))
fig.subplots_adjust(hspace =0.5)

x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,4,5,6,7,8,9,10]

axes[0].scatter(x,y)
axes[0].set_title("sub-1")
axes[0].set_ylabel("sub-1 y")
axes[0].set_xlabel("sub-1 x")


axes[1].scatter(x,y)
axes[1].set_title("sub-2")
axes[1].set_ylabel("sub-2 y")
axes[1].set_xlabel("sub-2 x")

# random image

img = np.random.random((50,50))
plt.imshow(img,cmap ="gray")
plt.axis("off")
plt.show()

#%% OS Kütüphanesi

"""
pc içerisindeki dosyalara ulaşmamızı ve resimleri yüklememizi sağlayacak.
"""
import os

print(os.name)

currentDir= os.getcwd() #şu an çalışılan dosya
print(currentDir)

folder_name = "new_folder" #new_folder oluşturduk.
os.mkdir(folder_name)

new_folder_name = "new_folder_name"
os.rename(folder_name,new_folder_name)

os.chdir(currentDir + "\\" + "new_folder_name")
print(os.getcwd())

os.chdir(currentDir)
print(os.getcwd())


files = os.listdir() #dosya içindeki fileslar sıralandı.
print(files)

os.rmdir(new_folder_name) #new_folder_name silindi

for i in os.walk(currentDir):
    print(i)





































































