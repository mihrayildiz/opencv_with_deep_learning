import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

col_list = ["frame_number","identity_number","left","top","width","height","score","class","visibility"] 
#gt.txt içerisinde belirlenen numaraların isimleri

data = pd.read_csv("gt.txt", names = col_list)

plt.figure()
sns.countplot(data["class"]) #class dağılımları

car = data[data["class"] == 3]

video_path = "deneme.mp4"

cap =cv2.VideoCapture(video_path)

id1 = 29

number_image =np.max(data["frame_number"])
fps = 25
bund_box = []
for i in range(number_image-1):
    ret,frame = cap.read()
    if ret:
        frame =cv2.resize (frame,dsize=(960,540))
        filter_id = np.logical_and(car["frame_number"] ==i+1 , car["identity_number"] == id1) 
        #car identity_number = 29, yani 29 identity değei ile 29.id li arabanın olduğu frameleri al 
       
         if len(car[filter_id1]) != 0:
            
            x = int(car[filter_id1].left.values[0]/2)
            y = int(car[filter_id1].top.values[0]/2)
            w = int(car[filter_id1].width.values[0]/2)
            h = int(car[filter_id1].height.values[0]/2)
            
            
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame, (int(x+w/2),int(y+h/2)), 2, (0,0,255),-1)
        
                    # frame, x, y, genis, yuksek, center_x, center_y
            bound_box_list.append([i, x,y,w,h,int(x+w/2),int(y+h/2)])
            
        cv2.putText(frame, "Frame num:"+str(i+1), (10,30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,255),2)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"): break
    else: break

cap.release()
cv2.destroyAllWindows()        
  

df = pd.DataFrame(bound_box_list, columns = ["frame_no", "x", "y", "w", "h", "center_x", "center_y"])
      
df.to_csv("gt_new.txt",index=False)