import os
import cv2

face = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

def crop_face(img,img_name):
    
    img_resized = cv2.resize(img,(700,750))
    gray = cv2.cvtColor(img_resized,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    if len(faces) != 0:
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img_resized[y:y+h, x:x+w]
            eyes = eye.detectMultiScale(roi_gray)
            if len(eyes)>= 2:
                return roi_color
            else : 
                print(f"{img_name}.......eyes not detected!!!")
                return None
    else: 
        print(f"{img_name}.......face not detected!!!")
        return None
    

def filter_face(datapath):

    """
    datapath would be the file path of the dataset where classes folder contains the images are located
    """
    
    cropped_datapath = os.path.join(datapath,'Cropped')
    if not os.path.exists(cropped_datapath):
        os.mkdir(cropped_datapath)
    img_dirs = []
    for x in os.scandir(datapath):
        if x.is_dir():
            img_dirs.append(x.path)
    for img_dir in img_dirs: 
        celebrity_name = img_dir.split("\\")[-1]
        count = 0
        for y in os.scandir(img_dir):
            if y.is_file() and y.name.endswith(('.jpg', '.jpeg', '.png')): 
                path = y.path
                img_name_list = path.split("\\")[-2:]
                img_name = "/".join(img_name_list)
                img = cv2.imread(path)
                extension = y.path.split('.')[-1]
                crop_img = crop_face(img,img_name)
                count += 1
                if crop_img is not None: 
                    cropped_folder = os.path.join(cropped_datapath,celebrity_name)
                    if not os.path.exists(cropped_folder):
                        os.mkdir(cropped_folder)
                    if not os.path.exists(os.path.join(cropped_folder,f"{celebrity_name}_{count}.{extension}")):
                        cv2.imwrite(os.path.join(cropped_folder,f"{celebrity_name}_{count}.{extension}"),crop_img)