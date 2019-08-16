import os
import shutil
#import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

size = 20
width = int(size / 1.3)
height = int(size/0.9)
alfabet_all = "-|0123456789 qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
labels_all = {"X":128, "O":255}

def create_subimage(sym, name):
    font = ImageFont.truetype("fonts/CourierNew-B.ttf", size)
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    draw.text((0,0), sym,font=font)
    img = img.resize((width, int(height*1.5)))
    img.save("fonts/"+name+".png")
    return

def create_alfabet():
    if not os.path.exists("fonts"):
        os.mkdir("fonts")
    for i,s in enumerate(alfabet_all):
        create_subimage(s, str(i))
    return

def load_alfabet():
    a = {}
    for i,s in enumerate(alfabet_all):
        a[s] = ( i ,"fonts/" + str(i) + ".png")
    return a

def get_image_size(w_c, h_c):
    return int((size / 1.3)*w_c), int((1.5 * size/0.9)*h_c)

def get_subimage_step():
    return int(size / 1.3), int(1.5*size/0.9)

def create_image_list():
    root_dir = "data_segnet/"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    mask_path = root_dir + "masks/"
    image_path = root_dir + "images/"
    for path in [mask_path, image_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    src_dir = "data/"
    if not os.path.exists(src_dir):
        return
    f = open(root_dir + "dataset.txt", 'w')
    for l in os.listdir(src_dir):
        if not os.path.isdir(os.path.join(src_dir, l)):
            continue
        src_image_ = os.path.join(src_dir, l, "image/")
        src_mask_ = os.path.join(src_dir, l, "label/")
        for img in os.listdir(src_image_):
            shutil.copyfile(src_image_ + img, image_path + l + "_" + img)
            shutil.copyfile(src_mask_ + img, mask_path + l + "_" + img)
            f.write(l + "_" + img + "\n")
    f.close()
    return

#create_alfabet()
#create_image_list()

def resize_dataset(w,h):
    root_dir = "data_segnet/"
    image_dir = root_dir + "images/"
    mask_dir = root_dir + "masks/"
    f = open(root_dir + "dataset.txt",'r')
    lines = f.readlines()[:-1]
    for l in lines:
        img = Image.open(image_dir + l)
        img = img.resize((w,h))
        img.save(image_dir + l)
        mask = Image.open(mask_dir + l)
        mask = mask.resize((w, h))
        mask.save(image_dir + l)
    return

