import cv2
from PIL import Image
import os
from feature_extract import fire

def resize(path1,path2):
    resolution = (100,80)
    scaler = Image.ANTIALIAS
    if not os.path.exists(path2):
        os.makedirs(path2)
    listing=os.listdir(path1)
    for i,file_1 in enumerate(listing):
        img=Image.open(path1 + file_1).convert("RGB")
        if img:
            res=img.resize(resolution , Image.ANTIALIAS)
            res.save(path2+"{}.jpg".format(i))


def main():

    global q

    before = "./source/"
    after = "./image/"

    folders = ["forest/","fire/","maple/","cloud/"]

    for f in folders:
        resize(before+f,after+f)
        print("done")
    for f in folders:
        resize(before+'test/'+f,after+'test/'+f)
        print("done")

if __name__ == "__main__":main()
