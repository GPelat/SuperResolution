from click import FileError
from model_postup import FSRCNN, SuperResolutor
from model_preup import SRCNN, VDSR, SuperResolutorUpscaled
import numpy as np
import cv2
import os
from tqdm import tqdm
import json
import torch
from torchvision import transforms, utils


model_args = {}
pth_file = ""
target_folder = ""
filename = 0


def save(downsized32, downsized, original, total):
    for i in range(len(downsized)):
        cv2.imwrite(os.path.join(target_folder, "downsizedsmall", f"{total}.jpg".zfill(13)), downsized32[i])
        cv2.imwrite(os.path.join(target_folder, "downsized", f"{total}.jpg".zfill(13)), downsized[i])
        cv2.imwrite(os.path.join(target_folder, "original", f"{total}.jpg".zfill(13)), original[i])
        total += 1
    return total

def process_frame(model, img_tensor):
    # print(img_tensor.shape)
    with torch.no_grad():
        x = model(img_tensor)
    # print(x.shape)
    # print(x)
    return x[0].cpu().numpy()

def process():
    model = get_model()
    feed = cv2.VideoCapture(0)
    if not feed.isOpened():
        print("Error while loading file.")
        raise FileError(filename)
    
    tt = transforms.ToTensor()
    while(True):
        ret, img = feed.read()
        if not ret: 
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = (640,480)
        res_shape = (160,120)
        img = cv2.resize(img, res_shape)
        img = cv2.resize(img, shape)
        img_processed = process_frame(model, tt(img))
        # utils.save_image(tt(img), "img_before.jpg")
        # utils.save_image(img_processed, "img.jpg")
        cv2.imshow("original", img)
        cv2.imshow("upscaled", img_processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    feed.release()

    

def get_model():
    available_models = ["SRCNN", "VDSR", "FSRCNN"]
    if model_args["name"] not in available_models:
        print(f"The model {model_args['name']} is not (yet?) implemented.")
        print(f"Possible values: {available_models}")
        raise NotImplementedError
    
    if model_args["name"] == "SRCNN":
        model = SRCNN()
    elif model_args["name"] == "VDSR":
        model = VDSR(model_args["n_layers"])
    elif model_args["name"] == "FSRCNN":
        # model = FSRCNN()
        model = SuperResolutor()

    try:
        model.load_state_dict(torch.load(pth_file))
        print('Checkpoint loaded')
    except Exception as e:
        print(e)

    return model

def read_args():
    global model_args
    global pth_file
    global target_folder
    global filename
    
    with open("parameters.json", "r") as read_file:
        args = json.load(read_file)

    model_args = args["model"]
    filename = args["upscale"]["filename"]
    target_folder = args["upscale"]["target_folder"]
    pth_file = args["upscale"]["model_pth_path"]

if __name__ == "__main__":
    read_args()
    process()