import sys
import cv2
import imageio
import pylab
import numpy as np
import skimage.transform
from torchvision import models 
from torchvision.models import googlenet 
from PIL import Image 
import torch 
from torchvision import transforms 
import torch.nn as nn 
import os
import pandas as pd


def extract_feats(filename,output_path):
    
    if not os.path.exists(output_path):
        os.system('mkdir '+output_path)
        #print('mkdir '+output_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    model1 = googlenet(pretrained=True)
    lenet = nn.Sequential(*list(model1.children())[:-2])
    vid = imageio.get_reader(filename,'ffmpeg')
    curr_frames = []
    features = []
    pathOut=output_path+'/frames'
    #print(pathOut)
    if not os.path.exists(pathOut):
        #print('in fe mkdir '+pathOut)
        os.system('mkdir '+pathOut)
    count=0
    #print(pathOut)
    #print('extracting features')
    try:
        for frame in vid:
            name = os.path.join(pathOut, "frame{:d}.jpg".format(count)) 
            cv2.imwrite(name, frame)
            count+=1 
            #print(count)
    except Exception as e:
        print(e)
    count2=0
    for name in os.listdir(output_path+'/frames'):
        #print(count2)
        name=output_path+'/frames/'+name
        input_image = Image.open(name) 
        input_tensor = preprocess(input_image) 
        input_batch = input_tensor.unsqueeze(0)
        fe = lenet(input_batch)
        fe = torch.reshape(fe, (1, 1024)) 
        fe=fe[0]
        fe=fe.detach().cpu().numpy()
        features.append(fe)
        count2+=1
    #print(features)
    df = pd.DataFrame(features)
    df.to_csv(output_path+'/feature.csv')
    curr_frames = []
    features = []
    return output_path+'/feature.csv'