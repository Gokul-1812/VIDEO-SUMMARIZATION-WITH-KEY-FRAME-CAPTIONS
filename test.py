#importing packages required
from torchvision import transforms
#Transforms are common image transformations.
import numpy as np
#NumPy is a Python library that provides a simple yet powerful data structure:
#the n-dimensional array.
import time
import glob as gl
#Glob is a general term used to define techniques to match specified patterns
# according to rules related to Unix shell
import random
import argparse
import shutil,os.path

#The argparse module makes it easy to write user-friendly 
#command-line interfaces.
import h5py
#The h5py package is a Pythonic interface to the HDF5 binary data format. 
#It lets you store huge amounts of numerical data, and easily manipulate that 
#data from NumPy
from tensorflow.keras.preprocessing import image
import json
import torch.nn.init as init
import pandas as pd
from pandas.core.common import flatten
import cv2
import os
from shutil import copy
from PIL import Image
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import keras
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import sys, time, os, warnings 
from collections import Counter 
import warnings
warnings.filterwarnings("ignore")
import logging #Disable Tensorflow debugging information
tf.get_logger().setLevel(logging.ERROR)
# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re
import pickle
import requests
from keras.applications.vgg16 import VGG16, preprocess_input
import torch
from sklearn.cluster import KMeans
from PIL import Image as pil_image
import re
import moviepy
from moviepy.editor import VideoFileClip
from datetime import datetime

from config import  *
from sys_utils import *
from vsum_tools import generate_summary
from vasnet_model import  *
from feat_extract import extract_feats
from generate_summarized_vid import generate_summarized_vid
from cpd_auto import cpd_auto
from cpd_nonlin import cpd_nonlin
# from image_cap import *
# from image_captioning import predict
from caption import *


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)
def parse_splits_filename(splits_filename):
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  
    dataset_type = sfname.split('_')[1]  
    if dataset_type == 'splits':
        dataset_type = ''
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)
    return dataset_name, dataset_type, splits
def lookup_weights_splits_file(path, dataset_name, dataset_type, split_id):
    dataset_type_str = '' if dataset_type == '' else dataset_type + '_'
    weights_filename = path + '/models/{}_{}splits_{}_*.tar.pth'.format(dataset_name, dataset_type_str, split_id)
    weights_filename = glob.glob(weights_filename)
    if len(weights_filename) == 0:
        print("Couldn't find model weights: ", weights_filename)
        return ''
    weights_filename = weights_filename[0]
    splits_file = path + '/splits/{}_{}splits.json'.format(dataset_name, dataset_type_str)
    return weights_filename, splits_file
class AONet:
    def __init__(self, hps: HParameters):
        self.hps = hps
        self.model = None
        self.log_file = None
        self.verbose = hps.verbose
    def load_model(self, model_filename):
        self.model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage), strict=False)
        return
    def initialize(self, cuda_device=None):
        rnd_seed = 12345
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)
        self.model = VASNet()
        #model is a VASNet model
        self.model.eval()
        self.model.apply(weights_init)
        return
    def lookup_weights_file(self, data_path):
        dataset_type_str = '' if self.dataset_type == '' else self.dataset_type + '_'
        weights_filename = data_path + '/models/{}_{}splits_{}_*.tar.pth'.format(self.dataset_name, dataset_type_str, self.split_id)
        weights_filename = glob.glob(weights_filename)
        if len(weights_filename) == 0:
            print("Couldn't find model weights: ", weights_filename)
            return ''
        weights_filename = weights_filename[0]
        splits_file = data_path + '/splits/{}_{}splits.json'.format(self.dataset_name, dataset_type_str)
        return weights_filename, splits_file
def centering(K):
    """Apply kernel centering"""
    mean_rows = np.mean(K, 1)[:, np.newaxis]
    return K - mean_rows - mean_rows.T + np.mean(mean_rows)

def test(source_vid,hps,weights_filename,fe_csv_path,fe_output_path,req_clsuter):
  #hps contains hyperparameters, weigths_filename contain the file containing weights
    #fe_csv_path contains extracted features
    #fe_output_path contains the path to which we want the video to be saved to.
    import torch
    import pandas as pd
    df = pd.read_csv(fe_csv_path)
    fe = df.to_numpy()
    fe=fe[:,1:]
    ao = AONet(hps)
    #creating an AOTet with the given parameters.
    ao.initialize()
    ao.load_model(weights_filename)
    seq = fe
    seq = torch.from_numpy(seq).unsqueeze(0)
    if ao.hps.use_cuda:
        seq = seq.float()
    y, att_vec = ao.model(seq, seq.shape[1])
    att_vec=att_vec.cpu().detach().numpy()
    y=y.cpu().detach().numpy()
    K = np.dot(y.T, y)
    n = K.shape[0]
    num_frames = y.shape[1]
    vmax = np.trace(centering(K)/n)
    cps,scores = cpd_auto(K,num_frames//2,1,vmax)
    import pandas as pd
    df=pd.DataFrame(cps)
    df.to_csv(fe_output_path+'/cps.csv')
    df = pd.read_csv(fe_output_path+'/cps.csv')
    df = df.to_numpy()
    c_names=[]
    f_arr=df[:,1]
    #get summary frames with change points
    for i in f_arr:
        c_names.append('frame'+str(i)+'.jpg')
    #print(c_names)
    # main_dir= "/content/drive/MyDrive/VSWKC/Summariser"
    main_dir = "videos/"
    folder = source_vid.split('.')[0]
    op = os.path.join(main_dir, folder+'/cps_sum')
    if not os.path.exists(op):
        print("No diriectory cps_sum")
        os.system('mkdir '+op)
    pa=main_dir+'/'+folder+'/frames'
    fig=os.listdir(pa)
    file_a=c_names
    for file in file_a:
      if file in fig:
        copy(os.path.join(pa,file),op)
      else:
        print("not in frames\n")
    orgin=main_dir+'/'+folder+'/cps_sum_rgb'
    if not os.path.exists(orgin):
        os.system('mkdir '+orgin)
    for g in os.listdir(op):
      imUMat = cv2.imread(os.path.join(op, g))
      im_rgb = cv2.cvtColor(imUMat,cv2.COLOR_BGR2RGB)
      cv2.imwrite(orgin+'/'+g, im_rgb)
#highest score
    reg_score_list=[]
    reg_score_arr=(y.flatten())
    reg_score_list=reg_score_arr.tolist()
    print(reg_score_list)
#end highest score
#cluster
    image.LOAD_TRUNCATED_IMAGES = True 
    model = VGG16(weights='imagenet', include_top=False)
    imdir=orgin
    targetdir=os.path.join(main_dir, folder+'/cluster')
    #imdir = '/content/drive/MyDrive/summarizer/sunset/cps_sum_rgb'
    #targetdir = '/content/drive/MyDrive/summarizer/sunset/cluster'
    number_clusters = int(req_clsuter)
    filelist = gl.glob(os.path.join(imdir, '*.jpg'))
    filelist.sort()
    featurelist = []
    for i, imagepath in enumerate(filelist):
        print("    Status: %s / %s" %(i, len(filelist)), end="\r")
        # img = image.load_img(imagepath, target_size=(224, 224))
        img = image.load_img(imagepath, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = np.array(model.predict(img_data))
        featurelist.append(features.flatten())
    kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(featurelist))
    label=(kmeans.labels_).tolist()
    print(label)
    print(len(kmeans.labels_))
    if not os.path.exists(targetdir):
        print('in fe mkdir '+targetdir)
        os.system('mkdir '+targetdir)
    print("\n")
    for i, m in enumerate(kmeans.labels_):
       print("    Copy: %s / %s" %(i, len(kmeans.labels_)), end="\r")
       shutil.copy(filelist[i], targetdir + str(m) + "_" + str(i) + ".jpg")
    
    clr_list=[]
    key_f=[]
    src_dir=os.path.join(main_dir, folder)
    c_files=gl.glob(os.path.join(src_dir, '*.jpg'))
    for f in c_files:
     clr_list.append(os.path.basename(f))
    print(clr_list)
    lib=[]
    for i in range(number_clusters):
      lib.append([])
    for fil in sorted(clr_list):
      temp=re.findall(r'\d+',fil)
      lofl=list(map(int,temp))
      lib[lofl[0]].append(fil)
    print(lib)
    for lists in sorted(lib):
      tot_length=len(lists)
      mid=int(tot_length/2)
      key_f.append(lists[mid])
    print(key_f)
    flee=os.listdir(src_dir) 
    par_c= os.path.join(main_dir,folder+'/clustered') 
    #p=par+'/sum_frames'
    if not os.path.exists(par_c):
      print('in fe mkdir '+par_c)
      os.system('mkdir '+par_c)
    for file in key_f:
     if file in flee:
        copy(os.path.join(src_dir,file),par_c)
     else:
        print("not in frames\n")
    need=os.listdir(par_c)
    need_fldr=main_dir+'/static'
    if not os.path.exists(need_fldr):
      print('in fe mkdir '+need_fldr)
      os.system('mkdir '+need_fldr)
    rest=need_fldr+'/clustered_images'
    if not os.path.exists(rest):
      print('in fe mkdir '+rest)
      os.system('mkdir '+rest)
    inside_rest=os.path.join(rest,folder)
    if not os.path.exists(inside_rest):
      print('in fe mkdir '+inside_rest)
      os.system('mkdir '+inside_rest)
    for file in need:
      par_c = "videos/static/clustered_images/sample15"
      inside_rest = "videos/static/clustered_images/sample15"
      print(par_c," -- ", file, " -- ", inside_rest)
      src = os.path.join(par_c,file)
    #   copy(src, inside_rest)
      inside_rest = src

#cluster_end
#color histo 

#end color histo

#summary short video
    vid_par= ""
    vid_fold = source_vid.split('.')[0]
    vid_p = os.path.join(vid_par,vid_fold+'/sum_video')
    orgin2=vid_par+'/'+vid_fold+'/key_frame_rgb'
    if not os.path.exists(vid_p):
        os.system('mkdir '+vid_p)
    img_array = []
    for filename in gl.glob(orgin2+'/'+'*.jpg'):
       img = cv2.imread(filename)
       height, width, layers = img.shape
       size1 = (width,height)
       img_array.append(img)
       out = cv2.VideoWriter(vid_p+'/'+'output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size1)
       for i in range(len(img_array)):
         out.write(img_array[i])
       out.release()
#end summary short video
    cps=df[:,1]
    start=cps[0]
    cps_pair=[]
    cnt=1
    frame=df[:,1]
    start=frame[0]
    dummy_start=frame[0]
    frame_pair=[]
    cnt=1
    while cnt!=len(cps):
        end = cps[cnt]
        if dummy_start+1!=end:
            cps_pair.append([start,cps[cnt-1]])
            start = end
            dummy_start = start
            cnt=cnt+1
        else:
            dummy_start = end
            cnt=cnt+1
    if cps_pair[-1][1] != y.shape[1]:
        cps_pair.append([cps_pair[-1][1],y.shape[1]])
    positions = range(0,y.shape[1],15)
    positions = np.asarray(positions)
    probs = np.asarray(y)
    nfps = []
    for i in cps_pair:
        nfps.append(i[1]-i[0])
    cps_pair = np.asarray(cps_pair)
    nfps = np.asarray(nfps)
    machine_summary = generate_summary(probs[0], cps_pair, num_frames, nfps, positions)
    machine_summary = machine_summary.tolist()
    summary_frames = [i for i in range(len(machine_summary)) if machine_summary[i] == 1]
    f_names = []
    for i in summary_frames:
        f_names.append('frame'+str(i)+'.jpg')
    par= "videos/"
    dire = source_vid.split('.')[0]
    p = os.path.join(par, dire+'/sum_frames')
    if not os.path.exists(p):
        os.system('mkdir '+p)
    path=par+'/'+dire+'/frames'
    fl=os.listdir(path)
    file_arr=f_names
    for file in file_arr:
      if file in fl:
        copy(os.path.join(path,file),p)
      else:
        print("not in frames\n")
    org=par+'/'+dire+'/final_rgb'
    if not os.path.exists(org):
        os.system('mkdir '+org)
    for fil in os.listdir(p):
      imgUMat = cv2.imread(os.path.join(p, fil))
      image_rgb = cv2.cvtColor(imgUMat,cv2.COLOR_BGR2RGB)
      cv2.imwrite(org+'/'+fil, image_rgb)
    del sys.modules['torch']
    del torch
    return cps_pair,summary_frames

def run_sum(source_vid,req_clsuter):
    hps = HParameters()	
    #configuration of parameters loaded
    print(source_vid.split('./process/')[-1].split('.')[0])
    parent_dir = "videos/"
    directory = source_vid.split('.')[0]
    path = os.path.join(parent_dir, directory)
    print(str(path) + " created at time : " + str(datetime.now().strftime("%H:%M:%S")))
    # os.mkdir(path)
    video_path = ROOT_DIR + "/videos/" + source_vid
    fe_output_path= path
    fe_csv_path = extract_feats(video_path,fe_output_path)
    weights_filename = "model_weight/summe_aug_splits_1_0.443936558699067.tar.pth"
    #load attention weights
    cps_pair,summary_frames = test(source_vid,hps,weights_filename,fe_csv_path,fe_output_path,req_clsuter)
    #hps contains hyperparameters, weigths_filename contain the file containing weights
    #fe_csv_path contains extracted features
    #fe_output_path contains the path to which we want the video to be saved to.
    #code to create summary video
    df=pd.DataFrame(summary_frames)
    df.to_csv(fe_output_path+'/summary_frames.csv')
    df=pd.read_csv(fe_output_path+'/summary_frames.csv')
    df = df.to_numpy()
    frame=df[:,1]
    start=frame[0]
    dummy_start=frame[0]
    frame_pair=[]
    cnt=1
    while cnt!=len(frame):
        end = frame[cnt]
        if dummy_start+1!=end:
            frame_pair.append([start,frame[cnt-1]])
            start = end
            dummy_start = start
            cnt=cnt+1
        else:
            dummy_start = end
            cnt=cnt+1    
    lenFramePair = len(frame_pair)
    count = 0
    cnt=0
    finalFramePair = []
    frames = list(range(frame_pair[0][0],frame_pair[0][1]+1,7))
    finalFramePair.append(frames)
    count=count+1
    while count!=lenFramePair:
        last = frame_pair[count-1][1]
        next_ = frame_pair[count][0]
        if ((next_-last<=40) and (len(finalFramePair[cnt])<=20)):
            frames = list(range(frame_pair[count][0],frame_pair[count][1]+1,1))
            finalFramePair[cnt].append(frames)
        else:
            l = finalFramePair[cnt]
            finalFramePair[cnt] = list(flatten(l))
            cnt=cnt+1
            frames = list(range(frame_pair[count][0],frame_pair[count][1]+1,1))
            finalFramePair.append(frames)
        count=count+1
    for count,i in enumerate(finalFramePair):
        frames = np.asarray(i)
        generate_summarized_vid(frames,fe_output_path,count,video_path)

def summarise(source_vid):
    # root="/content/drive/MyDrive/VSWKC/Summariser"
    # clip = VideoFileClip(root+'/'+source_vid)

    # name = str(datetime.now().microsecond) + str(datetime.now().month) + '-' + str(datetime.now().day) +  '.jpg'
    name = "sample15.mp4"
    # photo = request.files['photo']
    path = os.path.join("videos",name)
    
    clip = VideoFileClip(path)
    video_duration=clip.duration
    req_cluster=int(0.15*video_duration)
    if (req_cluster==0):
      req_cluster=5
    run_sum(source_vid,req_cluster)
    return "finished"
