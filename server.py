from flask import Flask ,request, url_for, jsonify
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename

import caption as cp

import os
import time
import pandas as pd

CAP_FOLDER = "videos/sample/static/clustered_images"

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def home():
  return "server look-alike ON"



@app.route('/caption/<video_name>', methods = ['GET'])
def work(video_name):

  print("Starting captioner.....")
  
  file_name = video_name.split(".mp4")[0]
  captions_folder = os.path.join(CAP_FOLDER,file_name)
  found = False

  while(True):
    cap_exists = os.path.exists(captions_folder)
    if cap_exists:
      found = True
      print("Folder Found")
      break
    print("Searching"+str(captions_folder))
    time.sleep(5)
  
  if found:
    
    csv_path = 'videos/sample/static/clustered_images/'+str(file_name)+'/file.csv'
    csv_path = os.path.join(csv_path)

    cp.caption(video_name)

    if os.path.exists(csv_path):
      captions_csv = pd.read_csv(csv_path)
      images = [x for x in captions_csv['Filename']]
      image = []
      for i in images:
        temp = ".." + i.split("/Summariser")[1]
        image.append(temp)
      captions =[x for x in captions_csv['Caption']]
      captions_1 = [(x.split("['<start>")[1]).split("', '<end>']")[0] for x in captions]
      caption = []
      for i in captions_1:
        temp = i
        if "a close up of " in i:
          temp = i.replace("a close up of "," ")
        caption.append(temp)
      return jsonify([image,caption])
  
  return "404"


if __name__ == '__main__': 
  app.run()
