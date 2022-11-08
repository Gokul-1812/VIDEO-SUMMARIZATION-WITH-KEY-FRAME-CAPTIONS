from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
from werkzeug.serving import WSGIRequestHandler
from datetime import datetime
import imghdr
import json
import requests
import os
import pandas as pd
from test import *

#AUTH_CODE = "4/0AY0e-g7QkYMQ12tMhDNLuZHKllnWaD18BwiMbK4W43Z6XPppXMo7CX38_iKLTI-sdCZneg"
ACCESS_CODE = "ya29.a0AfH6SMAjQLMQeQXAaHQ1q_URhqywmDv5oLQlTASzqt6zjbZKh4Vetp4nguAJ_VvAgaxyuw0_YLTOyYRWCrArS9bjeis9qcSv_kbMJ5-IZaxAEwHH_MU2zGUM1ljipsgd2V_9toAuG8LQuvMgKM85r0-M_JVg"
IMG_FOLDER = "/content/drive/MyDrive/VSWKC/Summariser/static/clustered_images"
CAP_FOLDER = "/content/drive/MyDrive/VSWKC/Summariser/static/clustered_images"


ALLOWED_EXTENSIONS = {'mp4', '3gp'}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
run_with_ngrok(app)


def delete_all_in_folder(dir_name):
    test = os.listdir(dir_name)

    for images in test:
        if images.endswith(".jpg"):
            os.remove(os.path.join(dir_name, images))

        if images.endswith(".mp4"):
            os.remove(os.path.join(dir_name, images))

        if images.endswith(".csv"):
            os.remove(os.path.join(dir_name, images))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/working')
def work():
    summarise("vdeo.mp4")
    return render_template('process.html')


@app.route('/img')
def get_image():
    if request.args.get('file') == '':
        return 'error'

    filename = request.args.get('file')

    return send_file("D:/mini/summariser/videos/sample15/clustered/"+filename, mimetype='image/jpg')


@app.route('/vids')
def get_videos():
    if request.args.get('file') == '':
        return 'error'

    filename = request.args.get('file')

    return send_file("D:/mini/summariser/videos/sample15/output/"+filename, mimetype='video/mp4')



@app.route('/summ', methods=['POST'])
def start_summ():
    f = request.files['video']

    if f and not allowed_file(f.filename):
        response = {"status": 500,
                    "status_msg": "File extension is not permitted"}
        return jsonify(response)

    print("\nGot post request from form at time : " +
          str(datetime.now().strftime("%H:%M:%S")) + "\n")

    if os.path.exists('videos/sample15.mp4'):
        os.remove('videos/sample15.mp4')

    delete_all_in_folder('videos/sample15')
    delete_all_in_folder('videos/sample15/clustered')
    delete_all_in_folder('videos/sample15/cps_sum')
    delete_all_in_folder('videos/sample15/cps_sum_rgb')
    delete_all_in_folder('videos/sample15/final_rgb')
    delete_all_in_folder('videos/sample15/frames')
    delete_all_in_folder('videos/sample15/sum_frames')
    delete_all_in_folder('videos/sample15/summary_frames')
    delete_all_in_folder('videos/sample15/output')
    # return "AAAA"

    f.filename = "sample15.mp4"
    video_name = f.filename

    f.save(os.path.join('videos', video_name))
    print(f.filename)

  

    # return "1111"
    #headers = {"Authorization": "Bearer " + ACCESS_CODE}

    # parameter = {
    #        "name" : f.filename,
    #       "parents" : ["1eV0RTsMq2lbka2hvRrpH95y7DqTX5hVV"]
    #         }
    # files = {
    #          'data': ('metadata', json.dumps(parameter), 'application/json; charset=UTF-8'),
    #         'file': open(f.filename, "rb")
    #      }
    # r = requests.post(
    #                     "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
    #                     headers=headers,
    #                       files=files)

    summarise(video_name)


    clustered = os.listdir('videos/sample15/clustered')
    outputs = os.listdir('videos/sample15/output')

    clustered_list = []
    for images in clustered:
        if images.endswith(".jpg"):
            clustered_list.append(images)

    outputs_list = []
    for videos in outputs:
        if videos.endswith(".mp4"):
            outputs_list.append(videos)

    response = {"status": 200, "status_msg": "Keyframes Loaded",
                "imgs": clustered_list, "videos": outputs_list}
    return jsonify(response)

    

    print("**-----------------**")
    video_name = video_name[:-4]
    image_folder = os.path.join(IMG_FOLDER, video_name)

    req_captions = []
    req_images = []

    # change server link here
    r = requests.get("http://127.0.0.1:5000/caption/"+str(f.filename))

    # print(r)

    # response = {"status": 200, "status_msg": "Clustered images and Videos loaded !!"}
    # return jsonify(response)

    if r.status_code == 200:

        caption_image = r.json()
        req_images = caption_image[0]
        req_captions = caption_image[1]

        return render_template('output.html', file_name=video_name, count=len(req_captions), folder=req_images, cap=req_captions)

    return "Clustered images not exists"


if __name__ == '__main__':
    app.debug = True
    app.use_reloader = False
    app.threaded = True
    app.run()
