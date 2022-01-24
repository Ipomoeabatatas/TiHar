# reference codes: https://github.com/robertklee/COCO-Human-Pose/blob/main/human_pose_app.py
# https://github.com/robertklee/COCO-Human-Pose/blob/main/human_pose_app.py
# https://github.com/rejexx/Parkingspot_Vacancy/blob/main/src/streamlit_app.py

from tkinter import image_names
import streamlit as st
import os
import cv2
import numpy as np
import time
import tensorflow as tf
from PIL import Image

#MODEL = './model/lightweight-ltc-cnn-model-10.h5'  # Nov 2021 version
MODEL = './model/lightweight-multiscale-ltc-cnn-model-all-frames-13.h5'  # LTC-CNN
VIDEO_EXTENSIONS = ["mp4", "ogv", "m4v", "webm"]
VIDEO_DIR = "./data"
UPLOAD_DIR = "./upload"  

PROJECT_INFO = "./project-info/about.md"
TEAM_DIR = './project-info'

# Constants for sidebar dropdown
SIDEBAR_OPTION_PROJECT_INFO = "Show Project Info"
SIDEBAR_OPTION_DEMO_IMAGE = "Select a Demo Video"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload Your Video"
SIDEBAR_OPTION_MEET_TEAM = "Meet the Team"
SIDEBAR_OPTIONS = [SIDEBAR_OPTION_PROJECT_INFO, SIDEBAR_OPTION_DEMO_IMAGE, SIDEBAR_OPTION_UPLOAD_IMAGE, SIDEBAR_OPTION_MEET_TEAM]

# Parameters
clip_frames = 30
total_frames = 9000  # original 900
inference = True
frame_count = 0

def get_video_files_in_dir(directory):
    out = ["< --Select-- >"]
    for item in os.listdir(directory):
        try:
            name, ext = item.split(".")
        except:
            continue
        if name and ext:
            if ext in VIDEO_EXTENSIONS:
                out.append(item)
    return out

def display_team():
       st.sidebar.write(" ------ ")
       stframe_1 = st.empty()
       stframe_1.subheader("Project Team")
       stframe_2 = st.empty()
       first_column, second_column , third_column, forth_column, fifth_column = stframe_2.columns(5)
       first_column.image("https://via.placeholder.com/300x300?text=Fu%20Wei%20Jiang",  use_column_width = True, caption = "Fu Wei Jiang")
       second_column.image("https://via.placeholder.com/300x300?text=Jose%20Daniel%20Castor",  use_column_width = True, caption = "Jose Daniel Castor")
       third_column.image("https://via.placeholder.com/300x300?text=Ching%20Hui%20Lee",  use_column_width = True, caption = "Ching Hui Lee")
       forth_column.image("https://via.placeholder.com/300x300?text=Eng%20Mong%20Goh",  use_column_width = True, caption = "Eng Mong Goh")
       fifth_column.image("https://via.placeholder.com/300x300?text=Ming%20Zhang",  use_column_width = True, caption = "Ming Zhang")
     
       stframe_2a = st.empty()
       first_column, second_column , third_column, forth_column, fifth_column = stframe_2a.columns(5)
       first_column.image(os.path.join( TEAM_DIR, 'tanpohkeam.jpeg'),  use_column_width = True, caption = "Tan Poh Keam")
       second_column.image(os.path.join( TEAM_DIR, 'loosailam.jpeg'),  use_column_width = True, caption = "Loo Sai Lam")
       third_column.image("https://via.placeholder.com/300x300?text=Seow%20Khee%20Wei",  use_column_width = True, caption = "Seow Khee Wei")
      # forth_column.image("https://via.placeholder.com/300x300?text=Jose%20Daniel%20Castor",  use_column_width = True, caption = "Jose Daniel Castor")
       
       stframe_3 = st.empty()
       expandar_contact = stframe_3.expander('Contact Information')
       expandar_contact.write('Jiang Fu Wei  |   fuwei.jiang@sg.panasonic.com') 
       expandar_contact.write('Tan Poh Keam  |   tan_poh_keam@rp.edu.sg  |  https://www.linkedin.com/in/tan-pohkeam/')
 
def  display_projectinfo():
        st.sidebar.write(" ------ ")
        #st.sidebar.success("Project information showing on the right!")
        with open(PROJECT_INFO, 'r', encoding="utf8") as f:
           proj_info = f.read()
           st.write(proj_info)

def run_demo(model):
       st.sidebar.write("------")
       files = get_video_files_in_dir(VIDEO_DIR)
       if len(files) == 1:
          st.sidebar.write("There are no demo videos avaialble in  ( %s) " % VIDEO_DIR )
       else:
          filename = st.sidebar.selectbox("Select a demo video for action detection" , files )
          if filename != '< --Select-- >':
          #with st.spinner("Performing Inferfence... this may take a few seconds. Please don't interrupt it."):
             stframe_1 = st.empty()
             stframe_1.info("Started")
             inference_actions(model, os.path.join(VIDEO_DIR, filename) )
             stframe_1.success("Ended")

def run_upload_demo(model):
     # This is not an upload with actual file transfer, but to state the path of the video path"
     #st.write('Placholder: SIDEBAR_OPTION_UPLOAD_IMAGE ')
     uploaded_file = st.sidebar.file_uploader("Choose a file", type=['mp4'])

     if uploaded_file:
        #st.write("Filename: ", uploaded_file.name)
        upload_video_path = os.path.join(UPLOAD_DIR ,uploaded_file.name)
        with open(upload_video_path,"wb") as f:
            stframe_1 = st.empty()
            stframe_1.info("Started")
            f.write((uploaded_file).getbuffer())
            inference_actions(model, upload_video_path )
            stframe_1.success("Ended")

def inference_actions(model, video):
    frame_count = 0
    cap = cv2.VideoCapture(video)
    frames_list = []
    pred_cls_text = '-'
    preds = ['-', '-', '-']
    stframe = st.empty()

    while cap.isOpened():
 #   # Obtain the first frame
       success, image = cap.read()  #was .read()
       if not success:
           break 

       frame_count += 1
       image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
       resized_image = cv2.resize(image, dsize=(60, 60), interpolation=cv2.INTER_AREA)
       h, w, ch = resized_image.shape

       if inference:
           if len(frames_list) != clip_frames:
             # continue
             frames_list.append(resized_image)
             cv2.putText(image, 'Prediction: {}'.format(pred_cls_text), org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Probability', org=(500, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Exercise: {}'.format(preds[0]), org=(500, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Concentrate: {}'.format(preds[1]), org=(500, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Relaxation: {}'.format(preds[2]), org=(500, 160), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)

             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
             #stframe.image( image_rgb)

             filename = './tmp/savedImage.jpg'
             cv2.imwrite(filename, image)
             reload_image = Image.open(filename)
             stframe.image(reload_image)


           elif len(frames_list) == clip_frames:
             # continue
             frames_array = np.stack(frames_list, axis=2)
             frames_list = []

            # Resize and scale/normalize the data
             frames_array = frames_array[..., [2, 1, 0]]

             m = np.mean(frames_array)
             std = np.std(frames_array)
             frames_array = (frames_array - m) / std

             preds = model.predict(frames_array[np.newaxis])
             pred_cls = np.argmax(preds)
             preds = [round(i*100, 2) for i in preds[0]]
             if pred_cls == 0:
                 pred_cls_text = 'Exercise'
             elif pred_cls == 1:
                 pred_cls_text = 'Concentration'
             elif pred_cls == 2:
                 pred_cls_text = 'Relaxation'

             cv2.putText(image, 'Prediction: {}'.format(pred_cls_text), org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Probability', org=(500, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Exercise: {:.2f}'.format(preds[0]), org=(500, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Concentration: {:.2f}'.format(preds[1]), org=(500, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Relaxation: {:.2f}'.format(preds[2]), org=(500, 160), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             #cv2.imshow('Inference', image)   # old CV codes
             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
             #stframe.image( image_rgb)

             filename = 'savedImage.jpg'
             cv2.imwrite(filename, image)
             reload_image = Image.open(filename)
             stframe.image(reload_image)
             
       if not inference:
           stframe.imshow('No Inference', image)

       #if cv2.waitKey(25) & 0xFF == ord('q'):
       #    break

       if frame_count == total_frames:
           frame_count = 0
           break

    cap.release()

def main():
   # show side bar 
   st.sidebar.write(" ------ ")
   st.sidebar.title("Explore the Following")
   app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)
   model = tf.keras.models.load_model(MODEL)

   if app_mode == SIDEBAR_OPTION_PROJECT_INFO:
        display_projectinfo()

   elif app_mode ==  SIDEBAR_OPTION_DEMO_IMAGE:
        run_demo(model)
            
   elif app_mode ==  SIDEBAR_OPTION_UPLOAD_IMAGE :
        run_upload_demo(model)

   elif app_mode ==  SIDEBAR_OPTION_MEET_TEAM:   
       display_team()


# run the main program         
main()