# reference codes: https://github.com/robertklee/COCO-Human-Pose/blob/main/human_pose_app.py
# https://github.com/robertklee/COCO-Human-Pose/blob/main/human_pose_app.py
# https://github.com/rejexx/Parkingspot_Vacancy/blob/main/src/streamlit_app.py

import streamlit as st
import os
import cv2
import numpy as np
import time
import tensorflow as tf

MODEL = './model/lightweight-ltc-cnn-model-10.h5'
VIDEO_EXTENSIONS = ["mp4", "ogv", "m4v", "webm"]
VIDEO_DIR = "./data"
UPLOAD_DIR = "./tmp_upload"

PROJECT_INFO = "./project_info.md"
TEAM_DIR = './team'

# Constants for sidebar dropdown
SIDEBAR_OPTION_PROJECT_INFO = "Show Project Info"
SIDEBAR_OPTION_DEMO_IMAGE = "Select a Demo Video"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload Your Video"
SIDEBAR_OPTION_MEET_TEAM = "Meet the Team"
SIDEBAR_OPTIONS = [SIDEBAR_OPTION_PROJECT_INFO, SIDEBAR_OPTION_DEMO_IMAGE, SIDEBAR_OPTION_UPLOAD_IMAGE, SIDEBAR_OPTION_MEET_TEAM]

# Initialize time
start = time.time()


# Parameters
clip_frames = 20
total_frames = 900  # original 900
inference = True
frame_count = 0

# st.write("Demo thermal videos files should be stored in  ./data")

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

def inference_actions(model, video):
    frame_count = 0
    # st.video(video)
    cap = cv2.VideoCapture(video)
    frames_list = []
    pred_cls_text = '-'
    preds = ['-', '-', '-', '-']

    stframe = st.empty()

    while cap.isOpened():
 #   # Obtain the first frame
       success, image = cap.read()
       if not success:
           st.write("Ignoring empty camera frame.")
           # For video, use 'break' instead of 'continue'
           continue


       frame_count += 1
       image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
       resized_image = cv2.resize(image, dsize=(58, 58), interpolation=cv2.INTER_AREA)
       h, w, ch = resized_image.shape

       if inference:
           if len(frames_list) != clip_frames:
             # continue
             frames_list.append(resized_image)
             cv2.putText(image, 'Prediction: {}'.format(pred_cls_text), org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Probability', org=(500, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Clap: {}'.format(preds[0]), org=(500, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Squat: {}'.format(preds[1]), org=(500, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Twist: {}'.format(preds[2]), org=(500, 160), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Windmill: {}'.format(preds[3]), org=(500, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             stframe.image(image)

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
                 pred_cls_text = 'Clap'
             elif pred_cls == 1:
                 pred_cls_text = 'Squat'
             elif pred_cls == 2:
                 pred_cls_text = 'Twist'
             elif pred_cls == 3:
                pred_cls_text = 'Windmill'

             cv2.putText(image, 'Prediction: {}'.format(pred_cls_text), org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Probability', org=(500, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Clap: {:.2f}'.format(preds[0]), org=(500, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Squat: {:.2f}'.format(preds[1]), org=(500, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Twist: {:.2f}'.format(preds[2]), org=(500, 160), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             cv2.putText(image, 'Windmill: {:.2f}'.format(preds[3]), org=(500, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
             #cv2.imshow('Inference', image)
             stframe.image(image)


       if not inference:
           stframe.imshow('No Inference', image)

       if cv2.waitKey(25) & 0xFF == ord('q'):
           break

       if frame_count == total_frames:
           frame_count = 0
           break

    # cap.release()
    # end = time.time()
    # print('Time elapsed: {:.2f}'.format(end-start))



###########################
# Main
###########################
def main():
   ## st.title("What Is The Person Doing?")
   st.sidebar.write(" ------ ")
   st.sidebar.title("Explore the Following")
   app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)

   model = tf.keras.models.load_model(MODEL)

   if app_mode == SIDEBAR_OPTION_PROJECT_INFO:
        st.sidebar.write(" ------ ")
        #st.sidebar.success("Project information showing on the right!")
        with open(PROJECT_INFO, 'r') as f:
           proj_info = f.read()
           st.write(proj_info)
      

   elif app_mode ==  SIDEBAR_OPTION_DEMO_IMAGE:
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
            
   elif app_mode ==  SIDEBAR_OPTION_UPLOAD_IMAGE :
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
            stframe_1("Ended")

            #st.write(upload_video_path)
          
     
     # filename = 'sub01_clap_dv_setting02.mp4'
     
     #with st.spinner("Performing Inference... this may take a few seconds. Please don't interrupt it."):
     #   inference_actions(model, upload_video_path )
     #    Do you wish to try another demo file?")

   elif app_mode ==  SIDEBAR_OPTION_MEET_TEAM:   
       st.sidebar.write(" ------ ")
       stframe_1 = st.empty()
       stframe_1.subheader("Project Team")

       stframe_2 = st.empty()
       first_column, second_column , third_column, forth_column = stframe_2.columns(4)
       first_column.image(os.path.join( TEAM_DIR, 'tanpohkeam.jpeg'),  use_column_width = True, caption = "Tan Poh Keam")
       second_column.image(os.path.join( TEAM_DIR, 'loosailam.jpeg'),  use_column_width = True, caption = "Loo Sai Lam")
       third_column.image("https://via.placeholder.com/300x300?text=Jiang%20Fu%20Wei",  use_column_width = True, caption = "Jiang Fu Wei")
       forth_column.image("https://via.placeholder.com/300x300?text=Jose%20Daniel%20Castor",  use_column_width = True, caption = "Jose Daniel Castor")
       
       stframe_3 = st.empty()
       expandar_contact = stframe_3.expander('Contact Information')
       expandar_contact.write('Tan Poh Keam | tan_poh_keam@rp.edu.sg | https://www.linkedin.com/in/tan-pohkeam/')
       expandar_contact.write('Loo Sai Lam  | loo_sai_lam@rp.edu.sg  | https://www.linkedin.com/in/sai-lam-loo-711b22182/')
       expandar_contact.write('Jiang Fu Wei | fuwei.jiang@sg.panasonic.com')
       expandar_contact.write('Jose Daniel Castor | josedaniel.castor@sg.panasonic.com')
       expandar_contact.write('Corey: Lorem ipsum dolor sit amet ...')
       

main()