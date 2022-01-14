import os
import cv2
import numpy as np
import time
import tensorflow as tf


# Initialize time
start = time.time()

# Parameters
clip_frames = 20
total_frames = 900
inference = True
frame_count = 0

# Set the data directory
video_data_dir = os.path.join('D:', 'Datasets', 'Thermal_videos', 'raw_videos')
cls_video_data_dir = os.path.join(video_data_dir, 'exercise')
video_file_name = 'sub01_twist_fv_setting01.mp4'

# Load the video file
#cap = cv2.VideoCapture(os.path.join(cls_video_data_dir, video_file_name))
cap = cv2.VideoCapture('./data/relaxing-02.mp4')

# Load the model
if inference:
    #model = tf.keras.models.load_model('lightweight-ltc-cnn-model-10.h5')
    #model = tf.keras.models.load_model('./model/lightweight-ltc-cnn-model-10.h5')
    model = tf.keras.models.load_model('./model/lightweight-multiscale-ltc-cnn-model-all-frames-13.h5')

frames_list = []
pred_cls_text = '-'
preds = ['-', '-', '-']

while cap.isOpened():
    # Obtain the first frame
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # For video, use 'break' instead of 'continue'
        continue


    frame_count += 1
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    resized_image = cv2.resize(image, dsize=(60,60), interpolation=cv2.INTER_AREA)
    #resized_image = cv2.resize(image, dsize=(58,58), interpolation=cv2.INTER_AREA)
    h, w, ch = resized_image.shape
    #print('\n\n')
    print (frame_count, h,w,ch)

    if inference:
        if len(frames_list) != clip_frames:
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
            #cv2.putText(image, 'Windmill: {}'.format(preds[3]), org=(500, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #            fontScale=0.5, color=(0, 255, 255), thickness=2)
            cv2.imshow('Inference', image)

        elif len(frames_list) == clip_frames:
            frames_array = np.stack(frames_list, axis=2)
            frames_list = []

            # Resize and scale/normalize the data
            frames_array = frames_array[..., [2, 1, 0]]

            m = np.mean(frames_array)
            std = np.std(frames_array)
            frames_array = (frames_array - m) / std
            
            print('\n\n Debug #2 Frames Array content')
            print(frames_array)


            preds = model.predict(frames_array[np.newaxis])
            pred_cls = np.argmax(preds)
            preds = [round(i*100, 2) for i in preds[0]]
            if pred_cls == 0:
                pred_cls_text = 'Exercise'
            elif pred_cls == 1:
                pred_cls_text = 'Concentrate'
            elif pred_cls == 2:
                pred_cls_text = 'Relaxation'
            elif pred_cls == 3:
                pred_cls_text = 'Windmill'

            cv2.putText(image, 'Prediction: {}'.format(pred_cls_text), org=(20, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 255, 255), thickness=2)
            cv2.putText(image, 'Probability', org=(500, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
            cv2.putText(image, 'Exercise: {:.2f}'.format(preds[0]), org=(500, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
            cv2.putText(image, 'Concentrate: {:.2f}'.format(preds[1]), org=(500, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
            cv2.putText(image, 'Relaxation: {:.2f}'.format(preds[2]), org=(500, 160), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 255), thickness=2)
            #cv2.putText(image, 'Windmill: {:.2f}'.format(preds[3]), org=(500, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #            fontScale=0.5, color=(0, 255, 255), thickness=2)
            cv2.imshow('Inference', image)

    if not inference:
        cv2.imshow('No Inference', image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    if frame_count == total_frames:
        frame_count = 0
        break

cap.release()
end = time.time()
print('Time elapsed: {:.2f}'.format(end-start))




