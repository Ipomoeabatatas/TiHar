

#import streamlit as st
import os
#import cv2
import numpy as np
import time
import tensorflow as tf

print(tf.__version__)

MODEL = './model/lightweight-ltc-cnn-model-10.h5'  # Nov 2021 version
model = tf.keras.models.load_model(MODEL)

print ("\n\nNov Model: " + MODEL)
model.summary()

print ("\n\nJan Model: " + MODEL)
MODEL = './model/lightweight-multiscale-ltc-cnn-model-all-frames-13.h5'  # LTC-CNN
model = tf.keras.models.load_model(MODEL)
model.summary()

