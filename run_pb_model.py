import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
import os
import scipy
import matplotlib.pyplot as plt
import numpy as np
import time

""" 
Script loads model from 'model_file_path', and extracts features from that model 
at the node 'feature_tensor_name'. Input images read from paths in below txt files.
Writes these features to the gender write dirs. 

Note: Images are upsampled by factor 1.75 to match model input size
"""

model_file_path ='../tf_model_info/tf_out/'
male_files_path = './all_male_paths.txt'
male_write_dir = './feature_data/male/'
female_write_dir = './feature_data/female/'
female_files_path = './all_female_paths.txt'

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], model_file_path)
    
    input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
    feature_tensor = sess.graph.get_tensor_by_name(feature_tensor_name)

    start = time.time()
    print("Starting feature extraction")
    with open(male_files_path, 'r') as f:
        line_list = f.readlines()
        length = len(line_list)
        for i, line in enumerate(line_list):
            line = line.rstrip()
            path_to_file, age_str = line.split()
            file_name = os.path.basename(line)
            file_name = os.path.splitext(file_name)[0]
            img = scipy.ndimage.imread(path_to_file)
            img = scipy.misc.imresize(img, 1.75)
            img = np.expand_dims(img, 0)
            features = sess.run(feature_tensor, feed_dict = {input_tensor : img})
            np.save(male_write_dir + age_str + '_' + file_name + '.npy', features)
            if i % 100 == 0:
                elapsed = (time.time() - start)
                print("Elapsed ", elapsed / 60, " male ", i, "/", length)

    with open(female_files_path, 'r') as f:
        line_list = f.readlines()
        length = len(line_list)
        for i, line in enumerate(line_list):
            line = line.rstrip()
            path_to_file, age_str = line.split()
            file_name = os.path.basename(line)
            file_name = os.path.splitext(file_name)[0]
            img = scipy.ndimage.imread(path_to_file)
            img = scipy.misc.imresize(img, 1.75)
            img = np.expand_dims(img, 0)
            features = sess.run(feature_tensor, feed_dict = {input_tensor : img})
            np.save(female_write_dir + age_str + '_' + file_name + '.npy', features)
            if i % 100 == 0:
                elapsed = (time.time() - start)
                print("Elapsed ", elapsed / 60, " female ", i, "/", length)

