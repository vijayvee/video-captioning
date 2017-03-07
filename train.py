#!/usr/bin/python

import numpy as np
import tensorflow as tf
import glob
from preprocess import *
from VideoCap import S2VT

"""Python script to train the video captioning system"""

#Global initializations
n_lstm_steps = 80
DATA_DIR = None
VIDEO_DIR = None
TEXT_DIR = None
Vid2Cap = None
Vid2Url = None
Url2Vid = None
word_counts = None
word2id = None
video_files = None

def init():
    """Function to initialize directories, dictionaries and vocabulary"""
    global DATA_DIR,VIDEO_DIR,TEXT_DIR,Vid2Cap,Vid2Url,Url2Vid,word_counts,word2id,id2word,video_files
    DATA_DIR = './Data/'
    VIDEO_DIR = DATA_DIR + 'Features_VGG/'
    TEXT_DIR = 'text_files/'
    Vid2Cap = eval(open(TEXT_DIR + 'Vid2Cap.txt').read())
    Vid2Url = eval(open(TEXT_DIR + 'Vid2Url_train.txt').read())
    Url2Vid = eval(open(TEXT_DIR + 'Url2Vid_train.txt').read())
    word_counts = build_vocab(0)
    word2id,id2word = word_to_word_ids(word_counts)
    video_files = glob.glob(VIDEO_DIR + '*.npy')
    video_files = Vid2Url.keys()
    print "{0} files processed".format(len(video_files))

def fetch_data_batch(batch_size):
    """Function to fetch a batch of video features, captions and caption masks
        Input:
                batch_size: Size of batch to load
        Output:
                curr_vids: Features of the randomly selected batch of video_files
                curr_caps: Ground truth (padded) captions for the selected videos
                curr_masks: Mask for the pad locations in curr_caps"""
    curr_batch_vids = np.random.choice(video_files,batch_size)
    curr_batch_feats = [np.load(VIDEO_DIR + Vid2Url[vid] + '.npy') for vid in curr_batch_vids]
    curr_vids = np.array(curr_batch_feats)
    captions = [np.random.choice(Vid2Cap[vid],1)[0] for vid in curr_batch_vids]
    curr_caps,curr_masks = convert_caption(captions,word2id,n_lstm_steps)
    curr_caps,curr_masks = np.array(curr_caps),np.array(curr_masks)
    return curr_vids,curr_caps,curr_masks

def train(nIter,learning_rate,batch_size):
    init()
    vid2cap_s2vt = S2VT(hidden_dim=100,batch_size=batch_size,vocab_size=len(word_counts))
    print "Built model"
    for v in tf.trainable_variables():
        print v.name,v.get_shape()
    optim = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(vid2cap_s2vt.loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(nIter):
        vids,caps,masks = fetch_data_batch(batch_size=batch_size)
        print vids.shape,caps.shape,masks.shape
        _,loss = sess.run([optim,vid2cap_s2vt.loss],feed_dict={vid2cap_s2vt.video:vids,
                                                                vid2cap_s2vt.caption:caps,
                                                                vid2cap_s2vt.caption_mask:masks})
        gen_caption = vid2cap_s2vt.generate_caption()
        test_video = np.expand_dims(vids[0],0)
        caption = sess.run(gen_caption,feed_dict={vid2cap_s2vt.video_test:test_video})
        caption_eng = []
        for c in caption:
            caption_eng.append(id2word[c])
        print ' '.join(caption_eng)
        print loss

if __name__ == "__main__":
    train(10,0.001,1)
