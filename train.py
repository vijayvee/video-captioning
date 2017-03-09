#!/usr/bin/python

import numpy as np
import tensorflow as tf
import glob
from preprocess import *
from VideoCap import S2VT

"""Python script to train the video captioning system"""

#Global initializations
n_lstm_steps = 50

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
    curr_vids = np.array([np.load(VIDEO_DIR + Vid2Url[vid] + '.npy') for vid in curr_batch_vids])
    ind_50 = map(int,np.linspace(0,79,50))
    curr_vids = curr_vids[:,ind_50,:]
    captions = [np.random.choice(Vid2Cap[vid],1)[0] for vid in curr_batch_vids]
    curr_caps,curr_masks = convert_caption(captions,word2id,n_lstm_steps)
    return curr_vids,curr_caps,curr_masks
@profile
def train(nEpoch,learning_rate,batch_size):
    init()
    vid2cap_s2vt = S2VT(n_steps=n_lstm_steps,hidden_dim=256,batch_size=batch_size,vocab_size=len(word_counts))
    video,caption,caption_mask,loss = vid2cap_s2vt.build_model()
    print "Built model"
    for v in tf.trainable_variables():
        print v.name,v.get_shape()
    optim = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    saver = tf.train.Saver()
    nVideos = 1200
    nIter = nVideos*nEpoch/batch_size
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    sess.run(tf.initialize_all_variables())
    for i in range(11):
        vids,caps,masks = fetch_data_batch(batch_size=batch_size)
	
#       print vids.shape,caps.shape,masks.shape
        _,curr_loss = sess.run([optim,loss],feed_dict={video:vids,
                                                        caption:caps,
                                                        caption_mask:masks})
    	if i%10 == 0:
            gen_caption,video_test = vid2cap_s2vt.generate_caption()
            caption_ = sess.run(gen_caption,feed_dict={video_test:np.expand_dims(vids[0],0)})
            caption_eng = []
    	    caption_GT = []
            for l in range(len(caption_)):
    		    if id2word[caption_[l]]!='<EOS>':
                        caption_eng.append(id2word[caption_[l]])
            for l in range(len(caps[0])):
                if caps[0][l]!=2:
                    caption_GT.append(id2word[caps[0][l]])
            print ' '.join(caption_eng)
    	    print ' '.join(caption_GT)
            print curr_loss
    	if i%1000 == 0:
    	    saver.save(sess,'VideoCap_{}_{}_{}_{}.ckpt'.format(nEpoch,learning_rate,batch_size,i))
    	    print 'Saved {}'.format(i)

if __name__ == "__main__":
    train(50,0.01,10)
