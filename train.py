#!/usr/bin/python
import numpy as np
import tensorflow as tf
import glob
from preprocess import *
from VideoCap import S2VT

"""Python script to train the video captioning system"""

#Global initializations
n_lstm_steps = 80
DATA_DIR = './Data/'
VIDEO_DIR = DATA_DIR + 'Features_VGG/'
TEXT_DIR = 'text_files/'
Vid2Cap = eval(open(TEXT_DIR + 'Video2Caption.txt').read())
Vid2Url = eval(open(TEXT_DIR + 'Vid2Url_train.txt').read())
Url2Vid = eval(open(TEXT_DIR + 'Url2Vid_train.txt').read())
word_counts,unk_required = build_vocab(0)
word2id,id2word = word_to_word_ids(word_counts,unk_required)
video_files = Vid2Url.keys()
print "{0} files processed".format(len(video_files))

def fetch_data_batch_imgs(batch_size):
    """Function to fetch a batch of video features, captions and caption masks
        Input:
                batch_size: Size of batch to load
        Output:
                curr_vids: Features of the randomly selected batch of video_files
                curr_caps: Ground truth (padded) captions for the selected videos
                curr_masks: Mask for the pad locations in curr_caps"""
    curr_batch_vids = np.random.choice(video_files,batch_size)
    curr_vids = np.array([np.load(VIDEO_DIR + Vid2Url[vid] + '.npy') for vid in curr_batch_vids])
    ind = np.random.randint(0,79)
    curr_vids = curr_vids[:,ind,:]
    captions = [np.random.choice(Vid2Cap[vid],1)[0] for vid in curr_batch_vids]
    print captions
    curr_caps,curr_masks = convert_caption(captions,word2id,n_lstm_steps)
    return curr_vids,curr_caps,curr_masks

def get_bias_vector():
    """Function to return the initialization for the bias vector
       for mapping from hidden_dim to vocab_size.
       Borrowed from neuraltalk by Andrej Karpathy"""
    bias_init_vector = np.array([1.0*word_counts[id2word[i]] for i in id2word])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    return bias_init_vector

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
    ind_50 = map(int,np.linspace(0,79,n_lstm_steps))
    curr_vids = curr_vids[:,ind_50,:]
    captions = [np.random.choice(Vid2Cap[vid],1)[0] for vid in curr_batch_vids]
    curr_caps,curr_masks = convert_caption(captions,word2id,n_lstm_steps)
    return curr_vids,curr_caps,curr_masks

def print_in_english(caption_idx):
    """Function to take a list of captions with words mapped to ids and
        print the captions after mapping word indices back to words."""
    captions_english = [[id2word[word] for word in caption] for caption in caption_idx]
    for i,caption in enumerate(captions_english):
	if '<EOS>' in caption:
       	    caption = caption[0:caption.index('<EOS>')]
        print str(i+1) + ' ' + ' '.join(caption)
        print '..................................................'

def train(nEpoch,learning_rate,batch_size,saved_sess = None):
    init()
    gen_caption_idx = []
    vid2cap_s2vt = S2VT(n_steps=n_lstm_steps,hidden_dim=256,batch_size=batch_size,vocab_size=len(word2id))
    video,caption,caption_mask,loss = vid2cap_s2vt.build_model()
    #image,caption,caption_mask,loss,outs = vid2cap_s2vt.build_model_img()
    gen_caption,video_test = vid2cap_s2vt.generate_caption()
    print "Built model"
    for v in tf.trainable_variables():
        print v.name,v.get_shape()
    optim = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    saver = tf.train.Saver()
    nVideos = 1200
    nIter = nVideos*nEpoch/batch_size
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    if saved_sess:
    	saver_ = tf.train.import_meta_graph(saved_sess)
    	saver_.restore(sess,tf.train.latest_checkpoint('.'))
    	print "Restored"
    else:
        sess.run(tf.initialize_all_variables())

    for i in xrange(0,nIter):
        vids,caps,masks = fetch_data_batch(batch_size=batch_size)
	    #imgs,caps,masks = fetch_data_batch_imgs(batch_size)
        #print vids.shape,caps.shape,masks.shape
        _,curr_loss = sess.run([optim,loss],feed_dict={video:vids,
                                                        caption:caps,
                                                        caption_mask:masks})
    	if i%10 == 0:
    	    print 'Loss {}: {}\nCorresponding caption sample \n'.format(i,curr_loss)
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
            del caption_eng
            del caption_GT
    	if i%1000 == 0:
    	    saver.save(sess,'VideoCap_{}_{}_{}_{}.ckpt'.format(nEpoch,learning_rate,batch_size,i))
    	    print 'Saved {}'.format(i)

if __name__ == "__main__":
    train(250,0.01,10)
