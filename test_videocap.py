#!/usr/bin/python
import numpy as np
import tensorflow as tf
from utils import *
import sys
#GLOBAL VARIABLE INITIALIZATIONS TO BUILD MODEL
n_steps = 80
hidden_dim = 500
frame_dim = 4096
batch_size = 1
vocab_size = len(word2id)
bias_init_vector = get_bias_vector()

def build_model():
    """This function creates weight matrices that transform:
            * frames to caption dimension
            * hidden state to vocabulary dimension
            * creates word embedding matrix """

    print "Network config: \nN_Steps: {}\nHidden_dim:{}\nFrame_dim:{}\nBatch_size:{}\nVocab_size:{}\n".format(n_steps,
                                                                                                    hidden_dim,
                                                                                                    frame_dim,
                                                                                                    batch_size,
                                                                                                    vocab_size)

    #Create placeholders for holding a batch of videos, captions and caption masks
    video = tf.placeholder(tf.float32,shape=[batch_size,n_steps,frame_dim],name='Input_Video')
    caption = tf.placeholder(tf.int32,shape=[batch_size,n_steps],name='GT_Caption')
    caption_mask = tf.placeholder(tf.float32,shape=[batch_size,n_steps],name='Caption_Mask')
    dropout_prob = tf.placeholder(tf.float32,name='Dropout_Keep_Probability')

    with tf.variable_scope('Im2Cap') as scope:
        W_im2cap = tf.get_variable(name='W_im2cap',shape=[frame_dim,
                                                    hidden_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
        b_im2cap = tf.get_variable(name='b_im2cap',shape=[hidden_dim],
                                                    initializer=tf.constant_initializer(0.0))
    with tf.variable_scope('Hid2Vocab') as scope:
        W_H2vocab = tf.get_variable(name='W_H2vocab',shape=[hidden_dim,vocab_size],
                                                         initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
        b_H2vocab = tf.Variable(name='b_H2vocab',initial_value=bias_init_vector.astype(np.float32))

    with tf.variable_scope('Word_Vectors') as scope:
        word_emb = tf.get_variable(name='Word_embedding',shape=[vocab_size,hidden_dim],
                                                                initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
    print "Created weights"

    #Build two LSTMs, one for processing the video and another for generating the caption
    with tf.variable_scope('LSTM_Video',reuse=None) as scope:
        lstm_vid = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
        lstm_vid = tf.nn.rnn_cell.DropoutWrapper(lstm_vid,output_keep_prob=dropout_prob)
    with tf.variable_scope('LSTM_Caption',reuse=None) as scope:
        lstm_cap = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
        lstm_cap = tf.nn.rnn_cell.DropoutWrapper(lstm_cap,output_keep_prob=dropout_prob)

    #Prepare input for lstm_video
    video_rshp = tf.reshape(video,[-1,frame_dim])
    video_rshp = tf.nn.dropout(video_rshp,keep_prob=dropout_prob)
    video_emb = tf.nn.xw_plus_b(video_rshp,W_im2cap,b_im2cap)
    video_emb = tf.reshape(video_emb,[batch_size,n_steps,hidden_dim])
    padding = tf.zeros([batch_size,n_steps-1,hidden_dim])
    video_input = tf.concat([video_emb,padding],1)
    print "Video_input: {}".format(video_input.get_shape())
    #Run lstm_vid for 2*n_steps-1 timesteps
    with tf.variable_scope('LSTM_Video') as scope:
        out_vid,state_vid = tf.nn.dynamic_rnn(lstm_vid,video_input,dtype=tf.float32)
    print "Video_output: {}".format(out_vid.get_shape())

    #Prepare input for lstm_cap
    padding = tf.zeros([batch_size,n_steps,hidden_dim])
    caption_vectors = tf.nn.embedding_lookup(word_emb,caption[:,0:n_steps-1])
    caption_vectors = tf.nn.dropout(caption_vectors,keep_prob=dropout_prob)
    caption_2n = tf.concat([padding,caption_vectors],1)
    caption_input = tf.concat([caption_2n,out_vid],2)
    print "Caption_input: {}".format(caption_input.get_shape())
    #Run lstm_cap for 2*n_steps-1 timesteps
    with tf.variable_scope('LSTM_Caption') as scope:
        out_cap,state_cap = tf.nn.dynamic_rnn(lstm_cap,caption_input,dtype=tf.float32)
    print "Caption_output: {}".format(out_cap.get_shape())

    #Compute masked loss
    output_captions = out_cap[:,n_steps:,:]
    output_logits = tf.reshape(output_captions,[-1,hidden_dim])
    output_logits = tf.nn.dropout(output_logits,keep_prob=dropout_prob)
    output_logits = tf.nn.xw_plus_b(output_logits,W_H2vocab,b_H2vocab)
    output_labels = tf.reshape(caption[:,1:],[-1])
    caption_mask_out = tf.reshape(caption_mask[:,1:],[-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_logits,labels=output_labels)
    masked_loss = loss*caption_mask_out
    loss = tf.reduce_sum(masked_loss)/tf.reduce_sum(caption_mask_out)
    return video,caption,caption_mask,output_logits,loss,dropout_prob

if __name__=="__main__":
    with tf.Graph().as_default():
        learning_rate = 0.00001
        video,caption,caption_mask,output_logits,loss,dropout_prob = build_model()
        optim = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
        ckpt_file = 'S2VT_Dyn_10_0.0001_300_46000.ckpt.meta'
    	saver = tf.train.Saver()
        with tf.Session() as sess:
    	    if ckpt_file:
        		saver_ = tf.train.import_meta_graph(ckpt_file)
        		saver_.restore(sess,'./S2VT_Dyn_10_0.0001_300_46000.ckpt')
        		print "Restored model"
    	    else:
                sess.run(tf.initialize_all_variables())
            while(1):
                vid,caption_GT,_,video_urls = fetch_data_batch_val(1)
                caps,caps_mask = convert_caption(['<BOS>'],word2id,80)
                for i in range(n_steps):
                    o_l = sess.run(output_logits,feed_dict={video:vid,
                                                            caption:caps,
                                                            caption_mask:caps_mask,
                                                            dropout_prob:1.0})
                    out_logits = o_l.reshape([batch_size,n_steps-1,vocab_size])
                    output_captions = np.argmax(out_logits,2)
                    caps[0][i+1] = output_captions[0][i]
                    print_in_english(caps)
                    if id2word[output_captions[0][i]] == '<EOS>':
                        break
                print '............................\nGT Caption:\n'
                print_in_english(caption_GT)
                play_video = raw_input('Should I play the video? ')
                if play_video.lower() == 'y':
                    playVideo(video_urls)
                test_again = raw_input('Want another test run? ')
                if test_again.lower() == 'n':
                    break
