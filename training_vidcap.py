#!/usr/bin/python
import numpy as np
import tensorflow as tf
from train import *

#GLOBAL VARIABLE INITIALIZATIONS TO BUILD MODEL
n_steps = 80
hidden_dim = 500
frame_dim = 4096
batch_size = 10
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
    with tf.variable_scope('LSTM_Caption',reuse=None) as scope:
        lstm_cap = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)

    #Create placeholders for holding a batch of videos, captions and caption masks
    video = tf.placeholder(tf.float32,shape=[batch_size,n_steps,frame_dim],name='Input_Video')
    caption = tf.placeholder(tf.int32,shape=[batch_size,n_steps],name='GT_Caption')
    caption_mask = tf.placeholder(tf.float32,shape=[batch_size,n_steps],name='Caption_Mask')

    #Prepare input for lstm_video
    video_rshp = tf.reshape(video,[-1,frame_dim])
    video_emb = tf.nn.xw_plus_b(video_rshp,W_im2cap,b_im2cap)
    video_emb = tf.reshape(video_emb,[batch_size,n_steps,hidden_dim])
    padding = tf.zeros([batch_size,n_steps-1,hidden_dim])
    video_input = tf.concat(1,[video_emb,padding])

    #Run lstm_vid for 2*n_steps-1 timesteps
    with tf.variable_scope('LSTM_Video') as scope:
        out_vid,state_vid = tf.nn.dynamic_rnn(lstm_vid,video_input,dtype=tf.float32)

    #Prepare input for lstm_cap
    padding = tf.zeros([batch_size,n_steps,hidden_dim])
    caption_vectors = tf.nn.embedding_lookup(word_emb,caption[:,0:n_steps-1])
    caption_2n = tf.concat(1,[padding,caption_vectors])
    caption_input = tf.concat(2,[caption_2n,out_vid])

    #Run lstm_cap for 2*n_steps-1 timesteps
    with tf.variable_scope('LSTM_Caption') as scope:
        out_cap,state_cap = tf.nn.dynamic_rnn(lstm_cap,caption_input,dtype=tf.float32)

    #Compute masked loss
    output_captions = out_cap[:,n_steps:]
    output_logits = tf.reshape(output_captions,[-1,hidden_dim])
    output_logits = tf.nn.xw_plus_b(output_logits,W_H2vocab,b_H2vocab)
    output_labels = tf.reshape(caption[:,1:],[-1])
    caption_mask_out = tf.reshape(caption_mask[:,1:],[-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_logits,labels=output_labels)
    masked_loss = loss*caption_mask_out
    loss = tf.reduce_sum(masked_loss)/tf.reduce_sum(caption_mask_out)
    return video,caption,caption_mask,output_logits,loss

if __name__=="__main__":
    with tf.Graph().as_default():
        video,caption,caption_mask,output_logits,loss = build_model()
        optim = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            vids,caps,caps_mask = fetch_data_batch(batch_size=batch_size)
            _,curr_loss,o_l = sess.run([optim,loss,output_logits],feed_dict={video:vids,
                                                                            caption:caps,
                                                                            caption_mask:caps_mask})
            output_logits = o_l.reshape([batch_size,n_steps-1,vocab_size])
            output_captions = np.argmax(output_logits,2)
            print print_in_english(output_captions)
            print curr_loss,o_l.shape
