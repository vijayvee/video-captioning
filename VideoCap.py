#!/usr/bin/python
import numpy as np
import tensorflow as tf

class S2VT:
    """Main class file for Sequence to Sequence -- Video to Text
       Contains the S2VT model and training script"""
    def __init__(self,n_steps=80,frame_dim=4096,hidden_dim=256,batch_size=20,vocab_size=1000):
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.frame_dim = frame_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
	#self.gen_caption_idx = []
        self.create_RNNs()
	self.create_weights()

    def create_weights(self):
        """Function to create weight matrices for transforming image to hidden vector shape
        and hidden state to vocabulary shape"""
        with tf.variable_scope('Im2Cap') as scope:
            self.W_im2cap = tf.get_variable(name='W_im2cap',shape=[self.frame_dim,
                                                        self.lstm_cap.state_size],
                                                        initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
            self.b_im2cap = tf.get_variable(name='b_im2cap',shape=[self.lstm_cap.state_size],
                                                        initializer=tf.constant_initializer(0.0))
        with tf.variable_scope('Hid2Vocab') as scope:
            self.W_word_embed = tf.get_variable(name='W_H2vocab',shape=[self.hidden_dim,self.vocab_size],
                                                             initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))
            self.b_word_embed = tf.get_variable(name='b_H2vocab',shape=[self.vocab_size],
                                                                initializer=tf.constant_initializer(0.0))
        with tf.variable_scope('Word_Vectors') as scope:
            self.word_emb = tf.get_variable(name='Word_embedding',shape=[self.vocab_size,self.hidden_dim],
                                                                    initializer=tf.random_uniform_initializer(minval=-0.08,maxval=0.08))

    def create_RNNs(self):
        """Function to create 2 RNNs, one for processing the frames of a video and the other
        for generating the caption"""
        with tf.variable_scope('LSTM_Video') as scope:
            self.lstm_vid = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim,state_is_tuple=False)
        with tf.variable_scope('LSTM_Caption') as scope:
            self.lstm_cap = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim,state_is_tuple=False)

    def build_model(self):
        """Function to build the graph for S2VT"""
        video = tf.placeholder(tf.float32,shape=[self.batch_size,self.n_steps,self.frame_dim],name='Input_Video')
        caption = tf.placeholder(tf.int32,shape=[self.batch_size,self.n_steps],name='GT_Caption')
        caption_mask = tf.placeholder(tf.float32,shape=[self.batch_size,self.n_steps],name='Caption_Mask')
        video_rshp = tf.reshape(video,[self.batch_size*self.n_steps,self.frame_dim])
        video_emb = tf.nn.xw_plus_b(video_rshp,self.W_im2cap,self.b_im2cap)
        video_emb = tf.reshape(video_emb,[self.batch_size,self.n_steps,self.hidden_dim])
        state_vid = tf.zeros([self.batch_size,self.lstm_vid.state_size])
        state_cap = tf.zeros([self.batch_size,self.lstm_cap.state_size])
        padding = tf.zeros([self.batch_size,self.hidden_dim])
        loss = 0.0
        for i in range(self.n_steps): #process video
            if i>0:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('LSTM_Video') as scope:
                out_vid,state_vid = self.lstm_vid(video_emb[:,i,:],state_vid)
            with tf.variable_scope('LSTM_Caption') as scope:
                out_cap,state_cap = self.lstm_cap(tf.concat(1,[out_vid,padding]),state_cap)

        for i in range(self.n_steps-2): #generate caption
#            if i==0:
 #               curr_word = tf.zeros([self.batch_size,self.hidden_dim])
  #          else:
            curr_word = tf.nn.embedding_lookup(self.word_emb,caption[:,i])
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('LSTM_Video') as scope:
                out_vid,state_vid = self.lstm_vid(padding,state_vid)
            with tf.variable_scope('LSTM_Caption') as scope:
                out_cap,state_cap = self.lstm_cap(tf.concat(1,[curr_word,out_vid]),state_cap)
            curr_pred = tf.nn.xw_plus_b(out_cap,self.W_word_embed,self.b_word_embed)
            curr_cap = caption[:,i+1]
            labels = tf.one_hot(indices=curr_cap,depth=self.vocab_size,on_value=1.0,off_value=0.0)
            xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=curr_pred,labels=labels)
            loss += tf.reduce_sum(xentropy*caption_mask[:,i+1])
        loss = loss/tf.reduce_sum(caption_mask)
        return video,caption,caption_mask,loss

    def generate_caption(self):
        """Function to test S2VT. Takes a single video and generates a caption describing the video"""
	gen_caption_idx = []
        video_test = tf.placeholder(tf.float32,shape=[1,self.n_steps,self.frame_dim],name='Test_Video')
        video_rshp = tf.reshape(video_test,[self.n_steps,self.frame_dim])
        video_emb = tf.nn.xw_plus_b(video_rshp,self.W_im2cap,self.b_im2cap)
        video_emb = tf.reshape(video_emb,[1,self.n_steps,self.hidden_dim])
        state_vid = tf.zeros([1,self.lstm_vid.state_size])
        state_cap = tf.zeros([1,self.lstm_cap.state_size])
        padding = tf.zeros([1,self.hidden_dim])

        for i in range(self.n_steps): #process video
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('LSTM_Video') as scope:
                out_vid,state_vid = self.lstm_vid(video_emb[:,i,:],state_vid)
            with tf.variable_scope('LSTM_Caption') as scope:
                out_cap,state_cap = self.lstm_cap(tf.concat(1,[out_vid,padding]),state_cap)

        for i in range(self.n_steps): #generate caption
	    if i==0:
		curr_word = tf.nn.embedding_lookup(self.word_emb,0)
		curr_word = tf.reshape(curr_word,[1,self.hidden_dim])
	    else:
                curr_word = tf.nn.embedding_lookup(self.word_emb,gen_caption_idx[-1])
                curr_word = tf.reshape(curr_word,[1,self.hidden_dim])
            #print curr_word.get_shape()
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('LSTM_Video') as scope:
                out_vid,state_vid = self.lstm_vid(padding,state_vid)
            with tf.variable_scope('LSTM_Caption') as scope:
                out_cap,state_cap = self.lstm_cap(tf.concat(1,[curr_word,out_vid]),state_cap)
            curr_pred = tf.nn.xw_plus_b(out_cap,self.W_word_embed,self.b_word_embed)
            #print curr_pred.get_shape()
            gen_caption_idx.append(tf.argmax(curr_pred,1)[0])
        return gen_caption_idx,video_test

    def build_model_img(self):
        """Function to build the graph for S2VT"""
        image = tf.placeholder(tf.float32,shape=[self.batch_size,self.frame_dim],name='Input_Image')
        caption = tf.placeholder(tf.int32,shape=[self.batch_size,self.n_steps],name='GT_Caption')
        caption_mask = tf.placeholder(tf.float32,shape=[self.batch_size,self.n_steps],name='Caption_Mask')
        #video_rshp = tf.reshape(video,[self.batch_size*self.n_steps,self.frame_dim])
        video_emb = tf.nn.xw_plus_b(image,self.W_im2cap,self.b_im2cap)
        #video_emb = tf.reshape(video_emb,[self.batch_size,self.n_steps,self.hidden_dim])
        #state_vid = tf.zeros([self.batch_size,self.lstm_vid.state_size])
        state_cap = tf.zeros([self.batch_size,self.lstm_cap.state_size])
        padding = tf.zeros([self.batch_size,self.hidden_dim])
        loss = 0.0
	caption_input = tf.nn.embedding_lookup(self.word_emb,caption[:,0:self.n_steps-1])
	caption_output = caption[:,1:]
	output,state = tf.nn.dynamic_rnn(self.lstm_cap,inputs=caption_input,initial_state=video_emb)
	output_rshp = tf.reshape(output,[self.batch_size*(self.n_steps-1),self.hidden_dim])
	caption_mask_rshp = tf.reshape(caption_mask[:,1:],[-1])
	logits = tf.nn.xw_plus_b(output_rshp,self.W_word_embed,self.b_word_embed)
	labels = tf.reshape(caption_output,[-1])
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = labels)
	xentropy_masked = xentropy*caption_mask_rshp
	loss = tf.reduce_sum(xentropy_masked)/tf.reduce_sum(caption_mask)
	return image,caption,caption_mask,loss,logits
	

if __name__ == "__main__":
    s2vt = S2VT()
    for v in tf.trainable_variables():
        print v.name
