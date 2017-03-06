#!/usr/bin/python
import numpy as np
import tensorflow as tf

class S2VT:
    """Main class file for Sequence to Sequence -- Video to Text
       Contains the S2VT model and training script"""
    def __init__(self,n_steps=80,frame_dim=4096,hidden_dim=256,batch_size=2,vocab_size=1000):
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.frame_dim = frame_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.create_placeholders()
        self.create_weights()
        self.create_RNNs()
        self.loss = self.build_model()

    def create_placeholders(self):
        """Function to create placeholders for building the graph"""
        self.video = tf.placeholder(tf.float32,shape=[self.batch_size,self.n_steps,self.frame_dim])
        self.caption = tf.placeholder(tf.int32,shape=[self.batch_size,self.n_steps])

    def create_weights(self):
        """Function to create weight matrices for transforming image to hidden vector shape
        and hidden state to vocabulary shape"""
        with tf.variable_scope('Im2Cap') as scope:
            self.W_im2cap = tf.get_variable(name='W_im2cap',shape=[self.frame_dim,
                                                        self.hidden_dim],
                                                        initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))
            self.b_im2cap = tf.get_variable(name='b_im2cap',shape=[self.hidden_dim],
                                                        initializer=tf.constant_initializer(0.1))
        with tf.variable_scope('Hid2Vocab') as scope:
            self.W_word_embed = tf.get_variable(name='W_H2vocab',shape=[self.hidden_dim,self.vocab_size],
                                                             initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.01))
            self.b_word_embed = tf.get_variable(name='b_H2vocab',shape=[self.vocab_size],
                                                                initializer=tf.constant_initializer(0.1))
        with tf.variable_scope('Word_Vectors') as scope:
            self.word_emb = tf.get_variable(name='Word_embedding',shape=[self.vocab_size,self.hidden_dim],
                                                                    initializer=tf.random_uniform_initializer(minval=-0.01,maxval=0.01))

    def create_RNNs(self):
        """Function to create 2 RNNs, one for processing the frames of a video and the other
        for generating the caption"""
        with tf.variable_scope('LSTM_Video') as scope:
            self.lstm_vid = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim,state_is_tuple=False)
        with tf.variable_scope('LSTM_Caption') as scope:
            self.lstm_cap = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim,state_is_tuple=False)

    def build_model(self):
        """Function to build the graph for S2VT"""
        self.video_rshp = tf.reshape(self.video,[self.batch_size*self.n_steps,self.frame_dim])
        self.video_emb = tf.nn.xw_plus_b(self.video_rshp,self.W_im2cap,self.b_im2cap)
        self.video_emb = tf.reshape(self.video_emb,[self.batch_size,self.n_steps,self.hidden_dim])
        state_vid = tf.zeros([self.batch_size,self.lstm_vid.state_size])
        state_cap = tf.zeros([self.batch_size,self.lstm_cap.state_size])
        padding = tf.zeros([self.batch_size,self.hidden_dim])
        loss = 0.0
        for i in range(self.n_steps): #process video
            if i>0:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('LSTM_Video') as scope:
                out_vid,state_vid = self.lstm_vid(self.video_emb[:,i,:],state_vid)
            with tf.variable_scope('LSTM_Caption') as scope:
                out_cap,state_cap = self.lstm_cap(tf.concat(1,[out_vid,padding]),state_cap)

        for i in range(self.n_steps): #generate caption
            if i==0:
                curr_word = tf.zeros([self.batch_size,self.hidden_dim])
            else:
                curr_word = tf.nn.embedding_lookup(self.word_emb,self.`caption[:,i-1])
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('LSTM_Video') as scope:
                out_vid,state_vid = self.lstm_vid(padding,state_vid)
            with tf.variable_scope('LSTM_Caption') as scope:
                out_cap,state_cap = self.lstm_cap(tf.concat(1,[curr_word,out_vid]),state_cap)
            curr_pred = tf.nn.xw_plus_b(out_cap,self.W_word_embed,self.b_word_embed)
            curr_cap = self.caption[:,i]
            labels = tf.one_hot(indices=curr_cap,depth=self.vocab_size,on_value=1.0,off_value=0.0)
            loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=curr_pred,labels=labels))
        return loss

    def generate_caption(self):
        """Function to test S2VT. Takes a single video and generates a caption describing the video"""
        gen_caption_idx = []
        video_test = tf.placeholder(tf.float32,shape=[1,self.n_steps,self.frame_dim])
        video_rshp = tf.reshape(video_test,[self.n_steps,self.frame_dim])
        video_emb = tf.nn.xw_plus_b(video_rshp,self.W_im2cap,self.b_im2cap)
        video_emb = tf.reshape(video_emb,[self.n_steps,self.hidden_dim])
        state_vid = tf.zeros([1,self.hidden_dim])
        state_cap = tf.zeros([1,self.hidden_dim])
        padding = tf.zeros([1,self.hidden_dim])
        for i in range(n_steps): #process video
            out_vid,state_vid = self.lstm_vid(video_emb[i,:],state_vid)
            out_cap,state_cap = self.lstm_cap(tf.concat(1,[out_vid,padding]),state_cap)

        for i in range(n_steps): #generate caption
            if i==0:
                curr_word = tf.zeros([1,self.hidden_dim])
            else:
                curr_word = tf.nn.embedding_lookup(self.word_emb,gen_caption[-1])
            out_vid,state_vid = self.lstm_vid(padding,state_vid)
            out_cap,state_cap = self.lstm_cap(tf.concat(1,[curr_word,out_vid]),state_cap)
            curr_pred = tf.nn.xw_plus_b(out_cap,self.W_word_embed,self.b_word_embed)
            gen_caption.append(tf.argmax(curr_pred))
        return video_test,gen_caption

if __name__ == "__main__":
    s2vt = S2VT()
    for v in tf.trainable_variables():
        print v.name
