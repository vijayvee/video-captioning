#!/usr/bin/python
import numpy as np

"""Functions to do the following:
            * Create vocabulary
            * Create dictionary mapping from word to word_id
            * Map words in captions to word_ids"""

def build_vocab(word_count_thresh):
    """Function to create vocabulary based on word count threshold.
        Input:
                word_count_thresh: Threshold to choose words to include to the vocabulary
        Output:
                vocabulary: Set of words in the vocabulary"""

    sents_train = open('text_files/sents_train_lc_nopunc.txt','r').read().splitlines()
    sents_val = open('text_files/sents_val_lc_nopunc.txt','r').read().splitlines()
    sents_test = open('text_files/sents_test_lc_nopunc.txt','r').read().splitlines()
    unk_required = False
    all_captions = []
    word_counts = {}
    for sent in sents_train + sents_val + sents_test:
        caption = sent.split('\t')[-1]
        caption = '<BOS> ' + caption + ' <EOS>'
        all_captions.append(caption)
        for word in caption.split(' '):
            if word_counts.has_key(word):
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    for word in word_counts.keys():
        if word_counts[word] < word_count_thresh:
            word_counts.pop(word)
            unk_required = True
    return word_counts,unk_required

def word_to_word_ids(word_counts,unk_required):
    """Function to map individual words to their id's.
        Input:
                word_counts: Dictionary with words mapped to their counts
        Output:
                word_to_id: Dictionary with words mapped to their id's. """
    count = 0
    word_to_id = {}
    id_to_word = {}
    if unk_required:
        word_to_id['<UNK>'] = count
        id_to_word[count] = '<UNK>'
        count += 1
    for word in word_counts.keys():
        word_to_id[word] = count
        id_to_word[count] = word
        count+=1
    return word_to_id,id_to_word

def convert_caption(caption,word_to_id,max_caption_length):
    """Function to map each word in a caption to it's respective id and to retrieve caption masks
        Input:
                caption: Caption to convert to word_to_word_ids
                word_to_id: Dictionary mapping words to their respective id's
                max_caption_length: Maximum number of words allowed in a caption
        Output:
                caps: Captions with words mapped to word id's
                cap_masks: Caption masks with 1's at positions of words and 0's at pad locations"""
    caps,cap_masks = [],[]
    if type(caption) == 'str':
        caption = [caption] # if single caption, make it a list of captions of length one
    for cap in caption:
        nWords = cap.count(' ') + 1
        cap = cap + ' <EOS>'*(max_caption_length-nWords)
        cap_masks.append([1.0]*nWords + [0.0]*(max_caption_length-nWords))
        curr_cap = []
        for word in cap.split(' '):
            if word_to_id.has_key(word):
                curr_cap.append(word_to_id[word]) # word is present in chosen vocabulary
            else:
                curr_cap.append(word_to_id['<UNK>']) # word not present in chosen vocabulary
        caps.append(curr_cap)
    return np.array(caps),np.array(cap_masks)
