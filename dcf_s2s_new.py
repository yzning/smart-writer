# coding: utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
import codecs
import re
import email_lda as elda
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import warnings
import os
from tensorflow.contrib.seq2seq.python.ops.dynamic_attention_wrapper import SelfCustAttention
#print('TensorFlow Version: {}'.format(tf.__version__))

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings('ignore')

def over_sig(content):
	#content = content.strip(' ')
	content = content.replace(',',' ')
	content = content.replace('.',' ')
	content = content.replace('\n','')
	content = content.replace('+',' + ')
	content = content.replace('/',' / ')
	content = content.replace('<',' < ')
	content = content.replace('>',' > ')
	content = content.replace('"',' " ')
	content = content.replace('=',' = ')
	content = content.replace('!',' ! ')
	#content = content.replace("\"," \ ")
	content = content.replace('*','* ')
	content = content.replace(':',' : ')
	content = content.replace('__',' __ ')
	content = content.replace('[',' [ ')
	content = content.replace(']',' ] ')
	content = content.replace('-',' - ')
	content = content.replace('?',' ? ')
	content = content.replace('@',' @ ')
	content = content.replace('$',' $ ')
	content = content.replace('~',' ~ ')
	content = content.replace('|',' | ')
	content = content.replace('(',' ( ')
	content = content.replace(')',' ) ')
	content = content.replace('	',' ')
	content = content.replace('#',' # ')
	content = content.replace('{',' { ')
	content = content.replace('}',' } ')
	content = content.replace('^',' ^ ')
	content = content.replace('&',' & ')
	content = content.replace('%',' % ')
	content = content.replace('\\',' ')
	content = content.replace('1',' 1 ')
	content = content.replace('2',' 2 ')
	content = content.replace('3',' 3 ')
	content = content.replace('4',' 4 ')
	content = content.replace('5',' 5 ')
	content = content.replace('6',' 6 ')
	content = content.replace('7',' 7 ')
	content = content.replace('8',' 8 ')
	content = content.replace('9',' 9 ')
	content = content.replace('0',' 0 ')


	
	return content


contractions = { 
	"ain't": "am not",
	"aren't": "are not",
	"can't": "cannot",
	"can't've": "cannot have",
	"'cause": "because",
	"could've": "could have",
	"couldn't": "could not",
	"couldn't've": "could not have",
	"didn't": "did not",
	"doesn't": "does not",
	"don't": "do not",
	"hadn't": "had not",
	"hadn't've": "had not have",
	"hasn't": "has not",
	"haven't": "have not",
	"he'd": "he would",
	"he'd've": "he would have",
	"he'll": "he will",
	"he's": "he is",
	"how'd": "how did",
	"how'll": "how will",
	"how's": "how is",
	"i'd": "i would",
	"i'll": "i will",
	"i'm": "i am",
	"i've": "i have",
	"isn't": "is not",
	"it'd": "it would",
	"it'll": "it will",
	"it's": "it is",
	"let's": "let us",
	"ma'am": "madam",
	"mayn't": "may not",
	"might've": "might have",
	"mightn't": "might not",
	"must've": "must have",
	"mustn't": "must not",
	"needn't": "need not",
	"oughtn't": "ought not",
	"shan't": "shall not",
	"sha'n't": "shall not",
	"she'd": "she would",
	"she'll": "she will",
	"she's": "she is",
	"should've": "should have",
	"shouldn't": "should not",
	"that'd": "that would",
	"that's": "that is",
	"there'd": "there had",
	"there's": "there is",
	"they'd": "they would",
	"they'll": "they will",
	"they're": "they are",
	"they've": "they have",
	"wasn't": "was not",
	"we'd": "we would",
	"we'll": "we will",
	"we're": "we are",
	"we've": "we have",
	"weren't": "were not",
	"what'll": "what will",
	"what're": "what are",
	"what's": "what is",
	"what've": "what have",
	"where'd": "where did",
	"where's": "where is",
	"who'll": "who will",
	"who's": "who is",
	"won't": "will not",
	"wouldn't": "would not",
	"you'd": "you would",
	"you'll": "you will",
	"you're": "you are"}

def clean_text(text, remove_stopwords = True):
    text = text.lower()
    
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
        
    return text

def count_words(count_dict, text):
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1
                
def get_embedings(embeddings_index):
	#read all the embeddings
	with codecs.open('text.model.txt', encoding='utf-8') as f:
    	    for line in f:
        	values = line.split(' ')
        	word = values[0]
        	embedding = np.asarray(values[1:], dtype='float32')
        	embeddings_index[word] = embedding
        	
	#print('Word embeddings:', len(embeddings_index))
	
def get_vocab_int(vocab_to_int, word_counts, threshold, embeddings_index):
	#give each word an index
	value = 0
	for word, count in word_counts.items():
    	    if count >= threshold or word in embeddings_index:
        	vocab_to_int[word] = value
        	value += 1
        	
def get_int_vocab(int_to_vocab, vocab_to_int):
	#invert of the get_vocab_int()
	for word, value in vocab_to_int.items():
    	    int_to_vocab[value] = word
    	
def get_word_embedings(vocab_to_int, embeddings_index):
	# Need to use 300 for embedding dimensions to match YiZhao's vectors.
	embedding_dim = 300
	nb_words = len(vocab_to_int)

	word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
	for word, i in vocab_to_int.items():
    	    if word in embeddings_index:
        	word_embedding_matrix[i] = embeddings_index[word]
    	    else:
        	# If word not in YiZhao's vectors, create a random embedding for it
        	new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        	embeddings_index[word] = new_embedding
        	word_embedding_matrix[i] = new_embedding

	#Check if value matches len(vocab_to_int)
	#print(len(word_embedding_matrix))
	return word_embedding_matrix
	
def convert_to_ints(text, word_count, unk_count, eos=False):
    #change the text to numbers(index)
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count
    
def convert_to_ints_attention(atts, eos=False):
    #change the text to numbers(index)
    ints = []
    for i in range(0,len(atts)):
        att_ints = []
        for word in atts[i]:
            if word in vocab_to_int:
                att_ints.append(vocab_to_int[word])
            else:
                att_ints.append(vocab_to_int["<UNK>"])
        if eos:
            att_ints.append(vocab_to_int["<EOS>"])
        ints.append(att_ints)
    return ints
    
def convert_to_ints_attention_test(atts, eos=False):
    #change the text to numbers(index)
    ints = []
    for word in atts:
        if word in vocab_to_int:
        	ints.append(vocab_to_int[word])
        else:
            ints.append(vocab_to_int["<UNK>"])
    if eos:
        ints.append(vocab_to_int["<EOS>"])
    return ints
    
def create_lengths(text):
    #calculate the lengths format of a text
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])
    
def unk_counter(sentence):
    #Counts the number of time UNK appears in a sentence.
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count
    
def get_sorted_resource(sorted_summaries,sorted_texts, sorted_attentions_top,sorted_attentions_mid,sorted_attentions_less, int_summaries,int_texts, int_att_top,int_att_mid,int_att_less, lengths_texts):
	max_text_length = 900
	max_summary_length = 900
	min_length = 30
	unk_text_limit = 50
	unk_summary_limit = 50

	for length in range(min(lengths_texts.counts), max_text_length): 
    	    for count, words in enumerate(int_summaries):
        	if (len(int_summaries[count]) >= min_length and
            	len(int_summaries[count]) <= max_summary_length and
            	len(int_texts[count]) >= min_length and
            	unk_counter(int_summaries[count]) <= unk_summary_limit and
            	unk_counter(int_texts[count]) <= unk_text_limit and
            	length == len(int_texts[count])
           	):
            	    sorted_summaries.append(int_summaries[count])
            	    sorted_texts.append(int_texts[count])
            	    sorted_attentions_top.append(int_att_top[count])
            	    sorted_attentions_mid.append(int_att_mid[count])
            	    sorted_attentions_less.append(int_att_less[count])
            	    
            	    #sorted_attentions.append(int_attentions_vec[count])
            	    #print(sorted_attentions[0].shape)
        
	# Compare lengths to ensure they match
	#print(len(sorted_summaries))
	#print(len(sorted_texts))

def str_list(c_str):
	c_list = []
	for element in over_sig(c_str).lower().split(' '):
		c_list.append(element)
	
	return c_list

def get_resource(clean_summaries,
	clean_texts,
	word_counts,
	embeddings_index,
	missing_words,
	threshold,
	vocab_to_int, 
	codes,   
	int_to_vocab,
	word_count,
	unk_count,
	int_summaries,
	int_texts,
	int_attentions,
	sorted_summaries,
	sorted_texts,
	sorted_attentions_top,
	sorted_attentions_mid,
	sorted_attentions_less,
	attentions):
	reviews = pd.read_csv("/Users/ningyizhao/Desktop/DLI/dcf/traindata.csv")
	
	clean_atts_top = []
	clean_atts_mid = []
	clean_atts_less = []
	
	# Clean the summaries and texts
	for summary in reviews.Summary:
    	    clean_summaries.append(clean_text(summary, remove_stopwords=False))
	#print("Summaries are complete.", clean_summaries[0])
	
	for text in reviews.Text:
    	    clean_texts.append(clean_text(text))
    	    #print(str_list(clean_text(text)))
    	    #print(clean_text(text))
    	    attention_keys_top, attention_keys_mid, attention_keys_less = elda.get_attention_keys(str_list(clean_text(text)))
    	    #attention_key = 0.7 * attention_keys_top + 0.2 * attention_keys_mid + 0.1 * attention_keys_less
    	    #attentions.append(attention_key)
    	    clean_atts_top.append(attention_keys_top)
    	    clean_atts_mid.append(attention_keys_mid)
    	    clean_atts_less.append(attention_keys_less)
    	    
	# Find the number of times each word was used and the size of the vocabulary
	count_words(word_counts, clean_summaries)
	count_words(word_counts, clean_texts)
            
	#print("Size of Vocabulary:", len(word_counts))

	get_embedings(embeddings_index)
	
	for word, count in word_counts.items():
    	    if count > threshold:
        	if word not in embeddings_index:
            	    missing_words += 1
            
	missing_ratio = round(missing_words/len(word_counts),4)*100
	
	get_vocab_int(vocab_to_int, word_counts, threshold, embeddings_index)

	for code in codes:
    	    vocab_to_int[code] = len(vocab_to_int)

	get_int_vocab(int_to_vocab, vocab_to_int)

	word_embedding_matrix = get_word_embedings(vocab_to_int, embeddings_index)
	
	int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
	int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)
	int_att_top = convert_to_ints_attention(clean_atts_top)
	int_att_mid = convert_to_ints_attention(clean_atts_mid)
	int_att_less = convert_to_ints_attention(clean_atts_less)
	
	unk_percent = round(unk_count/word_count,4)*100

	lengths_summaries = create_lengths(int_summaries)
	lengths_texts = create_lengths(int_texts)

	get_sorted_resource(sorted_summaries,sorted_texts,sorted_attentions_top,sorted_attentions_mid,sorted_attentions_less,int_summaries,int_texts,int_att_top,int_att_mid,int_att_less, lengths_texts)
	return word_embedding_matrix
	

########## Building the Model

def model_inputs():    
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    test_attention_1 = tf.placeholder(tf.int32, [None, None], name='test_attention_1')
    test_attention_2 = tf.placeholder(tf.int32, [None, None], name='test_attention_2')
    test_attention_3 = tf.placeholder(tf.int32, [None, None], name='test_attention_3')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')

    return input_data, targets, test_attention_1,test_attention_2,test_attention_3, lr, keep_prob, summary_length, max_summary_length, text_length


def process_encoding_input(target_data, vocab_to_int, batch_size):
    #Remove the last word id from each batch and concat the <GO> to the begining of each batch
    
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length, dtype=tf.float32)
    # Join outputs since we are using a bidirectional RNN 
    enc_output = tf.concat(enc_output,2)
    
    return enc_output, enc_state



def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer, vocab_size, max_summary_length):
    #Create the training logits
    
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input, sequence_length=summary_length, time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, initial_state, output_layer) 

    training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, output_time_major=False, impute_finished=True, maximum_iterations=max_summary_length*4)
    return training_logits



def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer, max_summary_length, batch_size):
    #Create the inference logits
    
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens, end_token)
                
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, initial_state, output_layer)
                
    inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, output_time_major=False, impute_finished=True, maximum_iterations=max_summary_length*4)
    
    return inference_logits


def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, dec_attention):
    #Create the decoding cell and attention for the training and inference decoding layers
    
    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            d_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(d_cell, input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    #attn_mech = tf.contrib.seq2seq.SelfCustAttention(rnn_size, enc_output, dec_attention,text_length, normalize=False, name='SelfCustAttention')
    attn_mech = SelfCustAttention(rnn_size, enc_output, dec_attention,text_length, normalize=False, name='SelfCustAttention')

    dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell, attn_mech, rnn_size)
            
    initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state[0], _zero_state_tensors(rnn_size,batch_size,tf.float32)) 
    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input, 
                                                  summary_length, 
                                                  dec_cell, 
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size, 
                                                  max_summary_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,  
                                                    vocab_to_int['<GO>'], 
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell, 
                                                    initial_state, 
                                                    output_layer,
                                                    max_summary_length,
                                                    batch_size)

    return training_logits, inference_logits
  


def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length, vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, t_attention_1,t_attention_2,t_attention_3):    
	# Use Numberbatch's embeddings and the newly created ones as our embeddings
	
    embeddings = word_embedding_matrix
    
    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    #print('enc_embed_input',enc_embed_input.shape)
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
    dec_attention_1 = tf.nn.embedding_lookup(embeddings, t_attention_1)
    dec_attention_2 = tf.nn.embedding_lookup(embeddings, t_attention_2)
    dec_attention_3 = tf.nn.embedding_lookup(embeddings, t_attention_3)    
        
    dec_attention = 0.7*dec_attention_1 + 0.2*dec_attention_2 + 0.1*dec_attention_3
    #print('dec_attention',dec_attention.shape)

    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
                                                     
    training_logits, inference_logits  = decoding_layer(dec_embed_input, 
                                                        embeddings,
                                                        enc_output,
                                                        enc_state, 
                                                        vocab_size, 
                                                        text_length, 
                                                        summary_length, 
                                                        max_summary_length,
                                                        rnn_size, 
                                                        vocab_to_int, 
                                                        keep_prob, 
                                                        batch_size,
                                                        num_layers,
                                                        dec_attention)
    
    return training_logits, inference_logits


def pad_sentence_batch(sentence_batch):
    #Pad sentences with <PAD> so that each sentence of a batch has the same length
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    
def pad_sentence_batch1(sentence_batch, att_batch):
    #Pad sentences with <PAD> so that each sentence of a batch has the same length
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [att_sentence[0:max_sentence+1] for att_sentence in att_batch]

def get_batches(summaries, texts, attentions_1,attentions_2,attentions_3, batch_size):
    #Batch summaries, texts, attentions and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        attentions_batch_1 = attentions_1[start_i:start_i + batch_size]
        attentions_batch_2 = attentions_2[start_i:start_i + batch_size]
        attentions_batch_3 = attentions_3[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))
        pad_attentions_batch_1 = np.array(pad_sentence_batch1(summaries_batch, attentions_batch_1))
        pad_attentions_batch_2 = np.array(pad_sentence_batch1(summaries_batch, attentions_batch_2))
        pad_attentions_batch_3 = np.array(pad_sentence_batch1(summaries_batch, attentions_batch_3))
        
        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))
        #print pad_summaries_lengths
        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))
        #print pad_texts_lengths
        yield pad_summaries_batch, pad_texts_batch, pad_attentions_batch_1,pad_attentions_batch_2,pad_attentions_batch_3, pad_summaries_lengths, pad_texts_lengths


	# Set the Hyperparameters

clean_summaries = []
clean_texts = []
word_counts = {}
embeddings_index = {}
missing_words = 0
threshold = 1
vocab_to_int = {}
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]
int_to_vocab = {}
word_count = 0
unk_count = 0
int_summaries = []
int_texts = []
int_attentions = []
sorted_summaries = []
sorted_texts = []
sorted_attentions_top = []
sorted_attentions_mid = []
sorted_attentions_less = []
attentions = []

word_embedding_matrix = get_resource(clean_summaries,clean_texts,word_counts,embeddings_index,missing_words,threshold,vocab_to_int, codes,int_to_vocab,word_count,unk_count,int_summaries,int_texts, int_attentions, sorted_summaries,sorted_texts,sorted_attentions_top,sorted_attentions_mid,sorted_attentions_less, attentions)
#train_graph = tf.Graph()

def train_s2s():
	# Build the graph
	epochs = 100
	batch_size = 4
	rnn_size = 256
	num_layers = 2
	
	train_graph = tf.Graph()
	keep_probability = 0.75
	learning_rate = 0.005	
	# Set the graph to default to ensure that it is ready for training
	with train_graph.as_default():
    
    	# Load the model inputs    
    	    input_data, targets, test_attention_1,test_attention_2,test_attention_3, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

    	# Create the training and inference logits                               
            training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                      targets, 
                                                      keep_prob,   
                                                      text_length,
                                                      summary_length,
                                                      max_summary_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size, 
                                                      num_layers, 
                                                      vocab_to_int,
                                                      batch_size,
                                                      test_attention_1,
                                                      test_attention_2,
                                                      test_attention_3)

    	# Create tensors for the training logits and inference logits
    	    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    	    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
    
    	# Create the weights for sequence_loss
    	    masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

    	    with tf.name_scope("optimization"):
        	# Loss function
        	cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

        	# Optimizer
        	optimizer = tf.train.AdamOptimizer(learning_rate)

        	# Gradient Clipping
        	gradients = optimizer.compute_gradients(cost)
        	capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        	train_op = optimizer.apply_gradients(capped_gradients)
	
	print("Graph is built.")

	start = 0
	end = start + 20
	sorted_summaries_short = sorted_summaries[start:end]
	sorted_texts_short = sorted_texts[start:end]
	sorted_attentions_top_short = sorted_attentions_top[start:end]
	sorted_attentions_mid_short = sorted_attentions_mid[start:end]
	sorted_attentions_less_short = sorted_attentions_less[start:end]
	
	#print("The shortest text length:", len(sorted_texts_short[0]))
	#print("The longest text length:",len(sorted_texts_short[-1]))

	####### Train the Model
	learning_rate_decay = 0.95
	min_learning_rate = 0.0005
	display_step = 20 # Check training loss after every 20 batches
	stop_early = 0 
	stop = 3 # If the update loss does not decrease in 3 consecutive update checks, stop training
	per_epoch = 3 # Make 3 update checks per epoch
	#update_check = (len(sorted_texts_short)//batch_size//per_epoch)-1
	update_check = 2

	update_loss = 0 
	batch_loss = 0
	summary_update_loss = [] 

	checkpoint = "./best_model.ckpt" 
	with tf.Session(graph=train_graph) as sess:
	    sess.run(tf.global_variables_initializer())
    
    	    for epoch_i in range(1, epochs+1):
        	update_loss = 0
        	batch_loss = 0
        	for batch_i, (summaries_batch, texts_batch, attentions_batch_1,attentions_batch_2,attentions_batch_3, summaries_lengths, texts_lengths) in enumerate(
                	get_batches(sorted_summaries_short, sorted_texts_short,sorted_attentions_top_short,sorted_attentions_mid_short,sorted_attentions_less_short, batch_size)):
                    start_time = time.time()
            	    _, loss = sess.run(
                          	[train_op, cost],
                	        {input_data: texts_batch,
                 	        targets: summaries_batch,
                 	        test_attention_1: attentions_batch_1,
                 	        test_attention_2: attentions_batch_2,
                 	        test_attention_3: attentions_batch_3,
                 	        lr: learning_rate,
                  	        summary_length: summaries_lengths,
                 	        text_length: texts_lengths,
                 	        keep_prob: keep_probability})

            	    batch_loss += loss
            	    update_loss += loss
            	    end_time = time.time()
            	    batch_time = end_time - start_time

            	    if batch_i % display_step == 0 and batch_i > 0:
                	print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      	.format(epoch_i,
                              	epochs, 
                              	batch_i, 
                              	len(sorted_texts_short) // batch_size, 
                              	batch_loss / display_step, 
                              	batch_time*display_step))
                	batch_loss = 0

            	    if batch_i % update_check == 0 and batch_i > 0:
                	print("Average loss for this update:", round(update_loss/update_check,3))
                	summary_update_loss.append(update_loss)
                
                	# If the update loss is at a new minimum, save the model
                	if update_loss <= min(summary_update_loss):
                	    print('New Record! Good job!') 
                    	    stop_early = 0
                    	    saver = tf.train.Saver() 
                    	    saver.save(sess, checkpoint)
                        
                        else:
                            print("No Improvement......")
                    	    stop_early += 1
                    	    if stop_early == stop:
                        	break
                	update_loss = 0
            
                    
        	# Reduce learning rate, but not below its minimum value
        	learning_rate *= learning_rate_decay
        	if learning_rate < min_learning_rate:
            	    learning_rate = min_learning_rate
        
        	if stop_early == stop:
            	    print("Stopping Training.")
            	    break



def text_to_seq(text):
    #Prepare the input for predicting the trained model
    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]

def test_s2s(sentence_t):
	# Choose one of the keywords from the LDA trained dataset by YIZHAO
	batch_size = 4
	input_sentence = sentence_t
	text = text_to_seq(input_sentence)
	embeddings = word_embedding_matrix

	attention_keys_top_test, attention_keys_mid_test, attention_keys_less_test = elda.get_attention_keys(str_list(input_sentence))
	int_att_top_test = convert_to_ints_attention_test(attention_keys_top_test)
	int_att_mid_test = convert_to_ints_attention_test(attention_keys_mid_test)
	int_att_less_test = convert_to_ints_attention_test(attention_keys_less_test)
	
	int_att_top_test_cut = []
	int_att_mid_test_cut = []
	int_att_less_test_cut = []
	text_b = []
	
	for i in range(0,batch_size):
		text_b.append(text)
		int_att_top_test_cut.append(int_att_top_test[0:len(text)])
		int_att_mid_test_cut.append(int_att_mid_test[0:len(text)])
		int_att_less_test_cut.append(int_att_less_test[0:len(text)])
		
	int_att_top_test_cut_np = np.array(int_att_top_test_cut)
	int_att_mid_test_cut_np = np.array(int_att_mid_test_cut)
	int_att_less_test_cut_np = np.array(int_att_less_test_cut)
	text_b_np = np.array(text_b)
	

	#dec_attention_1_test = tf.nn.embedding_lookup(embeddings, int_att_top_test)
        #dec_attention_2_test = tf.nn.embedding_lookup(embeddings, int_att_mid_test)
        #dec_attention_3_test = tf.nn.embedding_lookup(embeddings, int_att_less_test) 
    
        #dec_attention_test = 0.7*dec_attention_1_test + 0.2*dec_attention_2_test + 0.1*dec_attention_3_test

	checkpoint = "./best_model.ckpt"
	loaded_graph = tf.Graph()
	#print([text]*batch_size)
	#print(attention_keys_top_test)

	#print(int_att_top_test)
	#print(len(int_att_top_test_cut_np[0]))

	#print([np.random.randint(5,8)])
	#print([len(text)]*batch_size)
	
	with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
    	    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    	    loader.restore(sess, checkpoint)

    	    input_data = loaded_graph.get_tensor_by_name('input:0')
    	    
    	    test_attention_1 = loaded_graph.get_tensor_by_name('test_attention_1:0')
    	    test_attention_2 = loaded_graph.get_tensor_by_name('test_attention_2:0')
    	    test_attention_3 = loaded_graph.get_tensor_by_name('test_attention_3:0')

    	    logits = loaded_graph.get_tensor_by_name('predictions:0')
    	    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    	    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    	    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
    	#Multiply by batch_size to match the model's input parameters
    	    answer_logits_m = sess.run(logits, {input_data: [text]*batch_size,
    	    				test_attention_1: int_att_top_test_cut_np,
    	    				test_attention_2: int_att_mid_test_cut_np,
    	    				test_attention_3: int_att_less_test_cut_np,
    	    				summary_length: [np.random.randint(5,8)], 
                                      	text_length: [len(text)]*batch_size,
                                      	keep_prob: 1.0})
            #print(answer_logits_m)
            answer_logits = answer_logits_m[0]

# Remove the padding from the tweet
	pad = vocab_to_int["<PAD>"] 
	print('  {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))
	
#train_s2s()
	
#test_s2s(" RTM MPI lsb.acct lsb.events")	


