import dcf_s2s_new as dcf_gen
#from tensorflow.contrib.seq2seq.python.ops.dynamic_attention_wrapper import SelfCustAttention

import warnings
import os

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings('ignore')

#dcf_gen.train_s2s()


aa = open('../email_key_word').readlines()
print('Test generation results:')
for line in aa:
	dcf_gen.test_s2s(''.join(line))
