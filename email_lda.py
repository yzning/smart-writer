#encoding:utf-8
from gensim import corpora, models 
import numpy as np
from operator import itemgetter
import string
import os
from gensim.models import word2vec
#import w_2_v
import matplotlib.pyplot as plt
import time
import warnings
import sys
import random as rd

reload(sys)
sys.setdefaultencoding('utf8')

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings('ignore')

#w_2_v.train_w2v()
#w_2_v.save_w2v('./vec_rec.txt')

#word_dict = corpora.Dictionary([['L', 'D', 'A']])
time_start=time.time()
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

def read_file(path):
	#path='/Users/ningyizhao/Desktop/DLI/dcf/data'
	files= os.listdir(path)    
	aaa = 0
	content = []
	for file in files:  
	     	if not os.path.isdir(file): 
     			if file != '.DS_Store':
					contents = open(path+"/"+file).read()
					file = open(path+"/"+file).read()
					file_contents = over_sig(file).lower().split(' ')
					#file_contents = file.replace('\n',' ').replace(',',' , ').replace('.',' . ').replace('!',' ! ').lower().split(' ')
				
					content_file = []
					#aaa += 1
					#print(aaa)
					for i in range(0,len(file_contents)):
						content_file.append(''.join(file_contents[i]))
						
					content.append(content_file)	
					
	print('read finished.')
	#print(len(content))
	return content
	
def read_test_file(txt_path):
	file = open(txt_path).read()
	content = over_sig(file).lower().split(' ')
	return content
	
def get_stop_word(content):
	num_word={}
	stop_words=[]
	for word in content:
		if word in num_word:
			num_word[word] += 1
		else:
			num_word[word] = 1
	
	num_word = sorted(num_word.values())
	
	i=0
	for word in num_word:
		stop_words.append([num_word])
		i += 1
		if i == 20:
			break
	
	return stop_words
	
	
def get_stop_words():
	f = open('./stop_words.txt').read()
	stop_contents = f.replace('\n',' ').lower().split()
	stop_word_list = []
	for i in range(0,len(stop_contents)):
		stop_word_list.append(''.join(stop_contents[i]).decode())
	stop_word_list.append('')
	#print stop_word_list
	return stop_word_list
		
	
def get_train_words(path):
	content = read_file(path)
	stop_words =  get_stop_words()
	lda_train = []
	for i in range(0,len(content)):
		lda_train_file = []
		lda_train_file.append([term for term in content[i] if term not in stop_words])
		lda_train.append(lda_train_file[0])
		
	return lda_train
		  
    	
def get_testfile_words(word_dict, test_file_path):
	
	lda = models.ldamodel.LdaModel.load('./lda.model')
	test_content = read_test_file(test_file_path)
	#print (test_content)
	test_bow = word_dict.doc2bow(test_content)
	test_lda = lda[test_bow]
	
	test_lda = sorted(test_lda, key=itemgetter(1), reverse=True)
	print test_lda 
	print test_lda[0]
	topic_word1 = lda.get_topic_terms(test_lda[0][0], topn=400)

	
	topic_word_ids1 = []

	for i in range(0,len(topic_word1)):
		topic_word_ids1.append(topic_word1[i][0])
	
	testfile_words_ids1 = []
	for i in range(0, len(test_bow)):
		if test_bow[i][0] in topic_word_ids1:
			testfile_words_ids1.append(test_bow[i][0])
			
	testfile_words1 = []	
	for t in testfile_words_ids1:
	    #testfile_words1.append(word_dict.id2token[t])
	    testfile_words1.append(word_dict[t])
	    #testfile_words1.append(word_dict.id2token[t].encode('ascii'))
	print(testfile_words1)

	testfile_words1_ordered = []
	for term in test_content:
		if term in testfile_words1:
			testfile_words1_ordered.append(term)
			testfile_words1.remove(term)

	return testfile_words1_ordered
		
def get_docs_topics(lda,word_dict,path):
	#path='/Users/ningyizhao/Desktop/DLI/dcf/data'
	files= os.listdir(path)    
	topic_lens = []
	first_topics = []
	for file in files:  
	     	if not os.path.isdir(file): 
     			if file != '.DS_Store':
     				test_content_1 = read_test_file(path+"/"+file)
     				test_bow_1 = word_dict.doc2bow(test_content_1)
				test_lda_1 = lda[test_bow_1]
				test_lda_1 = sorted(test_lda_1, key=itemgetter(1), reverse=True)
				topic_lens.append(len(test_lda_1))
				first_topics.append(test_lda_1[0][1])
				#print test_lda_1 
				#print test_lda[0]
	print topic_lens
	print first_topics
	avg_lens = np.mean(topic_lens)
	avg_topic_terming = np.mean(first_topics)
	print avg_lens
	print avg_topic_terming
	
	return avg_lens, avg_topic_terming
		
def get_attention_keys(test_file):
	lda = models.ldamodel.LdaModel.load('./lda.model')
	word_dict = corpora.Dictionary.load('./lda_word_dict.dict')
	test_content = test_file
	#print (test_content)
	test_bow = word_dict.doc2bow(test_content)
	test_lda = lda[test_bow]
	test_lda = sorted(test_lda, key=itemgetter(1), reverse=True)
		
	num_category = 5
	check_intval = len(test_lda)//num_category
	#print(check_intval)
	#print(len(test_lda))
	
	if (check_intval > 0):			
		pick_num_top = rd.randint(0, check_intval-1)
		pick_num_mid = rd.randint(check_intval, check_intval*4-1)
		pick_num_less = rd.randint(check_intval*4-1, len(test_lda)-1)
	if (len(test_lda) == 1):
		pick_num_top = 0
		pick_num_mid = 0
		pick_num_less = 0
	if ((len(test_lda) > 1) & (check_intval == 0)):
		pick_num_top = 0
		pick_num_mid = rd.randint(0, len(test_lda)-1)
		pick_num_less = rd.randint(0, len(test_lda)-1)
	
	lda_keys_top_index = lda.get_topic_terms(test_lda[pick_num_top][0], topn=350)
	lda_keys_mid_index = lda.get_topic_terms(test_lda[pick_num_mid][0], topn=350)
	lda_keys_less_index = lda.get_topic_terms(test_lda[pick_num_less][0], topn=350)
	
	attention_keys_top_index = []
	attention_keys_mid_index = []
	attention_keys_less_index = []

	for i in range(0,len(lda_keys_top_index)):
		attention_keys_top_index.append(lda_keys_top_index[i][0])
		
	for i in range(0,len(lda_keys_mid_index)):
		attention_keys_mid_index.append(lda_keys_mid_index[i][0])
		
	for i in range(0,len(lda_keys_less_index)):
		attention_keys_less_index.append(lda_keys_less_index[i][0])
	
	attention_keys_top = []
	attention_keys_mid = []
	attention_keys_less = []

	for t in attention_keys_top_index:
		attention_keys_top.append(word_dict[t])
		
	for t in attention_keys_mid_index:
		attention_keys_mid.append(word_dict[t])
		
	for t in attention_keys_less_index:
		attention_keys_less.append(word_dict[t])
	
	
	return attention_keys_top, attention_keys_mid, attention_keys_less
	
		
def word_2_ver(model, words):
	test_word_vec = []
	for term in words:
		test_word_vec.append(w_2_v.get_w2c(term,model))
		
	return test_word_vec

def train_lda():
		
	lda_train = get_train_words()
	word_dict = corpora.Dictionary(lda_train)  
	corpus_list = [word_dict.doc2bow(text) for text in lda_train]   
	lda = models.ldamodel.LdaModel(corpus=corpus_list,id2word=word_dict,num_topics=100,alpha='asymmetric', 
									distributed=False, chunksize=2000, passes=1, update_every=1,eta=None, 
									decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, 
									minimum_probability=0.01, random_state=None, ns_conf=None, 
									minimum_phi_value=0.01, per_word_topics=False, callbacks=None)  

	lda.save('./lda.model')
	#lda.print_topic(topic[0])
	#print lda.get_topic_terms(0, topn=10)
	
	fo = open("./word_dict.txt","a")
	fo.write(str(lda_train).lower())
	print('finished')
	fo.close()

def get_a_lda(word_dict, test_file_path,lda):
	
	test_content = read_test_file(test_file_path)
	test_bow = word_dict.doc2bow(test_content)
	test_lda = lda[test_bow]
	
	test_lda_list = []
	for i in range(0, len(test_lda)):
		element = list(test_lda[i])
		test_lda_list.append(element)
	
	return test_lda_list
	
def get_all_ldas(word_dict, path,lda):
	#path='/Users/ningyizhao/Desktop/DLI/dcf/data'
	#lda = models.ldamodel.LdaModel.load('./lda.model')
	files= os.listdir(path)    
	lda_distribute = []
	for file in files:  
	     	if not os.path.isdir(file): 
     			if file != '.DS_Store':
     				test_lda = get_a_lda(word_dict,path+"/"+file,lda)
     				lda_distribute.append(test_lda)
					
	return lda_distribute
	
def get_lda_distribution(word_dict):
	print('in get_lda_distribution')
	lda = models.ldamodel.LdaModel.load('./lda.model')
	lda_distribute = get_all_ldas(word_dict, '/Users/ningyizhao/Desktop/DLI/dcf/all_pmrs',lda)
	topic_distribute = np.zeros((lda.num_topics,2))
	for index in range(0,lda.num_topics):
		topic_distribute[index][0] = index
	for i in range(0,len(lda_distribute)):
		for j in range(0,len(lda_distribute[i])):
			topic_distribute[lda_distribute[i][j][0]][1] += lda_distribute[i][j][1]
			
	plt.axis([-5, 1005, 0, 1500])
	#plt.scatter(topic_distribute[:,0],topic_distribute[:,1])
	plt.plot(topic_distribute[:,0],topic_distribute[:,1])
	plt.savefig("./examples.jpg")

	plt.show()
			
	return topic_distribute, lda_distribute
	
def get_secondary_lda_distribution(topic_distribute,lda_distribute):
	#lda_distribute_tmp = lda_distribute
	print('in get_secondary_lda_distribution')
	topic_distribute_tmp = topic_distribute
	max_list = []
	for index in range(0,10):
		max = np.where(topic_distribute_tmp == np.max(topic_distribute_tmp[:,1]))
		max_list.append(max)
		#print(max)
		#print(max[0][0])
		for i in range(0,len(lda_distribute)):
			for j in range(0,len(lda_distribute[i])):
				#print(lda_distribute[i][j])
				#print(topic_distribute_tmp[max[0][0]])
				if lda_distribute[i][j][0] == topic_distribute_tmp[max[0][0]][0]:
					frac = (1-lda_distribute[i][j][1])/1
					#lda_distribute_tmp[i].remove(lda_distribute_tmp[i][j])
					for k in range(0,len(lda_distribute[i])):
						lda_distribute[i][k][1] = lda_distribute[i][k][1]/frac
					lda_distribute[i].remove(lda_distribute[i][j])
					break
					
		topic_distribute_tmp = np.delete(topic_distribute_tmp,max[0][0],0)
	
	for index in range(0,len(topic_distribute)):
		topic_distribute[index][1] = 0.0 
	
	for i in range(0,len(lda_distribute)):
		for j in range(0,len(lda_distribute[i])):
			topic_distribute[lda_distribute[i][j][0]][1] += lda_distribute[i][j][1]
	
	for i in range(0, len(max_list)):
		max = max_list[i]
		topic_distribute = np.delete(topic_distribute,max[0][0],0)
	
	plt.axis([-5, 1005, 0, 1000])
	#plt.scatter(topic_distribute[:,0],topic_distribute[:,1])
	plt.plot(topic_distribute[:,0],topic_distribute[:,1])
	plt.savefig("./examples1.jpg")

	plt.show()
			
	return topic_distribute, lda_distribute	

def save_key_word(testfile_words):
	f1 =  open('./email_key_word','a')
	for i in range(0,len(testfile_words)):
		f1.write(str(testfile_words[i])+' ')
	f1.write('\n')
	f1.close()
	print('lda, finished.')

'''
#train and test without inference

lda_train = get_train_words('/Users/ningyizhao/Desktop/DLI/dcf/2000')
#print(lda_train)
#print(len(lda_train[0]))
word_dict = corpora.Dictionary(lda_train) 
#print(word_dict.dfs)
print(len(word_dict))


corpus_list = [word_dict.doc2bow(text) for text in lda_train] 

corpora.MmCorpus.serialize('./corpus_lda.mm', corpus_list)
#corpus_list = corpora.MmCorpus('./corpus_lda.mm')
 
word_dict.save('./lda_word_dict.dict')

lda = models.ldamodel.LdaModel(corpus=corpus_list,id2word=word_dict,num_topics=200,alpha='asymmetric', 
									distributed=False, chunksize=2000, passes=1, update_every=1,eta=None, 
									decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001, 
									minimum_probability=0.01, random_state=None, ns_conf=None, 
									minimum_phi_value=0.01, per_word_topics=False, callbacks=None) 

time_end=time.time()
print(str(time_end-time_start) + "s")


static_dict = corpora.Dictionary.load('./lda_word_dict.dict')
static_dict.filter_extremes(no_below=1, no_above=1.0, keep_n=100)
print(static_dict.dfs)
print(static_dict.items())


lda.save('./lda.model')
lda = models.ldamodel.LdaModel.load('./lda.model')

lda_test_1 = get_train_words('/Users/ningyizhao/Desktop/DLI/dcf/data')
#word_dict_test_1 = corpora.Dictionary(lda_test_1) 
corpus_list_test_1 = [word_dict.doc2bow(text_test_1) for text_test_1 in lda_test_1]
print lda.log_perplexity(corpus_list_test_1)

topic_distribute, lda_distribute = get_lda_distribution(word_dict)
topic_distribute, lda_distribute = get_secondary_lda_distribution(topic_distribute,lda_distribute)
'''
'''
word_dict.filter_extremes(no_below=1, no_above=1.0, keep_n=50)
print(word_dict.dfs)
print(word_dict.items())

static_dict = corpora.Dictionary.load('./lda_word_dict.dict')
print(static_dict.dfs)
static_dict.filter_extremes(no_below=1, no_above=1.0, keep_n=50)
print(static_dict.dfs)
print(static_dict.items())

testfile_words1 = get_testfile_words(word_dict, './data/test.txt')
print testfile_words1
save_key_word(testfile_words1)

'''


'''
#test_inference	

word_dict = corpora.Dictionary.load('./lda_word_dict.dict')
#print(len(word_dict))
corpus_list = corpora.MmCorpus('./corpus_lda.mm')


static_dict = corpora.Dictionary.load('./lda_word_dict.dict')

static_dict.filter_extremes(no_below=1, no_above=1.0, keep_n=100)
print(static_dict.dfs)
print(static_dict.items())

lda = models.ldamodel.LdaModel.load('./lda.model')


lda_test_1 = get_train_words('/Users/ningyizhao/Desktop/DLI/dcf/data')
corpus_list_test_1 = [word_dict.doc2bow(text_test_1) for text_test_1 in lda_test_1]
print lda.log_perplexity(corpus_list_test_1)

topic_distribute, lda_distribute = get_lda_distribution(word_dict)
topic_distribute, lda_distribute = get_secondary_lda_distribution(topic_distribute,lda_distribute)

testfile_words1_ordered = get_testfile_words(word_dict, '/Users/ningyizhao/Desktop/DLI/dcf/data/98.txt')
print testfile_words1_ordered
print(len(testfile_words1_ordered))

time_end=time.time()
print(str(time_end-time_start) + "s")
save_key_word(testfile_words1)
'''

#test doc topics' distribution

#word_dict = corpora.Dictionary.load('./lda_word_dict.dict')
#print(len(word_dict))
#corpus_list = corpora.MmCorpus('./corpus_lda.mm')
#lda = models.ldamodel.LdaModel.load('./lda.model')

#avg_lens, avg_topic_terming = get_docs_topics(lda,word_dict,'/Users/ningyizhao/Desktop/DLI/dcf/test_200')
#for i in range(0, 1000):
	
#time_end=time.time()
#print(str(time_end-time_start) + "s")
