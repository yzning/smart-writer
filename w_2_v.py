from gensim.models import word2vec
import string
from gensim.corpora.textcorpus import TextCorpus
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
import codecs
import os

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
	
def train_w2v():
	#sentences = get_element()
	sentences = LineSentence('./clean_new')
	model = word2vec.Word2Vec(size=300, window=10, min_count=1, workers=10,sg=0,max_vocab_size=500,hs=1,iter=100)
	model.build_vocab(sentences)
	model.train(sentences, total_examples=model.corpus_count, epochs=100)
	model.save('./model_w2v')
	model.wv.save_word2vec_format("./text.model.txt",binary=False)

def get_element():
	path='/Users/ningyizhao/Desktop/DLI/dcf/data/'
	fo = open("./element.txt","a")
	files= os.listdir(path)    
	for file in files:  
     		if not os.path.isdir(file):
			if file != '.DS_Store':  
				contents = open(path+file).read()
				content = over_sig(contents).split(' ')
				fo.write(str(content).replace(',','').replace("'",'').lower())
	
	print('finished element.')
	fo.close()

	sentences = LineSentence('./element.txt')
	
	return sentences
		
def re_train();
	sentences = get_element()
	model = word2vec.Word2Vec.load('./model_w2v')
	model.build_vocab(sentences)
	model.train(sentences, total_examples=model.corpus_count, epochs=100)
	model.save('./model_w2v')
	model.wv.save_word2vec_format("./text.model.txt",binary=False)

train_w2v()



