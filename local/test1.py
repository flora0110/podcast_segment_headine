from transformers import pipeline
from textsplit.algorithm import split_optimal, split_greedy, get_total
from textsplit.tools import get_penalty, get_segments
from textsplit.tools import SimpleSentenceTokenizer
import os
from gensim.models import word2vec
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import logging

wrdvec_path = 'wrdvecs.bin'
'''
if not os.path.exists(wrdvec_path):
  sentences = word2vec.Text8Corpus('./text8')
  model = word2vec.Word2Vec(sentences, cbow_mean=1,
                            hs=1, sample=0.00001, window=15)
model.save(wrdvec_path)
'''
model = word2vec.Word2Vec.load(wrdvec_path)
wrdvecs = pd.DataFrame(model.wv.vectors, index=model.wv.key_to_index)
sentence_tokenizer = SimpleSentenceTokenizer()

segment_len = 25

book_path = "podcast_text.txt"
with open(book_path, 'rt') as f:
    text = f.read()  # .replace('\n', ' ')  # punkt tokenizer handles newlines not so nice

sentenced_text = sentence_tokenizer(text)
strs = " "
for i in range(len(sentenced_text)):
  if(sentenced_text[i] != " "):
    strs = sentenced_text[i]
  if(i+1 < len(sentenced_text)):
    if(strs == sentenced_text[i+1]):
      sentenced_text[i+1] = " "
vecr = CountVectorizer(vocabulary=wrdvecs.index)

sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)

penalty = get_penalty([sentence_vectors], segment_len)
print('penalty %4.2f' % penalty)


optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=250)
segmented_text = get_segments(sentenced_text, optimal_segmentation)

print('%d sentences, %d segments, avg %4.2f sentences per segment' % (
    len(sentenced_text), len(segmented_text), len(sentenced_text) / len(segmented_text)))
fo = open("segmented_text.txt", "w")
for i in range(len(segmented_text)):
    fo.write(str(i)+" : ")
    content = str(segmented_text[i])
    fo.write(content)
    fo.write('\n')
fo.close()
# 將每個段落內的句子合起來變成string
podcast_test = [""]*len(segmented_text)
for i in range(len(segmented_text)):
  for j in range(len(segmented_text[i])):
    podcast_test[i] += segmented_text[i][j]
headlineGenerator = pipeline(model="Michau/t5-base-en-generate-headline",
                             tokenizer="Michau/t5-base-en-generate-headline")
min_length = 5
max_length = 150
headlines = headlineGenerator(podcast_test, min_length, max_length)
for headline in headlines:
  print(headline)
  print(type(headline))
