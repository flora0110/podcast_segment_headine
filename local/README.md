# text8
http://mattmahoney.net/dc/text8.zip
# wrdvecs.bin
打開這個
'''
if not os.path.exists(wrdvec_path):   
  sentences = word2vec.Text8Corpus('./text8')   
  model = word2vec.Word2Vec(sentences, cbow_mean=1,   
                            hs=1, sample=0.00001, window=15)   
model.save(wrdvec_path)   
'''
