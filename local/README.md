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
# pip install
Package                      Version   
---------------------------- -------------------   
absl-py                      1.0.0   
aiohttp                      3.8.1   
aiosignal                    1.2.0   
astunparse                   1.6.3   
async-timeout                4.0.2   
attrs                        21.4.0   
cachetools                   5.0.0   
certifi                      2021.10.8   
charset-normalizer           2.0.12   
click                        8.1.2   
colorama                     0.4.4   
datasets                     2.1.0   
dill                         0.3.4   
filelock                     3.6.0   
flatbuffers                  2.0   
frozenlist                   1.3.0   
fsspec                       2022.3.0   
gast                         0.5.3   
gensim                       4.1.2   
google-auth                  2.6.5   
google-auth-oauthlib         0.4.6   
google-pasta                 0.2.0   
grpcio                       1.44.0   
h5py                         3.6.0   
huggingface-hub              0.5.1   
idna                         3.3   
joblib                       1.1.0   
keras                        2.8.0   
Keras-Preprocessing          1.1.2   
libclang                     13.0.0   
Markdown                     3.3.6   
multidict                    6.0.2   
multiprocess                 0.70.12.2   
nltk                         3.7   
nose                         1.3.7   
numpy                        1.22.3   
oauthlib                     3.2.0   
opt-einsum                   3.3.0   
packaging                    21.3   
pandas                       1.4.2   
pip                          22.0.4   
protobuf                     3.20.0   
pyarrow                      7.0.0   
pyasn1                       0.4.8   
pyasn1-modules               0.2.8   
pyparsing                    3.0.8   
python-dateutil              2.8.2   
pytz                         2022.1   
PyYAML                       6.0   
regex                        2022.3.15   
requests                     2.27.1   
requests-oauthlib            1.3.1   
responses                    0.18.0   
rsa                          4.8   
sacremoses                   0.0.49   
scikit-learn                 1.0.2   
scipy                        1.8.0   
sentencepiece                0.1.96   
setuptools                   58.1.0   
six                          1.16.0   
smart-open                   5.2.1   
tensorboard                  2.8.0   
tensorboard-data-server      0.6.1   
tensorboard-plugin-wit       1.8.1   
tensorflow                   2.8.0   
tensorflow-io-gcs-filesystem 0.24.0   
termcolor                    1.1.0   
textsplit                    0.5   
tf-estimator-nightly         2.8.0.dev2021122109   
threadpoolctl                3.1.0   
tokenizers                   0.12.1   
tqdm                         4.64.0   
transformers                 4.18.0   
typing_extensions            4.1.1   
urllib3                      1.26.9   
Werkzeug                     2.1.1   
wheel                        0.37.1   
word2vec                     0.11.1   
wrapt                        1.14.0   
xxhash                       3.0.0   
yarl                         1.7.2   
