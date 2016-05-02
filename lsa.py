#Author: MeiXing Dong

import numpy as np

from gensim import corpora, models, similarities

# Original data, where each row is a document and each column represents
# a characteristic of the documents. 
data = [[1, 1, 1, 0, 0], 
        [2, 2, 2, 0, 0], 
        [1, 1, 1, 0, 0],
        [5, 5, 5, 0, 0],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 1, 1]]

print "Results from plain SVD --------------------------------"
# Apply SVD to the original data.
data_array = np.array(data)
u, s, v = np.linalg.svd(data, full_matrices=False)

# Each column in U is a singular vector, where the values
# reflect how much each document contributes to the representation
# of that topic. Each of the vectors corresponds to a topic.
print "U: ", u

# Each value along the diagonal represents the weight of the topic
# in the decomposition of the original data.
S = np.diag(s)
print "S: ", S

# Each row in V is a singular vector, where the values reflect how
# much each characteristic contributes to the representation of the
# topic. Each of the row vectors corresponds to a topic.
print "V: ", v
print ''

print "Results from LSA -------------------------------------"

# Written as text to make it more apparent how the components work with each other.
data_text = [['data', 'inf', 'retrieval'],
             ['data']*2 + ['inf']*2 + ['retrieval']*2,
             ['data', 'inf', 'retrieval'],
             ['data']*5 + ['inf']*5+['retrieval']*5,
             ['brain']*2 + ['lung']*2,
             ['brain']*3 + ['lung']*3,
             ['brain', 'lung']]

dictionary = corpora.Dictionary(data_text)
# Compute bag of words representation of the text data, which yields numbers
# corresponding to the original data.
corpus = [dictionary.doc2bow(text) for text in data_text]
#corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus) 
#corpus = corpora.MmCorpus('/tmp/corpus.mm', 
print "Corpus: "
for entry in corpus:
  print(entry)
print ''

lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
corpus_lsi = lsi_model[corpus]
print "Topics: "
for topic in lsi_model.print_topics(2):
  print(topic)
print ''
#print lsi_model.print_topics(2)

print "Corpus transformed wrt topics from LSI: " 
for doc in corpus_lsi:
  print(doc)
