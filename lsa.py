#Author: MeiXing Dong

import numpy as np

from gensim import corpora, models, similarities

# Original data
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

print "U: ", u
S = np.diag(s)
print "S: ", S
print "V: ", v
print ''

print "Results from LSA -------------------------------------"

#lsi_model = models.LsiModel(data, num_topics=2)
#corpus_lsi = lsi_model[data]

#lsi.print_topics(2)
