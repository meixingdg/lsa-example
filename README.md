# LSA (Latent Semantic Analysis) Using gensim

LSA is a technique in NLP for analyzing relationships between sets of documents and the terms they contain by producing a set of topics/concepts that are related to the documents and terms.

The input is a matrix containing the word counts from documents, where the rows and columns correspond to documents and unique words, respectively. Singular value decomposition (SVD) is used to perform dimension reduction.

## Example of SVD
The example used here is taken from page 15 of [http://www.cs.cmu.edu/~jimeng/papers/ICMLtutorial.pdf](these slides from Faloutsos, Koda, and Sun). 

## Usage
Run the following:
'''
python lsa.py
'''

The first portion of the output contains the three matrices from plain SVD using numpy's SVD implementation.

The second portion contains the corresponding three matrices from gensim's LSI module. The input used is a matrix with documents containing words that reflect the characteristics in the example (eg. "data", "inf", "retrieval", "brain", "lung"). The output shows how the documents are grouped properly into the two desired topics.
