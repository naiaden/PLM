#! /usr/bin/env python
 
import numpy as np
from scipy.sparse import *
from scipy import *
from heapq import nlargest
from itertools import izip
from sklearn.feature_extraction.text import CountVectorizer
 
old_settings = np.seterr(all='ignore') 
 
def logsum(x):
    """Computes the sum of x assuming x is in the log domain.
 
    Returns log(sum(exp(x))) while minimizing the possibility of
    over/underflow.
 
    Examples
    ========
 
    >>> import numpy as np
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsum(a)
    9.4586297444267107
    """
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = x.max(axis=0)
    out = np.log(np.sum(np.exp(x - vmax), axis=0))
    out += vmax
    return out
 
class ParsimoniousLM(object):
    def __init__(self, documents, weight=0.5, min_df=1, max_df=1.0):
        self.l = weight

        # analyses words, originally it counted the characters
        self.vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)

        # corpus frequency
        cf = np.array(self.vectorizer.fit_transform(documents).sum(axis=0))[0]
        
        # Equation (2): log(P(t_i|C) * (1-lambda)) 
        self.corpus_prob = (np.log(cf) - np.log(np.sum(cf))) + np.log(1-self.l)
 
    def topK(self, k, document, iterations=50, eps=1e-5):
        ptf = self.lm(document, iterations, eps)
        return nlargest(k, izip(self.vectorizer.get_feature_names(), ptf), lambda tp: tp[1])
 
    # Create a language model per document
    def lm(self, document, iterations, eps):
        # term frequency
        tf = self.vectorizer.transform([document]).toarray()[0]

        # Equation (3): log(P(t_i|D=d))
        doc_prob = (np.log(tf > 0) - np.log((tf > 0).sum()))# + np.log(self.l)

        doc_prob = self.EM(tf, doc_prob, iterations, eps)
        return doc_prob
 
    def EM(self, tf, doc_prob, iterations, eps):
        tf = np.log(tf)
        for i in xrange(1, iterations + 1):
            doc_prob += np.log(self.l)

            # follows E and M steps from the paper
            E = tf + doc_prob - np.logaddexp(self.corpus_prob, doc_prob)
            M = E - logsum(E)
 
            diff = M - doc_prob
            doc_prob = M
            if (diff < eps).all():
                break
        return doc_prob
 
    def fit(self, texts, iterations=50, eps=1e-5):
        self.background_model = dok_matrix((len(texts),len(self.vectorizer.vocabulary_)))
        for label, text in enumerate(texts):
            lm = self.lm(text, iterations, eps)
            for (m,n) in [ (x,y) for (x,y) in enumerate(lm) if not (np.isnan(y) or np.isinf(y)) ]:
                self.background_model[label,m] = n
 
    def fit_transform(self, texts, iterations=50, eps=1e-5):
        self.fit(texts, iterations, eps)
        return self.background_model
 
    # Equation (4)
    def cross_entropy(self, query_lm, background_lm):
        return -np.sum(np.exp(query_lm) * np.logaddexp(self.corpus_prob, background_lm * self.l))
 
    def predict_proba(self, query):
        if not hasattr(self, 'background_model'):
            raise ValueError("No Language Model fitted.")
        for i in range(len(self.background_model)):
            score = self.cross_entropy(query, self.background_model[i][1])
            yield self.background_model[i][0], score

    def word_prob(self, word):
        if not hasattr(self, 'background_model'):
            raise ValueError("No Language Model fitted.")
        word_prob = 0
        
        word_id = self.vectorizer.vocabulary_.get(word)
        if word_id is not None:
            occurences = 0
            for (d_id, w_id) in self.background_model.keys():
                if w_id == word_id:
                    word_prob += self.background_model[d_id, w_id]
                    occurences += 1
            #word_prob = np.nansum([row[1][word_id] for row in self.background_model])
            #print [row[word_id] for row in self.background_model]
            return np.log(pow(np.exp(word_prob), 1.0/occurences))
        return word_prob
 
 
def demo():
    documents = ['er loopt een man op straat', 'de man is vies', 'allemaal nieuwe woorden', 'de straat is vies', 'de man heeft een gek hoofd', 'de hele straat kijkt naar de man']
    request = 'op de straat is vies lol'
    # initialize a parsimonious language model
    plm = ParsimoniousLM(documents, 0.1)
    # compute a LM for each document in the document collection
    plm.fit(documents)

    print "--- Parsimony at index time"
    query_lm = plm.lm(request, 50, 1e-5)
    print "    --- word probs"
    for word in request.split():
        print word, plm.word_prob(word)

    #print "--- Parsimony at request time"
    ## compute a LM model for the test or request document
    #qlm = plm.lm(request, 50, 1e-5)
    ## compute the cross-entropy between the LM of the test document and all training document LMs
    ## sort by increasing entropy
    #print [(documents[i], score) for i, score in sorted(plm.predict_proba(qlm), key=lambda i: i[1])]
   
if __name__ == '__main__':
    demo()
