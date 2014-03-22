#! /usr/bin/env python
 
import numpy as np
from scipy.sparse import *
#from numba import jit
import os
import sys
from colorama import init
from colorama import Fore, Back, Style
from scipy import *
import time
import argparse
from heapq import nlargest
from itertools import izip
from sklearn.feature_extraction.text import CountVectorizer
 
old_settings = np.seterr(all='ignore') 


def read_files(directory, extension, verbose=False, name=""):
    file_read_start = time.time()
    found_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("." + extension):
                found_files.append(root + '/' + file)
    if verbose:
        if found_files:
            file_read_spent = time.time() - file_read_start
            print Fore.GREEN + "Read %d %sfiles in %f seconds (avg %fs per file)" % (len(found_files), name.rstrip() + ' ', file_read_spent, file_read_spent/len(found_files))
        else:
            print Fore.RED + "No %sfiles read. Is it the right directory?" % (name.rstrip() + ' ')
            system.exit(8)
    return found_files

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
        corpus = []
        for doc in documents:
            with open(doc, 'r') as f:
                for line in f:
                    corpus.append( line.rstrip())
        cf = np.array(self.vectorizer.fit_transform(corpus).sum(axis=0))[0]
        
        # Equation (2): log(P(t_i|C) * (1-lambda)) 
        self.corpus_prob = (np.log(cf) - np.log(np.sum(cf))) + np.log(1-self.l)
 
    # Create a language model per document
    def lm(self, document, iterations, eps):
        # term frequency
        tf = self.vectorizer.transform([document]).toarray()[0]

        # Equation (3): log(P(t_i|D=d))
        doc_prob = (np.log(tf > 0) - np.log((tf > 0).sum()))# + np.log(self.l)

        doc_prob = self.EM(tf, doc_prob, iterations, eps)
        return (tf, doc_prob)
 
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
 
    def fit(self, texts, iterations=50, eps=1e-5, files=False):
        self.document_model = dok_matrix((len(texts),len(self.vectorizer.vocabulary_)))
        self.document_freq = dok_matrix((len(texts),len(self.vectorizer.vocabulary_)))

        if files:
            for label, file_name in enumerate(texts):
                if args.verbose:
                    sys.stdout.write(Fore.GREEN + "\r%d/%d %d%%" %(label,len(texts),float(label)/len(texts)*100.0))
                    sys.stdout.flush()
                file_content = ""
                with open(file_name, 'r') as f:
                    for line in f:
                        file_content += line.rstrip()
                tf, lm = self.lm(file_content, iterations, eps)

                for (m,n) in [ (x,y) for (x,y) in enumerate(tf) if y > 0 ]:
                    self.document_freq[label,m] = n
                for (m,n) in [ (x,y) for (x,y) in enumerate(lm) if not (np.isnan(y) or np.isinf(y)) ]:
                    self.document_model[label,m] = n
            if args.verbose:
                sys.stdout.write("\r")
                sys.stdout.flush()
        else:
            for label, text in enumerate(texts):
                tf, lm = self.lm(text, iterations, eps)
                
                for (m,n) in [ (x,y) for (x,y) in enumerate(tf) if y > 0 ]:
                    self.document_freq[label,m] = n
                for (m,n) in [ (x,y) for (x,y) in enumerate(lm) if not (np.isnan(y) or np.isinf(y)) ]:
                    self.document_model[label,m] = n
 
    def word_prob(self, word):
        if not hasattr(self, 'document_model'):
            raise ValueError("No Language Model fitted.")
        word_prob = 0
       
        # the word frequency is implicit in the background_model
        # by virtue of the self.lm(...) function. Here we just 
        # go over the documents and aggregate the probabilities
        word_id = self.vectorizer.vocabulary_.get(word)
        if word_id is not None:
            occurences = 0
            for ((d_id, w_id), val) in self.document_model.items():
                if w_id == word_id:
                    word_prob += self.document_model[d_id, w_id]
                    occurences += self.document_freq[d_id, w_id]
            # geometric average (sum over logs)
            return np.log(pow(np.exp(word_prob), 1.0/occurences)) if occurences > 0 else word_prob
        return word_prob

init(autoreset=True)

argparser = argparse.ArgumentParser()
argparser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
argparser.add_argument("-b", "--background", help="the directory with background files")
argparser.add_argument("-f", "--foreground", help="the directory with foreground files")
argparser.add_argument("-r", "--recursive", help="traverse the directories recursively", action="store_true")
argparser.add_argument("-e", "--extension", help="only use files with this extension", default="txt")
argparser.add_argument("-w", "--weight", help="the mixture parameter between background and foreground", type=float, default="0.5")
argparser.add_argument("-i", "--iterations", help="the number of iterations for EM", type=int, default="50")

args = argparser.parse_args()

if args.verbose:
    print Fore.YELLOW + "Verbosity enabled"
    print Fore.YELLOW + "Background files in: %s" % args.background
    print Fore.YELLOW + "Foreground files in: %s" % args.foreground
    print Fore.YELLOW + "Recursive file search: %s" % ("Yes" if test else "No")
    print Fore.YELLOW + "Lambda: %f" % args.weight
    print Fore.YELLOW + "Extension: %s" % args.extension
    print Fore.YELLOW + "Iterations: %d" % args.iterations

if args.background is not None:
    background_files = read_files(args.background, args.extension, verbose=args.verbose, name="background")

if args.foreground is not None:
    foreground_files = read_files(args.foreground, args.extension, verbose=args.verbose, name="foreground")

lm_build_start = time.time()
plm = ParsimoniousLM(background_files, args.weight)
if args.verbose:
    lm_build_spent = time.time() - lm_build_start
    print Fore.GREEN + "Built a vocabulary with %d words from %d background corpus files in %f seconds (avg %fs per file/avg %fs per type)" % (len(plm.vectorizer.vocabulary_), len(background_files), lm_build_spent, lm_build_spent/len(background_files), lm_build_spent/len(plm.vectorizer.vocabulary_))

# Clean up some of the mess
background_files = None


lm_fit_start = time.time()
plm.fit(foreground_files, files=True)
if args.verbose:
    lm_fit_spent = time.time() - lm_fit_start
    nr_tokens = sum(plm.document_freq.sum(0))
    print Fore.GREEN + "Fitted %d document models with %d tokens in %f seconds (avg %fs per file/avg %fs per token)" % ((plm.document_freq.shape)[0], nr_tokens, lm_fit_spent, lm_fit_spent/len(foreground_files),lm_fit_spent/nr_tokens)


for (itr, (word, word_id)) in enumerate(plm.vectorizer.vocabulary_.iteritems()):
    print ("(%d) %s: %f" % (word_id, word, plm.word_prob(word))).encode('utf-8')
    #if itr > 10:
    #    break
