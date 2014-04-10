#! /usr/bin/env python
 
import numpy as np
from scipy.sparse import *
#from numba import jit
import os
import datetime
import sys
from colorama import init
from colorama import Fore, Back, Style
from scipy import *
import time
import argparse
from heapq import nlargest
from itertools import izip
from sklearn.feature_extraction.text import CountVectorizer
import cPickle as pickle


old_settings = np.seterr(all='ignore') 

def hms(seconds):
    """ Returns a string representation in h:mm:ss.ssssss of the input
    """
    return str(datetime.timedelta(seconds=seconds))

def read_serialised_file(serialised_file, verbose=False, name=""):
    """ This method returns data from a serialised binary pickle file. 
        The content of the file is not checked, and passed on as is. 
    
        name is only used in verbose output.
    """
    file_read_start = time.time()
    if verbose:
        sys.stdout.write(Fore.GREEN + "Reading serialised %sfile: %s" % (name.rstrip() + ' ', serialised_file))
    with open(serialised_file, 'rb') as f:
        unpickled_data = pickle.load(f)
    if verbose:
        shape = ""
        if type(unpickled_data) is dict:
            shape = len(unpickled_data)
        else:
            shape = unpickled_data.shape
        print Fore.GREEN + "\rRead the serialised %sfile in %s. The file's shape is %s" % (name.rstrip() + ' ', hms(time.time() - file_read_start), shape)

    return unpickled_data
        
def write_serialised_file(content, file_name, verbose=False, name=""):
    """ This method writes an object to a serialised binary pickle file.
        The content of the file is not checked, and written to file as is.
        If the file already exists, the current file is overwritten.
    
        name is only used in verbose output.
    """
    file_write_start = time.time()
    if verbose:
        sys.stdout.write(Fore.GREEN + "Writing serialised %sfile: %s" % (name.rstrip() + ' ', file_name))
    with open(file_name, 'wb') as f:
        pickle.dump(content, f, pickle.HIGHEST_PROTOCOL)

    if verbose:
        if type(content) is dict:
            shape = len(content)
        else:
            shape = content.shape
        print Fore.GREEN + "\rWrote the serialised %sfile in %s to %s with shape %s" % (name.rstrip() + ' ', hms(time.time() - file_write_start), file_name, shape)

def read_files(directory, extension, verbose=False, name=""):
    """ This method returns a list of files with a certain extension from a directory.
        The files themselves are not read.
    
        name is only used in verbose output.
    """
    file_read_start = time.time()
    found_files = []
    if verbose:
        sys.stdout.write(Fore.GREEN + "Listing all files... This might take a while")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("." + extension):
                found_files.append(root + '/' + file)
    if verbose:
        if found_files:
            file_read_spent = time.time() - file_read_start
            print Fore.GREEN + "\rRead %d %sfiles in %s (avg %fs per file)" % (len(found_files), name.rstrip() + ' ', hms(file_read_spent), file_read_spent/len(found_files))
        else:
            print Fore.RED + "\rNo %sfiles read. Is it the right directory?" % (name.rstrip() + ' ')
            sys.exit(8)
    return found_files

def logsum(x):
    """ Computes the sum of x assuming x is in the log domain.
     
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

    def __init__(self, weight=0.5, max_features=None, maxn=1, min_df=1, max_df=1.0, files=None, serialised=None, vocabulary=None, verbose=False):
        """ This constructor builds a vectorizer and a background corpus' frequency counts
            weight is the interpolation factor/mixture weight for a balance between the background and foreground proabilities.
            min_df, maxdf are the minimum and respectively maximum document frequencies for the word needed to be modelled.
            files is the list of files to be read from which a background corpus is built. Should be None if serialised is not None.
            serialised is a binary serialised file containing the background corpus' frequency counts. If serialised is not None, a vocabulary must be given as well. serialised should be None if files is not None. The serialised file should not be normalised, and not be multiplied by lamdba.
        """
        self.l = weight

        if serialised is not None and vocabulary is not None:
            self.cf = serialised

            self.vectorizer = CountVectorizer(ngram_range=(1,maxn), min_df=min_df, max_df=max_df, max_features=max_features, vocabulary=vocabulary)
        elif serialised is not None:
            print Fore.RED + "Thanks for the serialised background corpus, but we also need the accompanying vocabulary file"
            sys.exit(8)
        elif files is not None:
            # analyses words, originally it counted the characters
            self.vectorizer = CountVectorizer(ngram_range=(1,maxn), min_df=min_df, max_df=max_df, max_features=max_features)

            # corpus frequency
            corpus = []
            for doc in files:
                with open(doc, 'r') as f:
                    for line in f:
                        corpus.append( line.rstrip())
            self.cf = np.array(self.vectorizer.fit_transform(corpus).sum(axis=0))[0]
        else:
            print Fore.RED + "The parsimonious language model should at least have a background corpus!"
            sys.exit(8)
        
        # Equation (2): log(P(t_i|C) * (1-lambda)) 
        if verbose:
            Fore.Green + "Applying lamdba to background corpus"
            lamdba_application_start = time.time()
        self.corpus_prob = (np.log(self.cf) - np.log(np.sum(self.cf))) + np.log(1-self.l)
        if verbose:
            print "\r" + Fore.Green + "Done applying lambda in %f seconds" % (time.time() - lambda_application_start)
 
    def lm(self, document, iterations, eps):
        """ This methods creates a language model per document with n EM iterations and an eps treshold.
            Returns the term frequencies and the document probabilities for all the words in document D.
        """
        # term frequency
        self.print_debug(" lm::Transforming")
        tf = self.vectorizer.transform([document]).toarray()[0]

        # Equation (3): log(P(t_i|D=d))
        self.print_debug(" lm::Equation 3")
        doc_prob = (np.log(tf > 0) - np.log((tf > 0).sum()))# + np.log(self.l)

        self.print_debug(" lm::EM")
        doc_prob = self.EM(tf, doc_prob, iterations, eps)
        return (tf, doc_prob)
 
    def EM(self, tf, doc_prob, iterations, eps):
        """ This methods does the two EM steps with n EM iterations and an eps treshold, for P(t|D)
            Returns the document probabilities.
        """
        tf = np.log(tf)
        for i in xrange(1, iterations + 1):
            self.print_debug(" EM::iteration %d" % i)
            doc_prob += np.log(self.l)

            # follows E and M steps from the paper
            E = tf + doc_prob - np.logaddexp(self.corpus_prob, doc_prob)
            M = E - logsum(E)
 
            diff = M - doc_prob
            doc_prob = M
            if (diff < eps).all():
                break
        return doc_prob

    def get_word(self, wordId):
        return self.rdictionary[wordId]

    def build_reverse_dictionary(self):
        self.rdictionary = [None]*len(self.vectorizer.vocabulary_)#np.empty(len(self.vectorizer.vocabulary_), dtype=str)
        for key, value in plm.vectorizer.vocabulary_.iteritems():
            self.rdictionary[value] = key.encode('utf-8')
            #self.rdictionary[value] = "hoi"

    def fit(self, texts, iterations=50, eps=1e-5, files=False):
        """ This function fits a foreground model onto the background model. The foreground model consists of texts, and the EM parameters are iterations and epsilon for the convergence regulation.
            If files is False, then texts contains an array of strings, each string comprising a document. If files is True, then each entry in texts is a file name pointing to a file that contains a text.

            Returns no explicit value. The results are stored in document_model and document_freq, with the first one being the probabilities for the words, and the second the raw counts.

        """
        self.document_model = dok_matrix((len(texts),len(self.vectorizer.vocabulary_)))
        self.document_freq = dok_matrix((len(texts),len(self.vectorizer.vocabulary_)))

        #print self.rdictionary

        if files:
            last_iter_start = time.time()
            begin_iter_start = last_iter_start
            for label, file_name in enumerate(texts):
                ttf = hms((time.time() - begin_iter_start)/max(1,label)*(len(texts)-label))
                self.debug_output = Fore.GREEN + "\r%d/%d %d%% (TTF: %s)" %(label,len(texts),float(label)/len(texts)*100.0, ttf)
                last_iter_start = time.time()
                if args.verbose:
                    sys.stdout.write(self.debug_output)
                    sys.stdout.flush()
                file_content = ""
                self.print_debug(" fit::reading file")
                with open(file_name, 'r') as f:
                    for line in f:
                        file_content += line.rstrip()
                tf, lm = self.lm(file_content, iterations, eps)

                if args.fwpdir:
                    #print file_name
                    ## Dirty hack, and only applicable when there is a hierarchical structure
                    ## Should also be a regex or the like. This is just plain stupid.
                    file_dir, ext_file_name = file_name.split('/')[-2:]
                    bare_file_name, file_ext = ext_file_name.split('.')

                    doc_prob_dir = args.fwpdir + "/" + file_dir
                    if not os.path.exists(doc_prob_dir):
                        os.makedirs(doc_prob_dir)

                    doc_prob_file_name = doc_prob_dir + "/" + bare_file_name + ".wp"

                    self.print_debug(" fit::writing doc probs to file")
                    with open(doc_prob_file_name, 'w') as f:
                        for (index,), value in np.ndenumerate(lm):
                            if not np.isinf(value):
                                #print self.get_word(index), value
                                f.write("%s\t%s\n" % (self.get_word(index), value))

                self.print_debug(" fit::frequency and model update")
                for (m,n) in [ (x,y) for (x,y) in enumerate(tf) if y > 0 ]:
                    self.document_freq[label,m] = n
                for (m,n) in [ (x,y) for (x,y) in enumerate(lm) if not (np.isnan(y) or np.isinf(y)) ]:
                    self.document_model[label,m] = n
                self.print_debug(" fit::done")
            if args.verbose:
                sys.stdout.write("\r")
                sys.stdout.flush()
        else:
            print Fore.RED + "You have found a dark place in the code, this might work. Or it might not."
            for label, text in enumerate(texts):
                tf, lm = self.lm(text, iterations, eps)
                
                for (m,n) in [ (x,y) for (x,y) in enumerate(tf) if y > 0 ]:
                    self.document_freq[label,m] = n
                for (m,n) in [ (x,y) for (x,y) in enumerate(lm) if not (np.isnan(y) or np.isinf(y)) ]:
                    self.document_model[label,m] = n
 
    def word_prob(self, word):
        """ This function computes the word probability over the collection of foreground documents.
        """
        if not hasattr(self, 'document_model'):
            raise ValueError("No Language Model fitted.")
        word_prob = -1
       
        # the word frequency is implicit in the background_model
        # by virtue of the self.lm(...) function. Here we just 
        # go over the documents and aggregate the probabilities
        word_id = self.vectorizer.vocabulary_.get(word)
        if word_id is not None:
            occurences = 0
            for ((d_id, w_id), val) in self.document_model.items():
                if w_id == word_id:
                    if word_prob < 0:
                        word_prob = 0
                    word_prob += self.document_model[d_id, w_id]
                    occurences += self.document_freq[d_id, w_id]
            # geometric average (sum over logs)
            #return np.log(pow(np.exp(word_prob), 1.0/occurences)) if occurences > 0 else log(word_prob)
            # if word_prob < 0, then occurences 
            return np.log(pow(np.exp(word_prob), 1.0/occurences)) if (word_prob > 0 and occurences > 0) else 0.0
        return word_prob

    def print_debug(self, string):
        if args.debug:
            sys.stdout.write(self.debug_output + string)
            sys.stdout.flush()


################## Things to process cmd-line arguments and stuff

init(autoreset=True)

argparser = argparse.ArgumentParser(description="This (partial) implementation of the 'Parsimonious Language Model for Information Retrieval' (Hiemstra et al., 2004) is forked from F. Karsdorp's implementation (https://github.com/fbkarsdorp/PLM). This fork can be found on https://github.com/naiaden/PLM. You can direct your comments to l.onrust@let.ru.nl")
backgroundgroup_argparser = argparser.add_mutually_exclusive_group()
foregroundgroup_argparser = argparser.add_mutually_exclusive_group()

argparser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
backgroundgroup_argparser.add_argument("-b", "--background", help="the directory with background files", metavar="dir", dest="bdir")
backgroundgroup_argparser.add_argument("-B", "--serialisedbackground", help="the serialised background file", metavar="dir", dest="rserb")
foregroundgroup_argparser.add_argument("-f", "--foreground", help="the directory with foreground files", metavar="dir", dest="fdir")
argparser.add_argument("-wb", "--writeserialisedbackground", help="write the background to a serialised file", metavar="file", dest="wserb")
foregroundgroup_argparser.add_argument("-F", "--serialisedforeground", help="the serialised foreground file. Not yet implemented", metavar="file", dest="rserf")
argparser.add_argument("-V", "--vocabulary", help="the vocabulary, only used for in/output of serialised files", metavar="file", dest="vocabulary")
argparser.add_argument("-wf", "--writeserialisedforeground", help="write the foreground to a serialised file", metavar="file", dest="wserf")
argparser.add_argument("-P", "--allwordprobs", help="write word probs to this file. This takes forever, be prepared!", metavar="file", dest="wpfile")
argparser.add_argument("-p", "--docwordprobs", help="write a file per foreground document with the word probabilities in this directory", dest="fwpdir")
argparser.add_argument("-m", "--maxfeatures", help="the n most frequent words", metavar="n", dest="maxfeatures", type=int, default="-1")
argparser.add_argument("--mindf", help="ignore terms that have a document frequency strictly lower than x, or a proportion of documents", metavar="x", dest="mindf", default="1", type=int) 
argparser.add_argument("--maxdf", help="ignore terms that have a document frequency strictly higher than x, or a proportion of documents", metavar="x", dest="maxdf", default="1.0", type=float) 
argparser.add_argument("-r", "--recursive", help="traverse the directories recursively", action="store_true")
argparser.add_argument("-e", "--extension", help="only use files with this extension", default="txt", metavar="ext")
argparser.add_argument("-w", "--weight", help="the mixture parameter between background and foreground", type=float, default="0.5", metavar="float")
argparser.add_argument("-i", "--iterations", help="the number of iterations for EM", type=int, default="50", metavar="n")
argparser.add_argument("--debug", help="enable debug output. Warning, can be a lot of info", action="store_true")
argparser.add_argument("-n", "--maxn", help="the max number of n for generating n-grams. Default n=1 (unigrams)", type=int, default="1", metavar="n", dest="maxn")

args = argparser.parse_args()

if args.maxfeatures < 0:
    args.maxfeatures = None

if args.verbose:
    print Fore.YELLOW + "Verbosity enabled"
    if args.bdir is not None:
        print Fore.YELLOW + "Background files in: %s" % args.bdir
    else:
        print Fore.YELLOW + "Serialised background files in: %s" % args.rserb 
    if args.fdir is not None:
        print Fore.YELLOW + "Foreground files in: %s" % args.fdir
    else:
        print Fore.YELLOW + "Serialised foreground files in: %s" % args.rserf 
    print Fore.YELLOW + "Word probability output file: %s" % args.wpfile
    print Fore.YELLOW + "Debug enabled: %s" % ("Yes" if args.debug else "No")
    print Fore.YELLOW + "Recursive file search: %s" % ("Yes" if args.recursive else "No")
    print Fore.YELLOW + "Lambda: %f" % args.weight
    print Fore.YELLOW + "Extension: %s" % args.extension
    print Fore.YELLOW + "Max size n-grams: %d" % args.maxn
    print Fore.YELLOW + "Iterations: %d" % args.iterations
    if args.mindf:
        print (Fore.YELLOW + "Min df: %s" % args.mindf)
    if args.maxdf:
        print (Fore.YELLOW + "Max df: %s" % args.maxdf)
    if args.maxfeatures:
        print (Fore.YELLOW + "Max features: %s" % args.maxfeatures)


## Background part
##
if args.bdir:
    background_files = read_files(args.bdir, args.extension, verbose=args.verbose, name="background")

    lm_build_start = time.time()
    plm = ParsimoniousLM(args.weight, files=background_files, maxn=args.maxn, min_df=args.mindf, max_df=args.maxdf, max_features=args.maxfeatures)
elif args.rserb:
    background_serialised_vocabulary = read_serialised_file(args.vocabulary, verbose=args.verbose, name="vocabulary")
    background_serialised = read_serialised_file(args.rserb, verbose=args.verbose, name="background")

    lm_build_start = time.time()
    plm = ParsimoniousLM(args.weight, serialised=background_serialised, vocabulary=background_serialised_vocabulary, min_df=args.mindf, max_df=args.maxdf, max_features=args.maxfeatures, maxn=args.maxn)
else:
    print Fore.RED + "No background input is given. Halting the execution!"
    sys.exit(8)

if args.verbose:
    lm_build_spent = time.time() - lm_build_start
    print Fore.GREEN + "Built a vocabulary with %d words from in %s" % (len(plm.vectorizer.vocabulary_), hms(lm_build_spent))

if args.wserb and args.bdir:
    write_serialised_file(plm.cf, args.wserb, verbose=args.verbose, name="background")
if args.vocabulary and args.bdir:
    write_serialised_file(plm.vectorizer.vocabulary_, args.vocabulary, verbose=args.verbose, name="vocabulary")
background_files = None
plm.cf = None
#
## 

if args.verbose:
    print Fore.GREEN + "Building reverse dictionary"

plm.build_reverse_dictionary()
rd_start = time.time()

if args.verbose:
    print Fore.GREEN + "Done building reverse dictionary in %s" % (hms(time.time() - rd_start))

## Foreground part
##
if args.fdir:
    foreground_files = read_files(args.fdir, args.extension, verbose=args.verbose, name="foreground")
elif args.rserf:
    print Fore.RED + "Although there is some code already, this does not work. Try the directory method. Halting"
    sys.exit(8)
    #foreground_serialised = read_serialised_file(args.rserf, verbose=args.verbose, name="foreground")
else:
    print Fore.RED + "No foreground input file given. Halting the execution!"
    sys.exit(8)

lm_fit_start = time.time()
plm.fit(foreground_files, iterations=args.iterations, files=True)
if args.verbose:
    lm_fit_spent = time.time() - lm_fit_start
    nr_tokens = sum(plm.document_freq.sum(0))
    print Fore.GREEN + "Fitted %d document models with %d tokens in %s (avg %fs per file/avg %fs per token)" % ((plm.document_freq.shape)[0], nr_tokens, hms(lm_fit_spent), lm_fit_spent/len(foreground_files),lm_fit_spent/nr_tokens)
#
##

plm.debug_output = "\rWP "

## Word probs part
##
if args.wpfile:
    if args.verbose:
        print Fore.GREEN + "Writing word probabilities to %s" %  args.wpfile
    wp_start = time.time()
    with open(args.wpfile, 'w') as f:
        for (itr, (word, word_id)) in enumerate(plm.vectorizer.vocabulary_.iteritems()):
            plm.print_debug(" Processing word: %d" % itr)
            f.write(("(%d) %s: %f\n" % (word_id, word, plm.word_prob(word))).encode('utf-8'))
            #print ("(%d) %s: %f" % (word_id, word, plm.word_prob(word))).encode('utf-8')
            #pass
    sys.stdout.write("\r")
    sys.stdout.flush()

    if args.verbose:
        wp_spent = time.time() - wp_start
        print Fore.GREEN + "Writing word probabilities took %s" % hms(wp_spent)
