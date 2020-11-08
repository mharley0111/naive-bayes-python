import sys
import string
import math

class NbClassifier(object):

    """
    A Naive Bayes classifier object has three parameters, all of which are populated during initialization:
    - a set of all possible attribute types
    - a dictionary of the probabilities P(Y), labels as keys and probabilities as values
    - a dictionary of the probabilities P(F|Y), with (feature, label) pairs as keys and probabilities as values
    """
    def __init__(self, training_filename, stopword_file):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}   

        self.collect_attribute_types(training_filename)
        if stopword_file is not None:
            self.remove_stopwords(stopword_file)
        self.train(training_filename)


    """
    A helper function to transform a string into a list of word strings.
    You should not need to modify this unless you want to improve your classifier in the extra credit portion.
    """
    def extract_words(self, text):
        no_punct_text = "".join([x for x in text.lower() if not x in string.punctuation])
        return [word for word in no_punct_text.split()]


    """
    Given a stopword_file, read in all stop words and remove them from self.attribute_types
    Implement this for extra credit.
    """
    def remove_stopwords(self, stopword_file):
        self.attribute_types.difference(set())

    """
    Given a training datafile, add all features that appear at least m times to self.attribute_types
    """
    def collect_attribute_types(self, training_filename, m=1):
    	#print("m: {}".format(m))
    	self.attribute_types = set()
    	input = open(training_filename, 'r')
    	words = dict()
    	message = list()

    	for line in input.readlines():
        	# get list of words in each message
        	message = self.extract_words(line)

        	# add word to words dict
        	for word in message[1:]:

        		if word in words.keys() and words[word] < m:
        			words[word] += 1
        		elif word not in words.keys():
        			words[word] = 1

		# add features (words) to NB attribute
    	for feature in words.keys():
        	if words[feature] >= m:
        		self.attribute_types.add(feature)


    """
    Given a training datafile, estimate the model probability parameters P(Y) and P(F|Y).
    Estimates should be smoothed using the smoothing parameter k.
    """
    def train(self, training_filename, k=0.05):
    	#print("k: {}".format(k))
    	self.label_prior = {}
    	self.word_given_label = {}
    	label = ""
    	label_count = dict()
    	message = list()
    	word_count = dict()
    	samples = 0
    	input = open(training_filename, 'r')

        # collect label count and word given label count
    	for line in input.readlines():
        	message = self.extract_words(line)
        	label = message[0]

        	if label in label_count.keys():
        		label_count[label] += 1
        	else:
        		label_count[label] = 1

        	for word in message[1:]:
        		if word in self.attribute_types:
        			if (word, label) in word_count.keys():
        				word_count[(word, label)] += 1
        			else:
        				word_count[(word, label)] = 1

        	samples += 1

        # get priors
    	for label in label_count.keys():
        	self.label_prior[label] = label_count[label] / samples

        # smooth parameters
    	for label in self.label_prior.keys():
        	for word in self.attribute_types:
        		if (word, label) not in word_count.keys():
        			word_count[(word, label)] = 0
        		self.word_given_label[(word, label)] = \
        		(word_count[(word, label)] + k) / (label_count[label] + (k * abs(len(self.attribute_types))))


    """
    Given a piece of text, return a relative belief distribution over all possible labels.
    The return value should be a dictionary with labels as keys and relative beliefs as values.
    The probabilities need not be normalized and may be expressed as log probabilities. 
    """
    def predict(self, text):
    	belief = 0
    	beliefs = dict()
    	message = self.extract_words(text)

    	for label in self.label_prior.keys():
    		belief = math.log(self.label_prior[label])
    		for word in message:
    			if word in self.attribute_types:
    				belief += math.log(self.word_given_label[(word, label)])
    		beliefs[label] = belief

    	return beliefs


    """
    Given a datafile, classify all lines using predict() and return the accuracy as the fraction classified correctly.
    """
    def evaluate(self, test_filename):
    	correct, total, accuracy = 0, 0, 0
    	beliefs = dict()
    	max_belief = float('-inf')
    	prediction = ""
    	label = ""
    	input = open(test_filename, 'r')

    	for line in input.readlines():
    		message = self.extract_words(line)
    		label = message[0]

    		# get prediction
    		beliefs = self.predict(" ".join(message[1:]))
    		for belief in beliefs.keys():
    			if beliefs[belief] > max_belief:
    				max_belief = beliefs[belief]
    				prediction = belief
    		max_belief = float('-inf')

    		# see if we have a correct prediction
    		if prediction == label:
    			correct += 1

    		total += 1

    	accuracy = correct / total

    	return accuracy


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nusage: ./hmm.py [training data file] [test or dev data file] [(optional) stopword file]")
        exit(0)
    elif len(sys.argv) == 3:
        classifier = NbClassifier(sys.argv[1], None)
    else:
        classifier = NbClassifier(sys.argv[1], sys.argv[3])
    print(classifier.evaluate(sys.argv[2]))