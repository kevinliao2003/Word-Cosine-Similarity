import sys
from collections import Counter, defaultdict
import math

class PPMIObject(object):

	def __init__(self, filename, wsize=5):
		self.filename = filename
		self.wsize = wsize # window size
		self.PC = self.get_pairwise_counts() # table of pairwise counts
		self.tc = self.return_pairwise_count_total() # sum over all pairwise counts
		self.MTC, self.MCC = self.get_marginal_counts() # table of marginal target counts
		self.PPMI = self.compute_ppmi_table() # table of PPMI values
		self.CT = self.make_cosine_table() # table of cosine similarities between word pairs


	def get_pairwise_counts(self):
		""" Gather pairwise counts of all (target_word, context_word) word co-occurrences with
		windowsize self.wsize. Return as a full lookup table of counts.
		Details: The input file is already tokenized and lower-cased; no further preprocessing
		is required. Use a windowsize of self.wsize to determine the number of tokens to the left
		AND right of the current target word to include as a context word.
		Truncate windows at the end of a line (do not have context windows run between lines.)
		If the target_word is the first word in the line, then it will have zero left-context tokens;
		if the target_word is the second word in the line, then it will have one left-context token, etc.
		"""
		PC = Counter()
		with open(self.filename, 'r', encoding="utf8") as file:
			for line in file:
				tokens = line.split()
				size = len(tokens)

				for i, target_word in enumerate(tokens):
					left = max(0, i - self.wsize) #determines number of tokens to the left
					right = min(size, i + self.wsize + 1) #determines number of tokens to the right
					context_words = tokens[left:i] + tokens[i+1:right]

					#update (target_word, context_word) pair
					for context_word in context_words:
						PC[(target_word, context_word)] += 1
		return PC

	def return_pairwise_count(self, target_word, context_word):
		""" Function that returns the pairwise counts (int) of a target word and context word. 
		See .get_pairwise_counts() method for details.
		"""
		pc = self.PC.get((target_word, context_word), 0)
		return pc

	def return_pairwise_count_total(self):
		""" Returns the sum (int) over all pairwise counts, i.e. sum_xy Count(x,y). """
		tc = sum(self.PC.values())
		return tc

	def get_marginal_counts(self):
		""" Taking the pairwise count table you constructed in get_pairwise_counts,
		construct tables of marginal target word counts, and marginal context word counts."""

		MTC = Counter() # Marginal target counts, i.e. sum_x Count(target_word, x)
		MCC = Counter() # Marginal context counts, i.e. sum_y Count(y, context_word)
		PC = self.PC

		#iterates over the PC table and accumulates the marginal counts
		for (target_word, context_word), count in PC.items():
			MTC[target_word] += count
			MCC[context_word] += count
		return MTC, MCC

	def return_marginal_target_count(self, target_word):
		""" Function that returns marginal counts (int) of a target word, i.e. sum_x Count(target_word, x) . """
		mtc = self.MTC[target_word]
		return mtc	

	def return_marginal_context_count(self, context_word):
		""" Function that returns marginal counts (int) of a context word, i.e. sum_y Count(y, context_word). """
		mcc = self.MCC[context_word]
		return mcc

	def compute_ppmi_table(self):
		""" Given pairwise and marginal tables, compute Positive PMI for all observed pairs """

		PPMI = Counter() # Note that values in Counters do not need to be integers
		PC = self.PC
		MTC, MCC = self.MTC, self.MCC
		
		pairwise_count_total = self.tc  #total count of all pairwise co-occurrences
		for (target_word, context_word), count in PC.items():
			#computes the P(target_word, context_word) joint probability
			joint = count / pairwise_count_total

			#computes the P(target_word) and P(context_word) marginal probabilities
			prob_target = MTC[target_word] / pairwise_count_total
			prob_context = MCC[context_word] / pairwise_count_total

			#computes PPMI
			pmi = max(0, math.log(joint / (prob_target * prob_context)))

			#stores pair in Counter
			if not target_word in PPMI:
				PPMI[target_word] = {}
			PPMI[target_word][context_word] = pmi

		return PPMI

	def return_ppmi_value(self, target_word, context_word):
		""" Return positive PMI, PPMI(x,y), (a float value) for target word x and context word y. 
		Use natural log. """

		PPMI = self.PPMI
		if target_word in PPMI and context_word in PPMI[target_word]:
			return PPMI[target_word][context_word]
		elif context_word in PPMI and target_word in PPMI[context_word]:
			return PPMI[context_word][target_word]
		else:
			return 0.0

	def return_topk_ppmi(self, word, k):
		""" Return ordered list of the k words (as strings) with the highest PPMI to the given word.
		The list of strings should be in decreasing order, starting with the highest-PPMI word. """
		PPMI = self.PPMI

		#creates a list of tuples (context_word, ppmi) for the given target word
		ppmi_list = [(context_word, ppmi) for context_word, ppmi in PPMI[word].items()]
		
		#sorts by PPMI values in decreasing order
		ppmi_list.sort(key=lambda x: x[1], reverse=True)

		#extracts the top k highest PPMI values
		topk_ppmi = [context_word for context_word, _ in ppmi_list[:k]]
		return topk_ppmi

	def return_cosine_similarity(self, word1, word2):
		""" Return cosine similarity of word1 and word2, where their vector representation is their PPMI
		score with all other observed words """
		CT = self.CT
		if word1 in CT and word2 in CT[word1]:
			return CT[word1][word2]
		elif word2 in CT and word1 in CT[word2]:
			return CT[word2][word1]
		else:
			return 0

	def make_cosine_table(self):
		""" Create a table lookup for cosine similarity of all pairs of words in the vocabulary. """
		CT = defaultdict(Counter)
		PPMI = self.PPMI

		#maps each word to a list of other words, PPMI values of which aren't 0
		tmp = Counter()
		for target in PPMI.keys():
			for context, ppmi in PPMI[target].items():
				if ppmi > 0:
					if not target in tmp:
						tmp[target] = {}
					tmp[target][context] = ppmi
	
		vectors = {} #creates vector for each target word
		magnitudes = {} #computes magnitude for each target word's vector
		for target, dic in tmp.items():
			vectors[target] = [context for (context, _) in dic.items()]
			magnitudes[target] = math.sqrt(sum(PPMI[target][context] * PPMI[target][context] 
									  for context in vectors[target]))

		for i, word1 in enumerate(vectors):
			for j, word2 in enumerate(vectors):
				if i < j:
					common = set(vectors[word1]) & set(vectors[word2]) #finds common words between vals and vals2
					vector1 = [PPMI[word1][context] for context in common]
					vector2 = [PPMI[word2][context] for context in common]
					dot_product = sum(x * y for x, y in zip(vector1, vector2))
					a_magnitude, b_magnitude, = magnitudes[word1], magnitudes[word2]
					if a_magnitude == 0 or b_magnitude == 0: #prevent divide by 0
						CT[word1][word2] = 0.0
					else:
						CT[word1][word2] = dot_product/(a_magnitude*b_magnitude)

		return CT

	def return_topk_cosine(self, word, k):
		""" Given a word, eturn ordered list of k nearest neighboring words. I.e. return a list of 
		lenght k with the words with the highest cosine similarity score to the given word.
		The list should be in decreasing order, starting with the closest neighboring word/highest
		cosine similarity. """
		CT = self.CT

		# Hint: the .most_common() method of the Counter() type may be useful here
		if word not in CT: #edge case
			return []
		
		#gets cosine similarities for the given word and sorts them in decreasing order
		similarities = CT[word]
		decreasing = similarities.most_common()

		#extracts top k neighboring words
		topk_cos = [neighbor for neighbor, _ in decreasing[:k]]
		return topk_cos

if __name__=="__main__":
	P = PPMIObject("HW2/odyssey.short.tok.txt", wsize=5)
	target_word, context_word = "spear", "sword"
	pc = P.return_pairwise_count(target_word, context_word)
	print("Pairwise count of \"%s\" and \"%s\": %d" % (target_word, context_word, pc))
	print(P.return_topk_cosine('sea', 10))
	
	P2 = PPMIObject("HW2/odyssey.short.tok.txt", wsize=20)
	print(P2.return_topk_cosine('sea', 10))
