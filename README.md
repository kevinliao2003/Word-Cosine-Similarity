# PPMIObject
PPMIObject is a Python class that computes Positive Pointwise Mutual Information (PPMI) and cosine similarity for word pairs based on co-occurrence statistics.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Methods](#methods)
- [License](#license)

## Installation
Clone the repository to your local machine:

```
bash
git clone https://github.com/your-username/PPMIObject.git
```

## Usage
Instantiate the PPMIObject by providing the filename and window size:

```
from PPMIObject import PPMIObject
P = PPMIObject("your_corpus.txt", wsize=5)
```

You can then use the various methods provided by the class to analyze co-occurrence data.

## Example
```
target_word, context_word = "spear", "sword"
pc = P.return_pairwise_count(target_word, context_word)
print("Pairwise count of \"%s\" and \"%s\": %d" % (target_word, context_word, pc))

top_k_cosine = P.return_topk_cosine('sea', 10)
print("Top 10 words with highest cosine similarity to 'sea':", top_k_cosine)
```

## Methods
- return_pairwise_count(target_word, context_word): Returns the pairwise count of a target word and context word.
- return_topk_cosine(word, k): Returns the top k words with the highest cosine similarity to the given word.
...
Refer to the source code or docstrings for more details on available methods.