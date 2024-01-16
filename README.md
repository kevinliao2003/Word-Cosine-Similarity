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

```bash
git clone https://github.com/your-username/PPMIObject.git
```

## Usage

Instantiate the PPMIObject by providing the filename and window size:

```python
from PPMIObject import PPMIObject
P = PPMIObject("your_corpus.txt", wsize=5)
```

You can then use the various methods provided by the class to analyze co-occurrence data.

## Example

```python
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

## License
This project is licensed under the MIT License - see the LICENSE file for details.

```javascript
Remember to replace `"your-username"` and `"your_corpus.txt"` with your GitHub username and the actual filename of your corpus. Save this content as `README.md` in the root directory of your project.
```