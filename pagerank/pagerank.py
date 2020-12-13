import os
import random
import re
import sys
import numpy as np
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    transition_dict = {}
    transition_dict[page] = (1 - damping_factor) / len(corpus)
    for i in corpus[page]:
        transition_dict[i] = transition_dict[page] + damping_factor / len(corpus[page])
    for i in corpus.keys():
        transition_dict[i] = transition_dict.get(i, transition_dict[page])
    return transition_dict
    


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample = random.choice(list(corpus.keys()))
    probabilities = {}
    for i in range(n):
        transition_probs = transition_model(corpus, sample, damping_factor)
        probs = []
        for j in corpus.keys():
            probs.append(transition_probs[j])
        suma = sum(probs)
        if (suma != 1):
            for j in probs:
                j /= suma
            suma = sum(probs)
            probs[random.randint(0, len(probs) - 1)] += (1 - suma)
        sample = np.random.choice(list(corpus.keys()), 1, p = probs)[0]
        probabilities[sample] = probabilities.get(sample, 0) + (1 / n)
        
    return probabilities

def PR(page, pagerank, corpus, damping_factor):
    suma = 0
    links = []
    for i in corpus.keys():
        if (page in corpus[i]):
            links.append(i)
    for i in links:
        suma += (pagerank[i] / len(corpus[i]))
    suma *= damping_factor
    pagerank[page] = suma + (1 - damping_factor) / len(corpus)

def iterate_pagerank(corpus, damping_factor, threshold = 0.001):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {}
    for i in corpus.keys():
        pagerank[i] = 1 / (len(corpus))
    while True:
        flag = 1
        old_values = copy.deepcopy(pagerank)
        for i in corpus.keys():
            PR(i, pagerank, corpus, damping_factor)
        for i in pagerank.keys():
            if (abs(old_values[i] - pagerank[i]) > threshold):
                flag = 0
                break
        if (flag):
            break

    suma = sum(list(pagerank.values()))
    for v in pagerank.keys():
        pagerank[v] /= suma
    return pagerank




if __name__ == "__main__":
    main()
