import re, math
from collections import Counter
from math import log
from prettytable import PrettyTable
import wikipedia
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Compute Term Frequency (TF)
def compute_tf(tokens, vocab):
    count = Counter(tokens)
    total_terms = len(tokens)
    return { term: count[term] / total_terms for term in vocab }

# Compute Inverse Document Frequency (IDF)
def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        df = sum(term in doc for doc in tokenized_docs)
        idf_dict[term] = log(N / (df or 1))
    return idf_dict

# Compute TF-IDF for each term
def compute_tfidf(tf_vector, idf, vocab):
    return { term: tf_vector[term] * idf[term] for term in vocab }

# Compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2, vocab):
    dot_product = sum(vec1[term] * vec2[term] for term in vocab)
    vec1Length = math.sqrt(sum(vec1[term]**2 for term in vocab))
    vec2Length = math.sqrt(sum(vec2[term]**2 for term in vocab))

    if vec1Length == 0 or vec2Length == 0:
        return 0.0

    return dot_product / (vec1Length * vec2Length)

def main():
    topics = ['Albert Einstein', 'Isaac Newton', 'Alexander the Great', 'Mahatma Gandhi', 'Martin Luther King Jr.']
    documents = []

    for topic in topics:
        try:
            page = wikipedia.page(topic)
            documents.append(page)
        except wikipedia.exceptions.PageError:
            print(f"Page not found for topic: {topic}")
            documents.append(None)

    STOP_WORDS = stopwords.words('english')
    wiki_texts = []

    for document in documents:
        if document:
            # Fixed line: escaped quotes and special characters
            wiki_text = re.sub(r"[?,().[\]\"'`\-\/;=_:]", " ", document.content[:8000].lower())
        else:
            wiki_text = ""
        wiki_texts.append(wiki_text)

    wiki_texts_tokens = [word_tokenize(text, language='english') for text in wiki_texts]
    wiki_texts_tokens = [[word for word in text if word not in STOP_WORDS] for text in wiki_texts_tokens]

    vocabulary = set(token for wiki_text_tokens in wiki_texts_tokens for token in wiki_text_tokens)

    tf_vectors = [compute_tf(wiki_text_tokens, vocabulary) for wiki_text_tokens in wiki_texts_tokens]
    idf_vectors = compute_idf(wiki_texts_tokens, vocabulary)
    tfidf_vectors = [compute_tfidf(tf, idf_vectors, vocabulary) for tf in tf_vectors]

    similarity = cosine_similarity(tfidf_vectors[2], tfidf_vectors[4], vocabulary)
    print(f"\nCosine Similarity between \"{topics[2]}\" and \"{topics[4]}\":")
    print(similarity)

if __name__ == "__main__":
    main()
