from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

from nltk.metrics import edit_distance

def calculate_difference_rate(true_answer, output_answer):
    # Calculate the Levenshtein distance between the true answer and output answer
    distance = edit_distance(true_answer.lower(), output_answer.lower())

    # Calculate the difference rate as a percentage
    max_length = max(len(true_answer), len(output_answer))
    difference_rate = (distance / max_length) * 100

    return difference_rate
def calculate_sts(true_answer, output_answer):
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Encode the true answer and output answer into embeddings
    true_embedding = model.encode(true_answer)
    output_embedding = model.encode(output_answer)

    # Calculate the cosine similarity between the embeddings
    similarity_score = 1 - cosine(true_embedding, output_embedding)

    return similarity_score
