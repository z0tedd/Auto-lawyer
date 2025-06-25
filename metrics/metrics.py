from sklearn.metrics import f1_score

from sentence_transformers import SentenceTransformer, util
import numpy as np


def calculate_mrr(questions, ground_truths, model):
    """
    Calculates the Mean Reciprocal Rank (MRR) for the given questions and ground truths.

    Parameters:
    questions: List of questions.
    ground_truths: List of correct answers corresponding to the questions.
    model: Instance of ResponseModel.

    Returns:
    MRR score.
    """
    reciprocal_ranks = []
    for question, correct_answer in zip(questions, ground_truths):
        # Get the ranked responses from the model
        results = model.get_answers([question])

        # Extract the answers and their similarity scores
        ranked_answers = [
            result[0][0] for result in results
        ]  # Extracting the answer text

        # Find the rank of the first correct answer
        try:
            rank = (
                ranked_answers.index(correct_answer) + 1
            )  # +1 because ranks start at 1
        except ValueError:
            rank = float("inf")  # Correct answer not found

        # Append reciprocal rank
        if rank != float("inf"):
            reciprocal_ranks.append(1 / rank)

    # Calculate MRR
    mrr_score = sum(reciprocal_ranks) / len(questions) if reciprocal_ranks else 0
    return mrr_score


def calculate_f1_score(questions, ground_truths, model, threshold=0.6):
    """
    Calculates the F1-score for the given questions and ground truths.

    Parameters:
    questions: List of questions.
    ground_truths: List of correct answers corresponding to the questions.
    model: Instance of ResponseModel.
    threshold: Similarity threshold to determine relevance.

    Returns:
    F1-score.
    """
    tp, fp, fn = 0, 0, 0

    for question, correct_answer in zip(questions, ground_truths):
        # Get the ranked responses from the model
        results = model.get_answers([question])

        # Extract the top answer and its similarity score
        top_answer, similarity_score = results[0]
        top_answer_text = top_answer[0]  # Extracting the answer text

        # Determine relevance based on similarity score
        if similarity_score >= threshold:
            if top_answer_text == correct_answer:
                tp += 1  # True positive
            else:
                fp += 1  # False positive
        else:
            fn += 1  # False negative (relevant answer not retrieved)

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate F1-score
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1_score


def calculate_f1_with_sklearn(y_true, y_pred):
    return f1_score(y_true, y_pred, average="binary", pos_label=1)


model_st = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def semantic_match(answer, correct_answer, threshold=0.8):
    sim = util.cos_sim(model_st.encode(answer), model_st.encode(correct_answer))
    return sim.item() >= threshold


def calculate_f1_semantic(
    questions, ground_truths, model, threshold=0.6, sim_threshold=0.8
):
    tp, fp, fn = 0, 0, 0
    for question, correct_answer in zip(questions, ground_truths):
        results = model.get_answers([question])
        top_answer, score = results[0]
        answer_text = top_answer[0]

        if score >= threshold:
            if semantic_match(answer_text, correct_answer, sim_threshold):
                tp += 1
            else:
                fp += 1
        else:
            fn += 1
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate F1-score
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    # Расчет F1 (как в оригинале)
    return f1_score
