import os
import time
import tracemalloc

# Import your modules (uncomment if this is a standalone test)
from tree.tree_builder import build_decision_tree
from models.sbert_ru_response_model import ResponseModel


def benchmark():
    print("Starting benchmark...\n")

    # --- 1. Decision Tree Building ---
    tracemalloc.start()
    start_time = time.time()

    decision_tree = build_decision_tree("./KnowledgeBase")

    tree_build_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"[Decision Tree] Built in {tree_build_time:.4f} seconds")
    print(f"               Memory used: {current / 10**6:.2f}MB")
    print(f"               Peak memory: {peak / 10**6:.2f}MB\n")

    # --- 2. Question Loading ---
    start_time = time.time()

    questions_pull = []
    with open("questions.txt", "r", encoding="utf-8") as file:
        content = file.readlines()
        for line in content:
            questions_pull.append(line.strip().split("|")[1])

    question_load_time = time.time() - start_time

    print(f"[Question Load] Completed in {question_load_time:.4f} seconds")
    print(f"                Loaded {len(questions_pull)} questions\n")

    # --- 3. Model Initialization ---
    tracemalloc.start()
    start_time = time.time()

    model = ResponseModel(decision_tree)

    model_init_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"[Model Init] Initialized in {model_init_time:.4f} seconds")
    print(f"             Memory used: {current / 10**6:.2f}MB")
    print(f"             Peak memory: {peak / 10**6:.2f}MB\n")

    # --- 4. Answer Generation ---
    tracemalloc.start()
    start_time = time.time()

    answers = model.get_answers(questions_pull)

    answer_gen_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"[Answer Gen] Generated in {answer_gen_time:.4f} seconds")
    print(f"             Memory used: {current / 10**6:.2f}MB")
    print(f"             Peak memory: {peak / 10**6:.2f}MB\n")

    print(f"Total number of answers generated: {len(answers)}")

    # --- Optional: Save results ---
    model_name = "Sentence transformers\n"
    with open("answers_test_model_name.txt", "w", encoding="utf-8") as file:
        file.write(model_name)
        count = 0
        for answer in answers:
            count += 1
            file.write(
                f"Test question {count}, similarity - {answer[1]:.4f}, len - {len(answer[0])}\n\n"
            )
            file.write(answer[0][0])
            file.write("\n\nEnd of test question\n")

    print("Benchmark completed and answers saved.")


if __name__ == "__main__":
    benchmark()
