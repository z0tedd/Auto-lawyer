import os
import models.bm25_response_model as model
import metrics.metrics as metrics  # Import the metrics module
import tree.tree_builder as tree_builder

# Build the decision tree
decision_tree = tree_builder.build_decision_tree("./KnowledgeBase")

# Load questions from the file
questions_pull = []
with open("metrics/questions.txt", "r", encoding="utf-8") as file:
    content = file.readlines()
    for i in content:
        questions_pull.append(i.strip().split("|")[1])

# Instantiate the ResponseModel
response_model = model.ResponseModel(decision_tree)

# Load ground truths (correct answers) from a file with '------' as a separator
ground_truths = []
with open("metrics/ground_truths.txt", "r", encoding="utf-8") as file:
    content = file.read()
    # Split the content by the delimiter '------'
    raw_answers = content.split("------")
    # Strip whitespace and filter out empty entries
    ground_truths = [answer.strip() for answer in raw_answers if answer.strip()]

# Ensure the lengths of questions_pull and ground_truths match
if len(ground_truths) < len(questions_pull):
    # Fill missing ground truths with empty strings
    # ground_truths.extend([""] * (len(questions_pull) - len(ground_truths)))
    questions_pull = questions_pull[: len(ground_truths)]
elif len(ground_truths) > len(questions_pull):
    raise ValueError(
        f"The number of ground truths ({len(ground_truths)}) exceeds the number of questions ({len(questions_pull)})."
    )
print(len(ground_truths), len(questions_pull))

# Calculate MRR
mrr_score = metrics.calculate_mrr(questions_pull, ground_truths, response_model)
print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")

# Calculate F1-score
f1_score = metrics.calculate_f1_score(questions_pull, ground_truths, response_model)
print(f"F1-Score: {f1_score}")

f1_score = metrics.calculate_f1_semantic(questions_pull, ground_truths, response_model)
print(f"F1-Score: {f1_score}")
# f1_score = metrics.calculate_f1_score(questions_pull, ground_truths, response_model)
# print(f"F1-Score: {f1_score}")
# Optionally, write the metrics to a file
with open("metrics_results.txt", "w", encoding="utf-8") as file:
    file.write(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}\n")
    file.write(f"F1-Score: {f1_score:}\n")


# import os
# import response_model as model
#
# # import response_model as model
# import tree_builder
#
# decision_tree = tree_builder.build_decision_tree("./KnowledgeBase")
# # Открываем файл для чтения
# questions_pull = []
# with open("questions.txt", "r", encoding="utf-8") as file:
#     content = file.readlines()
#     for i in content:
#         questions_pull.append(i.strip().split("|")[1])
#
# # # print(decision_tree)
# #
# model = model.ResponseModel(decision_tree)
# questions = ["кредитные каникулы"]
# answers = model.get_answers(questions_pull)
#
# model_name = "Sentence transformers\n"
# with open("answers_test_model_name.txt", "w", encoding="utf-8") as file:
#     file.write(model_name)
#     count = 0
#     for answer in answers:
#         count += 1
#
#         file.write(
#             f"Test question {count}, similiarity - {answer[1]}, len - {len(answer[0])}\n\n"
#         )
#         file.write(answer[0][0])
#         file.write("\n\nEnd of test question\n")
