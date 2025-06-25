import os
import json
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

nltk.download("punkt")


class ResponseModel:
    def __init__(self, decision_tree, config_path="config.json"):
        self.decision_tree = decision_tree
        # Загрузка конфигурации
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
                self.similarity_threshold = config.get("similarity_threshold", 0.60)
        else:
            print(f"Configuration file {config_path} not found. Using default value.")
            self.similarity_threshold = 10

        # Инициализация стеммера для русского языка
        self.stemmer = SnowballStemmer("russian")

        # Подготовка данных
        self.flat_structure, self.answers = self.flatten_tree(decision_tree)
        self.tokenized_corpus = [
            self.preprocess_text(text) for text in self.flat_structure
        ]
        self.bm25_model = BM25Okapi(self.tokenized_corpus)

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower(), language="russian")
        stemmed_tokens = [
            self.stemmer.stem(token) for token in tokens if token.isalnum()
        ]
        return stemmed_tokens

    # Остальные методы остаются без изменений
    def flatten_tree(self, node, path=""):
        if isinstance(node, dict):
            result = []
            answers = []
            for key, value in node.items():
                new_path = f"{path} {key}" if path else key
                flattened_result, flattened_answers = self.flatten_tree(value, new_path)
                result.extend(flattened_result)
                answers.extend(flattened_answers)
            return result, answers
        elif isinstance(node, tuple):
            description, _ = node
            answer = node
            return [f"{path} {description}"], [answer]
        else:
            return [], []

    def get_answers(self, questions):
        results = []
        for question in questions:
            tokenized_question = self.preprocess_text(question)
            scores = self.bm25_model.get_scores(tokenized_question)
            best_match_index = scores.argmax()
            similarity_score = scores[best_match_index]

            if similarity_score >= self.similarity_threshold:
                results.append((self.answers[best_match_index], similarity_score))
            else:
                results.append(
                    (
                        (
                            "Не удалось найти подходящий ответ на ваш вопрос.",
                            ["no_files"],
                        ),
                        similarity_score,
                    )
                )
        return results
