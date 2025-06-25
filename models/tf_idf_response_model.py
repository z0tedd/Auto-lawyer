import re
import os
import json
import pymorphy3
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download("stopwords")


class ResponseModel:
    """
    Класс ResponseModel используется для поиска наиболее подходящего ответа на основе
    предварительно построенного дерева решений, используя обработку текста, лемматизацию
    и метод векторизации TF-IDF с вычислением косинусного сходства.
    """

    def __init__(self, decision_tree, config_path="config.json", lemmatization=True):
        """
        Инициализирует модель ответов.

        Параметры:
        decision_tree: Дерево решений, содержащее ответы и описание условий для их выбора.
        lemmatization: Флаг, указывающий, нужно ли использовать лемматизацию.
        """
        self.decision_tree = decision_tree
        # Загружаем threshold из файла конфигурации
        # Проверка наличия файла конфигурации
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
                self.similarity_threshold = config.get(
                    "similarity_threshold", 0.25
                )  # по умолчанию 0.25
        else:
            # Если файл не найден
            print(f"Configuration file {config_path} not found. Using default value.")
            self.similarity_threshold = 0.25
        self.lemmatization = lemmatization
        self.stop_words = set(stopwords.words("russian"))
        self.morph = pymorphy3.MorphAnalyzer() if lemmatization else None
        self.flat_structure, self.answers = self.flatten_tree(decision_tree)
        self.flat_structure = [
            self.preprocess_text(item) for item in self.flat_structure
        ]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.flat_structure)

    def flatten_tree(self, node, path=""):
        """
        Рекурсивно преобразует дерево решений в плоскую структуру.

        Параметры:
        node: Текущий узел дерева (может быть словарем или кортежем).
        path: Путь к текущему узлу (для отслеживания иерархии).

        Возвращает:
        Два списка: плоскую структуру узлов + ответов и ответы.
        """
        if isinstance(node, dict):
            result = []
            answers = []
            for key, value in node.items():
                new_path = f"{path} {key}" if path else key
                flattened_result, flattened_answers = self.flatten_tree(value, new_path)
                result.extend(flattened_result)
                answers.extend(flattened_answers)
            return result, answers
        elif isinstance(node, tuple):  # Если узел - это ответ
            description, _ = node
            answer = node
            return [f"{path} {description}"], [answer]
        else:
            return [], []

    def preprocess_text(self, text):
        """
        Предобрабатывает текст: приводит к нижнему регистру, удаляет лишние пробелы,
        удаляет специальные символы (кроме чисел), фильтрует стоп-слова и выполняет лемматизацию.

        Параметры:
        text: Исходный текст для обработки.

        Возвращает:
        Обработанный текст.
        """
        # приведение к нижнему регистру
        text = text.lower()
        # удаление специальных символов, кроме цифр и пробелов
        text = re.sub(r"[^а-яА-ЯёЁ0-9\s]", "", text)
        # удаление лишних пробелов
        text = re.sub(r"\s+", " ", text).strip()
        # токенизация с использованием регулярных выражений
        words = re.findall(r"\w+", text)
        # фильтрация стоп-слов
        filtered_words = [word for word in words if word not in self.stop_words]
        # лемматизация
        if self.lemmatization:
            lem_words = [
                self.morph.parse(word)[0].normal_form for word in filtered_words
            ]
            return " ".join(lem_words)
        return " ".join(filtered_words)

    def get_answers(self, questions):
        """
        Получает ответы на заданные вопросы, используя косинусное сходство.

        Параметры:
        questions: Список вопросов, на которые нужно получить ответы.

        Возвращает:
        Словарь, содержащий соответствующий ответ и его схожесть для каждого вопроса.
        """
        results = []
        for question in questions:
            # Обработка и векторизация вопроса
            vector_question = self.vectorizer.transform(
                [self.preprocess_text(question)]
            )
            # Вычисление косинусного сходства
            similarity = cosine_similarity(vector_question, self.tfidf_matrix).flatten()
            # Индекс наибольшего сходства
            best_match_index = np.argmax(similarity)
            # Сохранение ответа
            results.append(
                (self.answers[best_match_index], similarity[best_match_index])
                if similarity[best_match_index] >= self.similarity_threshold
                else (
                    ("Не удалось найти подходящий ответ на ваш вопрос.", ["no_files"]),
                    similarity[best_match_index],
                )
            )
        return results
