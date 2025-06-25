import os
import json
from sentence_transformers import SentenceTransformer, util


class ResponseModel:
    """
    Класс ResponseModel используется для поиска наиболее подходящего ответа на основе
    предварительно построенного дерева решений, используя обработку текста и семантический поиск SBERT.
    """

    def __init__(self, decision_tree, config_path="config.json"):
        """
        Инициализирует модель ответов.

        Параметры:
        decision_tree: Дерево решений, содержащее ответы и описание условий для их выбора.
        """
        self.decision_tree = decision_tree
        # Загружаем threshold из файла конфигурации
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
                self.similarity_threshold = config.get(
                    "similarity_threshold", 0.60
                )  # по умолчанию 0.65
        else:
            # Если файл не найден
            print(f"Configuration file {config_path} not found. Using default value.")
            self.similarity_threshold = 0.60
        self.flat_structure, self.answers = self.flatten_tree(decision_tree)
        # Инициализация модели SBERT с поддержкой русского языка
        self.sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        # Генерация эмбеддингов для всех элементов плоской структуры
        self.embeddings = self.sbert_model.encode(
            self.flat_structure, convert_to_tensor=True
        )

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

    def get_answers(self, questions):
        """
        Получает ответы на заданные вопросы, используя семантический поиск SBERT.

        Параметры:
        questions: Список вопросов, на которые нужно получить ответы.

        Возвращает:
        Словарь, содержащий соответствующий ответ и его схожесть для каждого вопроса.
        """
        results = []
        for question in questions:
            # Генерация эмбеддинга для вопроса
            question_embedding = self.sbert_model.encode(
                question, convert_to_tensor=True
            )
            # Выполнение семантического поиска
            hits = util.semantic_search(question_embedding, self.embeddings, top_k=1)
            # Получение лучшего совпадения
            best_match = hits[0][0]  # Первый результат (наиболее похожий)
            best_match_index = best_match["corpus_id"]
            similarity_score = best_match["score"]

            # Сохранение ответа
            results.append(
                (self.answers[best_match_index], similarity_score)
                if similarity_score >= self.similarity_threshold
                else (
                    ("Не удалось найти подходящий ответ на ваш вопрос.", ["no_files"]),
                    similarity_score,
                )
            )
        return results
