import os
import json
from transformers import AutoTokenizer, AutoModel
import torch


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

        # Преобразование дерева решений в плоскую структуру
        self.flat_structure, self.answers = self.flatten_tree(decision_tree)

        # Инициализация модели HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence"
        )
        self.model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")

        # Генерация эмбеддингов для всех элементов плоской структуры
        self.embeddings = self.generate_embeddings(self.flat_structure)

    def mean_pooling(self, model_output, attention_mask):
        """
        Выполняет mean pooling для получения эмбеддингов предложений.

        Параметры:
        model_output: Выход модели (все токен-эмбеддинги).
        attention_mask: Маска внимания для корректного усреднения.

        Возвращает:
        Тензор с эмбеддингами предложений.
        """
        token_embeddings = model_output[
            0
        ]  # Первый элемент содержит все токен-эмбеддинги
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def generate_embeddings(self, texts):
        """
        Генерирует эмбеддинги для списка текстов.

        Параметры:
        texts: Список текстов для которых нужно сгенерировать эмбеддинги.

        Возвращает:
        Тензор с эмбеддингами текстов.
        """
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, max_length=24, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        return sentence_embeddings

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

    def cosine_similarity(self, vec1, vec2):
        """
        Вычисляет косинусное сходство между двумя векторами.

        Параметры:
        vec1: Первый вектор.
        vec2: Второй вектор.

        Возвращает:
        Значение косинусного сходства.
        """
        return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)

    def get_answers(self, questions):
        """
        Получает ответы на заданные вопросы, используя семантический поиск.

        Параметры:
        questions: Список вопросов, на которые нужно получить ответы.

        Возвращает:
        Словарь, содержащий соответствующий ответ и его схожесть для каждого вопроса.
        """
        results = []
        for question in questions:
            # Генерация эмбеддинга для вопроса
            question_embedding = self.generate_embeddings([question])[0]

            # Поиск наиболее похожего ответа
            best_match_index = -1
            best_similarity_score = -1
            for i, embedding in enumerate(self.embeddings):
                similarity_score = self.cosine_similarity(question_embedding, embedding)
                if similarity_score > best_similarity_score:
                    best_similarity_score = similarity_score
                    best_match_index = i

            # Сохранение ответа
            if best_similarity_score >= self.similarity_threshold:
                results.append(
                    (self.answers[best_match_index], best_similarity_score.item())
                )
            else:
                results.append(
                    (
                        (
                            "Не удалось найти подходящий ответ на ваш вопрос.",
                            ["no_files"],
                        ),
                        best_similarity_score.item(),
                    )
                )

        return results
