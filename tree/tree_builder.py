import os


def build_decision_tree(directory_path):
    """
    Строит дерево решений на основе структуры директорий.

    Метод проходит по заданному каталогу, собирая информацию о текстовых файлах
    (.txt) и подкаталогах. Если текстовые файлы найдены, возвращает содержимое
    первого текстового файла и список файлов формата .docx в текущем каталоге.
    Если текстовые файлы не найдены, но есть подкаталоги, рекурсивно строит
    поддерево для каждого подкаталога.

    Параметры:
    directory_path (str): Путь к директории, которую необходимо обработать.

    Возвращает:
    dict: Словарь, представляющий дерево решений, где ключи - это названия
    подкаталогов, а значения - соответствующие поддеревья. Если найдены текстовые
    файлы, возвращает кортеж (содержимое первого текстового файла, список файлов
    .docx и .jpg [или метку 'no_files' в случае их отсутствия]).
    """
    decision_tree = {}
    has_txt_files = False
    has_subdirectories = False
    files = []
    entries = sorted(os.scandir(directory_path), key=lambda item: item.name.upper())
    for entry in entries:
        if entry.is_dir():
            has_subdirectories = True
            subtree = build_decision_tree(entry.path)
            if subtree:
                decision_tree[entry.name] = subtree
        elif entry.is_file() and entry.name.endswith(".txt"):
            has_txt_files = True
            try:
                with open(entry.path, "r", encoding="utf-8") as file:
                    response = file.read().strip()
                    files = [
                        os.path.join(directory_path, f)
                        for f in os.listdir(directory_path)
                        if f.endswith((".docx", ".jpg"))
                    ]
                    if not files:
                        files = ["no_files"]
                    return response, files
            except UnicodeDecodeError as e:
                print(f"UnicodeDecodeError in file {entry.path}: {e}")
    if not has_txt_files and not has_subdirectories:
        print(f"Empty directory found: {directory_path}")
    return decision_tree
