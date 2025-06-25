import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
import tree.tree_builder as tree_builder
import models.sentence_transformer_response_model as model

# Загружаем дерево решений один раз при запуске бота
decision_tree = tree_builder.build_decision_tree("./KnowledgeBase")
response_model = model.ResponseModel(decision_tree)

# Токен вашего бота (вставьте свой)
TELEGRAM_BOT_TOKEN = "7738868585:AAET93TDRMjaLlJjVotIYYXCwjB-e61xKlo"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    print(f"Получено сообщение: {user_input}")

    # Получаем ответ и файлы
    answers = response_model.get_answers([user_input])
    answer_text, files = answers[0][0]  # Берём первый (и единственный) ответ
    similarity_score = answers[0][1]

    if similarity_score < response_model.similarity_threshold:
        await update.message.reply_text("Извините, я не нашёл подходящего ответа.")
        return

    # Отправляем текст ответа
    await update.message.reply_text(answer_text)

    # Если есть файлы — отправляем их
    if isinstance(files, list) and files and files[0] != "no_files":
        for file_path in files:
            try:
                if file_path.endswith(".docx"):
                    with open(file_path, "rb") as docx_file:
                        await update.message.reply_document(document=docx_file)
                elif file_path.endswith(".jpg"):
                    with open(file_path, "rb") as jpg_file:
                        await update.message.reply_photo(photo=jpg_file)
            except Exception as e:
                print(f"Ошибка при отправке файла {file_path}: {e}")
                await update.message.reply_text(
                    f"Не удалось отправить файл: {os.path.basename(file_path)}"
                )
    else:
        await update.message.reply_text("Нет прикреплённых файлов.")


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Регистрируем обработчик текстовых сообщений
    message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    app.add_handler(message_handler)

    print("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
