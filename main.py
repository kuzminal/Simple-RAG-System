from typing import Dict

from answer import AnswerGenerator
from prepare_data import DataPreparation
from search import SearchSystem


class RAGSystem:
    def __init__(self, api_key: str, base_path='knowledge_base'):
        self.search = SearchSystem(base_path)
        self.generator = AnswerGenerator(api_key)

    def answer(self, question: str, num_chunks=3) -> Dict:
        """Основная функция: поиск + генерация ответа"""
        print(f"Обрабатываем вопрос: {question}")
        # Находим релевантные фрагменты
        chunks = self.search.find_similar(question, num_chunks)
        print("\nНайденные фрагменты:")
        for i, chunk in enumerate(chunks, 1):
            similarity = int(chunk['similarity'] * 100)
            preview = chunk['text'][:100] + "..."
            print(f"{i}. Сходство {similarity}%: {preview}")
        # Генерируем ответ
        answer = self.generator.get_answer(question, chunks)
        return {
            'question': question,
            'answer': answer,
            'sources': chunks,
            'num_sources': len(chunks)
        }


# Пример использования
if __name__ == "__main__":
    # Подготавливаем тестовые документы
    documents = [
        "Python – высокоуровневый язык программирования общего назначения. Создан в 1991 году Гвидо ван Россумом в Центре математики и информатики в Нидерландах.",
        "RAG, или генерация с дополненной выборкой, – это архитектурный подход в обработке естественного языка, который объединяет поиск информации с генерацией текста.",
        "Векторные базы данных специально разработаны для хранения и поиска высокомерных векторов.Они используют специальные алгоритмы индексирования для быстрого поиска похожих векторов."
    ]

    # Создаём и сохраняем базу знаний
    data_prep = DataPreparation()
    data_prep.process_documents(documents)
    data_prep.save_knowledge_base()

    # Создаём RAG-систему
    rag = RAGSystem(api_key="ваш-openai-ключ")

    # Задаём вопросы
    questions = [
        "Кто создал Python?",
        "Что такое RAG?",
        "Как работают векторные базы данных?"
    ]
    for question in questions:
        result = rag.answer(question)
        print(f"\nВопрос: {result['question']}")
        print(f"Ответ: {result['answer']}")
        print("-" * 60)
