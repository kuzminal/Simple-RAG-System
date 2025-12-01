from langchain_ollama import ChatOllama
from typing import List, Dict


class AnswerGenerator:
    def __init__(self):
        self.system_message = """Ты – помощник, который отвечает на вопросы, опираясь
        строго на предоставленный контекст.
        Правила:
        1. Используй только информацию из контекста
        2. Если ответа нет в контексте, честно скажи об этом
        3. Указывай степень уверенности в ответе
        4. Будь кратким и точным"""

    def create_prompt(self, query: str, chunks: List[Dict]) -> str:
        """Формируем промпт с контекстом"""
        context = ""
        for i, chunk in enumerate(chunks, 1):
            similarity_percent = int(chunk['similarity'] * 100)
            context += f"\n--- Фрагмент {i} (релевантность {similarity_percent}%) ---\n"
            context += chunk['text']
            context += "\n"
        prompt = f"""Контекст:
        {context}
        Вопрос: {query}
        Ответ:"""
        return prompt

    def get_answer(self, query: str, chunks: List[Dict]) -> str:
        """Генерируем ответ с помощью языковой модели"""
        prompt = self.create_prompt(query, chunks)
        client = ChatOllama(
            model="gemma3:12b",
            base_url="http://localhost:11434/",
            temperature=0.3,
            # max tokens
            num_predict=300,
            repeat_penalty=1.1
        )
        try:
            response = client.invoke(
                input=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content.strip()
        except Exception as error:
            return f"Ошибка при генерации ответа: {str(error)}"
