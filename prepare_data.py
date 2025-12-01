import numpy as np
from sentence_transformers import SentenceTransformer
import json
from typing import List

class DataPreparation:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        # Загружаем модель для создания векторных представлений
        self.model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.vectors = None


    def split_into_chunks(self, text: str, chunk_size=500, overlap=50) -> List[str]:
        """Разбиваем текст на логические фрагменты с перекрытием"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50: # Игнорируем слишком короткие фрагменты
                chunks.append(chunk.strip())
        return chunks


    def process_documents(self, documents: List[str]):
        """Обрабатываем список документов"""
        print("Разбиваем документы на фрагменты...")
        for i, document in enumerate(documents):
            doc_chunks = self.split_into_chunks(document)
            # Сохраняем фрагменты с метаинформацией
            for j, chunk in enumerate(doc_chunks):
                self.chunks.append({
                'text': chunk,
                'document_id': i,
                'chunk_id': j,
                    'length': len(chunk)
                })
                print(f"Создано {len(self.chunks)} фрагментов")
                # Создаём векторные представления
                print("Создаём векторные представления...")
                chunk_texts = [chunk['text'] for chunk in self.chunks]
                self.vectors = self.model.encode(chunk_texts)
                print(f"Создано {len(self.vectors)} векторов размерности {self.vectors.shape[1]}")

    def save_knowledge_base(self, base_path='knowledge_base'):
        """Сохраняем базу знаний в файлы"""

        # Сохраняем фрагменты
        with open(f'{base_path}_chunks.json', 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        # Сохраняем векторы
        np.save(f'{base_path}_vectors.npy', self.vectors)
        print(f"База знаний сохранена в файлы {base_path}_*")