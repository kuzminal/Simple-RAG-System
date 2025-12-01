import json
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


class SearchSystem:
    def __init__(self, base_path='knowledge_base', embedding_model='all-MiniLM-L6-v2'):
        # Загружаем сохранённую базу знаний
        self.model = SentenceTransformer(embedding_model)
        # Загружаем фрагменты
        with open(f'{base_path}_chunks.json', 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        # Загружаем векторы
        self.vectors = np.load(f'{base_path}_vectors.npy')
        print(f"Загружена база знаний: {len(self.chunks)} фрагментов")

    def find_similar(self, query: str, top_k=3) -> List[Dict]:
        """Находим наиболее релевантные фрагменты"""
        # Векторизуем запрос
        query_vector = self.model.encode([query])
        # Вычисляем косинусное сходство
        similarities = np.dot(self.vectors, query_vector.T).flatten()
        base_norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        cosine_similarities = similarities / (base_norms * query_norm)
        # Находим индексы наиболее похожих фрагментов
        top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
        # Формируем результат
        results = []
        for idx in top_indices:
            result = self.chunks[idx].copy()
            result['similarity'] = float(cosine_similarities[idx])
            results.append(result)
        return results
