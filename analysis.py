import os
import re
from collections import Counter

import pandas as pd
import pymorphy3
import nltk
import spacy

from nltk.corpus import stopwords

# ---------------------------------
# Загрузка stopwords
# ---------------------------------
nltk.download('stopwords')

# ---------------------------------
# Загрузка моделей
# ---------------------------------
morph = pymorphy3.MorphAnalyzer()
nlp = spacy.load("en_core_web_sm")

# ---------------------------------
# Стоп-слова
# ---------------------------------
ru_stopwords = set(stopwords.words("russian"))
en_stopwords = set(stopwords.words("english"))

# Дополнительные стоп-слова RU
extra_ru = {
    "это", "который", "очень", "также",
    "ещё", "весь", "свой", "наш",
    "тысяча", "мочь", "стать"
}

# Дополнительные стоп-слова EN
extra_en = {
    "would", "could", "also",
    "one", "two", "first",
    "say", "get", "make"
}

ru_stopwords.update(extra_ru)
en_stopwords.update(extra_en)

# ---------------------------------
# Ручная нормализация RU
# ---------------------------------
ru_normalization = {

    "российский": "россия",
    "россиянин": "россия",
    "русский": "россия",

    "американский": "сша",
    "американец": "сша",

    "британский": "британия",

    "французский": "франция",

    "спортсмен": "спорт",
    "спортсменка": "спорт",

    "игрок": "игра"
}

# ---------------------------------
# Ручная нормализация EN
# ---------------------------------
en_normalization = {

    "russian": "russia",
    "russians": "russia",

    "british": "britain",

    "american": "usa",
    "americans": "usa",

    "french": "france",

    "players": "player",
    "athletes": "athlete",

    "games": "game",
    "matches": "match"
}

# ---------------------------------
# Обработка русского корпуса
# ---------------------------------
def process_russian(folder_path):

    all_tokens_before = []
    filtered_lemmas = []

    print("\n=== ОБРАБОТКА РУССКОГО КОРПУСА ===\n")

    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):

            print("Обрабатывается:", filename)

            filepath = os.path.join(folder_path, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().lower()

            # Извлечение слов
            tokens = re.findall(r"\b[а-яё]+\b", text)

            # До фильтрации
            all_tokens_before.extend(tokens)

            # Лемматизация
            for token in tokens:

                lemma = morph.parse(token)[0].normal_form

                # Нормализация
                if lemma in ru_normalization:
                    lemma = ru_normalization[lemma]

                # Фильтрация
                if lemma not in ru_stopwords and len(lemma) > 2:
                    filtered_lemmas.append(lemma)

    return build_statistics(
        all_tokens_before,
        filtered_lemmas
    )

# ---------------------------------
# Обработка английского корпуса
# ---------------------------------
def process_english(folder_path):

    all_tokens_before = []
    filtered_lemmas = []

    print("\n=== ОБРАБОТКА АНГЛИЙСКОГО КОРПУСА ===\n")

    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):

            print("Обрабатывается:", filename)

            filepath = os.path.join(folder_path, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().lower()

            doc = nlp(text)

            # Слова до фильтрации
            for token in doc:

                if token.is_alpha:
                    all_tokens_before.append(token.text)

            # Лемматизация
            for token in doc:

                if token.is_alpha:

                    lemma = token.lemma_.lower()

                    # Нормализация
                    if lemma in en_normalization:
                        lemma = en_normalization[lemma]

                    # Фильтрация
                    if lemma not in en_stopwords and len(lemma) > 2:
                        filtered_lemmas.append(lemma)

    return build_statistics(
        all_tokens_before,
        filtered_lemmas
    )

# ---------------------------------
# Подсчёт статистики
# ---------------------------------
def build_statistics(all_tokens_before, filtered_lemmas):

    total_words_before = len(all_tokens_before)

    unique_lemmas_before = len(set(all_tokens_before))

    functional_words_count = (
        total_words_before - len(filtered_lemmas)
    )

    functional_words_percent = round(
        functional_words_count / total_words_before * 100,
        2
    )

    total_after_filtering = len(filtered_lemmas)

    freq_counter = Counter(filtered_lemmas)

    rows = []

    # ТОП-100
    for lemma, abs_freq in freq_counter.most_common(100):

        rel_freq = round(
            abs_freq / total_after_filtering * 1000,
            2
        )

        rows.append([
            lemma,
            abs_freq,
            rel_freq
        ])

    freq_df = pd.DataFrame(
        rows,
        columns=[
            "Lemma",
            "Absolute Frequency",
            "Relative Frequency per 1000 words"
        ]
    )

    stats_df = pd.DataFrame({

        "Metric": [

            "Total words before filtering",
            "Unique lemmas before filtering",
            "Functional words count",
            "Functional words percentage",
            "Words after filtering"
        ],

        "Value": [

            total_words_before,
            unique_lemmas_before,
            functional_words_count,
            functional_words_percent,
            total_after_filtering
        ]
    })

    return freq_df, stats_df

# ---------------------------------
# Анализ RU
# ---------------------------------
ru_freq, ru_stats = process_russian("ru_articles")

# ---------------------------------
# Анализ EN
# ---------------------------------
en_freq, en_stats = process_english("en_articles")

# ---------------------------------
# Сохранение в Excel
# ---------------------------------
output_file = "corpus_analysis.xlsx"

with pd.ExcelWriter(output_file) as writer:

    ru_freq.to_excel(
        writer,
        sheet_name="RU Top 100",
        index=False
    )

    en_freq.to_excel(
        writer,
        sheet_name="EN Top 100",
        index=False
    )

    ru_stats.to_excel(
        writer,
        sheet_name="RU Statistics",
        index=False
    )

    en_stats.to_excel(
        writer,
        sheet_name="EN Statistics",
        index=False
    )

print("\n===================================")
print("АНАЛИЗ ЗАВЕРШЁН")
print("Файл сохранён:", output_file)
print("===================================\n")