# Методы распознавания сообщений от ботов на основе машинного обучения

**Выпускная квалификационная работа бакалавра**  
Дзахмишев Камбулат Заурович  
РУДН, направление 02.03.01 «Математика и компьютерные науки», 2026 г.  
Руководитель: Молодченков Алексей Игоревич, к.т.н., доцент

---

## О работе

Сравнительное исследование методов машинного обучения для автоматического распознавания ботов в мессенджере Telegram. В качестве прокси-задачи используется бинарная классификация русскоязычных сообщений на классы «спам» и «не спам».

Реализованы и сопоставлены пять подходов к векторизации текста в сочетании с тремя классическими классификаторами, а также дообучение BERT. Итого — 15 конфигураций + дообученный BERT.

---

## Результаты

| Подход | Классификатор | F1-macro | ROC-AUC |
|---|---|---|---|
| BERT fine-tuned | rubert-tiny2 | **0,9813** | 0,9977 |
| Hybrid (TF-IDF + FastText + BERT) | LinearSVC | 0,9727 | 0,9970 |
| BERT-CLS | LinearSVC | 0,9588 | 0,9952 |
| TF-IDF | LinearSVC | 0,9586 | 0,9934 |
| BERT-mean | LinearSVC | 0,9546 | 0,9948 |
| FastText | RandomForest | 0,9342 | 0,9881 |
| FastText | LogisticRegression | 0,8846 | 0,9905 |

**Ключевые выводы:**
- Дообученный BERT показал наивысшее качество (F1 = 0,981), но требует GPU и в 35 раз медленнее при инференсе
- Гибридный подход — лучший без дообучения (F1 = 0,973), статистически значимо превосходит одиночные методы (тест МакНемара, p < 0,001)
- TF-IDF + LinearSVC и BERT-CLS + LinearSVC статистически неразличимы (p = 0,845) — BERT без дообучения не даёт преимущества на коротких текстах
- LLM в режиме zero-shot (Qwen3.5-Plus, DeepSeek, GigaChat) достигают F1 ≈ 87–89%, уступая обученным моделям на 9–11 п.п.

---

## Датасет

**ruSpamModels/russian-spam-detection** — русскоязычные сообщения из Telegram, размеченные как спам / не спам.

- Платформа: [Hugging Face](https://huggingface.co/datasets/ruSpamModels/russian-spam-detection)
- Объём: 4,5+ млн сообщений, для экспериментов использована стратифицированная подвыборка 300 000
- Соотношение классов: ~92% не спам / ~8% спам
- Разбиение: train/val/test = 70/15/15, random_state=42, стратификация по метке

---

## Структура репозитория

```
VKR_Dzakhmishev/
├── notebooks/
│   ├── 01_eda.ipynb               # Разведочный анализ данных
│   ├── 03_fasttext.ipynb          # FastText эмбеддинги + классификаторы
│   ├── 04_bert.ipynb              # BERT как экстрактор признаков
│   ├── 05_hybrid.ipynb            # Гибридный подход
│   ├── 06_analysis.ipynb          # Итоговый анализ и визуализации
│   └── 07_bert_finetuning.ipynb   # Дообучение BERT
├── figures/                       # Графики из экспериментов
├── results/                       # CSV с метриками всех экспериментов
└── README.md
```

> **Примечание:** тяжёлые файлы (данные `.parquet`, эмбеддинги `.npy`, обученные модели `.pkl`) не включены в репозиторий. Датасет доступен на Hugging Face по ссылке выше.

---

## Методы

**Векторизация текста:**
- TF-IDF с биграммами (`max_features=10000`, `ngram_range=(1,2)`, `sublinear_tf=True`)
- FastText — модель GeoWAC, 300-мерные векторы, mean pooling (библиотека `compress-fasttext`)
- BERT — модель `cointegrated/rubert-tiny2`, 29 МБ, два варианта агрегации: CLS-токен и mean pooling
- Гибрид — конкатенация TF-IDF (300 компонент после TruncatedSVD) + FastText (300) + BERT-CLS (312) = 912 измерений
- BERT fine-tuned — дообучение `rubert-tiny2` через `AutoModelForSequenceClassification`

**Классификаторы:** LogisticRegression, LinearSVC, RandomForestClassifier (все с `class_weight="balanced"`)

**Метрики:** F1-macro (основная), ROC-AUC, Accuracy, Precision, Recall

**Статистическое сравнение:** критерий МакНемара с поправкой Йетса

---

## Стек

```
Python 3.10+
scikit-learn
transformers
torch
compress-fasttext
pandas, numpy
matplotlib, seaborn
```

---

## Апробация и публикации

Результаты представлены на **XVI Всероссийской конференции ITTMM 2026** (РУДН, Москва, апрель 2026 г.).

Публикация: Dzakhmishev K. Z. Evaluation of Large Language Models for Russian-Language Spam Detection: A Comparative Study of Qwen, GigaChat, and DeepSeek // ITTMM 2026.

---

## Список литературы (основные источники)

1. Cresci et al. The paradigm-shift of social spambots // WWW 2017
2. Varol et al. Online Human-Bot Interactions // ICWSM 2017
3. Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers // NAACL 2019
4. Bojanowski et al. Enriching Word Vectors with Subword Information // TACL 2017
5. Reimers & Gurevych. Sentence-BERT // EMNLP 2019
6. Vaswani et al. Attention Is All You Need // NeurIPS 2017
7. Sanh et al. DistilBERT // NeurIPS EMC² Workshop 2019
