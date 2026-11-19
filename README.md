# DontStopGreging

Гібридна RAG-система для пошуку інформації по ігровій збірці **GregTech: New Horizons (GTNH)**. Поєднує точний пошук предметів і рецептів у **PostgreSQL** (`nesql`) та векторний пошук по статтях вікі й книзі квестів у **Qdrant** (ембединги `BAAI/bge-base-en-v1.5`), а генерацію відповідей планується виконувати локальною моделлю `Qwen 2.5 7B`.

## Сценарії використання

- Пошук опису конкретного предмета, квесту або мультиблочної споруди.
- Пошук правила чи механіки, згаданої на вікі, у книзі квестів або в описі предмета.
- Визначення ланцюжка крафту предмета чи схеми виведення бджіл з урахуванням рівня напруги та побічних виходів.
- Порівняння альтернативних способів виробництва й оцінка можливих обмежень.

## Підготовка та запуск інфраструктури

### 1. Запуск Qdrant

> **Важливо:** команду обов'язково запускати з кореня проєкту.

```bash
docker run -d --name gtnh-vectors -p 6333:6333 \
    -v $(pwd)/data/qdrant:/qdrant/storage \
    qdrant/qdrant
```

### 2. Запуск бази даних PostgreSQL

```bash
docker run --name gtnh-recepties \
      -e POSTGRES_DB=gtnh-recepties \
      -e POSTGRES_USER=postgres \
      -e POSTGRES_PASSWORD=gthn \
      -p 5432:5432 \
      -d postgres
```

### 3. Завантаження дампу рецептів (`nesql`)

В розробці... Посилання та готовий дамп буде окремо.

### 4. Python-середовище

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Основні параметри шляхів, БД, моделі ембедингів, Qdrant та пошуку налаштовуються у [config/pipeline.toml](config/pipeline.toml).

## Запуск

Усі команди запускаються з кореня проєкту в активованому `.venv`:

- **Завантаження, очистка, збагачення та чанкування:**
  ```bash
  python3 src/pipeline_main.py
  ```
- **Векторизація та індексація чанків у Qdrant:**
  ```bash
  python3 src/index_main.py
  ```
- **Сирий пошук по векторах (TUI):**
  ```bash
  python3 src/search.py
  ```
- **Оцінювання пошуку (Recall@5, Recall@10, MRR@10):**
  ```bash
  python3 tests/eval_retrieval.py
  ```

## Структура проєкту

- `config/pipeline.toml` — централізована конфігурація джерел, шляхів, БД, моделі ембедингів, Qdrant і параметрів пошуку.
- `src/config.py` — завантаження конфігурації та підключення до PostgreSQL/SQLite і Qdrant.
- `src/pipeline_main.py` — запуск пайплайна підготовки даних (`src/pipeline/`: очищення вікі й таблиць, очищення квестів, збагачення з `nesql`, стратегії чанкування та дата-класи).
- `src/index_main.py` — запуск векторизації та завантаження чанків у Qdrant (`src/index/`: кодування через `SentenceTransformer`, `QdrantVectorizer`).
- `src/search.py` — інтерактивна TUI-обгортка для пошуку по Qdrant.
- `tests/` — тестовий набір (`eval-set.jsonl`) і скрипт оцінювання (`eval_retrieval.py`).
- `report/` — звіти до лабораторних робіт (`lb1`, `lb2`, `lb3`) та `data-card.md`.
