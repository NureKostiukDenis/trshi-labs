# DontStopGreg

Гібридна RAG-система для пошуку інформації по ігровій збірці GregTech: New Horizons (GTNH). Поєднує точний пошук рецептів у PostgreSQL та текстовий пошук по статтях вікі й книзі квестів у Qdrant (ембединги `BAAI/bge-m3`), а відповіді генерує локальна модель `Qwen 2.5 7B`.

## Сценарії використання

- Пошук опису конкретного предмета або мультиблочної споруди.
- Пошук правила чи механіки, згаданої на вікі, у книзі квестів або в описі предмета.
- Визначення ланцюжка крафту предмета чи схеми виведення бджіл з урахуванням рівня напруги та побічних виходів.
- Порівняння альтернативних способів виробництва й оцінка можливих обмежень.

## Мінімальне встановлення стеку

### Arch Linux

```bash
sudo pacman -S python postgresql docker ollama
sudo systemctl enable --now postgresql docker ollama
```

Запуск Qdrant та завантаження моделі генерації:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
ollama pull qwen2.5:7b
```

Python-залежності у віртуальному середовищі:

```bash
python -m venv .venv
source .venv/bin/activate
pip install qdrant-client psycopg sentence-transformers
```

### Windows

```powershell
winget install Python.Python.3.13 PostgreSQL.PostgreSQL Docker.DockerDesktop Ollama.Ollama
```

Після запуску Docker Desktop та Ollama:

```powershell
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
ollama pull qwen2.5:7b
```

Python-залежності у віртуальному середовищі:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install qdrant-client psycopg sentence-transformers
```
