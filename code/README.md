# Crypto Sentiment Analysis — BTC / ETH / BNB

> **Тема ВКР:** «Исследование изменения стоимости криптовалют на основе сообщений участников рынка»
>
> **Студент:** Мугари Абдеррахим, группа НФИбд-01-22, РУДН, 2026
>
> **Науч. рук.:** доц., к.т.н. Молодченков А.И.

Исследование взаимосвязи **тональности сообщений Twitter** и **динамики стоимости BTC, ETH, BNB**
методами анализа тональности (VADER + FinBERT) и статистическими тестами временных рядов
(корреляция Пирсона, тест Грейнджера на исходных и дифференцированных рядах, тест ADF,
апостериорный анализ мощности, бинарная классификация направления доходности).

---

## Набор данных

**Источник:** Kaggle — «3 Million Tweets Cryptocurrencies BTC, ETH, BNB» (Mazzoli, 2022)  
**Сырых сообщений:** 1 708 862 (2 516 CSV-файлов)  
**Период:** 26 июля — 30 августа 2022 года (36 дней)  
**Язык:** английский (`lang == "en"`)

| Криптовалюта | Сырых | После фильтрации | Удалено, % |
|---|---|---|---|
| Bitcoin (BTC) | 737 089 | 321 627 | 56,4 % |
| Ethereum (ETH) | 739 618 | 306 738 | 58,5 % |
| Binance Coin (BNB) | 232 155 | 75 171 | 67,6 % |
| **Итого** | **1 708 862** | **703 536** | — |

Сырые данные **не включены в репозиторий**. Для воспроизведения эксперимента загрузите датасет с
[kaggle.com/datasets/ilariamazzoli/3-million-tweets-cryptocurrencies-btc-eth-bnb](https://www.kaggle.com/datasets/ilariamazzoli/3-million-tweets-cryptocurrencies-btc-eth-bnb)
и распакуйте в `data/raw/`.

---

## Структура репозитория

```
.
├── .github/workflows/ci.yml   # GitHub Actions: Python 3.10 / 3.11 / 3.12
├── data/
│   └── processed/             # Кэш дневных цен (BTC/ETH/BNB, коммитится)
├── src/
│   ├── data_loader.py         # Загрузка и валидация твитов
│   ├── preprocessor.py        # Очистка и токенизация текстов
│   ├── sentiment_analyzer.py  # Оценка тональности (VADER + FinBERT)
│   ├── price_fetcher.py       # Цены с Binance Vision / CoinGecko / fallback
│   ├── binance_loader.py      # Агрегация 1-минутных OHLCV Binance → дневные цены
│   ├── correlation_analyzer.py # Корреляция, Granger, ADF, дифференцированные ряды
│   ├── power_analysis.py      # Fisher z-трансформация, мощность, MDR
│   └── predictor.py           # Классификатор направления доходности (LR, GBM, Dummy)
├── tests/                     # 354 теста pytest
├── results/
│   ├── btc/                   # Таблицы и графики по BTC
│   ├── eth/                   # Таблицы и графики по ETH
│   ├── bnb/                   # Таблицы и графики по BNB
│   ├── cross_crypto/          # Сравнительный анализ трёх криптовалют
│   ├── prediction/            # Результаты baseline-классификатора
│   └── finbert_torchinfo.txt  # Архитектура FinBERT (torchinfo)
├── main.py                    # Сквозной конвейер (12 шагов)
├── convert_data.py            # Фильтрация сырых CSV → отфильтрованные наборы
├── requirements.txt
└── requirements-test.txt
```

---

## Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/iragoum/crypto-sentiment-analysis.git
cd crypto-sentiment-analysis

# 2. Виртуальное окружение (рекомендуется)
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# 3. PyTorch CPU-only (FinBERT работает без GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Все зависимости
pip install -r requirements.txt
```

> FinBERT (`ProsusAI/finbert`) загружается автоматически при первом запуске
> через HuggingFace Hub (~450 МБ).

---

## Запуск

```bash
# Полный конвейер по трём криптовалютам (VADER + FinBERT)
python main.py --crypto all

# Только VADER (быстро, ~5 мин)
python main.py --crypto all --skip-finbert

# Одна криптовалюта
python main.py --crypto btc
python main.py --crypto eth
python main.py --crypto bnb
```

### Шаги конвейера

| # | Шаг | Описание |
|---|-----|----------|
| 1 | Load | Загрузка отфильтрованных твитов из `data/raw/tweets_{crypto}.csv` |
| 2 | Preprocess | Очистка текста, два варианта (оригинал для VADER / чистый для FinBERT) |
| 3 | Sentiment VADER | Оценка тональности лексиконным методом |
| 4 | Sentiment FinBERT | Классификация трансформерной моделью (CPU) |
| 5 | torchinfo | Сводка архитектуры FinBERT |
| 6 | Prices | Загрузка дневных цен (Binance Vision / CoinGecko / fallback) |
| 7 | Aggregate | Агрегация тональности и объёма по дням |
| 8 | Correlation | Корреляция Пирсона/Спирмена, матрица, лаги 0–7 |
| 9 | Granger (original) | Тест причинности Грейнджера на исходных рядах |
| 10 | Granger (differenced) | Тест Грейнджера на дифференцированных рядах (после ADF) |
| 11 | Power analysis | Апостериорный анализ мощности (Fisher z, n=36) |
| 12 | Predictor | Baseline-классификатор (LR, GBM, Dummy), walk-forward CV, bootstrap CI |

---

## Ключевые результаты

### Корреляция тональности VADER и дневной доходности (lag 0)

| Криптовалюта | r Pearson | p-value | Значимо (α=0.05)? |
|---|---|---|---|
| BTC | 0.218 | 0.208 | Нет |
| ETH | 0.259 | 0.133 | Нет |
| BNB | 0.044 | 0.803 | Нет |

Мощность теста при n=36 и наблюдаемых r: BTC — 24,7 %, ETH — 33,1 %, BNB — 5,7 %.  
Минимально детектируемая корреляция при мощности 80 %: |r| ≈ 0,45.

### Тест Грейнджера (min p по лагам 1–4, дифференцированные ряды)

| Криптовалюта | min p (исх.) | min p (дифф.) |
|---|---|---|
| BTC | 0.610 | 0.480 |
| ETH | 0.321 | 0.233 |
| BNB | 0.211 | 0.211 |

Ни одно значение не достигает α = 0.05.

### VADER vs FinBERT (согласие меток)

| Криптовалюта | VADER pos % | FinBERT neu % | Согласие % |
|---|---|---|---|
| BTC | 44.8 % | 84.5 % | 42.0 % |
| ETH | 47.3 % | 87.9 % | 40.7 % |
| BNB | 57.2 % | 85.2 % | 40.0 % |

FinBERT систематически классифицирует неформальные криптовалютные тексты как нейтральные
(доменное смещение: модель обучена на формальных финансовых текстах).

### Baseline-классификатор направления доходности (панельный режим)

| Модель | Accuracy | ROC-AUC |
|---|---|---|
| DummyClassifier (most_frequent) | **0.563** | 0.500 |
| LogisticRegression | 0.538 | 0.362 |
| GradientBoosting | 0.425 | 0.400 |

Ни один метод машинного обучения не превзошёл наивный baseline.

---

## Тестирование

```bash
# Все тесты
pytest tests/ -v

# С покрытием
pytest tests/ -v --cov=src

# Без FinBERT и полного датасета (быстро)
pytest tests/ -v -m "not slow"
```

**354 теста** покрывают: загрузку данных, предобработку, VADER, FinBERT (mock),
загрузку цен (3 криптовалюты + fallback), корреляционный анализ, Granger (исходный
и дифференцированный), ADF, power analysis, predictor (look-ahead bias, walk-forward,
bootstrap CI, детерминизм).

---

## CI/CD

GitHub Actions запускает полный тест-сьют на push/PR в `main`:
- Python 3.10, 3.11, 3.12
- Flake8 lint

---

## Архитектура FinBERT

`ProsusAI/finbert` — BERT-base (12 слоёв, 109,5 млн параметров) дообученный на финансовых текстах.

```
BertForSequenceClassification
├─ BertModel
│   ├─ BertEmbeddings     — 23,835,648 параметров
│   ├─ BertEncoder (×12)  — 85,054,464 параметров
│   └─ BertPooler         —    590,592 параметров
├─ Dropout
└─ Linear (classifier)    —      2,307 параметров
Total: 109,484,547 параметров | 437.94 MB
```

Полный вывод `torchinfo`: `results/finbert_torchinfo.txt`

---

## Методология

- **VADER** получает *оригинальный* текст (регистр, пунктуация и эмодзи — тональные сигналы)
- **FinBERT** получает *очищенный* текст (URL и упоминания — шум для трансформера)
- Тест Грейнджера проводится **дважды**: на исходных рядах и на первых разностях (после проверки ADF)
- Walk-forward CV гарантирует отсутствие утечки будущего в признаки классификатора
- Все источники случайности зафиксированы (`numpy.random.seed`, `torch.manual_seed`, `PYTHONHASHSEED=0`)

---

## Лицензия

MIT
