# Мультимодальное определение контрафактных товаров Ozon

Репозиторий содержит код дипломной работы по задаче бинарной классификации товарных карточек Ozon: определить, относится ли товар к контрафактным/проблемным по целевой метке `resolution`.

В работе сравниваются модели, использующие разные источники данных:

- табличные признаки товара и продавца;
- текстовые поля карточки товара;
- изображения товаров;
- мультимодальные комбинации этих признаков.

## Структура проекта

- `src/ozon_counterfeit/` - основной код для загрузки данных, подготовки признаков, split, моделей и обучения.
- `scripts/` - скрипты для подготовки manifest, проверок данных, обучения, инференса и построения графиков.
- `*.ipynb` - ноутбуки с экспериментами и анализом.
- `docs/` - краткое описание пайплайна проекта.
- `artifacts/` - результаты запусков: manifest, метрики, графики, предсказания и промежуточные файлы.

## Данные

Ожидаемая структура данных:

```text
data/
  ml_ozon_сounterfeit_data/
    ml_ozon_сounterfeit_train.csv
    ml_ozon_сounterfeit_test.csv
  ml_ozon_сounterfeit_train_images/
    *.png
  ml_ozon_сounterfeit_test_images/
    *.png
```

Изображения связываются с таблицей по `ItemID`.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Основной запуск

Построить manifest и split:

```bash
python3 scripts/build_manifest.py
```

Проверить данные:

```bash
python3 scripts/run_data_checks.py
```

Запустить обучение базовых экспериментов:

```bash
python3 scripts/train_experiment.py --experiment e0
python3 scripts/train_experiment.py --experiment e1
python3 scripts/train_experiment.py --experiment e2
```

Собрать таблицу метрик:

```bash
python3 scripts/collect_results.py
```

Построить PR-AUC графики:

```bash
python3 scripts/plot_pr_auc_comparisons.py
```

## Результаты

Краткие результаты на внутреннем test split:

| Модель | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Tabular only | 0.225 | 0.323 | 0.311 | 0.317 |
| Text only | 0.509 | 0.449 | 0.716 | 0.552 |
| Tabular + text | 0.594 | 0.572 | 0.616 | 0.593 |
| All modalities, early fusion | 0.607 | 0.614 | 0.541 | 0.575 |
| All modalities, gated fusion | 0.663 | 0.645 | 0.563 | 0.601 |
| Final late fusion | 0.630 | 0.622 | 0.577 | 0.599 |

Финальная late fusion модель выбиралась по validation PR-AUC и достигла `0.808` на validation и `0.630` на test.

Основные артефакты сохраняются в `artifacts/`:

- `artifacts/experiments/` - обученные эксперименты и метрики;
- `artifacts/reports/` - итоговые таблицы и графики;
- `artifacts/predictions/` - предсказания для внешнего test.

Ключевая идея проекта - проверить, насколько объединение табличных, текстовых и визуальных признаков улучшает качество определения контрафактных товаров по сравнению с отдельными модальностями.
