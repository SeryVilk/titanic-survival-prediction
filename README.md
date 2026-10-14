# Titanic Survival Prediction

Пет-проект по решению начальной задачи Kaggle — предсказание выживаемости пассажиров Титаника. Реализован полный ML-пайплайн: feature engineering, обработка пропусков, one-hot encoding, подбор гиперпараметров и усреднение предсказаний по фолдам кросс-валидации.

## Стек

- Python 3.x
- pandas, numpy
- scikit-learn
- Jupyter Notebook

## Структура проекта

```
.
├── notebooks/
│   └── titanic_eda_modeling.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Как запустить

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

3. Скачайте данные с Kaggle:
   - Перейдите на [Kaggle Titanic](https://www.kaggle.com/c/titanic/data)
   - Скачайте `train.csv` и `test.csv`
   - Положите оба файла в папку `notebooks/` рядом с ноутбуком

4. Откройте ноутбук и запустите все ячейки:

   ```bash
   jupyter notebook notebooks/titanic_eda_modeling.ipynb
   ```

   После выполнения в папке `notebooks/` появится `submission.csv` — файл для отправки на Kaggle.

## Что реализовано

- Разведочный анализ и обработка пропусков (`Age`, `Fare`, `Embarked`, `Deck`)
- Feature engineering:
  - `Title` из имени пассажира
  - `FamilySize` и `IsAlone`
  - `Deck` из номера каюты
  - `TicketPrefix` из номера билета
- Биннинг числовых признаков (`AgeBin`, `FareBin`)
- One-hot encoding категориальных признаков
- Взаимодействия признаков (`Sex_Pclass`, `FamSize_Pclass`, `Age_Pclass`)
- Подбор гиперпараметров Gradient Boosting через `GridSearchCV` (5-fold `StratifiedKFold`)
- Усреднение предсказаний по фолдам + подбор оптимального порога классификации

## Результаты

| Метрика            | Значение |
|--------------------|--|
| ROC-AUC (CV)       | 	0.88879 |
| ROC-AUC (OOF)      | 0.88300 |
| Accuracy (OOF)     | 0.84175 |
| Оптимальный порог  |	0.53 |
| Public LB (Kaggle) | 0.78947 |

## Возможные улучшения

- Ансамбль моделей (XGBoost, LightGBM, CatBoost). Планируется при наличии времени.
- Более глубокий feature engineering (взаимодействия категорий, группировка по билетам)