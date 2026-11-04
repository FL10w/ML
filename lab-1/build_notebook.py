"""Build the standalone, Russian-language submission notebook from local sources."""

from pathlib import Path
import re

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
cells = []


def markdown(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


def embed(filename):
    source = (ROOT / filename).read_text(encoding="utf-8")
    # The notebook contains the definitions itself, so this local import is unnecessary.
    source = re.sub(r"^from linear_regression import .*\n", "", source, flags=re.M)
    for part in re.split(r"^# %%[^\n]*\n", source, flags=re.M):
        if part.strip():
            code(part)


markdown(r"""
# Лабораторная работа № 1. Линейная регрессия

**Задача:** предсказать `RiskScore` по `train.csv`, подготовить прогнозы для
`test.csv`. Основная метрика — **MSE**, требование соревнования — **MSE < 25**.
Реализованы обязательные задания из `cml_lab_1.pdf`. Дополнительные задания
с регуляризацией не выполняются. Другие модели не используются.

Ноутбук самостоятельный: все определения находятся в ячейках; рядом нужны
`train.csv`, `test.csv`, `ex.csv`. Запускать сверху вниз. В VS Code выбрать
Python из `.venv` этой папки; установка зависимостей описана в `README.md`.
`ex.csv` используется только для проверки формата отправки, не как правильные ответы.
""")

code(r"""
from pathlib import Path
import json
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LinearRegression as SklearnLinearRegression, SGDRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.preprocessing import StandardScaler, MinMaxScaler as SklearnMinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_predict, LeaveOneOut

blas_limits = threadpool_limits(limits=1)
SEED = 42
pd.set_option("display.max_columns", 40)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 12})

candidates = [Path.cwd(), Path.cwd() / "ML" / "lab-1",
              Path.cwd() / "ml" / "lab-1", Path.cwd() / "lab-1"]
DATA_DIR = next((p for p in candidates if (p / "train.csv").exists()
                 and (p / "test.csv").exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError("Откройте ноутбук из папки lab-1 или корня проекта.")
OUTPUT_DIR = DATA_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

def show_figure(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "ex.csv")
display(pd.DataFrame({"файл": ["train.csv", "test.csv", "ex.csv"],
                      "строк": [len(train_raw), len(test_raw), len(sample_submission)],
                      "столбцов": [train_raw.shape[1], test_raw.shape[1], sample_submission.shape[1]]}))
display(train_raw.head())
""")

markdown(r"""
## 1. Собственные алгоритмы

Определим метрики, обе нормализации и линейную регрессию. Все вычисления
обучения выполняются средствами NumPy. sklearn далее используется для сравнения.

\[
\mathrm{MSE}=\frac1n\sum_i(y_i-\hat y_i)^2,\qquad
\mathrm{MAE}=\frac1n\sum_i|y_i-\hat y_i|,
\]
\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2},\qquad
\mathrm{MAPE}=\frac1n\sum_i\frac{|y_i-\hat y_i|}{\max(|y_i|,\varepsilon)}.
\]

MAPE возвращается как доля; в таблицах дополнительно показываем проценты.
При нулевых целях знаменатель заменяется машинным epsilon, как в sklearn.
Для постоянной цели R² равен 1 при точном прогнозе и 0 иначе; для одного
наблюдения он не определён.

Нормализации: $z=(x-\mu_{train})/\sigma_{train}$ и
$x'=(x-\min_{train})/(\max_{train}-\min_{train})$.
Константные столбцы обрабатываются без деления на ноль. Новые данные могут
выходить за диапазон [0, 1] после min-max — это корректно.

Модель: $\hat y=Xw+b$. Аналитическое решение $w=X_c^+y_c$ вычисляем
через SVD; $b=\bar y-\bar Xw$. Это устойчивый вариант аналитической формулы,
работающий и при зависимых столбцах, без регуляризации.
Для GD используем $\nabla L=2X_c^T(X_cw-y_c)/n$.
SGD обновляет веса по одному случайно выбранному наблюдению, проходя
перемешанную выборку в каждой эпохе. Свободный коэффициент во всех трёх
методах исключается аналитически через центрирование.
""")

embed("linear_regression.py")

markdown("""
### Численная проверка метрик и нормализаций

Проверяем совпадение со sklearn, в том числе нулевую цель, постоянную цель
и константный признак. Это отдельные проверки формул, а не оценка модели.
""")

code(r"""
metric_functions = {"MSE": (mse, mean_squared_error),
                    "MAE": (mae, mean_absolute_error),
                    "R2": (r2, r2_score),
                    "MAPE": (mape, mean_absolute_percentage_error)}
metric_checks = []
for case, actual, predicted in [
    ("обычные значения", [2., 4., 8., 10.], [2.5, 3., 8.2, 9.]),
    ("нулевая цель", [0., 1., 3.], [0.1, 1.2, 2.8]),
    ("постоянная цель", [5., 5., 5.], [5., 4., 6.]),
    ("точный прогноз константы", [5., 5., 5.], [5., 5., 5.]),
]:
    for name, (own, reference) in metric_functions.items():
        ours, theirs = own(actual, predicted), reference(actual, predicted)
        np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)
        metric_checks.append({"случай": case, "метрика": name,
                              "своя": ours, "sklearn": theirs})
display(pd.DataFrame(metric_checks))

scaler_train = np.array([[1., 7., -2.], [3., 7., 0.], [8., 7., 4.]])
scaler_unseen = np.array([[12., 7., -5.], [2., 9., 1.]])
for own_cls, sklearn_cls in [(ZScoreScaler, StandardScaler),
                              (MinMaxScaler, SklearnMinMaxScaler)]:
    own, reference = own_cls().fit(scaler_train), sklearn_cls().fit(scaler_train)
    np.testing.assert_allclose(own.transform(scaler_train), reference.transform(scaler_train))
    np.testing.assert_allclose(own.transform(scaler_unseen), reference.transform(scaler_unseen))
    np.testing.assert_allclose(own.inverse_transform(own.transform(scaler_unseen)), scaler_unseen)
print("Метрики и обе нормализации совпали со sklearn.")
""")

markdown(r"""
## 2. Очистка и подготовка данных

Пустые строки не содержат наблюдений. Повторные строки удаляем **до** разбиения,
чтобы одна запись не оказалась одновременно в обучении и проверке.
Пропущенную цель не заполняем. Значения цели `-9999999` и `10000000` рассматриваем
как повреждённые записи: они изолированы на порядки от остального распределения.
Обычные большие значения денежных признаков автоматически не удаляем.

В отдельной группе строк одновременно отсутствуют девять полей. Такие строки
с повреждёнными входами исключаем; в `test.csv` этого блока пропусков нет.
Локальные оценки ниже относятся к оставшимся полным наблюдениям. Для будущих
неполных входов подготовка всё равно поддерживает заполнение медианами, но
качество на таких наблюдениях здесь не подтверждено.

Дата заявки и `ID` не используются как признаки. Категории кодируем one-hot
с удалением опорного уровня. Медианы, набор категорий, параметры преобразований
и нормализации определяются только на обучающей части каждого разбиения.
""")

embed("data_preparation.py")

code(r"""
clean, cleaning_audit = clean_training_data(train_raw)
display(cleaning_audit)
raw_unique = train_raw.dropna(how="all").drop_duplicates()
display(raw_unique.isna().sum().loc[lambda counts: counts > 0].to_frame("пропуски до очистки"))
display(pd.DataFrame({"тип": clean.dtypes.astype(str),
                      "пропусков": clean.isna().sum(),
                      "уникальных": clean.nunique()}))
print("Оставшиеся значения RiskScore:", clean.RiskScore.min(), "…", clean.RiskScore.max())
assert "RiskScore" not in test_raw.columns
assert test_raw.ID.is_unique
assert list(sample_submission.columns) == ["ID", "RiskScore"]
""")

markdown("""
## 3. Разведочный анализ (EDA)

Матрица корреляций содержит числовые признаки. Пропуски здесь учитываются
попарно средствами pandas; статистики EDA не используются для заполнения
проверочной выборки. На графиках показана очищенная выборка.
""")

code(r"""
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
sns.histplot(clean.RiskScore, bins=40, kde=True, ax=axes[0, 0], color="#246b8e")
axes[0, 0].set_title("Распределение RiskScore после очистки")
plot_sample = clean.sample(n=min(2500, len(clean)), random_state=SEED)
sns.scatterplot(data=plot_sample, x="CreditScore", y="RiskScore",
                alpha=.25, s=15, ax=axes[0, 1])
axes[0, 1].set_title("Кредитный рейтинг и риск")
sns.scatterplot(data=plot_sample, x="AnnualIncome", y="RiskScore",
                alpha=.25, s=15, ax=axes[1, 0])
axes[1, 0].set_xscale("log")
axes[1, 0].set_title("Доход и риск (логарифмическая ось дохода)")
sns.scatterplot(data=plot_sample, x="CreditScore", y="InterestRate",
                alpha=.25, s=15, ax=axes[1, 1])
axes[1, 1].set_title("Зависимость двух признаков: рейтинг и ставка")
show_figure("eda_distributions.png")

correlations = clean.select_dtypes(include="number").corr()
plt.figure(figsize=(15, 12))
sns.heatmap(correlations, cmap="vlag", center=0, vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": .75})
plt.title("Корреляционная матрица числовых признаков и цели")
show_figure("correlation_matrix.png")
target_correlations = correlations.RiskScore.drop("RiskScore")
display(target_correlations.reindex(target_correlations.abs().sort_values(ascending=False).index)
        .head(12).to_frame("корреляция с RiskScore"))

missing_counts = raw_unique.isna().sum().loc[lambda counts: counts > 0].sort_values()
if not missing_counts.empty:
    missing_counts.plot.barh(figsize=(9, 4), color="#d28c45")
    plt.xlabel("Количество пропусков")
    plt.title("Пропуски до исключения повреждённых строк")
    show_figure("missing_values.png")
strongest = target_correlations.abs().idxmax()
display(Markdown(
    f"**Выводы EDA.** Наибольшая по модулю линейная корреляция с целью у "
    f"`{strongest}` ({target_correlations[strongest]:.3f}). "
    "У денежных величин разные масштабы и длинные правые хвосты. "
    "Часть финансовых показателей связана между собой, поэтому матрица может быть "
    "вырожденной; аналитическое решение вычисляется через псевдообратную. "
    "На графиках видны зависимости, которые одна прямая описывает неполно. "
    "Корреляция сама по себе не доказывает причинное влияние."
))
""")

markdown("""
## 4. Отложенная выборка и сравнение способов обучения

Выделяем 20% строк для проверки. На одинаковых исходных признаках сравниваем
аналитику, полный GD, настоящий SGD и реализации sklearn. Это базовый вариант
без дополнительных базисных признаков. Итоговую улучшенную модель исследуем
отдельно. Выбор шага и остановка итераций не используют проверочные ответы.
""")

code(r"""
X = clean.drop(columns="RiskScore")
y = clean.RiskScore.to_numpy(dtype=float)
train_indices, valid_indices = train_test_split(np.arange(len(clean)), test_size=.2, random_state=SEED)
X_train_raw, X_valid_raw = X.iloc[train_indices], X.iloc[valid_indices]
y_train, y_valid = y[train_indices], y[valid_indices]
baseline_preprocessor = TabularPreprocessor(engineered=False)
X_train = baseline_preprocessor.fit_transform(X_train_raw)
X_valid = baseline_preprocessor.transform(X_valid_raw)
print("Обучение:", X_train.shape, "Проверка:", X_valid.shape)

def scores(actual, predicted):
    return {"MSE": mse(actual, predicted), "MAE": mae(actual, predicted),
            "R2": r2(actual, predicted), "MAPE, %": 100 * mape(actual, predicted)}

baseline_models = {}
comparison_rows = []
for method, iterations in [("analytic", 1), ("gd", 20000), ("sgd", 250)]:
    started = perf_counter()
    model = LinearRegression(method=method, max_iter=iterations, tol=1e-10, random_state=SEED)
    model.fit(X_train, y_train)
    elapsed = perf_counter() - started
    baseline_models[method] = model
    comparison_rows.append({"модель": "своя " + method,
                            **scores(y_valid, model.predict(X_valid)),
                            "MSE train": mse(y_train, model.predict(X_train)),
                            "итераций/эпох": model.n_iter_, "секунд": elapsed})

reference_ols = SklearnLinearRegression().fit(X_train, y_train)
np.testing.assert_allclose(baseline_models["analytic"].predict(X_valid),
                           reference_ols.predict(X_valid), rtol=1e-8, atol=1e-7)
comparison_rows.append({"модель": "sklearn LinearRegression",
                        **scores(y_valid, reference_ols.predict(X_valid)),
                        "MSE train": mean_squared_error(y_train, reference_ols.predict(X_train))})
reference_sgd = SGDRegressor(penalty=None, loss="squared_error", max_iter=10000,
                             tol=1e-5, random_state=SEED, learning_rate="adaptive",
                             eta0=min(.001, baseline_models["sgd"].learning_rate_),
                             average=True)
reference_sgd.fit(X_train, y_train)
comparison_rows.append({"модель": "sklearn SGDRegressor, penalty=None",
                        **scores(y_valid, reference_sgd.predict(X_valid)),
                        "MSE train": mean_squared_error(y_train, reference_sgd.predict(X_train)),
                        "итераций/эпох": reference_sgd.n_iter_})
optimizer_comparison = pd.DataFrame(comparison_rows).set_index("модель")
display(optimizer_comparison)
optimizer_comparison.to_csv(OUTPUT_DIR / "optimizer_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for axis, method in zip(axes, ["gd", "sgd"]):
    history = baseline_models[method].loss_history_
    axis.plot(np.arange(len(history)), history, label=method.upper())
    axis.axhline(baseline_models["analytic"].training_mse_, color="black",
                 linestyle="--", label="аналитический минимум")
    axis.set(xlabel="Шаг GD / эпоха SGD", ylabel="MSE на обучении", title=method.upper())
    axis.legend()
show_figure("optimizer_convergence.png")

normalization_rows = []
for name, scaler in [("z-score", ZScoreScaler()), ("min-max", MinMaxScaler())]:
    normalized_train = scaler.fit_transform(X_train)
    normalized_valid = scaler.transform(X_valid)
    normalized_model = LinearRegression().fit(normalized_train, y_train)
    normalization_rows.append({"нормализация": name,
                                **scores(y_valid, normalized_model.predict(normalized_valid))})
display(pd.DataFrame(normalization_rows).set_index("нормализация"))
""")

markdown(r"""
GD и SGD решают ту же задачу минимизации MSE, что аналитический метод,
но за конечное число итераций могут не достичь его результата. Зависимые
признаки допускают разные наборы весов с одинаковыми прогнозами, поэтому
сравниваем прогнозы и ошибки, а не требуем совпадения весов.

При наличии свободного коэффициента обратимое изменение масштаба признаков
не меняет пространство прогнозов OLS. Поэтому аналитика с z-score и min-max
даёт практически одинаковую ошибку. Для скорости GD и SGD масштабирование существенно.

## 5. Подготовка улучшенных признаков

Дополнительные признаки — детерминированные преобразования входных данных.
Они могут описывать нелинейную зависимость от исходного столбца, но модель
остаётся **линейной по обучаемым весам**: $\hat y=\sum_j w_j\phi_j(x)+b$.
Другие обучаемые модели и регуляризация не используются.
Добавлены логарифмы денежных величин, полиномиальные члены степеней 2–4
по шести исходным величинам и взаимодействия с признаками банкротства,
предыдущих дефолтов и занятости. Перед возведением в степень входы полиномов
ограничиваются обучающими квантилями 0.1% и 99.9%, затем центрируются
и масштабируются. Это ограничение применяется к базисным признакам;
наблюдения и обычные исходные денежные величины сохраняются.
Точный набор преобразований и их вычисление приведены в `TabularPreprocessor`
выше. Все параметры, требующие оценки по данным, переобучаются внутри каждого fold.
""")

code(r"""
improved_preprocessor = TabularPreprocessor(engineered=True)
X_train_improved = improved_preprocessor.fit_transform(X_train_raw)
X_valid_improved = improved_preprocessor.transform(X_valid_raw)
improved_model = LinearRegression().fit(X_train_improved, y_train)
improved_predictions = improved_model.predict(X_valid_improved)
holdout_scores = scores(y_valid, improved_predictions)
feature_comparison = pd.DataFrame([
    {"признаки": "исходные + one-hot", **scores(y_valid, baseline_models["analytic"].predict(X_valid)),
     "столбцов": X_train.shape[1]},
    {"признаки": "преобразованные + one-hot", **holdout_scores,
     "столбцов": X_train_improved.shape[1]},
]).set_index("признаки")
display(feature_comparison)
feature_comparison.to_csv(OUTPUT_DIR / "feature_comparison.csv")
np.testing.assert_allclose(improved_predictions,
                           SklearnLinearRegression().fit(X_train_improved, y_train).predict(X_valid_improved),
                           rtol=1e-7, atol=1e-6)
for name, (own, reference) in metric_functions.items():
    np.testing.assert_allclose(own(y_valid, improved_predictions),
                               reference(y_valid, improved_predictions), rtol=1e-12, atol=1e-12)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_valid, improved_predictions, alpha=.3, s=12)
limits = [min(y_valid.min(), improved_predictions.min()), max(y_valid.max(), improved_predictions.max())]
axes[0].plot(limits, limits, "k--")
axes[0].set(xlabel="Истинный RiskScore", ylabel="Прогноз", title="Отложенная выборка")
axes[1].scatter(improved_predictions, y_valid - improved_predictions, alpha=.3, s=12)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set(xlabel="Прогноз", ylabel="Остаток: факт − прогноз", title="Остатки модели")
show_figure("validation_predictions.png")
""")

markdown("""
## 6. Самостоятельная k-fold кросс-валидация

Пять непересекающихся проверочных частей; каждая строка ровно один раз служит
проверочной. На каждой итерации заново обучаем всю подготовку и модель.
Сравниваем базовый и улучшенный варианты на одних разбиениях. OOF-метрики
считаются по всем прогнозам вне обучения; среднее MSE fold приводится отдельно.
""")

code(r"""
def evaluate_splits(frame, target, splits, engineered):
    predictions = np.full(len(target), np.nan)
    visits = np.zeros(len(target), dtype=int)
    rows = []
    for fold, (training, validation) in enumerate(splits, start=1):
        assert not np.intersect1d(training, validation).size
        preprocessor = TabularPreprocessor(engineered=engineered)
        train_design = preprocessor.fit_transform(frame.iloc[training])
        valid_design = preprocessor.transform(frame.iloc[validation])
        model = LinearRegression().fit(train_design, target[training])
        predicted = model.predict(valid_design)
        predictions[validation] = predicted
        visits[validation] += 1
        rows.append({"fold": fold, "train": len(training), "validation": len(validation),
                     **scores(target[validation], predicted)})
    assert np.all(visits == 1), "Каждая строка должна быть проверена ровно один раз"
    assert np.isfinite(predictions).all()
    return predictions, pd.DataFrame(rows)

kfold_results, oof_predictions = [], {}
for engineered, label in [(False, "исходные"), (True, "преобразованные")]:
    predicted, fold_table = evaluate_splits(
        X, y, k_fold_indices(len(y), n_splits=5, random_state=SEED), engineered=engineered)
    oof_predictions[label] = predicted
    fold_table.insert(0, "признаки", label)
    kfold_results.append(fold_table)
kfold_table = pd.concat(kfold_results, ignore_index=True)
display(kfold_table)
cv_summary = pd.DataFrame([
    {"признаки": label, **scores(y, predicted),
     "среднее MSE fold": kfold_table.loc[kfold_table["признаки"] == label, "MSE"].mean(),
     "std MSE fold": kfold_table.loc[kfold_table["признаки"] == label, "MSE"].std(ddof=1)}
    for label, predicted in oof_predictions.items()
]).set_index("признаки")
display(cv_summary)
kfold_table.to_csv(OUTPUT_DIR / "kfold_scores.csv", index=False)
cv_summary.to_csv(OUTPUT_DIR / "cross_validation_summary.csv")
""")

markdown(r"""
## 7. Самостоятельная leave-one-out кросс-валидация

**Полный пересчёт подготовки.** Для демонстрации обычного алгоритма берём
случайную подвыборку из 200 строк: 200 раз исключаем одну строку и заново
обучаем подготовку и модель на остальных 199. Здесь используем исходные
признаки: при 199 наблюдениях большой базис был бы избыточным.
Это именно LOO на подвыборке, а не оценка LOO на всех данных.

**Дополнительный быстрый расчёт.** На всей фиксированной числовой матрице
можно вычислить LOO аналитической OLS по формуле PRESS:
$\hat y_i^{(-i)}=y_i-e_i/(1-h_{ii})$.
Эта формула исключает строку из обучения регрессии, но не переобучает медианы
и кодирование. Поэтому результат явно обозначен как LOO **фиксированной матрицы**,
а основной оценкой полного решения служит k-fold выше. Диагональ проектора
получаем через SVD, не создавая матрицу размера $n\times n$.
""")

code(r"""
loo_indices = np.random.default_rng(SEED).choice(len(y), size=200, replace=False)
loo_frame = X.iloc[loo_indices].reset_index(drop=True)
loo_y = y[loo_indices]
loo_pipeline_predictions, _ = evaluate_splits(
    loo_frame, loo_y, leave_one_out_indices(len(loo_y)), engineered=False)
display(pd.DataFrame([{"вариант": "полный refit подготовки, 200 строк",
                       **scores(loo_y, loo_pipeline_predictions)}]).set_index("вариант"))

# Independent check: own PRESS and literal sklearn LOO on exactly the same design.
check_rng = np.random.default_rng(SEED)
check_design = check_rng.normal(size=(35, 5))
check_design = np.column_stack([check_design, check_design[:, 0] + check_design[:, 1]])
check_y = check_design @ check_rng.normal(size=6) + check_rng.normal(scale=.1, size=35)
check_press = loo_predictions_ols(check_design, check_y)
check_direct = cross_val_predict(SklearnLinearRegression(), check_design, check_y, cv=LeaveOneOut())
np.testing.assert_allclose(check_press, check_direct, rtol=1e-8, atol=1e-8)
print("PRESS совпал с буквальным sklearn LeaveOneOut, в том числе при зависимых столбцах.")

full_fixed_preprocessor = TabularPreprocessor(engineered=True)
full_fixed_design = full_fixed_preprocessor.fit_transform(X)
press_predictions = loo_predictions_ols(full_fixed_design, y)
press_scores = scores(y, press_predictions)
display(pd.DataFrame([{"вариант": "PRESS: фиксированная матрица, все строки",
                       **press_scores}]).set_index("вариант"))
""")

markdown("""
## 8. Итоговое обучение и файл для Kaggle

После проверки обучаем подготовку и аналитическую линейную регрессию на всех
очищенных размеченных строках. `test.csv` не участвует в оценке параметров
подготовки. Порядок строк и `ID` сохраняются. Прогнозы не ограничиваем искусственно.

Локальные MSE и кросс-валидация не являются результатом Kaggle. Проверка условия
**MSE < 25 на соревновании** возможна после загрузки `submission.csv`.
""")

code(r"""
final_preprocessor = TabularPreprocessor(engineered=True)
final_design = final_preprocessor.fit_transform(X)
final_model = LinearRegression().fit(final_design, y)
test_design = final_preprocessor.transform(test_raw.drop(columns="ID"))
test_predictions = final_model.predict(test_design)
submission = pd.DataFrame({"ID": test_raw.ID.to_numpy(), "RiskScore": test_predictions})
assert submission.shape == sample_submission.shape
assert list(submission.columns) == list(sample_submission.columns)
assert submission.ID.equals(test_raw.ID.reset_index(drop=True))
assert np.array_equal(submission.ID.to_numpy(), sample_submission.ID.to_numpy())
assert np.isfinite(submission.RiskScore).all()
submission.to_csv(DATA_DIR / "submission.csv", index=False)
display(submission.head())
display(submission.RiskScore.describe().to_frame("прогнозы test.csv"))

coefficients = pd.DataFrame({"признак": final_preprocessor.feature_names_,
                              "вес": final_model.coef_})
coefficients.to_csv(OUTPUT_DIR / "coefficients.csv", index=False)
summary = {
    "seed": SEED, "raw_train_rows": len(train_raw), "clean_train_rows": len(clean),
    "test_rows": len(test_raw), "final_features": final_design.shape[1],
    "holdout": holdout_scores,
    "kfold_oof": scores(y, oof_predictions["преобразованные"]),
    "loo_pipeline_200": scores(loo_y, loo_pipeline_predictions),
    "loo_fixed_design_press": press_scores,
    "submission": "submission.csv", "kaggle_score": None,
    "regularization": False,
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
display(Markdown(
    f"**Результат.** Отложенная MSE: **{holdout_scores['MSE']:.3f}**, "
    f"5-fold OOF MSE: **{summary['kfold_oof']['MSE']:.3f}**. "
    f"Файл `submission.csv` содержит {len(submission)} прогнозов. "
    "Результат на Kaggle пока не измерен."
))
""")

markdown(r"""
## Выводы и вопросы к защите

- Линейная регрессия обучает веса, минимизируя MSE. Аналитика находит решение
  через псевдообратную, GD использует весь набор, SGD — отдельные наблюдения.
- Масштабирование улучшает численное поведение и сходимость итерационных методов.
  Аналитическая OLS с intercept сохраняет прогнозы при обратимом масштабировании.
- Очистка повреждённых целей и предотвращение попадания дублей в разные части
  необходимы для содержательной оценки качества.
- Подготовка обучается заново внутри каждого fold. Ответы проверочной части
  не участвуют в обучении модели.
- MSE сильнее штрафует большие ошибки; MAE измеряется в баллах риска; R²
  сравнивает прогноз с постоянным средним; MAPE чувствительна к нулевым целям.
- Линейность модели определяется её зависимостью от обучаемых весов.
  Преобразованные признаки позволяют описывать более сложные зависимости.
- Обычный LOO и быстрый PRESS показаны с явным указанием объёма данных и
  различия между пересчётом всей подготовки и фиксированной матрицей признаков.

Для сдачи: отправить `submission.csv` на странице соревнования, проверить
MSE < 25, загрузить этот выполненный `.ipynb` на GitHub и указать свой ник
Kaggle в таблице оценок, как требует условие. Ссылки и аккаунты в файлах
задания не заданы; эти действия выполняются отдельно.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3 (ML lab-1)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.10"},
}
nbf.write(notebook, ROOT / "Lab1.ipynb")
print(f"Built {ROOT / 'Lab1.ipynb'} ({len(cells)} cells)")
