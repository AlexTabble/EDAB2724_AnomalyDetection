# Anomaly Detection on Time Series Data

---

## Benchmark Documentation

The most important function in the class is `create_anomaly_groups()`

#### `create_anomaly_groups`

**Parameters**

| Parameter               | Type                          | Description                                                                                                      |
| ----------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `data`                  | `pd.DataFrame` or `pd.Series` | Input data containing anomalies. If DataFrame, the column must be specified.                                     |
| `col`                   | `str`                         | Column name to check for anomalies (only needed if `data` is a DataFrame). Default is `'outlier'`.               |
| `include_single_groups` | `bool`                        | Whether to include anomalous regions of length 1. Default is `False`.                                            |
| `show_printout`         | `bool`                        | Whether to print the number of anomaly groups identified. Default is `True`.                                     |
| `merge_tolerance`       | `int`                         | Maximum gap between consecutive anomaly regions that will be merged. Default is `5`.                             |
| `noise_tolerance`       | `int`                         | Minimum length for a group to be considered a true anomaly; shorter groups are treated as noise. Default is `3`. |

**Output**
| Output | Type | Description |
|----------------|--------------------|-----------------------|
| `groups` | `list[tuple[int]]` | ${ (start_1,end_1),(start_2,end_2), \dots, (start_n, end_n)}$
This function can be used regardless of the chosen classifier.

### Does it generalise?

Most models have good accuracies and precision,recall values so actual classifiers
are not the main problem.

To accurately state where the anomalous regions are is a challenge as you will
have to tune the `merge_tolerance` and `noise_tolerance` parameters for the
grouper which could introduce possible bias.

If the nature of the data is known(i.e. Financial, Geogolicial, etc) those
parameters can be tuned with relevant business knowledge.

The dataset is called _ec2_utilization.zip_ which monitors the cpu usage of an
AWS EC2 instance over time. More research can be done to determine the norm
in terms of deviation frequency and length of deviation. (I'm assuming the
data is about cpu usage but we probably need to confirm)

To generalise the classification of anomalous period, I believe some leniency in
the classification of what an anomalous period is necessary.

For example: If the daily LIBOR rate decreased for 5 days during the 2007-2008
GFC, it does not mean the entire period is not anomalous. If this 5-day period
is not ignored as noise, the GFC would be classified as two anomalous periods(one
before the 5 days and one after) which essentially fragments the period.

This is seen extensively in the Z-Score predictor where a single period is
fragmented into multiple smaller periods with small gaps between them.

### Possible Improvements

`include_single_groups` might cause a fragmentation of a a single anomalous
region into two anomalous regions with incorrect $start_{n-1$ and $end_{n}$
values.

The reason I included the parameter in the first place is that if an anomaly has
length of 1, its start- and end points are the same like $(n,n)$ which I didn't
want to deal with.

After making the grouper more tolerant towards gaps and noise, I noticed that it
might not be a good parameter to include.

## Available Functions in Benchmarking.py

#### `evaluate_model`

**Input**

| Parameter | Type       | Description                                               |
| --------- | ---------- | --------------------------------------------------------- |
| `y_true`  | `np.array` | True binary anomaly labels (1 for anomaly, 0 for normal). |
| `y_pred`  | `np.array` | Predicted binary anomaly labels.                          |

**Output**

| Output    | DataType  |
| --------- | --------- |
| `metrics` | DataFrame |

| Metric                     | Description                                                           |
| -------------------------- | --------------------------------------------------------------------- |
| `Accuracy`                 | Standard classification accuracy.                                     |
| `Precision`                | Fraction of predicted anomalies that are true anomalies.              |
| `Recall`                   | Fraction of true anomalies that were detected.                        |
| `Balanced Accuracy`        | Average of recall per class (handles imbalance).                      |
| `Group Accuracy`           | Accuracy at the anomaly group level (based on `_evaluate_groups`).    |
| `Penalised Group Accuracy` | Group accuracy with penalty applied for mismatch in number of groups. |

**Example usage**

```python
metrics = Benchmarking.evaluate_model(y_true, y_pred)
print(metrics)
```

#### `print_evaluation`

**Input**
| Parameter | Type | Description |
| ------------ | ---------- | -------------------------------------------------------- |
| `y_true` | `np.array` | True anomaly labels. |
| `y_pred` | `np.array` | Predicted anomaly labels. |
| `model_name` | `str` | Name of the model, used for display in the table header. |

**Output**
| Output | DataType |
|-----------|------------|
| printed DataFrame | None|

**Example usage**

```python

Benchmarking.print_evaluation(y_true, y_pred, "ARIMA(Tuned)")
```

### Benchmarking usage

#### For Google Colab

Copy and paste the file contents into a cell

#### For Personal Machine

Make sure benchmarking.py file is in same directory as .ipynb/.qmd file

```
.
├── benchmarking.py
├── test
│   ├── 01.csv
│   ├── 02.csv
│   ├── 03.csv
│   ├── 04.csv
│   ├── 05.csv
│   ├── 06.csv
│   ├── 07.csv
│   ├── 08.csv
│   ├── 09.csv
│   └── 10.csv
└── Z*score_model.qmd -> \_Same directory level as benchmarking.py*

```

```python
from benchmarking import Benchmarking

metrics = Benchmarking.evaluate_model(y_true, y_pred)
```

---

## Z-Score StatsModel Documentation

The model classifies an anomaly based of the following metrics:

> [!Note] Assumptions
> Data is **Normally Distribted**.
> Metrics are not appropriate for other distributions

- Z-score
- IQR
- Rolling Z-score
- Mean Absolute Deviation

### Model API and Design

If follows the scikit-learn API of `fit`, `predict`. Currently, I have not
implemented it to predict on new data and it uses the provided training data
to predict as it is purely statistical.

I will modify the code to predict on new data using the thresholds from the
training data and compute new classifier masks.

## How it works

#### `StatsModel`

An ensemble Anomaly classifier using traditional statistics like:

- IQR
- Z-score
- Rolling Z-score
- MAD

| Parameter             | Default Value | Type    | Description                                                                                          |
| --------------------- | ------------- | ------- | ---------------------------------------------------------------------------------------------------- |
| `w_smooth`            | `51`          | `int`   | Window size for smoothing the input signal before computing residuals in the rolling Z-score method. |
| `w`                   | `61`          | `int`   | Window size for computing rolling mean and variance of residuals in rolling Z-score detection.       |
| `iqr_threshold`       | `1.5`         | `float` | Threshold multiplier for detecting outliers based on Interquartile Range (IQR).                      |
| `mad_threshold`       | `2.465`       | `float` | Threshold multiplier for detecting outliers using Mean Absolute Deviation (MAD).                     |
| `normal_z_threshold`  | `2`           | `float` | Z-score cutoff for normal distribution–based outlier detection.                                      |
| `rolling_z_threshold` | `2`           | `float` | Z-score cutoff for rolling window–based outlier detection.                                           |
| `metric_consensus`    | `3`           | `int`   | Number of metrics that must agree for a point to be labeled an outlier.                              |

#### `_determine_rolling_z_mask`

Takes the prediction data and computes the rolling z scores. This does not work
great when storing the metrics in the test fit as the convolution depends on the
actual data as well

| Parameter | Default Value | Type       | Description                                                              |
| --------- | ------------- | ---------- | ------------------------------------------------------------------------ |
| `X`       | —             | `np.array` | Input 1D numerical array (time series) to calculate rolling Z-scores on. |

#### `fit`

The normal sklearn API being adhered to. Note that the y parameter is passed but
not used during training. This is because the sklearn API requires it to be
passed for other wrappers like `GridSearchCV` to work.

Fun fact: it also returns self which is important as that's what allows tuning
algos to actually fit over and over again.

| Parameter | Default Value | Type         | Description                                             |
| --------- | ------------- | ------------ | ------------------------------------------------------- |
| `X`       | —             | `np.array`   | Input numerical array used to compute model statistics. |
| `y`       | `None`        | `array-like` | Optional labels (unused, for sklearn compatibility).    |

#### `predict`

The test set is passed here which returns the predictions as an array.
`_determine_rolling_z_mask` is also called here but not in fit due to the limitation
as mentioned above.

| Parameter | Default Value | Type       | Description                        |
| --------- | ------------- | ---------- | ---------------------------------- |
| `X`       | `None`        | `np.array` | Input numerical array to classify. |

#### `score`

Also a requirement in the sklearn API. The current scorer is recall. If hyperparameter
tuning for different scores, just change the `scorer` parameter in the tuning
algorithm to what you want.

| Parameter | Default Value | Type       | Description                                                                    |
| --------- | ------------- | ---------- | ------------------------------------------------------------------------------ |
| `X`       | —             | `np.array` | Input numerical array for prediction.                                          |
| `y_true`  | —             | `np.array` | True binary labels indicating whether each point is an outlier (1) or not (0). |

**Example usage**

```python

model = StatsModel()
model.fit(X_train)
y_pred = model.predict(X_test)

from benchmarking import Benchmarking

Benchmarking.evaluate_model(y_true, y_pred)
```
