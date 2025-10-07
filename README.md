# Anomaly Detection on Time Series Data

---

## Model Benchmarks

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
