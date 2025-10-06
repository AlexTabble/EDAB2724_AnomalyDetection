# Anomaly Detection on Time Series Data

---

## Model Benchmarks

Insert benchmarks or whatever here

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

```{python}
from benchmarking import Benchmarking

metrics = Benchmarking.evaluate_model(y_true, y_pred)
```

---
