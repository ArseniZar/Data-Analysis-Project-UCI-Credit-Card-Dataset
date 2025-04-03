Data Analysis Project

This project performs various data analysis and visualization tasks using Python. The codebase includes functions to generate multiple types of plots (boxplots, error bar charts, heatmaps, histograms, regression plots, and violin plots) that help you understand and interpret the data.


```
Project_1/
│── data/                       # Data files
│── new_venv/                   # Virtual environment
│── outputs/plots/              # Generated plots and plotting modules
│   ├── boxplots.py             # Box plot visualizations
│   ├── errorbars.py            # Error bar charts
│   ├── heatmap.py              # Heatmap visualizations
│   ├── histogram_with_hue.py   # Histograms with hue (colored by a category)
│   ├── histograma.py           # Basic histograms
│   ├── regression.py           # Regression plots
│   ├── violinplots.py          # Violin plots
│── src/                        # Source code
│   ├── data_loader.py          # Data loading utilities
│   ├── file_writer.py          # File writing utilities
│   ├── statistics.py           # Statistical analysis functions
│── data_analysis_main.py       # Main script for analysis
│── requirements.txt            # Required dependencies
│── README.md                   # Project documentation
│── requirements.txt            # Python library

```

## Setup & Installation

### 1. Clone the Repository
```bash
git clone <repository_url>
cd Project_1
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv new_venv
source new_venv/bin/activate  # On MacOS/Linux
new_venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project
```bash
python src/data_analysis_main.py
```

## Using Plot Functions
Each plot function is located in the `outputs/plots/` directory. Below is a list of available functions and their usage:

### 1. Box Plots (`boxplots.py`)
```python
from outputs.plots.boxplots import plot_boxplot
plot_boxplot(df, x_col='SEX', y_col='LIMIT_BAL', title="Credit Limit by Gender")
```

### 2. Error Bar Charts (`errorbars.py`)
```python
from outputs.plots.errorbars import plot_error_bars
plot_error_bars(df, x_col='default.payment.next.month', y_col='LIMIT_BAL')
```

### 3. Heatmaps (`heatmap.py`)
```python
from outputs.plots.heatmap import plot_heatmap
plot_heatmap(df, columns=['LIMIT_BAL', 'AGE', 'PAY_0'])
```

### 4. Histograms with Hue (`histogram_with_hue.py`)
```python
from outputs.plots.histogram_with_hue import plot_histogram_with_hue
plot_histogram_with_hue(df, x_col='EDUCATION', hue='default.payment.next.month')
```

### 5. Basic Histograms (`histograma.py`)
```python
from outputs.plots.histograma import plot_histogram
plot_histogram(df, column='BILL_AMT1')
```

### 6. Regression Plots (`regression.py`)
```python
from outputs.plots.regression import plot_regression
plot_regression(df, x_col='LIMIT_BAL', y_col='BILL_AMT1')
```

### 7. Violin Plots (`violinplots.py`)
```python
from outputs.plots.violinplots import plot_violinplot
plot_violinplot(df, x_col='default.payment.next.month', y_col='LIMIT_BAL')
```

## Outputs
- All generated plots are saved in the `outputs/plots/` directory.

## Dependencies
All required Python libraries are listed in `requirements.txt`. Install them using the command above.

## Notes
- Ensure the virtual environment is activated before running scripts.
- Modify the data directory path in `data_loader.py` if needed.
- If new dependencies are added, update `requirements.txt` using:
```bash
pip freeze > requirements.txt
```

