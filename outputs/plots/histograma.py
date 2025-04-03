import seaborn as sns
import matplotlib.pyplot as plt


def plot_histogram(df, x_col, title="Histogram", x_label=None, y_label="Count", 
                   bins=30, kde=True, figsize=(10, 6), save_path=None):
    """
    Function to plot a histogram.
    
    Parameters:
    - df: DataFrame with data
    - x_col: column for the X-axis (numerical variable)
    - title: plot title
    - x_label: X-axis label
    - y_label: Y-axis label
    - bins: number of bins
    - kde: whether to show the density line (True/False)
    - figsize: figure size
    - save_path: path to save the plot
    """
    plt.figure(figsize=figsize)
    
    if x_label is None:
        x_label = x_col
    
    sns.histplot(df[x_col], bins=bins, kde=kde)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()