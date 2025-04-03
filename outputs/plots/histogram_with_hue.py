import seaborn as sns
import matplotlib.pyplot as plt


def plot_histogram_with_hue(df, x_col, hue_col, title="Histogram with Hue", x_label=None, 
                            y_label="Count", bins=30, multiple="stack", figsize=(10, 6), save_path=None):
    """
    Function to plot a histogram with a hue parameter.
    
    Parameters:
    - df: DataFrame containing the data
    - x_col: column for the X-axis (numerical variable)
    - hue_col: column for color separation (categorical variable)
    - title: title of the plot
    - x_label: label for the X-axis
    - y_label: label for the Y-axis
    - bins: number of intervals
    - multiple: type of overlay ("stack", "layer", "dodge", "fill")
    - figsize: size of the plot
    - save_path: path to save the plot
    """
    plt.figure(figsize=figsize)
    
    if x_label is None:
        x_label = x_col
    
    sns.histplot(data=df, x=x_col, hue=hue_col, bins=bins, multiple=multiple)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()