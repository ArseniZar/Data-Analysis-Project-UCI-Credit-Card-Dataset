import seaborn as sns
import matplotlib.pyplot as plt


def plot_error_bars(df, x_col, y_col, title="Error Bars", x_label=None, y_label=None, 
                    figsize=(10, 6), y_formatter=None, save_path=None):
    """
    Function to plot error bars using Seaborn.
    
    Parameters:
    - df: DataFrame with data
    - x_col: column for the X-axis (categorical variable)
    - y_col: column for the Y-axis (numerical variable)
    - title: plot title
    - x_label: X-axis label
    - y_label: Y-axis label
    - figsize: figure size
    - y_formatter: format for Y-axis values
    - save_path: path to save the plot
    """
    plt.figure(figsize=figsize)
    
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col
    
    sns.pointplot(x=x_col, y=y_col, data=df, errorbar="ci")
    
    if y_formatter:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: y_formatter.format(x)))
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()