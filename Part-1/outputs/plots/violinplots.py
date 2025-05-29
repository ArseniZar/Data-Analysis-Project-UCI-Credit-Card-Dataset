import seaborn as sns
import matplotlib.pyplot as plt

def plot_violinplot(df, x_col, y_col, title="Violinplot", x_label=None, y_label=None, 
                    figsize=(10, 6), x_tick_labels=None, y_formatter=None, save_path=None):
    """
    Function to create a violin plot with an option to save it.
    
    Parameters:
    - df: DataFrame containing the data
    - x_col: column for the X-axis (categorical variable)
    - y_col: column for the Y-axis (numerical variable)
    - title: plot title
    - x_label: X-axis label (defaults to column name)
    - y_label: Y-axis label (defaults to column name)
    - figsize: figure size (width, height)
    - x_tick_labels: dictionary to replace X-axis tick labels (e.g., {0: "No Default", 1: "Default"})
    - y_formatter: format for Y-axis values (e.g., '{:,.0f}' for integers)
    - save_path: path to save the plot (e.g., "violinplot1.png")
    """
    plt.figure(figsize=figsize)
    
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col
    
    sns.violinplot(x=x_col, y=y_col, data=df)
    
    if x_tick_labels:
        plt.xticks(ticks=plt.xticks()[0], labels=[x_tick_labels.get(int(tick), tick) for tick in plt.xticks()[0]])
    
    if y_formatter:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: y_formatter.format(x)))
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()