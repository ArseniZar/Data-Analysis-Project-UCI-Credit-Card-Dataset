import seaborn as sns
import matplotlib.pyplot as plt

def plot_regression(df, x_col, y_col, hue_col=None, title="Linear Regression", 
                    x_label=None, y_label=None, figsize=(10, 6), save_path=None):
    """
    Function to plot a linear regression graph.
    
    Parameters:
    - df: DataFrame with data
    - x_col: column for the X-axis (numeric)
    - y_col: column for the Y-axis (numeric)
    - hue_col: column for color separation (categorical, optional)
    - title: graph title
    - x_label: X-axis label
    - y_label: Y-axis label
    - figsize: graph size (width, height)
    - save_path: path to save the graph
    """
    if hue_col:
        sns.lmplot(x=x_col, y=y_col, hue=hue_col, data=df, 
                   height=figsize[1], aspect=figsize[0]/figsize[1])
    else:
        plt.figure(figsize=figsize)
        sns.regplot(x=x_col, y=y_col, data=df)
    
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()
