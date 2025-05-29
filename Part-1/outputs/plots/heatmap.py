import seaborn as sns
import matplotlib.pyplot as plt


def plot_heatmap(df, columns, title="Correlation Heatmap", figsize=(12, 10), annot_fontsize=8, 
                 save_path=None):
    """
    Function to plot a correlation heatmap.
    
    Parameters:
    - df: DataFrame with data
    - columns: list of columns for correlation analysis
    - title: plot title
    - figsize: figure size
    - annot_fontsize: font size for annotations
    - save_path: path to save the plot
    """
    plt.figure(figsize=figsize)
    
    corr_matrix = df[columns].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, 
                annot_kws={"size": annot_fontsize}, fmt=".2f")
    
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()
    plt.close()
