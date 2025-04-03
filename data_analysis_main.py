from outputs.plots.boxplots import plot_boxplot
from outputs.plots.violinplots import plot_violinplot
from outputs.plots.errorbars import plot_error_bars
from outputs.plots.histograma import plot_histogram
from outputs.plots.histogram_with_hue import plot_histogram_with_hue
from outputs.plots.heatmap import plot_heatmap
from outputs.plots.regression import plot_regression

from src.data_loader import DataLoader
from src.file_writer import save_stats_to_csv
from src.calculate_statistics import calculate_statistics
 


def box(df):
    plot_boxplot(
        df,
        "MARRIAGE",
        "LIMIT_BAL",
        title="Credit Limit by Marital Status",
        x_label="Marital status (1=married, 2=single, 3=others)",
        y_label="Credit Limit (NT Dollars)",
        save_path="outputs/boxplot_limit_bal_by_default.png",
    )

    plot_boxplot(
        df,
        "default.payment.next.month",
        "AGE",
        title="Age Distribution by Default Status",
        x_label="Default (0 = No, 1 = Yes)",
        y_label="Age (Years)",
        save_path="outputs/boxplot_age_by_default.png",
    )

    plot_boxplot(
        df,
        "default.payment.next.month",
        "LIMIT_BAL",
        title="Credit Limit by Default Status",
        x_label="Default (0 = No, 1 = Yes)",
        y_label="Credit Limit (NT Dollars)",
        save_path="outputs/boxplot_limit_bal_by_default_status.png",
    )


def vio(df):
    plot_violinplot(
        df,
        "EDUCATION",
        "LIMIT_BAL",
        title="Credit Limit by Education Level",
        x_label="Education Level",
        y_label="Credit Limit (NT Dollars)",
        y_formatter="{:,.0f}",
        save_path="outputs/violinplot_limit_bal_by_education.png",
    )

    plot_violinplot(
        df,
        "EDUCATION",
        "AGE",
        title="Age Distribution by Education Level",
        x_label="Education Level",
        y_label="Age (Years)",
        save_path="outputs/violinplot_age_by_education.png",
    )

    plot_violinplot(
        df,
        "EDUCATION",
        "BILL_AMT1",
        title="Bill Amount (September) by Education Level",
        x_label="Education Level",
        y_label="Bill Amount (NT Dollars)",
        y_formatter="{:,.0f}",
        save_path="outputs/violinplot_bill_amt1_by_education.png",
    )


def errorbars(df):
    plot_error_bars(
        df,
        "default.payment.next.month",
        "LIMIT_BAL",
        title="Average Credit Limit by Default Payment Status",
        x_label="Default Payment Status",
        y_label="Average Credit Limit (NT Dollars)",
        y_formatter="{:,.0f}",
        save_path="outputs/error_bars_limit_bal_by_default.png",
    )

    plot_error_bars(
        df,
        "SEX",
        "AGE",
        title="Average Age by Gender",
        x_label="Gender",
        y_label="Average Age (Years)",
        save_path="outputs/error_bars_age_by_sex.png",
    )

    plot_error_bars(
        df,
        "EDUCATION",
        "BILL_AMT1",
        title="Average Bill Amount (September) by Education Level",
        x_label="Education Level",
        y_label="Average Bill Amount (NT Dollars)",
        y_formatter="{:,.0f}",
        save_path="outputs/error_bars_bill_amt1_by_education.png",
    )


def histograma(df):
    plot_histogram(
        df,
        "LIMIT_BAL",
        title="Distribution of Credit Limit",
        x_label="Credit Limit (NT Dollars)",
        save_path="outputs/histogram_limit_bal.png",
    )

    plot_histogram(
        df,
        "AGE",
        title="Distribution of Age",
        x_label="Age (Years)",
        save_path="outputs/histogram_age.png",
    )

    plot_histogram(
        df,
        "BILL_AMT1",
        title="Distribution of Bill Amount (September)",
        x_label="Bill Amount (NT Dollars)",
        save_path="outputs/histogram_bill_amt1.png",
    )


def histogram_with_hue(df):
    plot_histogram_with_hue(
        df,
        "LIMIT_BAL",
        "default.payment.next.month",
        title="Distribution of Credit Limit by Default Status",
        x_label="Credit Limit (NT Dollars)",
        save_path="outputs/histogram_hue_limit_bal_by_default.png",
    )

    plot_histogram_with_hue(
        df,
        "AGE",
        "SEX",
        title="Age Distribution by Sex",
        x_label="Age (Years)",
        save_path="outputs/histogram_hue_age_by_sex.png",
    )

    plot_histogram_with_hue(
        df,
        "BILL_AMT1",
        "default.payment.next.month",
        title="Bill Amount (September) by Default Status",
        x_label="Bill Amount (NT Dollars)",
        save_path="outputs/histogram_hue_bill_amt1_by_default.png",
    )

    plot_histogram_with_hue(
        df,
        "PAY_AMT1",
        "default.payment.next.month",
        title="Payment Amount (September) by Default Status",
        x_label="Payment Amount (NT Dollars)",
        save_path="outputs/histogram_hue_pay_amt1_by_default.png",
    )

    plot_histogram_with_hue(
        df,
        "LIMIT_BAL",
        "EDUCATION",
        title="Distribution of Credit Limit by Education Level",
        x_label="Credit Limit (NT Dollars)",
        save_path="outputs/histogram_hue_limit_bal_by_education.png",
    )

    plot_histogram_with_hue(
        df,
        "BILL_AMT1",
        "MARRIAGE",
        title="Bill Amount (September) by Marriage Status",
        x_label="Bill Amount (NT Dollars)",
        save_path="outputs/histogram_hue_bill_amt1_by_marriage.png",
    )


def heatmap(df):
    all_numeric_cols = [
        "LIMIT_BAL",
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "default.payment.next.month",
        "PAY_0",
    ]

    plot_heatmap(
        df,
        all_numeric_cols,
        title="Correlation Heatmap of All Numeric Features",
        figsize=(14, 12),
        annot_fontsize=7,
        save_path="outputs/heatmap_all_numeric_features.png",
    )


def regress(df):
    plot_regression(
        df,
        "LIMIT_BAL",
        "BILL_AMT1",
        title="Linear Regression: LIMIT_BAL vs BILL_AMT1",
        x_label="Credit Limit (NT Dollars)",
        y_label="Bill Amount (NT Dollars)",
        save_path="outputs/regression_limit_bal_bill_amt1.png",
    )

    plot_regression(
        df,
        "AGE",
        "LIMIT_BAL",
        title="Linear Regression: AGE vs LIMIT_BAL",
        x_label="Age (Years)",
        y_label="Credit Limit (NT Dollars)",
        save_path="outputs/regression_age_limit_bal.png",
    )

    plot_regression(
        df,
        "BILL_AMT1",
        "PAY_AMT1",
        title="Linear Regression: BILL_AMT1 vs PAY_AMT1",
        x_label="Bill Amount (NT Dollars)",
        y_label="Payment Amount (NT Dollars)",
        save_path="outputs/regression_bill_amt1_pay_amt1.png",
    )


if __name__ == "__main__":
    

    loader = DataLoader("data/UCI_Credit_Card.csv")
    df = loader.data
    
    numeric_cols = [
        "LIMIT_BAL", "AGE",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    categorical_cols = [
        "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "default.payment.next.month"
    ]

    df_statistic = calculate_statistics(df, numeric_cols, categorical_cols)
    save_stats_to_csv(df_statistic,"outputs/statistic.csv")
    
    box(df)
    vio(df)
    errorbars(df)
    histograma(df)
    histogram_with_hue(df)
    heatmap(df)
    regress(df)
