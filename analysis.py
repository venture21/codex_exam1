import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def load_data(filepath):
    """Load Excel data into a pandas DataFrame."""
    df = pd.read_excel(filepath)
    return df


def run_eda(df):
    """Perform basic EDA and return summary information."""
    summary = df.describe(include='all')
    print("Data Head:\n", df.head())
    print("\nSummary Statistics:\n", summary)
    return summary


def correlation_with_target(df, target):
    """Calculate correlation between a target column and all numeric columns."""
    numeric_df = df.select_dtypes(include='number')
    if target not in numeric_df:
        raise ValueError(f"Target '{target}' must be numeric for correlation analysis.")
    correlations = numeric_df.corr()[target].sort_values(ascending=False)
    print(f"\nCorrelations with {target}:\n", correlations)
    return correlations


def test_hypothesis(df):
    """Example hypothesis: Survival rate differs by gender."""
    contingency = pd.crosstab(df['sex'], df['survived'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print("\nChi-squared test p-value:", p)
    if p < 0.05:
        print("Reject the null hypothesis: Survival depends on gender.")
    else:
        print("Fail to reject the null hypothesis: No evidence of dependency.")
    return p


def generate_report(summary, correlations, hypothesis_p):
    """Generate a simple text report and save to file."""
    with open('report.txt', 'w', encoding='utf-8') as f:
        f.write("Exploratory Data Analysis Summary\n")
        f.write(str(summary))
        f.write("\n\nCorrelation with target:\n")
        f.write(str(correlations))
        f.write("\n\nHypothesis Test p-value: {:.4f}\n".format(hypothesis_p))


if __name__ == '__main__':
    # 1. Load data
    data = load_data('titanic.xls')

    # 2. EDA
    summary_stats = run_eda(data)

    # 3. Correlation analysis
    corr_survived = correlation_with_target(data, 'survived')

    # 4. Hypothesis testing
    p_value = test_hypothesis(data)

    # 5. Report
    generate_report(summary_stats, corr_survived, p_value)

