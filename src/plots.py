import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style():
    sns.set_theme(style="whitegrid")


def plot_sentiment_and_volume_distributions(pdf):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(pdf["Sentiment"], bins=50, ax=ax[0], kde=True)
    ax[0].set_title("Distribution of Post Sentiment")

    sns.histplot(pdf["Volume"], bins=50, ax=ax[1], kde=True, log_scale=(False, True))
    ax[1].set_title("Distribution of Volume (Log Scale)")

    plt.tight_layout()
    return fig


def plot_sentiment_volume_scatter(pdf):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=pdf, x="Sentiment", y="Volume", hue="Ticker", alpha=0.5, ax=ax)
    ax.set_title("Hourly Volume vs. Post Sentiment")
    ax.set_yscale("log")
    return fig


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Down (0)", "Predicted Up (1)"],
        yticklabels=["Actual Down (0)", "Actual Up (1)"],
        ax=ax,
    )
    ax.set_title("Baseline Model Confusion Matrix")
    return fig
