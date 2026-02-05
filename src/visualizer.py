"""
The visualizer is just to explore our data. Useful visuals are located in the Python Notebooks.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from wordcloud import WordCloud



class Visualizer:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report_dir = Path("reports/visuals")
        self.report_dir.mkdir(exist_ok=True)

    # just thought of a heatmap too see how everything correlates with each other
    def correlation_heatmap(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.df.corr(numeric_only=True),
            annot=False,
            cmap="coolwarm",
            square=True
        )
        plt.title("Correlation Heatmap of Lyrical Features")
        plt.tight_layout()
        plt.savefig(self.report_dir / "correlation_heatmap.png")
        plt.close()

    # distributions for each numeric feature (lots of plots)
    def feature_distributions(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(self.report_dir / f"{col}_distribution.png")
            plt.close()

    # sentiment-only distribution (kinda interesting)
    def sentiment_distribution(self):
        if "sentiment" not in self.df.columns:
            return
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df["sentiment"], kde=True, bins=30)
        plt.title("Sentiment Score Distribution")
        plt.tight_layout()
        plt.savefig(self.report_dir / "sentiment_distribution.png")
        plt.close()

    # scatterplot matrix — warning: kinda heavy
    def pairplot(self):
        try:
            sns.pairplot(
                self.df.select_dtypes(include=[np.number]).sample(200),
                diag_kind="kde"
            )
            plt.savefig(self.report_dir / "pairplot.png")
            plt.close()
        except:
            print("pairplot was skipped (dataset too big or missing data)")

    # average emotion levels across all songs (bar chart)
    def emotion_bars(self):
        emotions = [
            "fear", "anger", "anticipation", "trust", "surprise",
            "positive", "negative", "sadness", "disgust", "joy"
        ]
        if not all(e in self.df.columns for e in emotions):
            return
        emo_means = self.df[emotions].mean().sort_values()
        plt.figure(figsize=(10, 6))
        emo_means.plot(kind="barh", color="skyblue")
        plt.title("Average Emotion Scores Across All Songs")
        plt.xlabel("Score")
        plt.tight_layout()
        plt.savefig(self.report_dir / "emotion_bar_chart.png")
        plt.close()

    # sentiment comparison across artists (top 10)
    def artist_sentiment(self, top_n=10):
        if "artist" not in self.df.columns:
            return
        artist_sent = (
            self.df.groupby("artist")["sentiment"]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(x=artist_sent.values, y=artist_sent.index)
        plt.title(f"Top {top_n} Artists by Average Sentiment")
        plt.xlabel("Sentiment Score")
        plt.tight_layout()
        plt.savefig(self.report_dir / "artist_sentiment.png")
        plt.close()

    # shows how repetition relates to emotional tone
    def repetition_vs_sentiment(self):
        if "repetition_ratio" not in self.df.columns:
            return
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=self.df,
            x="repetition_ratio",
            y="sentiment",
            alpha=0.5
        )
        plt.title("Repetition Ratio vs Sentiment")
        plt.tight_layout()
        plt.savefig(self.report_dir / "repetition_vs_sentiment.png")
        plt.close()

    # vocab richness vs sentiment — honestly a cool relationship
    def richness_vs_sentiment(self):
        if "vocab_richness" not in self.df.columns:
            return
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=self.df,
            x="vocab_richness",
            y="sentiment",
            alpha=0.5
        )
        plt.title("Vocab Richness vs Sentiment")
        plt.tight_layout()
        plt.savefig(self.report_dir / "richness_vs_sentiment.png")
        plt.close()

    # emotion heatmap for a specific artist
    def emotion_heatmap_artist(self, artist_name):
        emotions = [
            "fear", "anger", "anticipation", "trust", "surprise",
            "positive", "negative", "sadness", "disgust", "joy"
        ]
        if not all(e in self.df.columns for e in emotions):
            return

        artist_df = self.df[self.df["artist"].str.lower() == artist_name.lower()]
        if artist_df.empty:
            return  # oops artist not found

        emo_matrix = artist_df[emotions]
        plt.figure(figsize=(12, 8))
        sns.heatmap(emo_matrix, cmap="viridis")
        plt.title(f"Emotion Heatmap for {artist_name}")
        plt.tight_layout()
        plt.savefig(self.report_dir / f"{artist_name}_emotion_heatmap.png")
        plt.close()
        
    # word cloud of ALL lyrics combined (very aesthetic)
    def wordcloud(self, max_words=100):
        if "clean_lyrics" not in self.df.columns:
            return

        full_text = " ".join(self.df["clean_lyrics"].dropna())

        wc = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        max_words=max_words
        ).generate(full_text)

        plt.figure(figsize=(16, 8))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(self.report_dir / "wordcloud.png")
        plt.close()


    # boxplots for numerical features
    def feature_boxplots(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=self.df[numeric_cols])
        plt.xticks(rotation=45)
        plt.title("Boxplots of Lyrical Features")
        plt.tight_layout()
        plt.savefig(self.report_dir / "feature_boxplots.png")
        plt.close()

if __name__ == "__main__":
    DF_PATH = "data/processed/spotify_lyrics_features.csv"
    df = pd.read_csv(DF_PATH).dropna()

    viz = Visualizer(df)

    viz.correlation_heatmap()
    viz.feature_distributions()
    viz.sentiment_distribution()
    viz.pairplot()
    viz.emotion_bars()
    viz.artist_sentiment()
    viz.repetition_vs_sentiment()
    viz.richness_vs_sentiment()
    viz.emotion_heatmap_artist("abba")  # just testing with abba
    viz.wordcloud()
    viz.feature_boxplots()

    print("all visualization files saved in /reports!") 
    