import pandas as pd
import numpy as np
import re
import spacy
import nltk
from collections import Counter
from nrclex import NRCLex
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import cmudict

# NLTK resources required by this script:
# Vader lexicon for sentiment analysis (SentimentIntensityAnalyzer)
# CMU Pronouncing Dictionary for rhyme detection
# These will be downloaded on first run (cached afterward).

nltk.download("vader_lexicon", quiet=True)
nltk.download("cmudict", quiet=True)

class Features:
    """
    Extracts linguistic and emotional features from song lyrics.
    This class computes:
    - Token level features (after lemmatization and stopword removal)
        * word frequency
        * total words
        * unique words
        * vocabulary richness (unique/total)
    - Line-level features
        * average line length (in words)
        * rhyme density (based on CMU phonemes of consecutive line endings)
    - Repetition metrics
        * repetition ratio ((total-unique)/total)
    - Sentiment (VADER compound score)
    - Emotion distribution (NRCLex: fear, anger, anticipation, trust, surprise,
      positive, negative, sadness, disgust, joy)
    """

    def __init__(self):
        # Load a lightweight spaCy English model.
        # We disable NER and parser to speed up tokenization & lemmatization.

        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        # Start from spaCy's built-in English stopwords set.
        self.stopwords = spacy.lang.en.stop_words.STOP_WORDS.copy()

        # Add music-specific fillers/interjections often present in lyrics,
        # which we generally want to ignore for analysis (domain-specific)
        custom_music_stopwords = {
            "oh", "yeah", "ooh", "ah", "na", "la", "woo", "hey", "ha",
            "uh", "mmm", "whoa", "yah", "ay", "sha", "doo", "dum", "bam",
            "ba", "da", "pa", "fa", "ho", "woah", "yea", "ye", "mm",
            "like", "to"
        }
        self.stopwords.update(custom_music_stopwords)

        #Initialize sentiment analyzer (VADER) and CMU pronouncing dictionary.
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.cmu = cmudict.dict()

        # Fixed set of emotion keys we will report
        self.emotion_keys = [
            "fear", "anger", "anticipation", "trust",
            "surprise", "positive", "negative",
            "sadness", "disgust", "joy"
        ]

    #Text preprocessing: lemmatize, lowercase, drop stopwords, and non-alpha.
    def _preprocess(self, lyrics: str):
        """
        Tokenize and normalize lyrics using spaCy:
        - keep only alphabetic tokens (drop numbers/punctuation)
        - lemmatize and lowercase
        - remove stopwords (including custom music fillers)

        Returns:
            List[str]: cleaned tokens suitable for word-level features.
        """
        doc = self.nlp(lyrics)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and token.lemma_.lower() not in self.stopwords
        ]
        return tokens

    #Word frequency and counts
    def word_frequency(self, lyrics: str):
        """
        Compute lemma-level word frequency after preprocessing.
        Returns:
            Counter: mapping token -> count
        """
        tokens = self._preprocess(lyrics)
        return Counter(tokens)

    def _unique_words(self, lyrics: str) -> int:
        """
        Count unique tokens after preprocessing.
        """
        tokens = self._preprocess(lyrics)
        return len(set(tokens))

    def _total_words(self, lyrics: str) -> int:
        """
        Count total tokens after preprocessing.
        """
        tokens = self._preprocess(lyrics)
        return len(tokens)

    def _vocab_richness(self, lyrics: str) -> float:
        """
        Vocabulary richness = unique tokens / total tokens
        Retuns 0.0 if there are no tokens
        """
        tokens = self._preprocess(lyrics)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)


    # Line-based features
    def _avg_line_len(self, lyrics: str) -> float:
        """
        Average line length measured in words not characters
        - Splits on newline
        - Ignores empty/whitespace-only lines

        Returns:
            float: average number of words per non-empty line.
        """
        lines = [line.strip().split() for line in lyrics.split("\n") if line.strip()]
        if not lines:
            return 0.0
        lengths = [len(words) for words in lines]
        return float(sum(lengths)) / len(lengths)


    def _repetition_ratio(self, lyrics: str) -> float:
        """
        Measures repetition in the token stream:
        repetition_ratio = (total_tokens - unique_tokens) / total_tokens

        Returns 0.0 if there are no tokens
        """
        tokens = self._preprocess(lyrics)
        if not tokens:
            return 0.0
        unique = len(set(tokens))
        total = len(tokens)
        return float(total - unique) / total


    def _rhyme_density(self, lyrics: str) -> float:
        """
        Approximate rhyme density across consecutive lines:
        - Take the last word of each non-empty line.
        - Clean to letters (ASCII a-z) for CMU dictionary lookup.
        - Compare consecutive pairs (line i vs line i+1)
        - Count a rhyme if last two phonemes match for any pronunciation pair
        
        Returns:
            float: rhymed_pairs / total_pairs across consecutive lines.
        """
        lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
        if len(lines) < 2:
            return 0.0

        endings = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            last = parts[-1].lower()
            # Strip everything except letters for CMU lookup (cmudict is ASCII-based)
            last_clean = re.sub(r"[^a-z]", "", last)
            if last_clean:
                endings.append(last_clean)

        if len(endings) < 2:
            return 0.0

        rhymes = 0
        total_pairs = 0

        # Compare each consecutive pair of line-ending words
        for i in range(len(endings) - 1):
            w1, w2 = endings[i], endings[i + 1]
            total_pairs += 1
            if self._words_rhyme(w1, w2):
                rhymes += 1

        return float(rhymes) / total_pairs if total_pairs > 0 else 0.0

    def _words_rhyme(self, w1: str, w2: str) -> bool:
        """
        Check rhyme between two words using CMU pronunciations:
        - IF either word is not in the CMU dict, return False
        - Compare all pronunciation variants; if any share the last two phonemes, treat as a rhyme

        Returns:
            bool: True if a rhyme is detected, else False
        """
        if w1 not in self.cmu or w2 not in self.cmu:
            return False

        w1_phones = self.cmu[w1] #list of posssible pronunciations
        w2_phones = self.cmu[w2]

        # compare last 2 phonemes across pronunciation pairs
        for p1 in w1_phones:
            for p2 in w2_phones:
                if len(p1) >= 2 and len(p2) >= 2 and p1[-2:] == p2[-2:]:
                    return True
        return False

    # Emotion and sentiment features
    def _emotion(self, lyrics: str):
        """
        Compute normalized emotion scores using NRClex.
        - raw_emotion_scores returns count for detected emotions
        - we normalize by total count so scores sum to ~1 across known keys
        Returns:
            Dict[str, float]: emotion scores for keys in self.emotion_keys
        """
        emo = NRCLex(lyrics)
        raw = emo.raw_emotion_scores

        # Initialize all target emotions to zero
        scores = {k: 0.0 for k in self.emotion_keys}

        total = sum(raw.values())
        if total > 0:
            for k, v in raw.items():
                if k in scores:
                    scores[k] = v / total   # normalize to proportions

        return scores

   
    def _sentiment(self, lyrics: str) -> float:
        """
        VADER sentiment compound score in [-1.0, 1.0]
        """
        return self.sentiment_analyzer.polarity_scores(lyrics)["compound"]

    
    def analyze_song(self, lyrics: str):
        #Compute all base and emotion features for a simple lyrics string.
        base_features = {
            "vocab_richness": self._vocab_richness(lyrics),
            "sentiment": self._sentiment(lyrics),
            "repetition_ratio": self._repetition_ratio(lyrics),
            "rhyme_density": self._rhyme_density(lyrics),
            "avg_line_len": self._avg_line_len(lyrics),
            "unique_word_count": self._unique_words(lyrics),
            "total_word_count": self._total_words(lyrics),
        }

        emo_features = self._emotion(lyrics)

        #Merge and return a single dict.
        base_features.update(emo_features)
        return base_features

    def analyze_songs(self, df: pd.DataFrame):
        """
        Apply feature extraction to a df
        Assumes there is a clean lyrics column produced by cleaner
        """
        # assumes df has 'clean_lyrics'
        feature_rows = df["clean_lyrics"].apply(self.analyze_song)
        feature_df = pd.DataFrame(feature_rows.tolist())
        return pd.concat([df, feature_df], axis=1)


if __name__ == "__main__":
    # PAths for input (cleaned lyrics) and output (features)
    CLEAN_PATH = "data/cleaned/clean_spotify_lyrics.csv"
    OUT_PATH = "data/processed/spotify_lyrics_features.csv"

    df = pd.read_csv(CLEAN_PATH)
    # Sample subset for faster iterations and testing
    df = df.sample(n=3000, random_state=42).reset_index(drop=True)

    f = Features()

    out = f.analyze_songs(df)

    # to fill any remaining NaNs with zeros (esp. emotion keys not present)
    out = out.fillna(0)

    out.to_csv(OUT_PATH, index=False)

