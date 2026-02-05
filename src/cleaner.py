import re
import pandas as pd
from pathlib import Path

class Cleaner:
    """
    Text cleaner for song lyrics with the following constraints:
    - Preserves newline characters to retain line-based features such as rhyme, avg. line length
    - Removes section headers like [Chorus], [Verse 1], [Bridge], etc.
    - Removes parentheticals like (spoken), (intro), etc.
    - Removes general punctuation except apostrophes.
    - Preserves apostrophes inside words (she's, don't, i'm)
    - Collapse excessive whitespace within each line while keeping line breaks.
    """

    def __init__(self):
        # No initialization needed for this cleaner
        pass

    def clean(self, lyrics: str) -> str:
        """
        Apply all cleaning steps to the given lyrics string.
        Parameters:
        lyrics (str): Raw lyrics text.

        Returns:
        str: Cleaned lyrics text.
        """
        lyrics = str(lyrics) #Ensuring input is a string

        #Applied cleaning steps in a sequence.
        lyrics = self._remove_section_headers(lyrics)
        lyrics = self._remove_parentheticals(lyrics)
        lyrics = self._remove_punctuation_but_keep_apostrophes(lyrics)
        lyrics = self._lowercase(lyrics)
        lyrics = self._normalize_whitespace_per_line(lyrics)

        return lyrics

    def _remove_section_headers(self, lyrics: str) -> str:
        # Remove section headers such as [Chorus], [Verse 1], [Bridge]
        return re.sub(r"\[.*?\]", " ", lyrics)

    def _remove_parentheticals(self, lyrics: str) -> str:
        # Remove parenthetical annotations like (spoken), (intro).
        return re.sub(r"\(.*?\)", " ", lyrics)

    def _remove_punctuation_but_keep_apostrophes(self, lyrics: str) -> str:
        # Remove all characters except letters, spaces, and apostrophes
        # Preserves newlines - \n for line-based analysis
        return re.sub(r"[^a-zA-Z'\s]", " ", lyrics)

    def _lowercase(self, lyrics: str) -> str:
        # Convert all text to lowercase for consistency
        return lyrics.lower()

    def _normalize_whitespace_per_line(self, lyrics: str) -> str:
        """
        Normalize whitespace within each line:
        - Collapse multiple spaces into one. 
        - Remove leading/trailing spaces.
        - Keep line breaks intact.
        """
        lines = lyrics.split("\n")
        cleaned_lines = []
        for line in lines:
            # Replaces multiple spaces with a single space and strips edges.
            line = re.sub(r"\s+", " ", line).strip()
            if line: #Skip empty lines.
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)


if __name__ == "__main__":
    # Define input and output file paths.
    RAW_PATH = Path("data/raw/spotify_millsongdata.csv")
    CLEAN_PATH = Path("data/cleaned/clean_spotify_lyrics.csv")

    # Read raw dataset from CSV
    df = pd.read_csv(RAW_PATH)

    cleaner = Cleaner()
    
    # Apply cleaning function to the 'text' column and create 'clean_lyrics'
    df["clean_lyrics"] = df["text"].astype(str).apply(cleaner.clean)

    print(f"Saving cleaned dataset to {CLEAN_PATH} ...")
    # Ensure output directory exists.
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    #Save cleaned dataset to CSV.
    df.to_csv(CLEAN_PATH, index=False)


