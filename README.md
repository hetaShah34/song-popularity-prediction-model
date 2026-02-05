## Documentation

Submission for Team 10B //BANA

Raw lyrics are loaded in cleaner.py, cleaned there, and exported to data/cleaned/clean_spotify_lyrics.csv.
The cleaned lyrics are then used in features.py, where features are extracted.
The feature script samples 3000 songs (random_state 42), computes all linguistic and emotional features, and exports the processed dataset to data/processed/spotify_lyrics_features.csv.
The processed features file is then used for visualizations in visualizer.py, which saves all generated plots into reports/visuals/.
The same processed features file is also used inside the Jupyter notebooks (Modeling and Visuals.ipynb and popularity.ipynb), where the machine learning models and popularity analysis are run.
Because each stage produces an output that becomes the input for the next stage, the project contains multiple files to keep the workflow clear and reproducible. Multiple people worked on different stages, so separating the steps into their own scripts prevented conflicts and made the pipeline easier to manage.

## Dataset

The dataset used in this project is not included in the repository due to GitHub file size limits.

To run the project locally, place the data in a `data/` directory with the following structure:

- data/raw
- data/cleaned
- data/processed

The original dataset file data/raw was sourced from Kaggle.

Final Project/
│
├── README.md                    ← main project documentation  
│
├── data/
│   ├── raw/
│   │   └── spotify_millsongdata.csv
│   ├── cleaned/
│   │   └── clean_spotify_lyrics.csv
│   └── processed/
│       └── spotify_lyrics_features.csv
│
├── src/
│   ├── cleaner.py
│   ├── features.py
│   ├── visualizer.py
│   ├── Modeling and Visuals.ipynb
│   └── popularity.ipynb
│
├── reports/
│   └── visuals/
│       ├── emotion_bar_chart.png
│       ├── joy_distribution.png
│       ├── trust_distribution.png
│       └── (other generated visuals)
│
└── models/
    └── (empty — no .pkl file required)
