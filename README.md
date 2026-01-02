# ⚽ Football Midfielder Role Classification

Classifying midfielder playing styles in Europe's top 5 leagues using machine learning.

## Overview

This project uses **7 seasons of player statistics** (2017-18 to 2023-24) from Europe's Big 5 leagues to predict tactical roles for midfielders. Built as a final project for **UIT's CSBU204 AI & Machine Learning** course.

### What This Does

The model analyzes midfielder statistics (goals, tackles, passes, etc.) and classifies them into one of four tactical roles:

- **AM** (Attacking Midfielder) - Creative players who operate in advanced positions
- **DM** (Defensive Midfielder) - Defensive anchors who shield the backline
- **DLP** (Deep-Lying Playmaker) - Deep orchestrators who control tempo
- **B2B** (Box-to-Box) - All-action players contributing in both attack and defense

## Dataset

**Source:** FBref (via web scraping)  
**Size:** 18,243 player-seasons across 7 years  
**Leagues:** Premier League, Bundesliga, La Liga, Serie A, Ligue 1

**Role Distribution:**
- AM: 27.1% (1,701 players)
- DM: 26.9% (1,654 players)  
- DLP: 24.5% (1,011 players)
- B2B: 21.5% (940 players)

**Features:** 18 behavioral metrics (per 90 minutes):
- Attacking: shots, shot-creating actions
- Defensive: tackles, interceptions, blocks, clearances, defensive touches
- Passing: completion %, progressive passes, key passes, passes into final third
- Carrying: progressive carries, carries into final third and penalty area

**Filtering Criteria:**
- Position = MF only
- Minimum 450 minutes played per season

## Project Structure

```
Final Project/
├── Football_MF_Roles_Classification.ipynb  # Main analysis notebook
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation
├── datasets/                                # 7 seasons of data
│   ├── cleaned_2017-18.csv
│   ├── cleaned_2018-19.csv
│   ├── cleaned_2019-20.csv
│   ├── cleaned_2020-21.csv
│   ├── cleaned_2021-22.csv
│   ├── cleaned_2022-23.csv
│   └── cleaned_2023-24.csv
├── images/                                  # Generated plots
└── labs/                                    # Lab exercises
```

## Methodology

### 1. Data Preparation
- Merged 7 seasons of data
- Filtered for midfielders (MF position only)
- Normalized features to "per 90 minutes"
- Handled missing values and outliers

### 2. Feature Engineering
Created 18 normalized metrics grouped into:
- **Attacking behavior** (2 features)
- **Defensive behavior** (6 features)
- **Passing & progression** (6 features)
- **Dribbling & carrying** (3 features)

### 3. Role Assignment
Used **rule-based classification** with percentile thresholds based on composite scores:
- **Attacking Score:** goals, assists, xG, shots, shot-creating actions
- **Defensive Score:** tackles, interceptions, blocks, clearances, defensive touches
- **Playmaking Score:** progressive passes, key passes, pass completion

Assignment logic:
- High attacking + high playmaking → **AM**
- High defensive + high defensive touches → **DM**
- High playmaking + low attacking → **DLP**
- Balanced across all dimensions → **B2B**

### 4. Model Training
Trained and compared three models:
- **Logistic Regression** (baseline)
- **Random Forest**
- **XGBoost**

Evaluation metrics:
- Accuracy
- Weighted F1-score
- Per-class precision/recall/F1-score
- Confusion matrices
- 5-fold cross-validation
- Feature importance analysis

## Key Results

### Model Performance

| Model | Test Accuracy | F1-Score |
|-------|--------------|----------|
| **Logistic Regression** | **93.8%** | **93.7%** |
| XGBoost | 90.9% | 90.8% |
| Random Forest | 89.5% | 89.4% |

**Winner:** Logistic Regression achieved the best performance with excellent generalization.

### Key Insights

- **DM and AM are easiest to classify** (95% F1-score) due to distinct behavioral patterns
- **B2B is hardest to classify** (92% F1-score) as it overlaps with multiple roles
- **Top discriminating features (XGBoost):**
  1. shots_p90
  2. pass_completion_pct
  3. clearances_p90
  4. tackles_attempted_p90
  5. interceptions_p90

### Misclassification Patterns

- B2B sometimes confused with DLP (similar passing patterns)
- DLP occasionally misclassified as B2B (when more active defensively)
- Very few confusions between AM and DM (opposite ends of spectrum)

## How to Run

### Requirements
- Python 3.11+
- Jupyter Notebook

### Setup
```bash
# Navigate to project directory
cd "Final Project"

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook Football_MF_Roles_Classification.ipynb
```

### Running the Analysis
1. Open the notebook
2. Run all cells in order (`Cell > Run All`)
3. View results and visualizations inline

## Technologies Used

- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, xgboost
- **Environment:** Jupyter Notebook

## Limitations

1. **Role Assignment Subjectivity:** Roles assigned algorithmically, not by coaches/analysts
2. **No Temporal Tracking:** Each player-season treated independently
3. **Team Context Ignored:** Playing style influenced by team tactics not captured
4. **Positional Fluidity:** Players who switch roles mid-game/season not captured
5. **Feature Coverage:** Some tactical aspects (pressing intensity, positioning discipline) not in data

## Future Improvements

- Add ground truth labels from football analysts for validation
- Include event-level data and spatial data (heatmaps, pass networks)
- Use temporal models (LSTM) to track role evolution across seasons
- Incorporate team context (formation, playing style, opponent strength)
- Implement multi-label classification for hybrid roles
- Expand to other positions (defenders, forwards, wingers)

## References

- Data source: [FBref.com](https://fbref.com/)
- Course: UIT CSBU204 - Artificial Intelligence and Machine Learning

