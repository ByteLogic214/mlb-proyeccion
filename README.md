# MLB Proyecciones 🧢⚾

Advanced MLB game projection system powered by comprehensive statistical analysis.

## Features 🚀

- **Multi-Factor Analysis**: Combines pitcher efficiency, team offense/defense ratings, and recent performance
- **Intelligent Park Factors**: Stadium-specific run adjustment for all 30 MLB parks
- **Confidence Scoring**: Assesses projection reliability based on data completeness
- **Real-time Data**: Fetches live statistics from MLB Official StatsAPI
- **Telegram Integration**: Daily automated projections delivered via Telegram bot

## System Architecture 🏗️

```
data_fetcher.py → predictor.py → main.py → Telegram Bot
     ↓               ↓             ↓
  MLB API      Advanced Rating  Message
              System & Calcs   Formatting
```

### Core Components

#### 1. **data_fetcher.py** 📊
Fetches comprehensive MLB data:
- Daily schedule from MLB StatsAPI
- Pitcher statistics (ERA, WHIP, K9, BB9, Quality Starts %)
- Team statistics (ERA, WHIP, AVG, OBP, SLG)
- Recent performance trends (last 14 days)
- Retry logic and robust error handling

#### 2. **predictor.py** 🧮
Advanced projection engine with:
- **Pitcher Rating System** (0-100 scale):
  - ERA normalized vs league average
  - WHIP efficiency
  - Strikeout-to-Walk ratio
  - Quality starts percentage
- **Offense Rating** (0-100 scale):
  - Batting average, OBP, SLG
  - Recent runs per game
- **Defense Rating** (0-100 scale):
  - Team ERA and WHIP
- **Win Probability**: Logistic regression model with home field advantage
- **Total Runs Projection**: Park factors + pitcher efficiency + team offense metrics

#### 3. **main.py** 🎯
Orchestration and notification:
- Pipeline coordination
- Telegram message formatting
- Results persistence

## Installation 🛠️

```bash
# Clone repository
git clone <repo-url>
cd mlb-proyeccion

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"
```

## Usage 📱

```bash
# Run full pipeline
python src/main.py

# Run just data fetching
python src/data_fetcher.py

# Run just projections
python src/predictor.py
```

## Output Format 📤

### Console Output
Formatted projections with:
- Pitcher names and efficiency ratings
- Team offensive/defensive ratings
- Win probability percentages
- Projected total runs
- Projection confidence score

### Saved Data
- `data/today_games.csv`: Raw game data with all statistics
- `data/projections.csv`: Final projections with all ratings and probabilities

## Algorithm Details 🔬

### Win Probability Calculation
1. Calculate individual ratings for each team's offense/defense
2. Compute combined strength: (Offense × 0.45) + (Defense × 0.55)
3. Apply logistic function to strength differential
4. Add home field advantage (~3.5%)
5. Clip probability to [0.1, 0.9] range

### Total Runs Projection
1. Start with league average (8.8 runs)
2. Apply park factor (0.95-1.15 range)
3. Adjust by pitcher quality metrics
4. Scale by average offensive rating
5. Ensure result in [4.0, 15.0] range

### Confidence Scoring
- Penalized if pitcher stats missing (-15%)
- Penalized if team stats missing (-10%)
- Maximum confidence: 1.0 (100%)

## League Averages (2024 MLB) 📈

| Metric | Value |
|--------|-------|
| ERA | 4.09 |
| WHIP | 1.28 |
| K9 | 9.2 |
| BB9 | 2.8 |
| Batting AVG | 0.244 |
| OBP | 0.314 |
| SLG | 0.390 |
| Runs Per Game | 8.8 |

## Park Factors 🏟️

Included for all 30 MLB stadiums (0.95-1.15 range):
- Higher factors = hitter-friendly parks (Colorado, Kansas City, Miami)
- Lower factors = pitcher-friendly parks (Fenway, Nationals Park)

## Dependencies 📦

```
requests==2.32.3
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
xgboost==2.0.3
```

## Future Enhancements 🔮

- [ ] Season trend weighting
- [ ] Head-to-head historical matchup data
- [ ] Weather impact analysis
- [ ] Injury report integration
- [ ] Model performance backtesting
- [ ] Multiple projection algorithms ensemble
- [ ] Live in-game probability updates

## Error Handling 🛡️

- Automatic retries on API failures
- Graceful degradation with league averages
- Comprehensive logging
- Timeout protection (10s per request)
- Data validation before processing

## Precision Improvements (This Update) 📊

✅ Advanced multi-factor analysis instead of simple ERA comparison
✅ Pitcher quality rating system incorporating 5+ metrics
✅ Separate offensive and defensive team ratings
✅ Complete park factor database for all 30 MLB stadiums
✅ Recent performance trend integration (14-day lookback)
✅ Home field advantage weighting (3.5%)
✅ Confidence scoring for result reliability
✅ Enhanced data fetching with retry logic
✅ Comprehensive error handling and validation
✅ Better message formatting with detailed metrics

## License 📄

MIT License - feel free to use and modify!

## Contact 📧

For questions or suggestions, open an issue on GitHub.
