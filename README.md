# MLB Proyecciones 🧢⚾

Advanced MLB game projection system powered by comprehensive statistical analysis.

## Features 🚀

- **Multi-Factor Analysis**: Combines pitcher efficiency, team offense/defense ratings, and recent performance
- **Log5 Win Probability**: Bill James log5 model combining team strength AND starting pitcher quality, with home field advantage
- **Real Park Factors**: Calibrated multi-season park factors keyed to real MLB StatsAPI venue IDs
- **Confidence Scoring**: Assesses projection reliability based on data completeness
- **Real-time Data**: Fetches live statistics from MLB Official StatsAPI
- **Telegram Integration**: Daily automated projections delivered via Telegram bot
- **Backtesting**: Calibration metrics (Brier, LogLoss, Accuracy), Platt/isotonic recalibration, and EV betting simulation

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
Advanced projection engine (v2) with:
- **Pitcher Rating** (0-100, league average = 50):
  - ERA and WHIP vs league average
  - K-BB (strikeout minus walk rate)
  - Quality starts percentage
- **Offense Rating** (0-100, league average = 50):
  - Batting average, OBP, SLG
  - Recent runs per game
- **Defense Rating** (0-100, league average = 50):
  - Team ERA and WHIP
- **Win Probability**: log5 model on team+pitcher expected win%, with home field advantage (~54%)
- **Total Runs Projection**: per-team expected runs from offense × opposing starter × opposing defense × park factor

#### 3. **main.py** 🎯
Orchestration and notification:
- Pipeline coordination
- Telegram message formatting
- Results persistence

#### 4. **backtest.py** 📈
Evaluation harness:
- Calibration metrics: Brier score, LogLoss, Accuracy, reliability table
- Recalibration: Platt scaling and isotonic regression
- EV-based betting simulation against decimal odds

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

# Backtest against historical results
python src/backtest.py --projections data/projections.csv --results data/historical_results.csv

# Run tests
pytest tests/
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

### Rating System (v2)
All ratings live on a **0-100 scale centered at 50** (league average), with ~15 points per standard deviation. (The previous version centered at 100 and clipped half the scale.)

### Win Probability Calculation
1. Estimate each team's expected win% vs an average opponent:
   - Team strength: (Offense × 0.45) + (Defense × 0.55), mapped to win% (±0.07 std)
   - Starter adjustment: ±0.003 win% per rating point above/below 50
2. Combine both win% via **log5** (Bill James): P(H beats A) = (H − H·A) / (H + A − 2·H·A)
3. Apply home field advantage in logit space (~54% historical home win rate)
4. Clip probability to [0.20, 0.80] (realistic MLB moneyline range)

(The previous version ignored starting pitchers entirely in win probability.)

### Total Runs Projection
1. League average base: 8.8 runs per game (4.4 per team)
2. Each team's expected runs = base × park factor × offense factor × opposing starter factor × opposing defense factor
3. Better starters/defenses **reduce** runs (direction bug fixed in v2)
4. Home team gets a small ~3% batting boost
5. Clip result to [5.5, 13.5]

### Park Factors
Keyed to **real MLB StatsAPI venue IDs** (e.g. Coors Field = 19, Yankee Stadium = 3313), with calibrated multi-season values (0.90 pitcher-friendly T-Mobile/Oracle to 1.32 Coors Field). Unknown venues fall back to 1.00. (The previous version used incorrect venue IDs, so almost every game silently fell back to 1.00.)

### Confidence Scoring
- Penalized if pitcher stats missing (-15%)
- Penalized if team stats missing (-10%)
- Penalized if starter unresolved / bullpen day (-10% per side)
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
- [x] Model performance backtesting (Brier/LogLoss/Accuracy + recalibration)
- [ ] Multiple projection algorithms ensemble
- [ ] Live in-game probability updates

## Error Handling 🛡️

- Automatic retries on API failures
- Graceful degradation with league averages
- Comprehensive logging
- Timeout protection (10s per request)
- Data validation before processing

## Precision Improvements (Model v2) 📊

✅ Win probability now includes starting pitcher quality via log5 (was: ignored)
✅ Ratings centered at 50 with full 0-100 range (was: centered at 100, half the scale clipped)
✅ Park factors keyed to real StatsAPI venue IDs with calibrated values (was: wrong IDs → silent 1.00 fallback)
✅ Total runs direction fixed: better pitching/defense reduces projected runs (was: inverted)
✅ Realistic probability range [0.20, 0.80] and home edge calibrated to ~54%
✅ Backtest import fixed (`python src/backtest.py` now works directly) + Accuracy metric added
✅ 9 sanity tests locking in the model's directional properties

## License 📄

MIT License - feel free to use and modify!

## Contact 📧

For questions or suggestions, open an issue on GitHub.
