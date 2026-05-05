# Kaggle Competitions

Personal repository of Kaggle competition submissions. Each folder is a self-contained project with its own data, source code, and documentation.

---

## Projects

### [OrbitWars](./OrbitWars/)

**Competition**: [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)  
**Type**: Simulation / Game AI  
**Best score**: ~548 (leaderboard)

A real-time strategy game where agents compete to control planets orbiting a sun in a 100×100 continuous 2D space. The game lasts 500 turns; the player with the most total ships wins.

**What the agent does**
- Parses the game observation into planet and fleet dictionaries each turn
- Iterative orbital aim solver to lead moving targets accurately
- 2-ply branch search: generates multiple action candidates, simulates an enemy counter-move for each, and picks the branch with the best forward-rolled-out value
- Coordinated multi-source assaults with ETA-windowing to synchronize fleets
- Emergency defense reinforcement when planets are under threat

**Key files**
| File | Purpose |
|---|---|
| `main.py` | Submission entry point — `agent(obs)` is called each turn |
| `SRC_Documentation/README.md` | Full game rules and mechanics reference |
| `SRC_Documentation/agents.md` | Notes on other agents / strategies observed |

**Libraries**
- `math`, `collections` — standard library only (Kaggle environment restriction)

---

### [TitanicSurvivalSim](./TitanicSurvivalSim/)

**Competition**: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)  
**Type**: Binary Classification  
**Best score**: 77.5% accuracy (Kaggle public leaderboard)

Predicts which passengers survived the Titanic disaster. Classic beginner ML competition — the goal is to learn feature engineering and model selection on a small, well-understood dataset.

**What the model does**
1. Loads `train.csv` and `test.csv` from the `data/` folder
2. Exploratory analysis: missing values, survival rates by sex / class / embarkation
3. Correlation heatmap to rank features by relationship to survival
4. Feature engineering:
   - **Title** — extracted from passenger name (Mr, Mrs, Miss, Master, Rare)
   - **FamilySize** — SibSp + Parch + 1
   - **IsAlone** — 1 if traveling alone
   - **Has_Cabin** — 1 if a cabin number was recorded
5. scikit-learn `Pipeline` with median imputation (numeric) and one-hot encoding (categorical)
6. `GridSearchCV` over Random Forest hyperparameters (depth, estimators, min leaf size) using 5-fold cross-validation
7. Writes `submission.csv` with PassengerId and predicted Survived

**Key files**
| File | Purpose |
|---|---|
| `src/main.py` | Full ML workflow: EDA → feature engineering → training → submission |
| `data/train.csv` | Labeled training data (891 rows) |
| `data/test.csv` | Unlabeled test data (418 rows) |
| `submission.csv` | Most recent Kaggle submission file |

**Libraries**
| Library | Used for |
|---|---|
| `pandas` | Loading CSVs, data manipulation, groupby analysis |
| `numpy` | Numeric operations (indirect, via sklearn) |
| `matplotlib` | Survival rate bar charts, age distribution histogram |
| `seaborn` | Correlation heatmap |
| `scikit-learn` | Pipelines, imputers, OneHotEncoder, RandomForest, GridSearchCV |

---

## Setup

Both projects use the same Python interpreter. Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kaggle
```

To submit to Kaggle from the terminal you need a `~/.kaggle/kaggle.json` API token. Then:

```bash
# Titanic
cd TitanicSurvivalSim
kaggle competitions submit -c titanic -f submission.csv -m "your message"

# Orbit Wars
cd OrbitWars
kaggle competitions submit orbit-wars -f main.py -m "your message"
```
