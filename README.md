# 🏎️ F1 Lap Time & Classification Analysis

This project is an end-to-end Machine Learning pipeline designed to predict and classify Formula 1 race data. It utilizes the **CatBoost** library and advanced hyperparameter tuning to predict whether a driver will make a pit stop on the next lap (`PitNextLap`).

---

## 📊 Performance & Metrics

Since pit stops are relatively rare events (class imbalance), the model was optimized for the **F1-Score** rather than global accuracy. Evaluation on the holdout test set (`X_test`) yielded the following results:

*   **Overall Accuracy**: `89.47%`
*   **Class 1.0 (Pit Stop) F1-Score**: `0.73`
*   **Class 1.0 Recall**: `0.72` (Successfully captures 72% of all actual pit stops)
*   **Class 1.0 Precision**: `0.75` (When predicting a pit stop, the model is right 75% of the time)

### Full Classification Report:
```text
              precision    recall  f1-score   support

         0.0       0.93      0.94      0.93     70286
         1.0       0.75      0.72      0.73     17542

    accuracy                           0.89     87828
   macro avg       0.84      0.83      0.83     87828
weighted avg       0.89      0.89      0.89     87828
```

---

## 🧠 Feature Importance

According to CatBoost's internal feature importance calculation, the top driving factors behind the model's predictions are:
1.  **Year**: Represents global rule changes, performance baselines, and seasonal shifts.
2.  **Stint**: The current stint number (tires get older with higher stints, increasing pit probability).
3.  **TyreLife**: The physical age/wear of the current tire set.

---

## 🛠️ Tech Stack & Libraries

*   **Language**: Python 3.x
*   **Frameworks**: CatBoost, Scikit-learn
*   **Environment**: Kaggle / Jupyter Notebook
*   **Libraries**: NumPy, Pandas, Matplotlib, Pickle

---

## ⚙️ Installation & Setup

To set up the environment and install all dependencies, run the following commands:

### 1. Create a virtual environment
```bash
python -m venv venv
```

### 2. Activate the virtual environment
*   **On Windows (Command Prompt):**
    ```cmd
    venv\Scripts\activate
    ```
*   **On Windows (PowerShell):**
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
*   **On macOS / Linux (and Git Bash):**
    ```bash
    source venv/bin/activate
    ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏆 Results

*   🥇 **Kaggle Competition Result**: Participated in the "Predicting F1 Pit Stops" competition and achieved a performance score of **0.83041**.
*   📈 **Rank**: 757th out of 804 participants.
*   🚀 **Improvement**: Boosted the model performance from **0.803** to **0.830** through hyperparameter tuning (`RandomizedSearchCV`) and automated handling of categorical features (`['Driver', 'Compound', 'Race']`).

---

## 🚀 Production Usage

The optimal estimator is serialized to a production-ready file: `F1Classification_model.pkl`. To load the model and make predictions on a target dataset:

```python
import pickle
import pandas as pd

# Load the model
with open('F1Classification_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your evaluation data
df_test = pd.read_csv('test.csv')
X_test = df_test.drop(['id', 'PitNextLap'], axis=1, errors='ignore')

# Generate predictions
predictions = model.predict(X_test)
```
