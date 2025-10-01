from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def load_data():
    # Replace with real path; keep raw data outside repo if private
    # For demo, synthesize a tiny dataset
    n=500
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'experience_level': rng.integers(1,5, size=n),
        'remote_ratio': rng.choice([0,50,100], size=n),
        'company_size': rng.integers(50, 5000, size=n),
        'country_idx': rng.integers(0, 5, size=n),
        'salary': rng.normal(85000, 15000, size=n)
    })
    return df

def main():
    df = load_data()
    X = df.drop(columns=['salary'])
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    model = RandomForestRegressor(n_estimators=200, random_state=7)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("MAE:", round(float((abs(y_test - y_pred)).mean()), 2))
    print("R2 :", round(float(r2_score(y_test, y_pred)), 3))
    Path("models").mkdir(exist_ok=True, parents=True)
    joblib.dump(model, "models/rf_model.joblib")
    print("Saved model to models/rf_model.joblib")

if __name__ == "__main__":
    main()
