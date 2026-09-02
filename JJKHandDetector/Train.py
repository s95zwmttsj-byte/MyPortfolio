import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def train(csv_file, model_file):
    df = pd.read_csv(csv_file)
    print(df.head())        # shows first 5 rows
    print(df.columns)       # shows all column names
    print(df["Label"].unique())  # shows what values are in Label column

    X = df.drop("Label", axis=1)
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    print(f"{model_file} accuracy: {accuracy:.1f}%")

    f = open(model_file, "wb")
    pickle.dump(model, f)
    f.close()
    print("Saved to " + model_file)

train("one_jjk_data.csv", "model_one_hand.pkl")
