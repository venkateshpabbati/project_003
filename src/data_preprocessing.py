import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    X = df.drop(columns=['medv'])  # Features
    y = df['medv']  # Target variable

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test