import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.dummy import DummyClassifier



train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")
test_df = pd.read_csv("test.csv")

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(train_df["text"])
X_val = vectorizer.transform(val_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train, y_val, y_test = train_df["label"], val_df["label"], test_df["label"]


def eval(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # Making predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Calculating accuracy and balanced accuracy
    accuracy_train = accuracy_score(y_train, y_pred_train)
    balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)

    accuracy_val = accuracy_score(y_val, y_pred_val)
    balanced_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)

    # Printing the results
    print(f"Training Accuracy: {accuracy_train * 100:.2f}%")
    print(f"Validation Accuracy: {accuracy_val * 100:.2f}%")
    print(f"Test Accuracy: {accuracy_test * 100:.2f}%")




# Create a dummy classifier with the strategy to predict the most frequent class
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)

eval(dummy_clf, X_train, y_train, X_val, y_val, X_test, y_test)

print('-'*60)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
eval(model, X_train, y_train, X_val, y_val, X_test, y_test)