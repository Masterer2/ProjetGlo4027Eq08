from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler


PATH_TO_DATA = '../data/Comments.csv'


def main():
    # Create dataset
    df = pd.read_csv(PATH_TO_DATA)
    df['created_time'] = pd.to_datetime(df['created_time']).dt.hour
    df['has_reply'] = df['id'].isin(df['parent']).astype(int)
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(
        df[['IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'SEVERE_TOXICITY', 'THREAT', 'TOXICITY', 'created_time']],
        df['has_reply'])
    X = pd.DataFrame(X_resampled,
                     columns=['IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'SEVERE_TOXICITY', 'THREAT', 'TOXICITY',
                              'created_time'])
    y = pd.Series(y_resampled, name='has_reply')

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    mean_cv_score = cv_scores.mean()
    print(f"Mean cross-validation score: {mean_cv_score}")

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)

    # Show confusion matrix
    confusion = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion)
    disp.plot()
    plt.show()
    print(f"Confusion matrix: {confusion}")
    print(f"True positive: {confusion[1, 1]}")
    print(f"True negative: {confusion[0, 0]}")
    print(f"False positive: {confusion[0, 1]}")
    print(f"False negative: {confusion[1, 0]}")
    print(f"Precision: {confusion[1, 1] / (confusion[1, 1] + confusion[0, 1])}")
    print(f"Recall: {confusion[1, 1] / (confusion[1, 1] + confusion[1, 0])}")
    print(f"Error rate: {(confusion[1, 0] + confusion[0, 1]) / confusion.sum()}")
    print(f"F1 score: {2 * confusion[1, 1] / (2 * confusion[1, 1] + confusion[1, 0] + confusion[0, 1])}")
    print(f"Accuracy: {(confusion[1, 1] + confusion[0, 0]) / confusion.sum()}")
    print(f"Balanced accuracy: {(confusion[1, 1] / (confusion[1, 1] + confusion[1, 0]) + confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])) / 2}")
    print(f"ROC AUC: {model.score(X_test, y_test)}")


if __name__ == '__main__':
    main()
