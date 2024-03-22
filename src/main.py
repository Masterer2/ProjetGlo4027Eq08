from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler


PATH_TO_DATA = '../data/Comments.csv'
PATH_TO_NEW_DATA = '../data/Test.csv'


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

    # Calculate metrics
    true_positive = confusion[1, 1]
    true_negative = confusion[0, 0]
    false_positive = confusion[0, 1]
    false_negative = confusion[1, 0]
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    error_rate = (false_negative + false_positive) / confusion.sum()
    f1_score = 2 * true_positive / (2 * true_positive + false_negative + false_positive)
    accuracy = (true_positive + true_negative) / confusion.sum()
    balanced_accuracy = (recall + true_negative / (confusion[0, 0] + confusion[0, 1])) / 2
    evaluation = true_positive / (true_positive + false_positive + false_negative)
    print(f"True positive: {true_positive}")
    print(f"True negative: {true_negative}")
    print(f"False positive: {false_positive}")
    print(f"False negative: {false_negative}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Error rate: {error_rate}")
    print(f"F1 score: {f1_score}")
    print(f"Accuracy: {accuracy}")
    print(f"Balanced accuracy: {balanced_accuracy}")
    print(f"Evaluation: {evaluation}")

    if evaluation > 0.93:
        # Classify new data
        new_data = pd.read_csv(PATH_TO_NEW_DATA)
        new_data['created_time'] = pd.to_datetime(new_data['created_time']).dt.hour
        test_pred = model.predict(new_data[['IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'SEVERE_TOXICITY', 'THREAT',
                                            'TOXICITY', 'created_time']])
        with open('../data/predictions.txt', 'w') as f:
            for idx, pred in enumerate(test_pred):
                if pred == 1:
                    f.write(f"{new_data.iloc[idx]['id']}\n")


if __name__ == '__main__':
    main()
