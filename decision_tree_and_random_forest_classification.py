from itertools import cycle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from yellowbrick.classifier import ROCAUC

RANDOM_SEED = 0
TEST_DATA_PERCENTAGE = 0.3

def read_liver(
        test_data_percentage: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv('../data/indian_liver_patient_sort.csv')
    df.drop('Age', axis=1, inplace=True)

    # encode categorical columns
    categorical_columns = [
        'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
        'Total_Proteins', 'Albumin'
    ]
    encoder = LabelEncoder()
    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])

    # split train test
    X = df.drop('Liver_Disease', axis=1).to_numpy()
    y = df['Liver_Disease'].to_numpy()
    return train_test_split(X, y, test_size=test_data_percentage, random_state=RANDOM_SEED)

def read_gender(
        test_data_percentage: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv('../data/gender_classification.csv')
    X = df.drop('gender', axis=1).to_numpy()
    y = df['gender'].to_numpy()
    return train_test_split(X, y, test_size=test_data_percentage, random_state=RANDOM_SEED)

def read_sign_mnist(
        test_data_percentage: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv('../data/sign_mnist_train.csv')
    X = df.drop('label', axis=1).to_numpy()
    y = df['label'].to_numpy()
    return train_test_split(X, y, test_size=test_data_percentage, random_state=RANDOM_SEED)

def choice(y):
    dataset = []
    match y:
        case '1':
            dataset = read_liver(TEST_DATA_PERCENTAGE)
            print(dataset)
            classification(dataset, False)
        case '2':
            dataset = read_gender(TEST_DATA_PERCENTAGE)
            print(dataset)
            classification(dataset, False)
        case '3':
            dataset = read_sign_mnist(TEST_DATA_PERCENTAGE)
            print(dataset)
            classification(dataset, True)
        case _:
            print("ERROR")
            return



def classification(data_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], multiclass: bool):
    X_train, X_test, y_train, y_test = data_set
    criterion_range = ["gini", "entropy"]
    splitter_range = ["best", "random"]
    max_depth_range = [30, 25, 20, 18, 15, 13, 10, 8, 5, 3, 2, 1]



    print("decision tree")
    print()
    best_acc = 0
    best_params = {}
    best_classifier = None

    for max_depth in max_depth_range:
        tree = DecisionTreeClassifier(random_state=RANDOM_SEED,
                                      max_depth=max_depth)
        tree.fit(X_train, y_train)
        train_acc = (tree.score(X_train, y_train))
        acc = tree.score(X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_classifier = tree
            best_params = {
                "max depth": max_depth
            }

        print("max_depth:", max_depth, "\ttrain_acc:",
              train_acc, "\ttest_acc:", acc)




    for criterion in criterion_range:
        tree = DecisionTreeClassifier(random_state=RANDOM_SEED,
                                      criterion=criterion)
        tree.fit(X_train, y_train)
        train_acc = (tree.score(X_train, y_train))
        acc = tree.score(X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_classifier = tree
            best_params = {
                "criterion": criterion
            }

        print("criterion:", criterion, "\ttrain_acc:",
              train_acc, "\ttest_acc:", acc)



    for splitter in splitter_range:
        tree = DecisionTreeClassifier(random_state=RANDOM_SEED,
                                      splitter=splitter)
        tree.fit(X_train, y_train)
        train_acc = (tree.score(X_train, y_train))
        acc = tree.score(X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_classifier = tree
            best_params = {
                "splitter": splitter
            }

        print("splitter:", splitter, "\ttrain_acc:",
              train_acc, "\ttest_acc:", acc)

    print()

# find best random forest

    n_estimators_range = [10, 20, 50, 80, 100, 200, 500]

    print("random forest")

    for n_estimators in n_estimators_range:
        forest = RandomForestClassifier(random_state=47,
                                        n_jobs=-1,
                                        n_estimators=n_estimators)
        forest.fit(X_train, y_train)
        train_acc = (forest.score(X_train, y_train))
        acc = forest.score(X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_classifier = forest
            best_params = {
                "n_estimators": n_estimators
            }

        print("n_estimators:", n_estimators, "\ttrain_acc:",
              train_acc, "\ttest_acc:", acc)


    print()
    print("best accuracy: ", best_acc)
    print("best params: ", best_params)

    #CONFUSION MATRIX

    print()
    print("CONFUSION MATRIX")
    y_prediction = best_classifier.predict(data_set[1])
    confusion = metrics.confusion_matrix(data_set[3], y_prediction)
    confusion_multilabel = metrics.multilabel_confusion_matrix(data_set[3], y_prediction)
    print(confusion)
    #print(confusion_multilabel)
    print("True Negatives: ", confusion[0][0], "False Negatives: ", confusion[1][0], "True Positives: ", confusion[1][1], "False Positives: ", confusion[0][1])
    print()

    #EVALUATION

    #sensitivity
    print("Sensitivity")
    for i in range(len(confusion)):
        if(sum(confusion[i] != 0)):
            sensitivity = confusion[i][i]/sum(confusion[i])
            print(best_classifier.classes_[i], ": ", sensitivity)
        elif(confusion[i][i] == 0):
            print(best_classifier.classes_[i], ": Not a number")
        else:
            print(best_classifier.classes_[i], ": infinity")
    print()

    #specifity
    print("Specifity")

    for i in range(len(confusion)):
        tnfp = 0
        fp = 0
        for ii in range(len(confusion)):
            if(ii == i):
                continue
            tnfp += sum(confusion[ii])
            fp += confusion[ii][i]
        tn = tnfp - fp
        if (tnfp != 0):
            specifity = tn/tnfp
            print(best_classifier.classes_[i], ": ", specifity)
        elif (tn == 0):
            print(best_classifier.classes_[i], ": Not a number")
        else:
            print(best_classifier.classes_[i], ": infinity")
    print()

    #accuracy
    print("Accuracy")

    for i in range(len(confusion)):
        tnfp = 0
        tp = confusion[i][i]
        fp = 0
        for ii in range(len(confusion)):
            if(ii == i):
                continue
            tnfp += sum(confusion[ii])
            fp += confusion[ii][i]
        tn = tnfp - fp
        sumconfusion = sum(list(map(sum, confusion)))
        if sumconfusion != 0:
            accuracy = (tp + tn)/(sumconfusion)
            print(best_classifier.classes_[i], ": ", accuracy)
        elif ((tp + tn) == 0):
            print(best_classifier.classes_[i], ": Not a number")
        else:
            print(best_classifier.classes_[i], ": infinity")

    print()


    #precision
    print("Precision")

    for i in range(len(confusion)):
        tp = confusion[i][i]
        fp = 0
        for ii in range(len(confusion)):
            if(ii == i):
                continue
            fp += confusion[ii][i]
        if(tp + fp != 0):
            precision = tp/(tp + fp)
            print(best_classifier.classes_[i], ": ", precision)
        elif (tp == 0):
            print(best_classifier.classes_[i], ": Not a number")
        else:
            print(best_classifier.classes_[i], ": infinity")
    print()



    #ROC Receiver Operating Characteristic

    print("ROC curve")
    visualizer = ROCAUC(best_classifier, classes=best_classifier.classes_)


    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.show()
    # # Learn to predict each class against the other
    # classifier = OneVsRestClassifier(
    #     best_classifier)
    # classifier.fit(X_train, y_train)
    # y_score = classifier.predict(X_test)
    # # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(best_classifier.n_classes_):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    # for i, color in zip(range(best_classifier.n_classes_), colors):
    #     plt.plot(
    #         fpr[i],
    #         tpr[i],
    #         color=color,
    #         label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    #     )
    #
    # plt.plot([0, 1], [0, 1], "k--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Some extension of Receiver operating characteristic to multiclass")
    # plt.legend(loc="lower right")
    # plt.show()


    #learning curve

    print("Learning curve")


    fig, axes = plt.subplots(figsize=(10, 15))
    print(axes)
    plot_learning_curve(
       best_classifier,
        "Learnig curve",
        X_train,
        y_train,
        axes=axes,
        #tu skala wyswietlania
        ylim=(0.6, 1.01),
        cv=5,
        n_jobs=4
    )

    plt.show()


def plot_learning_curve(
        estimator,
        title,
        X,
        y,
        axes=None,
        ylim=None,
        cv=None,
        n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 5),
):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    print(train_sizes)
    print(train_scores)
    print(test_scores)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    #fit_times_mean = np.mean(fit_times, axis=1)
    #fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes.legend(loc="best")

    # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, "o-")
    # axes[1].fill_between(
    #     train_sizes,
    #     fit_times_mean - fit_times_std,
    #     fit_times_mean + fit_times_std,
    #     alpha=0.1,
    # )
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # # Plot fit_time vs score
    # fit_time_argsort = fit_times_mean.argsort()
    # fit_time_sorted = fit_times_mean[fit_time_argsort]
    # test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    # test_scores_std_sorted = test_scores_std[fit_time_argsort]
    # axes[2].grid()
    # axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    # axes[2].fill_between(
    #     fit_time_sorted,
    #     test_scores_mean_sorted - test_scores_std_sorted,
    #     test_scores_mean_sorted + test_scores_std_sorted,
    #     alpha=0.1,
    # )
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt


#KRZYWA ROC PLUS WYSW PARAMETROW

def main():

    print("decision trees and random forests classifier")
    print()

    option = input("Choose dataset\n 1. indian_liver_patient_sort\n 2. gender_classification\n 3. sign_mnist_train\n")
    choice(option)



# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
