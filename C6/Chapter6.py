# A50 データの入手・整形
import pandas as pd
from sklearn.model_selection import train_test_split


def A50():
    news_df = pd.read_csv('news_aggregator/newsCorpora.csv', header=None, sep='\t')

    publisher_news_df = news_df[
        news_df[3].isin(['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail'])]
    publisher_news_df = publisher_news_df[[4, 1]]

    train_df, test_df = train_test_split(publisher_news_df, stratify=publisher_news_df[4], test_size=0.2,
                                         random_state=42)
    test_df, valid_df = train_test_split(test_df, stratify=test_df[4], test_size=0.5, random_state=42)

    train_df.to_csv('train.txt', index=False, header=False, sep='\t')
    test_df.to_csv('test.txt', index=False, header=False, sep='\t')
    valid_df.to_csv('valid.txt', index=False, header=False, sep='\t')


# A51 特徴量抽出
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import joblib


def A51():
    vectorizer = CountVectorizer()
    label_encoder = LabelEncoder()

    train_df = pd.read_csv('train.txt', header=None, sep='\t')
    train_feature_df = pd.DataFrame()

    train_feature_df['category'] = label_encoder.fit_transform(train_df[0])
    train_feature_df['title'] = train_df[1]
    vectorizer.fit(train_df[1])

    train_feature_df.to_csv('train.feature.txt', index=False, sep='\t')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    for data_file in ['test.txt', 'valid.txt']:
        df = pd.read_csv(data_file, header=None, sep='\t')
        feature_df = pd.DataFrame()

        feature_df['category'] = label_encoder.transform(df[0])
        feature_df['title'] = df[1]

        # print(list(label_encoder.classes_))
        # 0 - b, 1 - e, 2 - m, 3 - t

        data_path = data_file.split('.')
        data_path.insert(1, 'feature')
        data_path = '.'.join(data_path)

        feature_df.to_csv(data_path, index=False, sep='\t')


vectorizer = joblib.load('vectorizer.pkl')

train_df = pd.read_csv('train.feature.txt', sep='\t')
train_X = vectorizer.transform(train_df['title'])
train_y = train_df['category']

test_df = pd.read_csv('test.feature.txt', sep='\t')
test_X = vectorizer.transform(test_df['title'])
test_y = test_df['category']

valid_df = pd.read_csv('valid.feature.txt', sep='\t')
valid_X = vectorizer.transform(valid_df['title'])
valid_y = valid_df['category']

# A52 学習
from sklearn.linear_model import LogisticRegression


def A52():
    model = LogisticRegression()

    model.fit(train_X, train_y)

    joblib.dump(model, 'model.pkl')


model = joblib.load('model.pkl')


# A53 予測
def A53(headlines: list):
    for headline in headlines:
        vectorized_headline = vectorizer.transform([headline])

        print(f"The probability of categories of news with headline '{headline}':")
        for category, prob in zip(['business', 'entertainment', 'health', 'technology'],
                                  model.predict_proba(vectorized_headline)[0]):
            print(f"{category} - {round(prob * 100, 2)}%")


# A54 正解率の計測
def A54():
    accuracy_train = model.score(train_X, train_y)
    accuracy_test = model.score(test_X, test_y)

    print(f"Train set accuracy: {round(accuracy_train * 100, 5)}%")
    print(f"Test set accuracy: {round(accuracy_test * 100, 5)}%")


# A55 混同行列の作成
from sklearn.metrics import confusion_matrix

train_pred = model.predict(train_X)
train_cm = confusion_matrix(train_y, train_pred)
test_pred = model.predict(test_X)
test_cm = confusion_matrix(test_y, test_pred)


def A55():
    print(f"Confusion matrix for train set:\n{train_cm}")
    print(f"Confusion matrix for test set:\n{test_cm}")


# A56 適合率，再現率，F1スコアの計測
import numpy as np


def A56():
    test_data = pd.DataFrame()
    test_data['precision'] = np.diag(train_cm) / np.sum(train_cm, axis=0)
    test_data['recall'] = np.diag(train_cm) / np.sum(train_cm, axis=1)
    test_data['f1'] = 2 * (test_data['precision'] * test_data['recall']) / (
            test_data['precision'] + test_data['recall'])
    test_data.loc['macro_avg'] = test_data.sum() / 4

    micro_precision = np.sum(np.diag(train_cm)) / np.sum(train_cm)
    test_data.loc['micro_avg'] = [
        micro_precision,
        micro_precision,
        2 * micro_precision * micro_precision / (micro_precision + micro_precision)
    ]

    print(test_data)


# A57 特徴量の重みの確認
def A57():
    word_look_up = vectorizer.get_feature_names_out()
    category_list = ['business', 'entertainment', 'health', 'technology']

    for feature_weight in model.coef_:
        top_10_idx = np.argsort(feature_weight)[-10:]
        top_10_word = [word_look_up[idx] for idx in top_10_idx]
        least_10_idx = np.argsort(feature_weight)[:10]
        least_10_word = [word_look_up[idx] for idx in least_10_idx]

        print(f"Top 10 word for category {category_list[0]}: {top_10_word}")
        print(f"Least 10 word for category {category_list[0]}: {least_10_word}")
        print('')

        category_list.pop(0)


# A58 正則化パラメータの変更
import matplotlib.pyplot as plt


def A58():
    reg_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
    model2 = LogisticRegression(C=0.1)
    model3 = LogisticRegression(C=0.3)
    model4 = LogisticRegression(C=0.5)
    model5 = LogisticRegression(C=0.7)
    model6 = LogisticRegression(C=0.9)
    model7 = LogisticRegression(C=1.1)
    model8 = LogisticRegression(C=1.3)
    model9 = LogisticRegression(C=1.5)
    model10 = LogisticRegression(C=1.7)
    model11 = LogisticRegression(C=1.9)

    train_plot_y = []
    test_plot_y = []
    valid_plot_y = []

    for model in [model2, model3, model4, model5, model6, model7, model8, model9, model10, model11]:
        model.fit(train_X, train_y)
        train_score = model.score(train_X, train_y)
        test_score = model.score(test_X, test_y)
        valid_score = model.score(valid_X, valid_y)

        train_plot_y.append(train_score)
        test_plot_y.append(test_score)
        valid_plot_y.append(valid_score)

    plt.subplot(1, 3, 1)
    plt.plot(reg_list, train_plot_y)
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.xlim(-0.2, 2.2)
    plt.ylim(0.9, 1.05)
    plt.title('Train set accuracy')
    plt.subplot(1, 3, 2)
    plt.plot(reg_list, test_plot_y)
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.xlim(-0.2, 2.2)
    plt.ylim(0.85, 0.95)
    plt.title('Test set accuracy')
    plt.subplot(1, 3, 3)
    plt.plot(reg_list, valid_plot_y)
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.xlim(-0.2, 2.2)
    plt.ylim(0.85, 0.95)
    plt.title('Validation set accuracy')

    plt.show()


# A59 ハイパーパラメータの探索
import itertools
import warnings
from scipy.linalg import LinAlgWarning


def A59():
    penalties = ['l2', 'l1', 'elasticnet']
    tols = [1e-3, 1e-5]
    Cs = [1.0, 2.0]
    solvers = ['lbfgs', 'liblinear', 'newton-cg']
    max_iters = [100, 200, 300]
    valid_scores = []
    combs = list(itertools.product(penalties, tols, Cs, solvers, max_iters))

    warnings.filterwarnings('error', category=LinAlgWarning)

    for comb in combs:
        print(f"current comb: {combs.index(comb)} / {len(combs)}")
        try:
            model = LogisticRegression(penalty=comb[0], tol=comb[1], C=comb[2], solver=comb[3], max_iter=comb[4])
            model.fit(train_X, train_y)
            valid_scores.append(model.score(valid_X, valid_y))
        except:
            valid_scores.append(0)

    best_accuracy = max(valid_scores)
    best_comb = combs[valid_scores.index(best_accuracy)]

    print(f"Best parameters are: {best_comb} with accuracy in valid set: {round(best_accuracy * 100, 5)}.")

    model = LogisticRegression(penalty=best_comb[0], tol=best_comb[1], C=best_comb[2], solver=best_comb[3],
                               max_iter=best_comb[4])
    model.fit(train_X, train_y)

    print(f"The accuracy in test set: {round(model.score(test_X, test_y) * 100, 5)}")

    joblib.dump(model, 'best_model.pkl')


def test_chapter6():
    # A50()
    # A51()
    # A52()
    # A53(["An anonymous scientist found a particle even smaller than quark.",
    #      "The new iPhone is released with a new feature.",
    #      "Russia has closed its border.",])
    # print('')
    # A54()
    # print('')
    # A55()
    # print('')
    # A56()
    # print('')
    # A57()
    # print('')
    # A58()
    A59()


if __name__ == '__main__':
    test_chapter6()
