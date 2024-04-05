import numpy as np

data_path = 'GoogleNews-vectors-negative300.bin'
word_analogy_path = 'questions-words.txt'
word_similarity_353_path = 'wordsim353/combined.csv'
A64_path = 'A64.txt'

# A60 単語ベクトルの読み込みと表示
from gensim.models import KeyedVectors

word2vec = KeyedVectors.load_word2vec_format(data_path, binary=True)


def A60():
    print(f"Word vector for United States:\n{word2vec['United_States']}")


# A61 単語の類似度
def A61():
    print(f"Cosine similarity between United States and U.S. : {word2vec.similarity('United_States', 'U.S.')}")


def generate_similar_words(positive, negative=[], topn=10):
    for word in word2vec.most_similar(positive=positive, negative=negative, topn=topn):
        yield word


# A62 類似度の高い単語10件
def A62():
    print(f"Most similar words with United States:")
    similar_words = generate_similar_words(['United_States'])
    for word in similar_words:
        print(word)


# A63 加法構成性によるアナロジー
def A63():
    print(f"Most similar words with Spain, Athens and not with Madrid:")
    similar_words = generate_similar_words(['Spain', 'Athens'], ['Madrid'])
    for word in similar_words:
        print(word)


# A64 アナロジーデータでの実験
def A64():
    with open(word_analogy_path, 'r') as input_file, open(A64_path, 'w') as output_file:
        for line in input_file:
            if line.startswith(':'):
                pass
            else:
                splitted_line = line.split(' ')
                word_similarity = generate_similar_words([splitted_line[1], splitted_line[2]], [splitted_line[0]], 1)
                for word in word_similarity:
                    output_file.write(line[:-1] + ' ' + word[0] + ' ' + str(word[1]) + '\n')


# A65 アナロジータスクでの正解率
def A65():
    syntactic_score = 0
    semantic_score = 0
    length = 0

    with open(A64_path, 'r') as file:
        for line in file:
            length += 1
            splitted_line = line.split(' ')
            if splitted_line[3] == splitted_line[4]:
                syntactic_score += 1
            semantic_score += float(splitted_line[5][:-1])

    print(f"syntactic score: {syntactic_score / length}")
    print(f"semantic score: {semantic_score / length}")


# A66 WordSimilarity-353での評価
import pandas as pd


def A66():
    df = pd.read_csv(word_similarity_353_path)
    df['word_vector'] = df.apply(lambda row: word2vec.similarity(row['Word 1'], row['Word 2']), axis=1)
    df['human_rank'] = df['Human (mean)'].rank(method='average', ascending=True)
    df['word_vector_rank'] = df['word_vector'].rank(method='average', ascending=True)

    print(f"Correlation between human eval. and word vector calc.: {df['word_vector_rank'].corr(df['human_rank'])}")


# A67 k-meansクラスタリング
from sklearn.cluster import KMeans

country_list = ['United_States', 'South_Korea', 'Japan', 'France', 'Cuba', 'Mexico', 'San_Marino', 'Australia',
                'Indonesia',
                'Argentina', 'Malaysia', 'Singapore', 'Taiwan', 'China', 'Burundi', 'Niger', 'Morocco', 'Canada',
                'Iceland',
                'Denmark', 'Finland', 'Peru', 'Bolivia', 'Chile', 'Iran', 'Israel', 'Egypt', 'Italia', 'Greece',
                'Turkey',
                'Norway', 'Yemen']
country_word2vec = word2vec[country_list]


def A67():
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(country_word2vec)

    country_df = pd.DataFrame()

    country_df['country'] = np.array(country_list)
    country_df['label'] = kmeans.predict(country_word2vec)
    print(f"Clustered by k-means with n=5:\n{country_df}")


from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

# A68 Ward法によるクラスタリング
def A68():
    plt.title("Clustering Dendrogram (ward)")

    plt.plot()

    linked = linkage(country_word2vec, method='ward')

    dendrogram(linked, labels=country_list)

    plt.show()


from sklearn.manifold import TSNE

# A69 t-SNEによる可視化
def A69():
    tsne = TSNE()
    tsne.fit(country_word2vec)

    plt.title("Visualization of country vectors by t-SNE")

    plt.plot(tsne.embedding_[:,0], tsne.embedding_[:,1], 'o')

    for i, label in enumerate(country_list):
        plt.text(tsne.embedding_[i, 0] + 0.01, tsne.embedding_[i, 1] + 0.01, label, fontsize=9, ha='left',
                 va='bottom')

    plt.show()


def test_chapter7():
    A60()
    print('')
    A61()
    print('')
    A62()
    print('')
    A63()
    A64()
    A65()
    print('')
    A66()
    print('')
    A67()
    A68()
    A69()


if __name__ == '__main__':
    test_chapter7()
