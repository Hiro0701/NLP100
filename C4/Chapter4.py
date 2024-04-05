neko_txt_path = 'neko.txt'

# A30 形態素解析結果の読み込み
import MeCab


def A30(txt_path: str) -> list:
    with open(txt_path, 'r') as file:
        pos_list = []
        tagger = MeCab.Tagger()

        for line in file:
            lined_list = []
            parsed = tagger.parse(line)
            if len(parsed) == 4:
                continue

            for parse in parsed.split('\n'):
                splitted_pos = parse.split('\t')
                if len(splitted_pos) <= 1:
                    continue

                if '補助記号' not in splitted_pos[4]:
                    lined_list.append({'表層形': splitted_pos[0],
                                       '基本形': splitted_pos[3],
                                       '品詞': splitted_pos[4].split('-')[0],
                                       '品詞細分類1': splitted_pos[4].split('-')[1] if len(
                                           splitted_pos[4].split('-')) != 1 else ''
                                       })

            if lined_list:
                pos_list.append(lined_list)

        return pos_list


# A31 動詞
def A31(pos_list: list) -> list:
    verb_list = []
    for sentence in pos_list:
        for word in sentence:
            if word['品詞'] == '動詞':
                verb_list.append(word['表層形'])

    return verb_list


# A32 動詞の基本形
def A32(pos_list: list) -> list:
    verb_list = []
    for sentence in pos_list:
        for word in sentence:
            if word['品詞'] == '動詞':
                verb_list.append(word['基本形'])

    return verb_list


# A33 「AのB」
def A33(pos_list: list) -> list:
    noun_list = []
    for sentence in pos_list:
        for i in range(len(sentence) - 2):
            if sentence[i]['品詞'] == '名詞' and \
                    sentence[i + 2]['品詞'] == '名詞' and \
                    sentence[i + 1]['表層形'] == 'の':
                noun_list.append(sentence[i]['表層形'] + sentence[i + 1]['表層形'] + sentence[i + 2]['表層形'])

    return noun_list


# A34 名詞の連接
# import re


def A34(pos_list: list) -> str:
    noun_list = []
    for sentence in pos_list:
        current_noun = ''
        noun_longest_list = []
        last_word = sentence[-1]

        for word in sentence:
            if word['品詞'] == '名詞':
                noun_check = True
                current_noun = current_noun + word['表層形']
                if word == last_word:
                    noun_longest_list.append(current_noun)
            else:
                noun_check = False
                noun_longest_list.append(current_noun)
                current_noun = ''
        noun_list.append(max(noun_longest_list, key=len))

    # noun_jp_list = [s for s in noun_list if not re.search(r'[a-zA-Z]', s)]

    return max(noun_list, key=len)


# A35 単語の出現頻度
# form='表層形' or '表層形', pos=None(全部出力) or '名詞' or '動詞', ...
def A35(pos_list: list, pos=None, form='表層形') -> dict:
    frequent_dict = {}
    for sentence in pos_list:
        for word in sentence:
            if not pos or word['品詞'] == pos:
                frequent_dict[word[form]] = frequent_dict.get(word[form], 0) + 1

    return dict(sorted(frequent_dict.items(), key=lambda item: item[1], reverse=True))


# A36 頻度上位10語
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Arial Unicode MS'


def A36(sorted_frequent_dict: dict, n=10):
    top_n_list = list(sorted_frequent_dict.items())[:n]
    plt.bar([i[0] for i in top_n_list], [i[1] for i in top_n_list])
    plt.title(f'A36: Top {n} frequent words')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.show()


# A37 「猫」と共起頻度の高い上位10語
def A37(pos_list: list, n=10, pos=None, form='表層形'):
    neko_co_dict = {}
    for sentence in pos_list:
        if [d for d in sentence if d.get('表層形') == '猫']:
            for word in sentence:
                if word['表層形'] != '猫' and (not pos or word['品詞'] == pos):
                    neko_co_dict[word[form]] = neko_co_dict.get(word[form], 0) + 1

    sorted_neko_co_dict = dict(sorted(neko_co_dict.items(), key=lambda item: item[1], reverse=True))
    top_n_list = list(sorted_neko_co_dict.items())[:n]
    plt.bar([i[0] for i in top_n_list], [i[1] for i in top_n_list])
    plt.title(f'A37: Top {n} frequent words with "猫"')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.show()


# A38 ヒストグラム
def A38(sorted_frequent_dict: dict):
    plt.hist(sorted_frequent_dict.values(),
             bins=30)
    plt.title("A38: 単語の出現頻度とその数")
    plt.xlabel('出現頻度')
    plt.ylabel('種類数')
    plt.ylim(0, 100)
    plt.show()


# A39 Zipfの法則
def A39(sorted_frequent_dict: dict):
    plt.loglog([i + 1 for i in range(len(sorted_frequent_dict))],
               [i for i in sorted_frequent_dict.values()])
    plt.title("A39: 単語の出現頻度と順位の両対数")
    plt.xlabel("出現頻度順位")
    plt.ylabel("出現頻度")
    plt.show()

def test_chapter4():
    A30_pos_list = A30(neko_txt_path)
    print("A30:", '\n', A30_pos_list)
    print('')
    print("A31:", '\n', A31(A30_pos_list))
    print('')
    print("A32:", '\n', A32(A30_pos_list))
    print('')
    print("A33:", '\n', A33(A30_pos_list))
    print('')
    print("A34:", '\n', A34(A30_pos_list))
    print('')
    A35_sorted_frequent_dict = A35(A30_pos_list)
    print("A35:", '\n', A35_sorted_frequent_dict)
    A36(A35_sorted_frequent_dict)
    A37(A30_pos_list, pos='動詞')
    A38(A35_sorted_frequent_dict)
    A39(A35_sorted_frequent_dict)

if __name__ == '__main__':
    test_chapter4()