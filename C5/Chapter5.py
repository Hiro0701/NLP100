ai_txt_path = 'ai/ai.ja.txt'
ai_output_path = 'ai/ai.ja.txt.parsed'
A42_output_txt_path = 'A42.txt'
A43_output_txt_path = 'A43.txt'
A45_output_txt_path = 'A45.txt'
A46_output_txt_path = 'A46.txt'
A47_output_txt_path = 'A47.txt'
A48_output_txt_path = 'A48.txt'
A49_output_txt_path = 'A49.txt'

import ginza
import spacy
import json
import os

nlp = spacy.load("ja_ginza")

# Build parsed txt

# with open(ai_txt_path, 'r') as input_txt:
#     with open(ai_output_path, 'w') as output_txt:
#
#         output_list = []
#         for line in input_txt:
#             doc = nlp(line)
#
#             for sent in doc.sents:
#                 sentence_dict = {}
#                 pos_list = []
#                 dep_list = []
#
#                 if sent.__len__() > 1:
#                     for span in ginza.bunsetu_spans(sent):
#                         pos_list2 = []
#
#                         for token in span:
#                             token_pos = token.tag_.split('-')
#                             # (order, surface, base, pos, pos1)
#                             pos_list2.append([token.i,
#                                               token.text,
#                                               token.lemma_,
#                                               token_pos[0],
#                                               token_pos[2] if len(token_pos) > 1 and token_pos[1] == "普通名詞" else (token_pos[1] if len(token_pos) > 1 else '-')
#                                               ])
#
#                         pos_list.append(pos_list2)
#
#                         for token in span.lefts:
#                             dep_list.append((ginza.bunsetu_spans(sent).index(ginza.bunsetu_span(token)),
#                                              ginza.bunsetu_spans(sent).index(span)))
#
#                 if pos_list and dep_list:
#                     sentence_dict['pos'] = pos_list
#                     sentence_dict['dep'] = dep_list
#
#                     output_list.append(sentence_dict)
#
#         json.dump(output_list, output_txt, ensure_ascii=False)

# A40 係り受け解析結果の読み込み（形態素）
class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def __repr__(self):
        return f'(表層形: {self.surface}, 基本形: {self.base}, 品詞: {self.pos}, 品詞細分類1: {self.pos1})'


# A41 係り受け解析結果の読み込み（文節・係り受け）
class Chunk:
    def __init__(self, morphs, dst, srcs):
        self.morphs = morphs
        self.dst = dst
        self.srcs = srcs

    def __repr__(self):
        return f'(文章: {self.get_sent()}, 係り先: {self.dst}, 係り元: {self.srcs})'

    def check_sent(self, condition: str) -> bool:
        if any(condition in morph.pos for morph in self.morphs):
            return True

        return False

    def get_sent(self) -> str:
        sent = ''
        for morph in self.morphs:
            if morph.pos != '補助記号':
                sent += morph.surface

        return sent

    def get_morphs_sent(self, condition: str) -> list:
        morph_list = []
        for morph in self.morphs:
            if condition in morph.pos:
                morph_list.append(morph.surface)

        return morph_list

    def get_morphs(self, condition: str) -> list:
        morph_list = []
        for morph in self.morphs:
            if morph.pos == condition:
                morph_list.append(morph)

        return morph_list


def A40(input_path: str) -> list:
    with open(input_path, 'r') as txt:
        output_list = []
        for line in json.load(txt):
            sent_dict = {}
            sent_morph_list = []
            sent_chunk_list = []
            bunsetu_dep_list = line['dep']
            current_bunsetu = 0

            for bunsetu in line['pos']:
                chunk_bunsetu_list = []
                src_list = []

                for word in bunsetu:
                    word_morph = Morph(word[1], word[2], word[3], word[4])
                    sent_morph_list.append(word_morph)
                    chunk_bunsetu_list.append(word_morph)

                for dep in bunsetu_dep_list:
                    if current_bunsetu in dep:
                        if dep.index(current_bunsetu) == 0:
                            dst = dep[1]
                        elif dep.index(current_bunsetu) == 1:
                            src_list.append(dep[0])

                sent_chunk_list.append(Chunk(chunk_bunsetu_list,
                                             dst if dst else '',
                                             src_list if src_list else []))
                current_bunsetu += 1

            sent_dict['morph'] = sent_morph_list
            sent_dict['chunk'] = sent_chunk_list
            output_list.append(sent_dict)

    return output_list


# A42 係り元と係り先の文節の表示
def A42(sent_list: list, output_path: str):
    with open(output_path, 'w') as output_txt:
        for sent in sent_list:
            for bunsetu in sent['chunk']:
                if sent['chunk'].index(bunsetu) != bunsetu.dst:
                    line = bunsetu.get_sent() + '\t' + sent['chunk'][bunsetu.dst].get_sent() + '\n'
                    output_txt.write(line)


# A43 名詞を含む文節が動詞を含む文節に係るものを抽出
def A43(sent_list: list, output_path: str):
    with open(output_path, 'w') as output_txt:
        for sent in sent_list:
            for bunsetu in sent['chunk']:
                if sent['chunk'].index(bunsetu) != bunsetu.dst:
                    if bunsetu.check_sent('名詞') and sent['chunk'][bunsetu.dst].check_sent('動詞'):
                        line = bunsetu.get_sent() + '\t' + sent['chunk'][bunsetu.dst].get_sent() + '\n'
                        output_txt.write(line)


# A44 係り受け木の可視化
# URL: localhost:port
def A44(sentence: list):
    words = []
    for bunsetu in sentence:
        words.append(bunsetu.get_sent())
    spaces = [True] * len(words)
    doc = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)

    for i, token in enumerate(doc):
        if sentence[i].dst == i:
            token.dep_ = "ROOT"
        else:
            token.dep_ = "dep"
        token.head = doc[sentence[i].dst]

    spacy.displacy.serve(doc, style="dep", auto_select_port=True)


# A45 動詞の格パターンの抽出
# A46 動詞の格フレーム情報の抽出
def A45(sent_list: list, output_path: str, predicate_condition: str, case_condition: str, print_bunsetu=False):
    with open(output_path, 'w') as output_txt:
        for sent in sent_list:
            for bunsetu in sent['chunk']:
                predicate_bool = False

                for word in bunsetu.morphs:
                    if word.pos == predicate_condition:
                        predicate = word.base
                        predicate_bool = True
                        break

                if predicate_bool:
                    if bunsetu.srcs:
                        bunsetu_cases = []

                        for idx in bunsetu.srcs:
                            dep_bunsetu = sent['chunk'][idx]
                            if dep_bunsetu.check_sent(case_condition):
                                bunsetu_cases.append(
                                    [dep_bunsetu.get_morphs_sent(case_condition), dep_bunsetu.get_sent()])

                        if len(bunsetu_cases) == 0:
                            break

                        morphs = [bunsetu_case[0] for bunsetu_case in bunsetu_cases]
                        morphs = [morph for morph_list in morphs for morph in morph_list]

                        if not print_bunsetu:
                            line = predicate + '\t' + '\t'.join(sorted(morphs)) + '\n'
                            output_txt.write(line)
                        else:
                            sorted_bunsetu_cases = sorted(bunsetu_cases, key=lambda x: x[0][0])

                            line = predicate + '\t' + '\t'.join(sorted(morphs)) + '\t' + '\t'.join(
                                [sorted_bunsetu_case[1] for sorted_bunsetu_case in sorted_bunsetu_cases]) + '\n'
                            output_txt.write(line)
                        break


# A47 機能動詞構文のマイニング
def A47(sent_list: list, output_path: str):
    with open(output_path, 'w') as output_txt:
        for sent in sent_list:
            for bunsetu in sent['chunk']:
                for word1, word2 in zip(bunsetu.morphs, bunsetu.morphs[1:]):
                    if word1.pos1 == 'サ変可能' and word2.surface == 'を':
                        base_predicate = word1.surface + word2.surface
                        verb_bool = False
                        for word in sent['chunk'][bunsetu.dst].morphs:
                            if word.pos == '動詞':
                                predicate = base_predicate + word.base
                                verb_bool = True
                                break

                        if verb_bool:
                            idx_list = sent['chunk'][bunsetu.dst].srcs + bunsetu.srcs
                            idx_list.remove(sent['chunk'].index(bunsetu))

                            if idx_list:
                                bunsetu_cases = []
                                for idx in idx_list:
                                    dep_bunsetu = sent['chunk'][idx]
                                    if dep_bunsetu.check_sent('助詞'):
                                        bunsetu_cases.append(
                                            [dep_bunsetu.get_morphs_sent('助詞'), dep_bunsetu.get_sent()])

                                if len(bunsetu_cases) == 0:
                                    break

                                sorted_bunsetu_cases = sorted(bunsetu_cases, key=lambda x: x[0][0])

                                morphs = [bunsetu_case[0] for bunsetu_case in bunsetu_cases]
                                morphs = [morph for morph_list in morphs for morph in morph_list]

                                line = predicate + '\t' + '\t'.join(morphs) + '\t' + '\t'.join(
                                    [sorted_bunsetu_case[1] for sorted_bunsetu_case in
                                     sorted_bunsetu_cases]) + '\n'

                                output_txt.write(line)
                                break


# A48 名詞から根へのパスの抽出
def A48(sent_list: list, output_path: str):
    with open(output_path, 'w') as output_txt:
        for sent in sent_list:
            for bunsetu in sent['chunk']:
                if bunsetu.check_sent("名詞"):
                    current_bunsetu = bunsetu
                    bunsetu_list = []

                    while True:
                        bunsetu_list.append(current_bunsetu.get_sent())
                        if current_bunsetu.dst == sent['chunk'].index(current_bunsetu):
                            break
                        else:
                            current_bunsetu = sent['chunk'][current_bunsetu.dst]

                    if len(bunsetu_list) > 1:
                        line = ' -> '.join(bunsetu_list) + '\n'
                        output_txt.write(line)


# A49 名詞間の係り受けパスの抽出
def A49(sent_list: list):
    def sub_X_Y(bunsetu: Chunk, X_Y='X') -> str:
        noun_chunks = bunsetu.get_morphs_sent('名詞')

        if len(noun_chunks) >= 1:
            start_idx = bunsetu.get_sent().find(noun_chunks[0])
            end_idx = bunsetu.get_sent().rfind(noun_chunks[-1]) + len(noun_chunks[-1])
            line = bunsetu.get_sent().replace(
                bunsetu.get_sent()[start_idx:end_idx],
                X_Y)
        else:
            line = bunsetu.get_sent().replace(noun_chunks[0].surface, X_Y)

        return line

    with open(A49_output_txt_path, 'w') as output_txt:
        for sent in sent_list:
            for idx in range(len(sent['chunk'])):
                if sent['chunk'][idx].check_sent('名詞'):
                    pair_idx_list = []
                    dep_idx_list = []
                    current_bunsetu = sent['chunk'][idx]

                    for idx2 in range(idx + 1, len(sent['chunk'])):
                        if sent['chunk'][idx2].check_sent('名詞'):
                            pair_idx_list.append(idx2)

                    if pair_idx_list:
                        while True:
                            if current_bunsetu.dst == sent['chunk'].index(current_bunsetu):
                                break
                            else:
                                dep_idx_list.append(current_bunsetu.dst)
                                current_bunsetu = sent['chunk'][current_bunsetu.dst]

                        if not dep_idx_list:
                            break

                        for pair_idx in pair_idx_list:
                            line = sub_X_Y(sent['chunk'][idx])
                            line1 = line
                            line2 = line

                            if pair_idx in dep_idx_list:
                                for idx3 in dep_idx_list:
                                    if pair_idx == idx3:
                                        line1 += ' -> ' + sub_X_Y(sent['chunk'][idx3], 'Y')
                                        break
                                    else:
                                        line1 += ' -> ' + sent['chunk'][idx3].get_sent()

                                output_txt.write(line1 + '\n')
                            else:
                                pair_dep_idx_list = []
                                current_pair_bunsetu = sent['chunk'][pair_idx]

                                while True:
                                    if current_pair_bunsetu.dst == sent['chunk'].index(current_pair_bunsetu):
                                        break
                                    else:
                                        pair_dep_idx_list.append(current_pair_bunsetu.dst)
                                        current_pair_bunsetu = sent['chunk'][current_pair_bunsetu.dst]

                                try:
                                    final_bunsetu_idx = min(list(set(dep_idx_list) & set(pair_dep_idx_list)))
                                except ValueError:
                                    break

                                for dep_idx in dep_idx_list:
                                    if dep_idx != final_bunsetu_idx:
                                        line2 += ' -> ' + sent['chunk'][dep_idx].get_sent()
                                    else:
                                        break

                                line3 = sub_X_Y(sent['chunk'][pair_idx], 'Y')

                                for pair_dep_idx in pair_dep_idx_list:
                                    if pair_dep_idx != final_bunsetu_idx:
                                        line3 += ' -> ' + sent['chunk'][pair_dep_idx].get_sent()

                                line2 = line2 + ' | ' + line3 + ' | ' + sent['chunk'][final_bunsetu_idx].get_sent()

                                output_txt.write(line2 + '\n')


def test_chapter5():
    A40_sent_list = A40(ai_output_path)
    print("A40:", '\n')
    print(A40_sent_list[0]['morph'])
    print(A40_sent_list[0]['chunk'])
    A42(A40_sent_list, A42_output_txt_path)
    A43(A40_sent_list, A43_output_txt_path)
    A44(A40_sent_list[0]['chunk'])
    A45(A40_sent_list, A45_output_txt_path, '動詞', '助詞')
    A45(A40_sent_list, A46_output_txt_path, '動詞', '助詞', True)
    os.system(
        """cat A45.txt | awk 'BEGIN {FS="\t"} {for(i=2; i<=NF; i++) print $1 " " $i}' | sort | uniq -c | sort -nr""")
    print("")
    os.system(
        """cat A45.txt | awk 'BEGIN {FS="\t"} {if ($1 == "行う") for(i=2; i<=NF; i++) print $1 " " $i}' | sort | uniq -c | sort -nr""")
    print("")
    os.system(
        """cat A45.txt | awk 'BEGIN {FS="\t"} {if ($1 == "なる") for(i=2; i<=NF; i++) print $1 " " $i}' | sort | uniq -c | sort -nr""")
    print("")
    os.system(
        """cat A45.txt | awk 'BEGIN {FS="\t"} {if ($1 == "与える") for(i=2; i<=NF; i++) print $1 " " $i}' | sort | uniq -c | sort -nr""")
    A47(A40_sent_list, A47_output_txt_path)
    A48(A40_sent_list, A48_output_txt_path)
    A49(A40_sent_list)


if __name__ == '__main__':
    test_chapter5()
