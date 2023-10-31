txt_path = 'C2/popular-names.txt'
A11_output_path = 'C2/A11.txt'
A12_output1_path = 'C2/A12_col1.txt'
A12_output2_path = 'C2/A12_col2.txt'
A13_output_path = 'C2/A13.txt'
A18_output_path = 'C2/A18.txt'
A19_output_path = 'C2/A19.txt'

# A10 行数のカウント
import os


def A10(txt_path: str) -> int:
    with open(txt_path, 'r') as file:
        return len(file.readlines())


# A11 タブをスペースに置換
def A11(txt_path: str, output_path: str):
    with open(txt_path, 'r') as file, \
            open(output_path, 'w') as output_file:
        for line in file:
            output_file.write(line.replace('\t', ' '))


# A12 １列目をcol1.txtに、２列目をcol2.txtに保存
def A12(txt_path: str, output_path1: str, output_path2: str):
    with open(txt_path, 'r') as file, \
            open(output_path1, 'w') as output_file1, \
            open(output_path2, 'w') as output_file2:
        for line in file:
            output_file1.write(line.split('\t')[0] + '\n')
            output_file2.write(line.split('\t')[1] + '\n')


# A13 col1.txtとcol2.txtをマージ
def A13(input_path1: str, input_path2: str, output_path: str):
    with open(input_path1, 'r') as input_file1, \
            open(input_path2, 'r') as input_file2, \
            open(output_path, 'w') as output_file:
        lines1 = input_file1.readlines()
        lines2 = input_file2.readlines()

        for line1, line2 in zip(lines1, lines2):
            output_file.write(line1.strip() + '\t' + line2)


# A14 先頭からN行を出力
def A14(input_path: str, n: int) -> list:
    assert os.path.exists(input_path) and isinstance(n, int)
    with open(input_path, 'r') as file:
        return file.readlines()[:n]


# A15 末尾のN行を出力
def A15(input_path: str, n: int) -> list:
    with open(input_path, 'r') as file:
        return file.readlines()[::-1][:n]


# A16 ファイルをN分割する
def get_idx(len: int, n: int) -> list:
    return [len // n + 1 * (i < len % n) for i in range(n)]


def A16(input_path: str, n: int):
    output_path = 'C2/A16_{}.txt'
    with open(input_path, 'r') as file:
        lines = file.readlines()
        split_idx = get_idx(len(lines), n)
        split_idx = [0] + [sum(split_idx[:i + 1]) for i in range(len(split_idx))]
        for i in range(len(split_idx) - 1):
            with open(output_path.format(i + 1), 'w') as output_file:
                for line in lines[split_idx[i]:split_idx[i + 1]]:
                    output_file.write(line)


# A17 １列目の文字列の異なり
def A17(input_path: str) -> set:
    with open(input_path, 'r') as file:
        return {line.replace("\n", "") for line in file}


# A18 各行を3コラム目の数値の降順にソート
def A18(input_path: str, output_path: str):
    with open(input_path, 'r') as input_file, \
            open(output_path, 'w') as output_file:
        txt_data = input_file.readlines()
        sorted_data = sorted(txt_data, key=lambda x: x.split('\t')[2])
        for line in sorted_data:
            output_file.write(line)


# A19 各行の1コラム目の文字列の出現頻度を求め，出現頻度の高い順に並べる
from collections import Counter


def A19(input_path: str, output_path: str):
    with open(input_path, 'r') as input_file, \
            open(output_path, 'w') as output_file:
        name_counts = Counter(input_file.readlines()).most_common()
        for name in name_counts:
            output_file.write(name[0])


if __name__ == "__main__":
    print("A10:", A10(txt_path), '\n')
    # os.system('wc -l C2/popular-names.txt')
    A11(txt_path, A11_output_path)
    # os.system("sed 's/\t/ /g' C2/popular-names.txt")
    A12(txt_path, A12_output1_path, A12_output2_path)
    # os.system("cut -f1 C2/popular-names.txt")
    # os.system("cut -f2 C2/popular-names.txt")
    A13(A12_output1_path, A12_output2_path, A13_output_path)
    # os.system("paste C2/A12_col1.txt C2/A12_col2.txt")
    print("A14:")
    for line in A14(txt_path, 5):
        print(line.replace('\n', ''))
    print('')
    # os.system("head -n 5 C2/popular-names.txt")
    print("A15:")
    for line in A15(txt_path, 5):
        print(line.replace('\n', ''))
    print('')
    # os.system("tail -n 5 C2/popular-names.txt")
    A16(txt_path, 7)
    # os.system("split -n 7 -d C2/popular-names.txt C2/A16_unix_")
    print("A17:", A17(A12_output1_path), '\n')
    # os.system("sort C2/A12_col1.txt | uniq")
    A18(txt_path, A18_output_path)
    # os.system("sort -k 2 -r C2/popular-names.txt")
    A19(A12_output1_path, A19_output_path)
    # os.system("sort C2/A12_col1.txt | uniq -c | sort -nr")
