line_sep = '*' * 10
txt_path = 'C2/popular-names.txt'

# A10 行数のカウント
with open(txt_path, 'r') as original_file:
    A10_line_count = 0
    for line in original_file:
        A10_line_count += 1

print('A10:', A10_line_count)
print(line_sep)

# A11 タブをスペースに置換
A11_txt_path = 'A11.txt'
with open(txt_path, 'r') as original_file, \
        open(A11_txt_path, 'w') as A11_file:
    for line in original_file:
        A11_file.write(line.replace('\t', ' '))

# A12 １列目をcol1.txtに、２列目をcol2.txtに保存
A12_col1_txt_path = 'col1.txt'
A12_col2_txt_path = 'col2.txt'

with open(txt_path, 'r') as original_file, \
        open(A12_col1_txt_path, 'w') as A12_col1_file, \
        open(A12_col2_txt_path, 'w') as A12_col2_file:
    for line in original_file:
        col = line.split('\t')
        A12_col1_file.write(col[0] + '\n')
        A12_col2_file.write(col[1] + '\n')

# A13 col1.txtとcol2.txtをマージ
A13_txt_path = 'A13.txt'
with open(A12_col1_txt_path, 'r') as A12_col1_file, \
        open(A12_col2_txt_path, 'r') as A12_col2_file, \
        open(A13_txt_path, 'w') as A13_file:
    lines1 = A12_col1_file.readlines()
    lines2 = A12_col2_file.readlines()

    for line1, line2 in zip(lines1, lines2):
        A13_file.write(line1.strip() + '\t' + line2)


# A14 先頭からN行を出力
def A14(input, n):
    count = 0
    for line in input:
        if count == n:
            break
        print(line)
        count += 1


with open(txt_path, 'r') as original_file:
    n = int(input('Type your number: '))
    print(f"A14 with n={n}:")
    A14(original_file, n)

print(line_sep)


# A15 末尾のN行を出力
def A15(input, n):
    for i in range(n):
        print(input[-(i + 1)])
        if i == 0:
            print('')


with open(txt_path, 'r') as original_file:
    n = int(input('Type your number: '))
    print(f"A15 with n={n}")
    datas = original_file.readlines()
    A15(datas, n)

print(line_sep)


# A16 ファイルをN分割する
def A16_get_idx(n, q, r):
    idx_list = [n for i in range(q)]
    if q >= r:
        for i in range(r):
            idx_list[i] += 1
    else:
        r2 = r % q
        for i in range(q):
            idx_list[i] += 1
        for i in range(r2):
            idx_list[i] += 1
    return idx_list


def A16(input, n):
    splitted_list = []
    q, r = divmod(len(input), n)
    if r == 0:
        for i in range(len(input) // q):
            splitted_lines = input[q * i:q * (i + 1)]
            splitted_list.append(splitted_lines)
            with open(f'C2/A16_{i}.txt', 'w') as A16_splitted_file:
                for line in splitted_lines:
                    A16_splitted_file.write(line)
    else:
        idx = A16_get_idx(n, q, r)
        current_idx = 0
        for i in idx:
            splitted_lines = input[current_idx: current_idx + i]
            splitted_list.append(splitted_lines)
            current_idx = current_idx + i + 1
            with open(f'C2/A16_{i}.txt', 'w') as A16_splitted_file:
                for line in splitted_lines:
                    A16_splitted_file.write(line)


    return splitted_list


with open(txt_path, 'r') as original_file:
    datas = original_file.readlines()
    A16(datas, 5)
