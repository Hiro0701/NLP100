line = '*' * 10

# 00 文字列の逆順
A00 = 'stressed'[::-1]

print('A00:', A00)
print(line)

# 01　「パタトクカシーー」
A01_str = 'パタトクカシーー'
A01 = ''

for i in range(4):
    A01 += A01_str[i * 2]
print('A01:', A01)
print(line)

# 02　「パトカー」 ＋ 「タクシー」 ＝ 「パタトクカシーー」
A02_str1 = 'パトカー'
A02_str2 = 'タクシー'
A02 = ''

for i in range(len(A02_str1)):
    A02 += A02_str1[i]
    A02 += A02_str2[i]
print('A02:', A02)
print(line)

# 03 円周率
A03_str = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
A03_list = A03_str.split()
A03 = []

import string

alphabets = string.ascii_letters
for i in A03_list:
    count = 0
    for j in i:
        if j in alphabets:
            count += 1
    A03.append(count)
print('A03:', A03)
print(line)

# 04　元素記号
A04_str = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
A04_list = A04_str.split()
A04 = {}
A04_idx1 = [1, 5, 6, 7, 8, 9, 15, 16, 19]
A04_idx2 = [2, 3, 4, 10, 11, 12, 13, 14, 17, 18, 20]
for i in range(21):
    if i + 1 in A04_idx1:
        A04[A04_list[i][0]] = i + 1
    elif i + 1 in A04_idx2:
        A04[A04_list[i][0:2]] = i + 1
print('A04:', A04)
print(line)


# 05 n-gram
def A05(sequence, n):
    A05_n_gram = []
    for i in range(len(sequence) - n + 1):
        n_gram = []
        for j in range(n):
            n_gram.append(sequence[i + j])
        A05_n_gram.append(tuple(n_gram))

    return A05_n_gram


print('A05 word bi-gram:', A05(['I', 'am', 'an', 'NLPer'], 2))
print('A05 letter bi-gram:', A05('I am an NLPer', 2))
print(line)

# 0　集合
A06_str1 = 'paraparaparadise'
A06_str2 = 'paragraph'
A06_X = set(A05(A06_str1, 2))
A06_Y = set(A05(A06_str2, 2))
print('A06 union:', A06_X | A06_Y)
print('A06 intersection:', A06_X & A06_Y)
print('A06 difference X - Y', A06_X - A06_Y)
print('A06 difference Y - X', A06_Y - A06_X)
print('A06; Is "s, e" in X?:', ('s', 'e') in A06_X)
print('A06; Is "s, e" in Y?:', ('s', 'e') in A06_Y)
print(line)


# 07 テンプレートによる文生成
def A07(x, y, z):
    return f'{x}時の{y}は{z}'


print('A07 with x=12, y="気温", z=22.4:', A07(12, '気温', 22.4))
print(line)


# 08　暗号文
def A08_cipher(string):
    new_string = []
    for i in string:
        if i.islower():
            new_string.append(chr(219 - ord(i)))
        else:
            new_string.append(i)
    return ''.join(new_string)


print('A08 with "I am a HUge PERSoN.":', A08_cipher("I am a HUge PERSoN."))
print(line)
# 09 Typoglycemia
import random


def A09(string):
    string_list = string.split()
    new_string = []
    for i in string_list:
        if len(i) <= 4:
            new_string.append(i)
        else:
            new_word = []
            new_word.append(i[0])
            middle_letters = list(i[1:-1])
            random.shuffle(middle_letters)
            new_word.append(''.join(middle_letters))
            new_word.append(i[-1])
            new_string.append(''.join(new_word))
    return ' '.join(new_string)


A09_example = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
print(f'A09 with {A09_example}:')
print(A09(A09_example))
print(line)
