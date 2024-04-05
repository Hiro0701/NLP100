# A00 文字列の逆順
def A00(str1: str) -> str:
    return str1[::-1]


# A01　「パタトクカシーー」
def A01(str1: str) -> str:
    return str1[::2]


# A02　「パトカー」 ＋ 「タクシー」 ＝ 「パタトクカシーー」
def A02(str1: str, str2: str) -> str:
    combined_str = ''
    for i in range(len(str1)):
        combined_str += str1[i]
        if str2[i]:
            combined_str += str2[i]
    return combined_str


# A03 円周率
import string


def A03(str1: str) -> list:
    alphabets = string.ascii_letters
    return [sum(1 for char in word if char in alphabets) for word in str1.split()]


# A04　元素記号
def A04(str1: str, list1: list) -> dict:
    splitted_str = str1.split()
    return {splitted_str[i][:1] if i + 1 in list1 else splitted_str[i][:2]: i + 1 for i, word in
            enumerate(splitted_str)}


# A05 n-gram
def A05(seq: str or list, n: int) -> list:
    return [tuple(seq[i:i + n]) for i in range(len(seq))][:-1]


# A06　集合
def A06(set1: set, set2: set) -> tuple:
    return set1 | set2, set1 & set2, set1 - set2


# 07 テンプレートによる文生成
def A07(x: int, y: str, z: float) -> str:
    return f'{x}時の{y}は{z}'


# 08　暗号文
def A08(str1: str) -> str:
    return ''.join([chr(219 - ord(i)) if i.islower() else i for i in str1])


# 09 Typoglycemia
import random


def A09(str1: str) -> str:
    return ' '.join(
        [i if len(i) <= 4 else i[0] + ''.join(random.sample(i[1:-1], len(i) - 2)) + i[-1] for i in str1.split()])

def test_chapter1():
    print("A00:", A00('stressed'), '\n')
    print("A01:", A01('パタトクカシーー'), '\n')
    print("A02:", A02('パトカー', 'タクシー'), '\n')
    print("A03:", A03("Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."),
          '\n')
    print("A04:",
          A04('Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.',
              [1, 5, 6, 7, 8, 9, 15, 16, 19]), '\n')
    print("A05 (words):", A05(['I', 'am', 'an', 'NLPer'], 2), '\n')
    print("A05 (letters):", A05('I am an NLPer', 2), '\n')
    print("A06 (union):", A06(set(A05("paraparaparadise", 2)), set(A05("paragraph", 2)))[0], '\n')
    print("A06 (intersection):", A06(set(A05("paraparaparadise", 2)), set(A05("paragraph", 2)))[1], '\n')
    print("A06 (difference)", A06(set(A05("paraparaparadise", 2)), set(A05("paragraph", 2)))[2], '\n')
    print("A06 ('se' in X):", A05('se', 2)[0] in set(A05("paraparaparadise", 2)), '\n')
    print("A06 ('se' in Y):", A05('se', 2)[0] in set(A05("paragraph", 2)), '\n')
    print("A07:", A07(12, "気温", 22.4), '\n')
    print("A08:", A08("I am A HuGe PERSON."), '\n')
    print("A09:",
          A09("I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."),
          '\n')



if __name__ == "__main__":
    test_chapter1()