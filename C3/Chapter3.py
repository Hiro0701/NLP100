json_path = 'jawiki-country.json'

# A20 JSONデータの読み込み
import json


def A20(path: str, country: str) -> dict:
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]

    return [i for i in data if i['title'] == country][0]


# A21 カテゴリ名を含む行を抽出
def A21(data: str, pattern: str) -> list:
    splitted_data = data.split('\n')

    return [re.search(pattern, line).group() for line in splitted_data if re.search(pattern, line)]


# A22 カテゴリ名の抽出
import re


def A22(splitted_data: list, pattern: str) -> list:
    return [re.findall(pattern, line)[0] for line in splitted_data if re.findall(pattern, line)]


# A23 セクション構造
def A23(data: str, pattern1: str, pattern2: str) -> list:
    list1 = [len(item) - 1 for item in re.findall(pattern1, data)]
    list2 = [item for item in re.findall(pattern2, data)]
    return list(zip(list1[::2], list2))


# A24 ファイル参照の抽出
def A24(data: str, pattern: str) -> list:
    return [item[0] for item in re.findall(pattern, data, re.IGNORECASE)]

# A25 テンプレートの抽出
def A25(data: str, pattern1: str, pattern2: str) -> list:
     info = re.findall(pattern1, data, re.DOTALL)[0].split('\n')
     temp = {}
     current_key = None
     for i in info:
         print(i)
         print('***')
         match = re.findall(pattern2, i)
         print(match)

         if len(match) > 0:
             temp[match[0][0]] = match[0][1]
             current_key = match[0][0]
         else:
             if current_key == None:
                 pass
             else:
                temp[current_key] = temp[current_key] + i
     print(temp)

if __name__ == "__main__":
    A20_json_data = A20(json_path, 'イギリス')['text']
    print('A20:', '\n', A20_json_data)
    print('')
    A21_data = A21(A20_json_data, r'\[\[Category:.*\]\]')
    print('A21:')
    for i in A21_data:
        print(i, '\n')
    print('')
    print('A22:', '\n', A22(A21_data, r':(\w*)[^\w]'))
    print('')
    print('A23:', '\n', A23(A20_json_data, r'==+', r'==([^=\n]+)=='))
    print('')
    print('A24:', '\n', A24(A20_json_data, r'(\w+\.(jpg|png|pdf|svg|ogg))'))
    print('')
    A25(A20_json_data, r'{{基礎情報.*\n}}', r'\|(.+?)\s*=\s*(.+)')