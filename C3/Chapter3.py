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
def A25(data: str, pattern1: str, pattern2: str) -> dict:
    info_dic = {}
    current_key = None

    for line in re.findall(pattern1, data, re.DOTALL)[0].split('\n'):
        match = re.findall(pattern2, line)

        if match:
            info_dic[match[0][0]] = match[0][1]
            current_key = match[0][0]
        else:
            if current_key == None:
                pass
            else:
                info_dic[current_key] = info_dic[current_key] + line

    return info_dic


# A26 強調マークアップの除去
def A26(info_data: dict, patterns: list,
        patterns2: list, pattern2: str) -> dict:
    for key in info_data:
        for p in patterns:
            info_data[key] = re.sub(p, '\\1', info_data[key])

        # A27 内部リンクの除去
        # A28 MediaWikiマークアップの除去
        for pattern in patterns2:
            matches1 = re.findall(pattern[0], info_data[key])

            if len(matches1) != 0:
                matches2 = []
                for match in matches1:
                    match2 = re.findall(pattern[1], match[1])
                    if len(match2) == 0:
                        info_data[key] = info_data[key].replace(match[0], match[1])
                    else:
                        info_data[key] = info_data[key].replace(match[0], match2[0])

        info_data[key] = re.sub(pattern2, ' ', info_data[key])

    return info_data

# A29 国旗画像のURLを取得する
import requests
def A29(URL: str, title: str) -> str:
    S = requests.Session()
    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "titles": 'File:' + title,
        "iiprop": 'url',
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()
    IMG_URL = DATA['query']['pages']['-1']['imageinfo'][0]['url']

    return(IMG_URL)

def test_chapter3(country: str):
    print(f'Data of {country}:')
    A20_json_data = A20(json_path, country)['text']
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
    A25_data = A25(A20_json_data, r'{{基礎情報.*\n}}', r'\|(.+?)\s*=\s*(.+)')
    print('A25:', '\n', A25_data)
    print('')
    A26_data = A26(A25_data,
                   [r"''''(.*?)''''", "'''(.*?)'''", "''(.*?)''"],
                   [[r"(\[\[(.*?)\]\])", r'\{\{.*\}\}|[^|]*$'],
                    [r"(\{\{(.*?)\}\})", r".*\|(.*)"]],
                   r"(<.*?>)")
    print('A26, A27, A28:', '\n', A26_data)
    print('')
    print('A29:', A29('https://ja.wikipedia.org/w/api.php', A26_data['国旗画像']))

if __name__ == "__main__":
    test_chapter3('イギリス')
    # test_chapter3('大韓民国')