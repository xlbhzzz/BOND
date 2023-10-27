import re
import pinyin
import unicodedata

names_wrong = [
    # find in train
    (['takahiro', 'toshiyuki', 'takeshi', 'toshiyuki', 'tomohiro', 'takamitsu', 'takahisa', 'takashi',
     'takahiko', 'takayuki'], 'ta(d|k)ashi'),
    (['akimasa', 'akio', 'akito'], 'akira'),
    (['kentarok'], 'kentaro'),
    (['xiaohuatony', 'tonyxiaohua'], 'xiaohua'),
    (['ulrich'], 'ulrike'),
    # find in valid
    (['naoto', 'naomi'], 'naoki'),
    (['junko'], 'junichi'),
    # find in test
    (['isaku'], 'isao')
]


# 检查是否含有中文字符
def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


def match_name(name, target_name):
    [first_name, last_name] = target_name.split('_')
    first_name = re.sub('-', '', first_name)
    # 中文名转化为拼音形式
    if is_contains_chinese(name):
        name = re.sub('[^ \u4e00-\u9fa5]', '', name).strip()
        name = pinyin.get(name, format='strip')
        # 两个字的人名中间可能有空格
        name = re.sub(' ', '', name)
        target_name = last_name + first_name
        return name == target_name
    else:
        # 处理带声调的拼音
        str_bytes = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore')
        name = str_bytes.decode('ascii')

        name = name.lower()
        name = re.sub('[^a-zA-Z]', ' ', name)
        tokens = name.split()

        if len(tokens) < 2:
            return False
        if len(tokens) == 3:
            # just ignore middle name
            if re.match(tokens[0], first_name) and re.match(tokens[-1], last_name):
                return True
            # ignore tail noise char
            if tokens[-1] == 'a' or tokens[-1] == 'c':
                tokens = tokens[:-1]

        if re.match(tokens[0], last_name):
            # 中文名两个字母缩写(如果只有一个字母原方案能解决)
            if len(tokens) == 2 and len(tokens[1]) == 2:
                if re.match(f'{tokens[1][0]}.*{tokens[1][1]}.*', first_name):
                    return True
            remain = '.*'.join(tokens[1:]) + '.*'

            if re.match(remain, first_name):
                return True
            if len(tokens) == 3 and len(tokens[1]) == 1 and len(tokens[2]) == 1:
                remain_reverse = f'{tokens[2]}.*{tokens[1]}.*'
                if re.match(remain_reverse, first_name):
                    return True
        if re.match(tokens[-1], last_name):
            candidate = ''.join(tokens[:-1])
            find_remain = False
            for (wrong_list, right_one) in names_wrong:
                if candidate in wrong_list:
                    remain = right_one
                    find_remain = True
                    break
            if not find_remain:
                remain = '.*'.join(tokens[:-1]) + '.*'

            if re.match(remain, first_name):
                return True
            if len(tokens) == 3 and len(tokens[0]) == 1 and len(tokens[1]) == 1:
                remain_reverse = f'{tokens[1]}.*{tokens[0]}.*'
                if re.match(remain_reverse, first_name):
                    return True
        return False