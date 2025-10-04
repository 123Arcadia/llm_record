from collections import defaultdict, Counter

from pygments.lexer import default


def test_daultdict():
    dd = defaultdict(int)
    dd['a'] = 1
    dd['a'] = 2
    dd['b'] = 1
    dd['c'] = 2
    print(f'{dd=}')
    # dd=defaultdict(<class 'int'>, {'a': 1, 'b': 1, 'c': 2})
    # dd=defaultdict(<class 'int'>, {'a': 2, 'b': 1, 'c': 2})
    print(f'{dd.items()}')
    #     dict_items([('a', 1), ('b', 1), ('c', 2)])
    # dict_items([('a', 2), ('b', 1), ('c', 2)])
    sorted(dd.items(), key=lambda x: (-x[1], x[0]))
    print(f'{dd=}')
    # dd=defaultdict(<class 'int'>, {'a': 2, 'b': 1, 'c': 2})


def test_get_dict():
    ddd = {'a': 1}
    b_o = ddd.get('b', 'unk')
    print(f'{b_o=}')


def test_get_dict_reverse():
    ddd = {'a': 1, "b": 2, 'c': 2}
    reverse_dict = {v:k for k,v in ddd.items()}
    print(f'{reverse_dict=}')
    # reverse_dict={1: 'a', 2: 'c'}

def test_join_in_list():
    text = "Hello, this is a text!"
    print(' '.join(text))
    print(list(text))
    print(' '.join(list(text)))
    # H e l l o ,   t h i s   i s   a   t e x t !
    # ['H', 'e', 'l', 'l', 'o', ',', ' ', 't', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 't', 'e', 'x', 't', '!']
    # H e l l o ,   t h i s   i s   a   t   e x t !
    counter = Counter()
    for w in ' '.join(list(text)):
        counter.update(w)

    print(f'{counter=}')
    # counter=Counter({' ': 25, 't': 3, 'e': 2, 'l': 2, 'i': 2, 's': 2, 'H': 1, 'o': 1, ',': 1, 'h': 1, 'a': 1, 'x': 1, '!': 1})

    state = counter.copy()
    print(f'{state=}')

    for pair ,freq in state.items():
        print(f'{pair} {freq}')

def test_Counter_update_addstr():
    texts = [
        "Hello, how are you today?",
        "I hope you are doing well.",
        "This is a test of the BPE tokenizer."
    ]
    token_counts = Counter()
    token_counts_2 = Counter()
    for text in texts:
        words = [' '.join(list(text))]
        # print(f'{words=}')
        for w in words:
            token_counts_2.update(w)
            token_counts[w] += 1
    print(f'{token_counts_2=}')
    print(f'{token_counts=}')
    # token_counts_2=Counter({' ': 100, 'e': 9, 'o': 9, 't': 5, 'l': 4, 'h': 4, 'a': 4, 'i': 4, 'r': 3, 'y': 3, 's': 3, 'w': 2, 'u': 2, 'd': 2, 'n': 2, '.': 2, 'H': 1, ',': 1, '?': 1, 'I': 1, 'p': 1, 'g': 1, 'T': 1, 'f': 1, 'B': 1, 'P': 1, 'E': 1, 'k': 1, 'z': 1})
    # token_counts=Counter({'H e l l o ,   h o w   a r e   y o u   t o d a y ?': 1, 'I   h o p e   y o u   a r e   d o i n g   w e l l .': 1, 'T h i s   i s   a   t e s t   o f   t h e   B P E   t o k e n i z e r .': 1})

def test_str_split():
    str = 'H e l l o ,   h o w   a r e   y o u   t o d a y ?'
    print(str.split())
    s = ('H', 'e')
    print(' '.join(s)) # H e

def test_list_add_item():
    l = ['zcw']
    n = ['zcww', 'zzz']
    o = ['1', '11'] + l + ['2']
    print(f'{o=}')
    l.insert(0, ['1'])
    l.insert(len(l), ['2'])
    print(f'{l=}')
    # [['1'], 'zcw', ['2']]
    print(l+n)