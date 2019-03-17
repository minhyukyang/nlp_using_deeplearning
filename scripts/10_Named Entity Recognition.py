'''
10. 개체명 인식(Named Entity Recognition) - https://wikidocs.net/25042
'''

'''
1) 개체명 인식(Named Entity Recognition)
1. BIO 표현
  B는 Begin의 약자로 개체명이 시작되는 부분, I는 Inside의 약자로 개체명의 내부 부분을 의미하며, O는 Outside의 약자로 개체명이 아닌 부분을 의미합니다.
   
  영화 제목 개체명 추출
  해 B
  리 I
  포 I
  터 I
  보 O
  러 O
  가 O
  자 O
  
  복수의 개체명 추출(using TAG)
  해 B-movie
  리 I-movie
  포 I-movie
  터 I-movie
  보 O
  러 O
  메 B-theater
  가 I-theater
  박 I-theater
  스 I-theater
  가 O
  자 O
2. 개체명 인식을 위한 데이터 전처리
  개체명 인식을 위한 전통적인 영어 데이터 셋 : https://github.com/Franck-Dernoncourt/NeuroNER/tree/master/data/conll2003/en
  
  [단어] [품사 태깅] [청크 태깅] [개체명 태깅]의 형식
  EU NNP B-NP B-ORG
  rejects VBZ B-VP O
  German JJ B-NP B-MISC
  call NN I-NP O
  to TO B-VP O
  boycott VB I-VP O
  British JJ B-NP B-MISC
  lamb NN I-NP O
  . . O O
  
  Peter NNP B-NP B-PER
  Blackburn NNP I-NP I-PER
'''

# 2. 개체명 인식을 위한 데이터 전처리
from collections import Counter
vocab=Counter()
import re

f = open('corpus_ner/train.txt', 'r')
# 예를 들어 윈도우 바탕화면에 해당 파일을 위치시킨 저자의 경우
# f = open(r"C:\Users\USER\Desktop\train.txt", 'r')
sentences = []
sentence = []
ner_set = set()
# 파이썬의 set은 중복을 허용하지 않는다. 개체명 태깅의 경우의 수. 즉, 종류를 알아내기 위함이다.

for line in f:
  if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
    if len(sentence) > 0:
      sentences.append(sentence)
      sentence=[]
    continue
  splits = line.split(' ')
  # 공백을 기준으로 속성을 구분한다.
  splits[-1] = re.sub(r'\n', '', splits[-1])
  # 개체명 태깅 뒤에 붙어있는 줄바꿈 표시 \n을 제거한다.
  word=splits[0].lower()
  # 단어들은 소문자로 바꿔서 저장한다. 단어의 수를 줄이기 위해서이다.
  vocab[word]=vocab[word]+1
  # 단어마다 빈도 수가 몇 인지 기록한다.
  sentence.append([word, splits[-1]])
  # 단어와 개체명 태깅만 기록한다.
  ner_set.add(splits[-1])
  # set에다가 개체명 태깅을 집어 넣는다. 중복은 허용되지 않으므로
  # 나중에 개체명 태깅이 어떤 종류가 있는지 확인할 수 있다.

sentences[:3]

vocab
# Counter({'eu': 24,
#          'rejects': 1,
#          'german': 101,
#          'call': 38,
#          'to': 3424,
#          'boycott': 5,
#          'british': 96,
#          'lamb': 3,
#          '.': 7374,
#          'peter': 31,
#          'blackburn': 12,
#          'brussels': 33,
#          '1996-08-22': 125,
#          'the': 8390,
#          'european': 94,
#          'commission': 67,
#          ... 중략 ...
#          })

len(vocab)
# 21009

print(ner_set)
# {'I-ORG', 'B-MISC', 'O', 'B-ORG', 'I-PER', 'I-LOC', 'I-MISC', 'B-LOC', 'B-PER'}

vocab_sorted=sorted(vocab.items(), key=lambda x:x[1], reverse=True)
# vocab을 빈도수 순으로 정렬한다.
vocab_sorted
# 출력
# [('the', 8390),
#  ('.', 7374),
#  (',', 7290),
#  ('of', 3815),
#  ('in', 3621),
#  ('to', 3424),
#  ('a', 3199),
#  ('and', 2872),
#  ('(', 2861),
#  (')', 2861),
#  ('"', 2178),
#  ('on', 2092),
#  ('said', 1849),
#  ("'s", 1566),
#  ('for', 1465),
#  ('1', 1421),
#  ... 중략 ...

word_to_index={'padding' : 0}
i=0
# 인덱스 0은 이후에 입력값들의 길이를 맞추기 위해 패딩에 사용될 것입니다.
for (word, frequency) in vocab_sorted :
    if frequency > 5 :
        i=i+1
        word_to_index[word]=i
print(word_to_index)
# {'padding': 0,
#  'the': 1,
#  '.': 2,
#  ',': 3,
#  'of': 4,
#  'in': 5,
#  'to': 6,
#  'a': 7,
#  'and': 8,
#  '(': 9,
#  ')': 10,
#  '"': 11,
#  'on': 12,
#  'said': 13,
#  "'s": 14,
#  'for': 15,
#  '1': 16,
#  ... 중략 ...

print(len(word_to_index))
# 3938

word_to_index['the']
# 1

ner_to_index={}
i=0
for ner in ner_set:
    i=i+1
    ner_to_index[ner]=i
print(ner_to_index)
# 5

x_data=[]
y_data=[]

for sentence in sentences:
    x=[]
    y=[]
    for(w, L) in sentence:
        x.append(word_to_index.get(w, -1))
        y.append(ner_to_index.get(L))

    x_data.append(x)
    y_data.append(y)

