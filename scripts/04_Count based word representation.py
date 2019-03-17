'''
4. 카운트 기반의 단어 표현(Count based word Representation) - https://wikidocs.net/24557
(1) Local representations
N-grams
원-핫 인코딩
Bag-of-words
(2) Continuous representations
Latent Semantic Analysis
Latent Dirichlet Allocation
Distributed Representations
'''

'''
1) 원-핫 인코딩(One-hot encoding) - https://wikidocs.net/22647
1. 원-핫 인코딩(One-hot encoding)이란?
2. 원-핫 인코딩(One-hot encoding)의 한계
  첫째는 카운트 기반으로 단어의 의미를 벡터화하는 LSA 등이 있으며, 둘째는 신경망으로 단어의 의미를 벡터화하는 NNLM, RNNLM, CBOW, Skip-gram 등
'''

from konlpy.tag import Okt
okt=Okt()
token=okt.morphs("나는 자연어 처리를 배운다")
print(token)
# ['나', '는', '자연어', '처리', '를', '배운다']

word2index={}
for voca in token:
  if voca not in word2index.keys():
    word2index[voca]=len(word2index)
print(word2index)
# {'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}

def one_hot_encoding(word, word2index):
  one_hot_vector = [0] * (len(word2index))
  index = word2index[word]
  one_hot_vector[index] = 1
  return one_hot_vector

one_hot_encoding("자연어",word2index)

'''
2) Bag of Words(BoW) - https://wikidocs.net/22650
1. Bag of Words란?
2. Bag of Words의 다른 예제들
3. CountVectorizer 클래스로 BoW 만들기
4. 불용어를 제거한 BoW 만들기
'''

# 문서1 : 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.
from konlpy.tag import Okt
import re
okt=Okt()

token=re.sub("(\.)","","정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")
# 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.
token=okt.morphs(token)
# OKT 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에, token에다가 넣습니다.

word2index={}
bow=[]
for voca in token:
  if voca not in word2index.keys():
    word2index[voca]=len(word2index)
    # token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.
    bow.insert(len(word2index),1)
    # BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 갯수는 최소 1개 이상이기 때문입니다.
  else:
    index=word2index.get(voca)
    # 재등장하는 단어의 인덱스를 받아옵니다.
    bow[index]=bow[index]+1
    # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 갯수를 세는 것입니다.)

print(word2index)
# ('정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9)
bow
# [1, 2, 1, 1, 2, 1, 1, 1, 1, 1]

# 문서2 : 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.
from konlpy.tag import Okt
import re
okt=Okt()

token=re.sub("(\.)","","소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.")
# 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.
token=okt.morphs(token)
# OKT 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에, token에다가 넣습니다.

word2index={}
bow=[]
for voca in token:
  if voca not in word2index.keys():
    word2index[voca]=len(word2index)
    # token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.
    bow.insert(len(word2index),1)
    # BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 갯수는 최소 1개 이상이기 때문입니다.
  else:
    index=word2index.get(voca)
    # 재등장하는 단어의 인덱스를 받아옵니다.
    bow[index]=bow[index]+1
    # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 갯수를 세는 것입니다.)

print(word2index)

# 문서3: 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다. 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.
from konlpy.tag import Okt
import re
okt=Okt()

token=re.sub("(\.)","","정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다. 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.")
# 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.
token=okt.morphs(token)
# OKT 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에, token에다가 넣습니다.

word2index={}
bow=[]
for voca in token:
  if voca not in word2index.keys():
    word2index[voca]=len(word2index)
    # token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.
    bow.insert(len(word2index),1)
    # BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 갯수는 최소 1개 이상이기 때문입니다.
  else:
    index=word2index.get(voca)
    # 재등장하는 단어의 인덱스를 받아옵니다.
    bow[index]=bow[index]+1
    # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 갯수를 세는 것입니다.)

print(word2index)

# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
# [[1 1 2 1 2 1]]
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
# {'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}

# example : user stopwords
from sklearn.feature_extraction.text import CountVectorizer
text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])
print(vect.fit_transform(text).toarray())
# [[1 1 1 1 1]]
print(vect.vocabulary_)
# {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}

# example : CountVectorizer stopwords
from sklearn.feature_extraction.text import CountVectorizer
text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")
print(vect.fit_transform(text).toarray())
# [[1 1 1]]
print(vect.vocabulary_)
# {'family': 0, 'important': 1, 'thing': 2}

# example : nltk stopwords
from sklearn.feature_extraction.text import CountVectorizer
text=["Family is not an important thing. It's everything."]
from nltk.corpus import stopwords
sw = stopwords.words("english")
vect = CountVectorizer(stop_words =sw)
print(vect.fit_transform(text).toarray())
# [[1 1 1 1]]
print(vect.vocabulary_)
# {'family': 1, 'important': 2, 'thing': 3, 'everything': 0}

'''
3) 단어 문서 행렬(Term-Document Matrix) - https://wikidocs.net/24559
1. 단어 문서 행렬(Term-Document Matrix)의 표기법
  불용어와 중요한 단어에 대해서 가중치를 줄 수 있는 방법은 없을까?
2. TF-IDF(단어 빈도-역문서 빈도, Term Frequency-Inverse Document Frequency)
  TF-IDF는 Term Frequency-Inverse Document Frequency의 줄임말로, 단어의 빈도와 역문서 빈도(문서의 빈도에 특정 식을 취함)를 사용하여 단어들마다 중요한 정도를 가중치를 주는 방법
  문서를 d, 단어를 t, 문서의 총 개수를 n라고 할 때,
  (1) tf(d,t) : 특정 문서 d에서의 특정 단어 t의 등장 횟수.
  (2) df(t) : 특정 단어 t가 등장한 문서의 수.
  (3) idf(t) : df(t)에 반비례하는 수. --> idf(d,t) = ln(n/(1+df(t))
  TF-IDF = TF * IDF
3. 사이킷 런을 이용한 TDM과 TF-IDF 실습
  사이킷런은 TF-IDF를 자동 계산해주는 TfidVectorizer 클래스를 제공합니다. 향후 실습을 하다가 혼란이 생기지 않도록 언급하자면, 
  사이킷런의 TF-IDF는 우리가 위에서 배웠던 보편적인 TF-IDF식에서 좀 더 조정된 다른 식을 사용합니다. 
  하지만 크게 다른 식은 아니며(IDF 계산 시 분자에다가도 1을 더해주며, TF-IDF에 L2 정규화라는 방법으로 값을 조정하는 등의 차이), 
  여전히 TF-IDF가 가진 의도를 그대로 갖고 있으므로 사이킷런의 TF-IDF를 그대로 사용하셔도 좋습니다.
  
'''

# 문서1 : 먹고 싶은 사과
# 문서2 : 먹고 싶은 바나나
# 문서3 : 길고 노란 바나나 바나나
# 문서4 : 저는 과일이 좋아요

# example : TF-IDF
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',
]
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
# [[0 1 0 1 0 1 0 1 1]
#  [0 0 1 0 0 0 0 1 0]
#  [1 0 0 0 1 0 1 0 0]]
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
# {'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}

# example : TF-IDF in sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',
]
tfidfv = TfidfVectorizer().fit(corpus)
# print(tfidfv.transform(corpus).toarray())
# [[0.         0.46735098 0.         0.46735098 0.         0.46735098
#   0.         0.35543247 0.46735098]
#  [0.         0.         0.79596054 0.         0.         0.
#   0.         0.60534851 0.        ]
#  [0.57735027 0.         0.         0.         0.57735027 0.
#   0.57735027 0.         0.        ]]
print(tfidfv.vocabulary_)
# {'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}

