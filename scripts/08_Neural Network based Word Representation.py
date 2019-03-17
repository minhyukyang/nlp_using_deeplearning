'''
8. 신경망 기반의 단어 표현(Neural Network based Word Representation) - https://wikidocs.net/22644
'''

'''
1) 워드투벡터(Word2Vec) - https://wikidocs.net/22660
[참고] https://ratsgo.github.io/natural%20language%20processing/2017/03/08/word2vec/
1. 희소 표현(Sparse Representation)
2. 분산 표현(Distributed Representation)
3. CBOW(Continuous Bag of Words)
  CBOW는 주변에 있는 단어들을 가지고, 중간에 있는 단어들을 예측하는 방법입니다.
  CBOW에서 hidden layer의 크기 N은 임베딩하고 난 벡터의 크기가 됩니다. 
  다시 말해, 위의 그림에서 hidden layer의 크기는 N=5이기 때문에 해당 CBOW를 수행하고나서 나오는 벡터의 크기는 5가 될 것입니다.
4. Skip-gram
  Skip-gram은 CBOW를 이해했다면, 메커니즘 자체는 동일하기 때문에 쉽게 이해할 수 있습니다. 
  앞서 CBOW에서는 주변 단어를 통해 중심 단어를 예측했다면, Skip-gram은 중심 단어에서 주변 단어를 예측하려고 합니다.
5. 영어 Word2Vec 만들기
6. 한글 Word2Vec 만들기
'''

# 5. 영어 Word2vec 만들기
# Download link : https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip
import re
from lxml import etree
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

targetXML=open(r'corpus_en/ted_en-20160408.xml', 'r', encoding='UTF8')
# 예를 들어 저자의 경우 윈도우 바탕화면에서 작업하여서 'C:\Users\USER\Desktop\ted_en-20160408.xml'이 해당 파일의 경로였습니다.
target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))
# xml 파일로부터 <content>와 </content> 사이의 내용만 가져옵니다.

content_text = re.sub(r'\([^)]*\)', '', parse_text)
# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거합니다.
# 해당 코드는 괄호로 구성된 내용을 제거하는 코드입니다.

sent_text=sent_tokenize(content_text)
# 입력 코퍼스에 대해서 nltk를 이용하여 문장 토큰화를 수행합니다.

normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)
# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환합니다.

result=[]
result=[word_tokenize(sentence) for sentence in normalized_text]
# 각 문장에 대해서 nltk를 이용하여 단어 토큰화를 수행합니다.

print(result[:10])
# 문장 10개만 출력

from gensim.models import Word2Vec
model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)

a=model.wv.most_similar("man")
print(a)
# [('woman', 0.8412898182868958), ('guy', 0.8099554181098938), ('boy', 0.7764801979064941), ('lady', 0.7613880634307861), ('girl', 0.7509098052978516), ('gentleman', 0.7428570985794067), ('soldier', 0.7122098207473755), ('kid', 0.6934249401092529), ('poet', 0.6682940721511841), ('friend', 0.6607728004455566)]

# 6. 한글 Word2Vec 만들기
# (1) 위키피디아 한글 덤프 파일 다운로드
# Download link : https://dumps.wikimedia.org/kowiki/latest/
# (2) 위키피디아 익스트랙터 다운로드
# https://github.com/attardi/wikiextractor 위 링크로 직접 이동하여 zip 파일로 다운로드 하고 압축을 푼 뒤에,
# 윈도우의 명령 프롬프트나 MAC과 리눅스의 터미널에서 python setup.py install 명령어를 치면 '위키피디아 익스트랙터'가 다운로드 됩니다.
# (3) 위키피디아 한글 덤프 파일 변환
# 위키피디아 익스트랙터와 위키피디아 한글 덤프 파일을 동일한 디렉토리 경로에 두고, 아래 명령어를 실행하면 위키피디아 덤프 파일이 텍스트 파일로 변환됩니다.
# 컴퓨터마다 다르지만 보통 10분 내외의 시간이 걸립니다.
# python WikiExtractor.py kowiki-latest-pages-articles.xml.bz2
# (4) 훈련 데이터 만들기
# 우선 AA 디렉토리 안의 모든 파일인 wiki00 ~ wiki90에 대해서 wikiAA.txt로 통합해보도록 하겠습니다.
# copy AA디렉토리의 경로\wiki* wikiAA.txt
# 해당 커맨드는 AA디렉토리 안의 wiki로 시작되는 모든 파일을 wikiAA.txt 파일에 전부 복사하라는 의미를 담고있습니다.
# 결과적으로 wiki00 ~ wiki90파일의 모든 내용은 wikiAA.txt 파일이라는 하나의 파일에 내용이 들어가게 됩니다.
# copy 현재 디렉토리의 경로\wikiA* wiki_data.txt
# (5) word2vec 작업
# 내용 없음

'''
2) 패스트텍스트(FastText) - https://wikidocs.net/22883
  단어를 벡터로 만드는 또 다른 방법으로는 페이스북에서 개발한 패스트텍스트(FastText)가 있습니다. 
  워드투벡터 이후에 나온 것이기 때문에, 메커니즘 자체는 워드투벡터의 확장이라고 볼 수 있습니다. 
  워드투벡터와 패스트텍스트와의 가장 큰 차이점이라면 워드투벡터는 단어를 쪼개질 수 없는 단위로 생각한다면, 
  패스트텍스트는 하나의 단어 안에도 여러 단어들이 존재하는 것으로 간주합니다. 즉 내부 단어(subword)를 고려하여 학습한다는 것입니다.
1. 모르는 단어에 대해서 대응이 가능한 경우가 생긴다
  패스트텍스트에서 각 단어는 글자들의 n-gram으로 나타냅니다.
  builder란 단어가 임베딩이 되었다면 build나 builds와 같은 단어는 builder와 n-gram에서 겹치기 때문에 둘 다 builder와 유사한 단어가 되어 
  해당 단어들이 단어 집합에 없었더라도 대응할 수 있게 됩니다.
2. 단어 집합 내 빈도 수가 적었던 단어에 대한 대응
3. 실습으로 비교하는 워드투벡터 Vs. 패스트텍스트
4. 이미 훈련 된 벡터(Pre-trained vector)의 활용
  페이스북의 패스트텍스트는 294개 언어에 대하여 위키피디아로 학습한 이미 훈련돤 벡터들을 제공합니다. 한글 또한 제공 대상입니다.
  https://fasttext.cc/docs/en/pretrained-vectors.html
5. 페이스북에서 제공하는 패스트텍스트
  위에서 실습에서 사용한 패스트텍스트는 파이썬의 gensim 패키지에서 패스트텍스트를 제공하는 것으로, 
  페이스북에서 직접 제공하는 C++ 기반의 패스트텍스트를 사용하려면 별도의 다운로드 절차를 거쳐야합니다. 
  다만, 현재 윈도우 환경에서는 실습이 불가능하며, MAC이나 리눅스에서만 실습이 가능합니다.
  $ git clone https://github.com/facebookresearch/fastText.git
  $ cd fastText
  $ make
  이후 추가 예정
6. 한글에 적용가능한 패스트텍스트
'''

a = model.wv.most_similar("electrofishing")
# KeyError: "word 'electrofishing' not in vocabulary"

from gensim.models import FastText
model = FastText(result, size=100, window=5, min_count=5, workers=4, sg=1)

a = model.wv.most_similar("electrofishing")
print(a)
# [('electrolux', 0.7934642434120178), ('electrolyte', 0.78279709815979), ('electro', 0.779127836227417), ('electric', 0.7753111720085144), ('airbus', 0.7648627758026123), ('fukushima', 0.7612422704696655), ('electrochemical', 0.7611693143844604), ('gastric', 0.7483425140380859), ('electroshock', 0.7477173805236816), ('overfishing', 0.7435552477836609)]
