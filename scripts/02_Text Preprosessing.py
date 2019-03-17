'''
2. 텍스트 전처리(Text preprocessing) - https://wikidocs.net/21694
'''

'''
1) 토큰화(Tokenization) - https://wikidocs.net/21698
1. 단어 토큰화(Word Tokenization)
2. 토큰화 중 생기는 선택의 순간
3. 토큰화에서 고려해야할 사항
4. 문장 토큰화(Sentence Tokenization)
5. 이진 분류기(Binary Classifier)
6. 한글에서의 토큰화의 어려움.
7. 품사 부착(Part-of-speech tagging)
8. NLTK와 KoNLPy를 이용한 영어, 한글 토큰화 실습
'''

# word_tokenize
import nltk
from nltk.tokenize import word_tokenize
print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
# ['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']

# WordPunctTokenizer
import nltk
from nltk.tokenize import WordPunctTokenizer
WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.")
# ['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']

# TreebankWordTokenizer
import nltk
from nltk.tokenize import TreebankWordTokenizer
tokenizer=TreebankWordTokenizer()
text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
tokenizer.tokenize(text)
# ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']

# Sentence Tokenization
import nltk
from nltk.tokenize import sent_tokenize
text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."
print(sent_tokenize(text))
# ['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to mae sure no one was near.']

text="I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))

# Eng/Kor Tokenizing using nltk and KoNLPy
import nltk
from nltk.tokenize import word_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
print(word_tokenize(text))
# ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
from nltk.tag import pos_tag
x=word_tokenize(text)
pos_tag(x)
# [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
# Penn Treebank POG Tags에서 PRP는 인칭 대명사, VBP는 동사, RB는 부사, VBG는 현재부사, IN은 전치사, NNP는 고유 명사, NNS는 복수형 명사, CC는 접속사, DT는 관사를 의미합니다.

from konlpy.tag import Okt
okt=Okt()
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# [('열심히','Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# ['코딩', '당신', '연휴', '여행']

from konlpy.tag import Kkma
kkma=Kkma()
print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# [('열심히','MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# ['코딩', '당신', '연휴', '여행']

'''
한글 형태소 분석기 중에 가장 속도가 빠른 Mecab은 konlpy 엔진에 포함되어 있지 않다.
아래는 eunjeon 패키지를 이용하여 python에서 mecab을 활용하는 예시이다.
'''
from eunjeon import Mecab  # KoNLPy style mecab wrapper
tagger = Mecab()
print(tagger.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# ['열심히', '코딩', '한', '당신', ',', '연휴', '에', '는', '여행', '을', '가', '봐요']
print(tagger.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# [('열심히', 'MAG'), ('코딩', 'NNG'), ('한', 'XSA+ETM'), ('당신', 'NP'), (',', 'SC'), ('연휴', 'NNG'), ('에', 'JKB'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가', 'VV'), ('봐요', 'EC+VX+EC')]
print(tagger.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# ['코딩', '당신', '연휴', '여행']

'''
2) 정제(Normalization) - https://wikidocs.net/21693
1. 규칙에 기반한 표기가 다른 단어들의 통합
2. 대, 소문자 통합
3. 정규 표현식(Regular Expression)
'''

'''
3) 어간 추출(Stemming) and 표제어 추출(Lemmatization) - https://wikidocs.net/21707
1. 표제어 추출(Lemmatization)
2. 어간 추출(Stemming)
'''

# WordNetLemmatizer
import nltk
from nltk.stem import WordNetLemmatizer
n=WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
[n.lemmatize(w) for w in words]
# ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']

n.lemmatize('dies', 'v')
# 'die'
n.lemmatize('watched', 'v')
# 'watch'
n.lemmatize('has', 'v')
# 'have'

# porterStemmer
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
s = PorterStemmer()
text="This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
words=word_tokenize(text)
print(words)
# ['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']
[s.stem(w) for w in words]
# ['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', 'with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']

import nltk
from nltk.stem import PorterStemmer
s = PorterStemmer()
words=['formalize', 'allowance', 'electricical']
[s.stem(w) for w in words]
# ['formal', 'allow', 'electric']

# LancasterStemmer
import nltk
from nltk.stem import PorterStemmer
s=PorterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
[s.stem(w) for w in words]
# ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
import nltk
from nltk.stem import LancasterStemmer
l=LancasterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
[l.stem(w) for w in words]
# ['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']

'''
4) 불용어(Stopword) - https://wikidocs.net/22530
1. nltk에서 불용어 확인하기
2. nltk를 통해서 불용어 제거하기
3. 한글에서 불용어 제거하기
'''

# stopword
import nltk
from nltk.corpus import stopwords
stopwords.words('english')[:10]
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']

# remove stopwords in English
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example)
result = []

for w in word_tokens:
    if w not in stop_words:
        result.append(w)

print(word_tokens)
# ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
print(result)
# ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']

# remove stopwords in Korean
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"
stop_words=stop_words.split(' ')

word_tokens = word_tokenize(example)
result = []

for w in word_tokens:
    if w not in stop_words:
        result.append(w)

print(word_tokens)
# ['고기를', '아무렇게나', '구우려고', '하면', '안', '돼', '.', '고기라고', '다', '같은', '게', '아니거든', '.', '예컨대', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']
print(result)
# ['고기를', '구우려고', '안', '돼', '.', '고기라고', '다', '같은', '게', '.', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']

'''
5) 정규 표현식(Regular Expression) - https://wikidocs.net/21703
1. 정규 표현식 문법과 모듈 함수
2. 정규 표현식 실습
3. 정규 표현식 모듈 함수 예제
4. 정규 표현식 텍스트 전처리 예제
5. 정규 표현식을 이용한 토큰화
'''

# example - .
import re
r=re.compile("a.c")
r.search("kkk") # 아무런 결과도 출력되지 않는다.
r.search("abc")
# <_sre.SRE_Match object; span=(0, 3), match='abc'>

# example - ?
import re
r=re.compile("ab?c")
r.search("abbc") # 아무런 결과도 출력되지 않는다.
r.search("abc")
# <_sre.SRE_Match object; span=(0, 3), match='abc'>
r.search("ac")
# <_sre.SRE_Match object; span=(0, 2), match='ac'>

# example - *
import re
r=re.compile("ab*c")
r.search("a") # 아무런 결과도 출력되지 않는다.
r.search("ac")
# <_sre.SRE_Match object; span=(0, 2), match='ac'>
r.search("abc")
# <_sre.SRE_Match object; span=(0, 3), match='abc'>
r.search("abbbbc")
# <_sre.SRE_Match object; span=(0, 6), match='abbbbc'>

# example - +
import re
r=re.compile("ab+c")
r.search("ac") # 아무런 결과도 출력되지 않는다.
r.search("abc")
# <_sre.SRE_Match object; span=(0, 3), match='abc'>
r.search("abbbbc")
# <_sre.SRE_Match object; span=(0, 6), match='abbbbc'>

# example - ^
import re
r=re.compile("^a")
r.search("bbc") # 아무런 결과도 출력되지 않는다.
r.search("ab")
# <_sre.SRE_Match object; span=(0, 1), match='a'>

# example - {num}
import re
r=re.compile("ab{2}c")
r.search("ac") # 아무런 결과도 출력되지 않는다.
r.search("abc") # 아무런 결과도 출력되지 않는다.
r.search("abbc")
# <_sre.SRE_Match object; span=(0, 4), match='abbc'>
r.search("abbbbbc") # 아무런 결과도 출력되지 않는다.

# example - {num1,num2}
import re
r=re.compile("ab{2,8}c")
r.search("ac") # 아무런 결과도 출력되지 않는다.
r.search("ac") # 아무런 결과도 출력되지 않는다.
r.search("abc") # 아무런 결과도 출력되지 않는다.
r.search("abbc")
# <_sre.SRE_Match object; span=(0, 4), match='abbc'>
r.search("abbbbbbbbc")
# <_sre.SRE_Match object; span=(0, 10), match='abbbbbbbbc'>
r.search("abbbbbbbbbc") # 아무런 결과도 출력되지 않는다.

# example - {num,}
import re
r=re.compile("a{2,}bc")
r.search("bc") # 아무런 결과도 출력되지 않는다.
r.search("aa") # 아무런 결과도 출력되지 않는다.
r.search("aabc")
# <_sre.SRE_Match object; span=(0, 4), match='aabc'>
r.search("aaaaaaaabc")
# <_sre.SRE_Match object; span=(0, 10), match='aaaaaaaabc'>

# example - [ ]
import re
r=re.compile("[abc]") # [abc]는 [a-c]와 같다.
r.search("zzz") # 아무런 결과도 출력되지 않는다.
r.search("a")
# <_sre.SRE_Match object; span=(0, 1), match='a'>
r.search("aaaaaaa")
# <_sre.SRE_Match object; span=(0, 1), match='a'>
r.search("baac")
# <_sre.SRE_Match object; span=(0, 1), match='b'>

import re
r=re.compile("[a-z]")
r.search("AAA") # 아무런 결과도 출력되지 않는다.
r.search("aBC")
# <_sre.SRE_Match object; span=(0, 1), match='a'>
r.search("111") # 아무런 결과도 출력되지 않는다.

# example - [^char]
import re
r=re.compile("[^abc]")
r.search("a") # 아무런 결과도 출력되지 않는다.
r.search("ab") # 아무런 결과도 출력되지 않는다.
r.search("b") # 아무런 결과도 출력되지 않는다.
r.search("d")
# <_sre.SRE_Match object; span=(0, 1), match='d'>
r.search("1")
# <_sre.SRE_Match object; span=(0, 1), match='1'>

# re.match(), re.search()
import re
r=re.compile("ab.")
r.search("kkkabc")
# <_sre.SRE_Match object; span=(3, 6), match='abc'>
r.match("kkkabc")  #아무런 결과도 출력되지 않는다.
r.match("abckkk")
# <_sre.SRE_Match object; span=(0, 3), match='abc'>

# re.split()
import re
text="사과 딸기 수박 메론 바나나"
re.split(" ",text)
# ['사과', '딸기', '수박', '메론', '바나나']

import re
text="""사과
딸기
수박
메론
바나나"""
re.split("\n",text)
# ['사과', '딸기', '수박', '메론', '바나나']
import re
text="사과+딸기+수박+메론+바나나"
re.split("\+",text)
# ['사과', '딸기', '수박', '메론', '바나나']

# re.findall()
import re
text="""이름 : 김철수
전화번호 : 010 - 1234 - 1234
나이 : 30
성별 : 남"""
re.findall("\d+",text)

re.findall("\d+", "문자열입니다.")

# re.sub()
import re
text="Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."
re.sub('[^a-zA-Z]',' ',text)

# example - regular expression
import re
text = """100 John    PROF
101 James   STUD
102 Mac   STUD"""
re.split('\s+', text)
# ['100', 'John', 'PROF', 101, 'James', 'STUD', '102', 'Mac', 'STUD']

re.findall('\d+',text)
# ['100', '101', '102]

re.findall('[A-Z]',text)
# ['J', 'P', 'R', 'O', 'F', 'J', 'S', 'T', 'U', 'D', 'M', 'S', 'T', 'U', 'D']

re.findall('[A-Z]{4}',text)
# ['PROF', 'STUD', 'STUD']

re.findall('[A-Z][a-z]+',text)
# ['John', 'James', 'Mac']
import re
letters_only = re.sub('[^a-zA-Z]', ' ', text)
print(letters_only)

# Tokenize using regular expression
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer("[\w]+")
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))
# ['Don', 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'Mr', 'Jone', 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']

import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer("[\s]+", gaps=True)
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))
# ["Don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name,', 'Mr.', "Jone's", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']

'''
6) 단어 분리(Subword Segmentation) - https://wikidocs.net/22592
내용 기재 예정
'''

'''
7) 어떤 Applicaiton에서 사용하는가? - https://wikidocs.net/21704
1. Sentiment Analysis (감정 분석)
2. Spell Correction (철자 교정)
3. Machine Translation (기계 번역)
추가 내용 기재 예정
'''

'''
8) 최소 편집 거리 - https://wikidocs.net/21705
내용 기재 예정
'''