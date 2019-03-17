'''
9. 텍스트 분류(Text Classification) - https://wikidocs.net/22891
'''

'''
1) 딥 러닝을 이용한 텍스트 분류 - https://wikidocs.net/24873
1. 훈련 데이터에 대한 이해
2. 훈련 데이터와 테스트 데이터
3. 단어에 대한 인덱싱
'''

from collections import Counter
vocab = Counter()
text= " His barber kept his word. But keeping such a huge secret to himself was driving him crazy. " \
      "Finally, the barber went up a mountain and almost to the edge of a cliff. " \
      "He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."

# 기재예정

'''
3) 로이터 뉴스 분류하기(Reuters News Classification) - https://wikidocs.net/22933
  이번 장에서는 케라스에서 제공하는 로이터 뉴스 데이터를 이용하여 텍스트 분류를 진행해보도록 하겠습니다. 
  로이터 뉴스 데이터는 총 11,258개의 뉴스 기사가 46개의 카테고리로 나누어지는 테스트 데이터입니다. 
  우선 데이터가 어떻게 구성되어있는지에 대해서 알아보도록 하겠습니다.
1. 로이터 뉴스 데이터의 구성
2. LSTM으로 학습하기
'''

# 1. 로이터 뉴스 데이터의 구성
from keras.datasets import reuters
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

# 여기서 num_words는 이 데이터에서 등장 빈도 순위로 몇 번째에 해당하는 단어까지를 갖고 올 것인지 조절하는 것입니다.
# 예를 들어서 저기에 100이란 값을 넣으면, 등장 빈도 순위가 1~100에 해당하는 단어만 갖고오게 됩니다. 모든 단어를 사용하고자 한다면 None으로 설정합니다.
# 정확하게 무슨 의미인지 이해가 안간다면, 아래에서 훈련 데이터를 출력할 때를 보시면 이해할 수 있습니다.
#
# test_split은 테스트 데이터를 전체 데이터 중 몇 퍼센트를 사용할 것인지를 의미합니다. 우리는 전체 데이터 중 20%를 테스트 데이터로 사용할 것이므로, 0.2로 설정합니다.

print('훈련 데이터: {}'.format(len(x_train)))
print('테스트 데이터: {}'.format(len(x_test)))
num_classes = max(y_train) + 1
print('카테고리: {}'.format(num_classes))
# 훈련용 데이터의 개수, 테스트 데이터의 개수, 카테고리의 수를 출력하는 코드입니다.
# 훈련 데이터: 8982
# 테스트 데이터: 2246
# 카테고리: 46

print(x_train[0])
print(y_train[0])
# [1, 27595, 28842, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124,
# 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579,
# 1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12]
# 3

# 첫번째 훈련 데이터에서 x에는 숫자들이 들어있습니다. 텍스트 데이터가 아니라서 의아할 수 있는데,
# 현재 이 데이터는 토크나이제이션과 단어의 등장 횟수를 세는 것과 단어에 인덱스를 부여하는 전처리가 끝난 상태라고 이해하시면 되겠습니다.

word_index = reuters.get_word_index()
# reuters.get_word_index는 각 단어와 그 단어에 부여된 인덱스를 리턴
print(word_index)
# {'mdbl': 10996, 'fawc': 16260, 'degussa': 12089, 'woods': 8803, 'hanging': 13796, 'localized': 20672, 'sation': 20673,
# 'chanthaburi': 20675, 'refunding': 10997, 'hermann': 8804, 'passsengers': 20676, 'stipulate': 20677, 'heublein': 8352,
# 'screaming': 20713, 'tcby': 16261, 'four': 185, 'grains': 1642, 'broiler': 20680, 'wooden': 12090, 'wednesday': 1220,
# 'highveld': 13797, 'duffour': 7593, '0053': 20681, 'elections': 3914, '270': 2563, '271': 3551, '272': 5113, '273': 3552,
# '274': 3400, 'rudman': 7975, '276': 3401, '277': 3478, '278': 3632, '279': 4309, 'dormancy': 9381, - 이하 생략 -}

index_to_word={}
for key, value in word_index.items():
    index_to_word[value] = key

print(index_to_word[28842])
# nondiscriminatory

# most frequent word
print(index_to_word[1])

print(' '.join([index_to_word[x] for x in x_train[0]]))
# the wattie nondiscriminatory mln loss for plc said at only ended said commonwealth could 1 traders now april 0 a after
# said from 1985 and from foreign 000 april 0 prices its account year a but in this mln home an states earlier and rise
# and revs vs 000 its 16 vs 000 a but 3 psbr oils several and shareholders and dividend vs 000 its all 4 vs 000 1 mln
# agreed largely april 0 are 2 states will billion total and against 000 pct dlrs

# 2. LSTM으로 학습하기
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

x_train = sequence.pad_sequences(x_train, maxlen=100)
x_test = sequence.pad_sequences(x_test, maxlen=100)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Embedding(10000, 120))
model.add(LSTM(120))
model.add(Dense(46, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=100, epochs=50, validation_data=(x_test,y_test))

print("\n 테스트 정확도: %.4f" % (model.evaluate(x_test, y_test)[1]))
# 테스트 정확도: 0.6803

'''
4) IMDB 리뷰 감성 분류하기(IMDB Movie Review Sentiment Analysis) - https://wikidocs.net/24586
  논문 링크 : http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf
1. IMDB 리뷰 데이터의 이해 
2. LSTM으로 학습하기
 
'''

# 1. IMDB 리뷰 데이터의 이해
from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
# 영화 리뷰는 x_train에, 감성 정보는 y_train에 저장된다.

print('훈련 데이터: {}'.format(len(x_train)))
print('테스트 데이터: {}'.format(len(x_test)))
num_classes = max(y_train) + 1
print('카테고리: {}'.format(num_classes))
# 훈련 데이터: 25000
# 테스트 데이터: 25000
# 카테고리: 2

print(x_train[0])
print(y_train[0])
# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]
# 1

word_index = imdb.get_word_index()
index_to_word={}
for key, value in word_index.items():
  index_to_word[value] = key

print(index_to_word[1])
# the
print(index_to_word[3941])
# journalist

print(' '.join([index_to_word[x] for x in x_train[0]]))
# the as you with out themselves powerful lets loves their becomes reaching had journalist of lot from anyone to have after out atmosphere never more room and it so heart shows to years of every never going and help moments or of every chest visual movie except her was several of enough more with is now current film as you of mine potentially unfortunately of you than him that with out themselves her get for was camp of you movie sometimes movie that with scary but and to story wonderful that in seeing in character to of 70s musicians with heart had shadows they of here that with her serious to have does when from why what have critics they is you that isn't one will very to as itself with other and in of seen over landed for anyone of and br show's to whether from than out themselves history he name half some br of and odd was two most of mean for 1 any an boat she he should is thought frog but of script you not while history he heart to real at barrel but when from one bit then have two of script their with her nobody most that with wasn't to with armed acting watch an for with heartfelt film want an

# 2. LSTM으로 학습하기
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

x_train = sequence.pad_sequences(x_train, maxlen=500)
x_test = sequence.pad_sequences(x_test, maxlen=500)

model = Sequential()
model.add(Embedding(5000, 32))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=64)
scores = model.evaluate(x_test, y_test, verbose=0) # 테스트 데이터에 대해서 정확도 평가
print("정확도: %.2f%%" % (scores[1]*100))

'''
5) 나이브 베이즈 분류기(Naive Bayes Classifier) - https://wikidocs.net/22892
1) 베이즈의 정리(Bayes' theorem)를 이용한 분류 메커니즘
  P(정상 메일 | 입력 텍스트) = P(w1 | 정상 메일) * P(w2 | 정상 메일) * P(w3 | 정상 메일) * P(정상 메일)
  P(스팸 메일 | 입력 텍스트) = P(w1 | 스팸 메일) * P(w2 | 스팸 메일) * P(w3 | 스팸 메일) * P(스팸 메일)
2) 스팸 메일 분류기(Spam Detection)
  P(정상 메일 | 입력 텍스트) = P(you | 정상 메일) * P(free | 정상 메일) * P(lottery | 정상 메일) * P(정상 메일)
  P(스팸 메일 | 입력 텍스트) = P(you | 스팸 메일) * P(free | 스팸 메일) * P(lottery | 스팸 메일) * P(스팸 메일)
  라플라스 스무딩 : 각 단어에 대한 확률의 분모, 분자에 전부 숫자를 더해서 분자가 0이 되는 것을 방지
3) 뉴스 데이터 분류하기(Classification of 20 News Group with Naive Bayes Classifier)
  (1) 뉴스 데이터에 대한 이해
  (2) 나이브 베이즈 분류
'''

# 3) 뉴스 데이터 분류하기(Classification of 20 News Group with Naive Bayes Classifier)
# (1) 뉴스 데이터에 대한 이해
from sklearn.datasets import fetch_20newsgroups
newsdata=fetch_20newsgroups(subset='train')
print(newsdata.keys())
# dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR', 'description'])

print (len(newsdata.data), len(newsdata.filenames), len(newsdata.target_names), len(newsdata.target))
# 11314 11314 20 11314

print(newsdata.target_names)
# ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

print(newsdata.target[0])
# 7
print(newsdata.target_names[7])
# rec.autos
print(newsdata.data[0])
# From: lerxst@wam.umd.edu (where's my thing)
# Subject: WHAT car is this!?
# Nntp-Posting-Host: rac3.wam.umd.edu
# Organization: University of Maryland, College Park
# Lines: 15
#
#  I was wondering if anyone out there could enlighten me on this car I saw
# the other day. It was a 2-door sports car, looked to be from the late 60s/
# early 70s. It was called a Bricklin. The doors were really small. In addition,
# the front bumper was separate from the rest of the body. This is
# all I know. If anyone can tellme a model name, engine specs, years
# of production, where this car is made, history, or whatever info you
# have on this funky looking car, please e-mail.
#
# Thanks,
# - IL
#    ---- brought to you by your neighborhood Lerxst ----

# (2) 나이브 베이즈 분류
from sklearn.feature_extraction.text import CountVectorizer
tdmvector = CountVectorizer()
X_train_tdm = tdmvector.fit_transform(newsdata.data)
print(X_train_tdm.shape)
# (11314, 130107)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
tfidfv = tfidf_transformer.fit_transform(X_train_tdm)
print(X_train_tdm.shape)
# (11314, 130107)

from sklearn.naive_bayes import MultinomialNB # 다항분포 나이브 베이즈 모델
mod = MultinomialNB()
mod.fit(tfidfv, newsdata.target)

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

from sklearn.metrics import accuracy_score #정확도 계산을 위한 함수
newsdata_test = fetch_20newsgroups(subset='test', shuffle=True) #테스트 데이터 갖고오기
X_test_tdm = tdmvector.transform(newsdata_test.data) #테스트 데이터를 TDM으로 변환
tfidfv_test = tfidf_transformer.transform(X_test_tdm) #TDM을 TF-IDF 행렬로 변환

predicted = mod.predict(tfidfv_test) #테스트 데이터에 대한 예측
print("정확도:", accuracy_score(newsdata_test.target, predicted)) #예측값과 실제값 비교
# 정확도: 0.7738980350504514
