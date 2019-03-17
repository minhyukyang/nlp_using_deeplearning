'''
5. 문서 유사도(Document Similarity) - https://wikidocs.net/24602
'''

'''
1) 코사인 유사도(Cosine Similarity) - https://wikidocs.net/24603
1. 코사인 유사도
  코사인 유사도는 두 벡터 간의 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도를 의미합니다. 
  두 벡터의 방향이 완전히 동일한 경우에는 1의 값을 가지며, 90°의 각을 이루면 0, 180°로 반대의 방향을 가지면 -1의 값을 갖게 됩니다. 
  즉, 결국 코사인 유사도는 -1 이상 1 이하의 값을 가지며 값이 1에 가까울수록 유사도가 높다고 판단할 수 있습니다.
2. 유사도를 이용한 추천 시스템 구현하기
  캐글에서 사용되었던 무비 데이터셋을 가지고 영화 추천 시스템을 만들어보겠습니다. 
  TF-IDF와 코사인 유사도만으로 영화의 줄거리에 기반해서 영화를 추천하는 추천 시스템을 만들 수 있습니다.
  다운로드 링크 : https://www.kaggle.com/rounakbanik/the-movies-dataset
'''

# 문서1 : 저는 사과 좋아요
# 문서2 : 저는 바나나 좋아요
# 문서3 : 저는 바나나 좋아요 저는 바나나 좋아요

# example : cosine similarity
from numpy import dot
from numpy.linalg import norm
import numpy as np

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))
# 코사인 유사도를 계산하는 함수를 만들었습니다.

doc1=np.array([0,1,1,1])
doc2=np.array([1,0,1,1])
doc3=np.array([2,0,2,2])
# 예를 들었던 문서1, 문서2, 문서3에 대해서 각각 BoW를 만들었습니다. 이제 각 문서에 대한 코사인 유사도를 계산해보겠습니다.

print(cos_sim(doc1, doc2)) # 문서1과 문서2의 코사인 유사도
# 0.6666666666666667
print(cos_sim(doc1, doc3)) # 문서1과 문서3의 코사인 유사도
# 0.6666666666666667
print(cos_sim(doc2, doc3)) # 문서2과 문서3의 코사인 유사도
# 1.0000000000000002

# example : movies analysis
import pandas as pd
data = pd.read_csv('d:/Analysis/data/the-movies-dataset/movies_metadata.csv', low_memory=False)
# 예를 들어 윈도우 바탕화면에 해당 파일을 위치시킨 저자의 경우
# pd.read_csv(r'C:\Users\USER\Desktop\movies_metadata.csv', low_memory=False)
# data.head(2)
data=data.head(20000)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
data['overview'] = data['overview'].fillna('')
# 줄거리에서 NaN 값을 가진 경우에는 값 제거

tfidf_matrix = tfidf.fit_transform(data['overview'])
# overview에 대해서 tf-idf 수행
print(tfidf_matrix.shape)
# (20000, 47487)
# 줄거리에 대해서 tf-idf를 수행했습니다. 20000개의 영화를 표현하기위해 총 47487개의 단어가 사용되었음을 보여주고 있습니다.

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# 코사인 유사도를 구합니다.

indices = pd.Series(data.index, index=data['title']).drop_duplicates()
print(indices.head())

idx = indices['Father of the Bride Part II']
print(idx)

# 이제 선택한 영화에 대해서 코사인 유사도를 이용하여, 가장 overview가 유사한 10개의 영화를 찾아내는 함수를 만듭니다.
def get_recommendations(title, cosine_sim=cosine_sim):
  # 선택한 영화의 타이틀로부터 해당되는 인덱스를 받아옵니다. 이제 선택한 영화를 가지고 연산할 수 있습니다.
  idx = indices[title]

  # 모든 영화에 대해서 해당 영화와의 유사도를 구합니다.
  sim_scores = list(enumerate(cosine_sim[idx]))

  # 유사도에 따라 영화들을 정렬합니다.
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  # 가장 유사한 10개의 영화를 받아옵니다.
  sim_scores = sim_scores[1:11]

  # 가장 유사한 10개의 영화의 인덱스를 받아옵니다.
  movie_indices = [i[0] for i in sim_scores]

  # 가장 유사한 10개의 영화의 제목을 리턴합니다.
  return data['title'].iloc[movie_indices]

# 영화 다크 나이트 라이즈와 overview가 유사한 영화들을 찾아보겠습니다.
get_recommendations('The Dark Knight Rises')
# 12481                            The Dark Knight
# 150                               Batman Forever
# 1328                              Batman Returns
# 15511                 Batman: Under the Red Hood
# 585                                       Batman
# 9230          Batman Beyond: Return of the Joker
# 18035                           Batman: Year One
# 19792    Batman: The Dark Knight Returns, Part 1
# 3095                Batman: Mask of the Phantasm
# 10122                              Batman Begins