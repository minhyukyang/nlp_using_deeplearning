'''
3. 언어 모델(Language Model) - https://wikidocs.net/21695
'''

'''
1) 언어 모델(Language Model)이란? - https://wikidocs.net/21668
1. 언어 모델(Language Model)이 뭘까?
2. 문장의 확률을 계산하는 일이 왜 필요하지?
3. 언어 모델은 문장의 확률 또는 단어 등장 확률을 계산하는 일.
4. 언어 모델의 간단한 직관
'''

'''
2) 확률 언어 모델(Statistical language model) - https://wikidocs.net/21687
1. 조건부 확률에 대한 상기.
2. 예제를 통해서 확인해보자.
  P(“An adorable little boy is spreading smiles”) =
    P(An) × P(adorable|An) × P(little|An adorable)
      × P(is|An adorable little) × P(spreading|An adorable little is)
      × P(smiles|An adorable little is spreading)
3. 실제로는 count 기반으로 접근한다.
  예를 들어, 갖고 있는 데이터를 기준으로 An adorable little boy가 100번 언급됐는데 그 다음에 is가 나온 것이 30개라고 한다면, P(is|An adorable little boy)는 30%로 계산됩니다.
4. count 기반의 접근은 사실 한계가 있다.
  count 기반으로 접근하려고 한다면, 우리가 갖고있는 코퍼스에는 엄청난 양의 데이터가 있어야 합니다.
  가령, 우리가 An adorable little boy가 나왔을 때 is가 나올 확률을 계산하고 싶었지만, 우리가 갖고있는 코퍼스에 An adorable little boy is라는 문장 자체가 없었다고 한다면 우리가 갖고있는 확률을 계산할 수 없습니다.
  위의 문제를 완전히 해결할 수는 없지만, 그래도 조금은 일반화(generalization)하는 방법이 있습니다. 바로 마르코프의 가정과 그 이론을 이용한 n-gram입니다. (물론, n-gram도 여전히 한계를 가지며 그 대안이 되는 것이 딥 러닝입니다. 이해의 흐름을 위해서 미리 언급합니다.)
'''

'''
3) N-grams - https://wikidocs.net/21692
1. 코퍼스에서 count하지 못하는 경우를 좀 줄여보자.
  확률을 계산하고 싶은 문장이 길어질수록, 갖고있는 코퍼스에서 그 문장이 존재하지 않을 가능성이 높고, count 할 수 없을 가능성이 높습니다. 그런데 마르코프의 가정을 사용하면 count를 할 수 있을 가능성이 높아집니다.
  An adorable little boy가 나왔을 때 is가 나올 확률을 그냥 boy가 나왔을 때 is가 나올 확률로 생각하면 어떨까요? 우리가 갖고있는 코퍼스에 An adorable little boy is가 있을 가능성 보다는, boy is라는 더 짧은 word sequence가 있을 가능성이 더 높지 않을까요?
2. N-gram
  기준 단어의 앞 단어를 전부 포함해서 세지말고, 앞 단어 중 임의의 n개만 포함해서 세보자는 것
  n을 크게 잡는다면? 정확도가 높아집니다. 즉, 근사의 식은 정확해집니다. 하지만 실제 훈련 코퍼스에서 해당 데이터를 count할 수 있는 확률은 적어집니다.
  n을 작게 잡는다면? 훈련 코퍼스에서 count는 잘 되겠지만 근사의 정확도는 점점 실제의 확률분포와 멀어집니다. 그렇기 때문에 적절한 n을 선택해야합니다.  
* 내용 추가 예정
'''

'''
4) 한글에서의 언어 모델(Language Model for Korean Sentences) - https://wikidocs.net/22533
1. 한글은 어순이 중요하지 않다.
2. 다양한 조사가 나올 수 있다.
3. 한글은 띄어쓰기가 제대로 되지 않습니다.
'''

'''
5), 6) 없음
'''

'''
7) 조건부 확률(Conditional Probability) - https://wikidocs.net/21681
'''