'''
7. NLP를 위한 딥 러닝 개요(Deep Learning for NLP) - https://wikidocs.net/22882

'''

'''
1) 퍼셉트론(Perceptron) - https://wikidocs.net/24958
1. 퍼셉트론(Perceptron)
2. 단층 퍼셉트론(Single-Layer Perceptron)
'''

'''
3) 순환 신경망(RNN) - https://wikidocs.net/22886
1. 순환 신경망(Recurrent Neural Network)
NN은 은닉층의 뉴런이 은닉층에서 활성화 함수를 통해 나온 결과값을 출력층으로도 보내면서, 다시 자기 자신의 다음 계산의 입력으로 보내는 특징을 갖고있습니다.
'''

'''
4) 장단기 메모리(LSTM) - https://wikidocs.net/22888
1. 기존 RNN의 한계
  전통적인 RNN은 비교적 짧은 시퀀스(sequence)에 대해서만 효과를 보이는 단점이 있습니다. 
  즉, RNN의 time-step이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생합니다. 
  이를 장기 의존성 문제(the problem of Long-Term Dependencies)라고 합니다.
2. LSTM(Long Short-Term Memory)
  LSTM은 은닉층의 메모리 셀에 입력 게이트, 망각 게이트, 출력 게이트를 추가하여 불필요한 기억을 지우고, 기억해야할 것들을 정합니다. 
  요약하자면 LSTM은 hidden state를 계산하는 식이 전통적인 RNN보다 조금 더 복잡해졌습니다. 
  LSTM은 RNN과는 달리 긴 시퀀스의 입력값을 처리하는데 탁월한 성능을 보입니다.
3. LSTM으로 문장 생성하기
  내용 없음
4. LSTM으로 문장 생성하기 (한글)
  내용 없음
'''

'''
5) 게이트 순환 유닛(GRU) - https://wikidocs.net/22889
  GRU(Gated Recurrent Unit)는 2014년 뉴욕대학교 조경현 교수님이 집필한 논문에서 제안되었습니다. 
  이 논문은 인코더-디코더 네트워크를 소개한 논문이기도 합니다. 
  GRU는 LSTM의 그래디언트 소실 문제에 대한 해결책을 유지하면서, 은닉 상태를 업데이트하는 계산을 줄였습니다. 
  다시 말해서, GRU는 성능은 LSTM과 유사하면서 복잡했던 LSTM의 구조를 간단화 시켰습니다.
'''
