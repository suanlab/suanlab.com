## 압축은 복호 규칙까지 포함한다

압축된 배열이 짧아 보이는 것만으로 무손실 부호가 되지는 않습니다. 수신자가 비트열을 원본으로 유일하게 되돌릴 규칙이 있어야 합니다. 여기서는 N=8 또는 16이 외부에서 공유되었다고 가정하고 길이 N의 이진 문자열을 부호화합니다. 원본 길이가 알려지지 않으면 그 길이를 보내는 비용도 추가해야 합니다.

실습의 mask는 정수이며 낮은 N비트만 사용합니다. 화면의 문자열은 보통의 높은 자리부터 읽습니다. 외부 길이, 비트 순서, 방식 헤더도 프로토콜의 일부입니다.

## 두 방식의 정확한 문법

Raw 방식은 헤더 0 뒤에 N비트를 그대로 붙입니다. RLE는 헤더 1 뒤에 첫 비트 한 개와 각 run의 길이−1을 붙입니다. N이 2의 거듭제곱이므로 각 길이는 w=log₂N비트로 표현할 수 있습니다. run 사이의 비트는 항상 반전되므로 따로 보내지 않습니다.

$$
\begin{aligned}
L_{\rm raw}&=1+N,\\
L_{\rm RLE}&=2+r\log_2N.
\end{aligned}
$$

N=16의 0000000000000000은 1 · 0 · 1111입니다. 마지막 1111은 정수 15에 1을 더한 길이 16입니다. 전체는 6비트이며 원본을 정확히 복원합니다. N=16의 0000000011111111은 1 · 0 · 0111 · 0111로 10비트입니다.

## 복호가 실제로 수행하는 일

수신자는 헤더로 방식을 선택하고, RLE이면 첫 비트를 각 길이만큼 반복한 뒤 반전합니다. 길이의 합이 정확히 N이어야 합니다. 부족한 토큰·N 초과·잘못된 문자는 오류로 거부합니다. Raw 헤더이면 나머지가 정확히 N비트인지 검사합니다.

$$
D(E(x),N)=x,\qquad x\in\{0,1\}^{N}.
$$

화면은 실제 전송 stream과 실제 복호 결과를 표시합니다. RLE가 더 길면 raw를 선택하며 동점도 raw로 처리합니다. 선택 규칙을 알고 있어도 어느 방식이 선택되었는지 전달할 헤더는 여전히 필요합니다.

## 모든 입력을 압축할 수 없는 이유

교대열 0101…은 run이 16개라 RLE 비용이 66비트입니다. 최종 인코더는 raw의 17비트를 택합니다. 원본 16비트보다 한 비트 길지만 공유 문법 아래에서 유일 복원됩니다. N비트 문자열은 2ᴺ개인데 길이가 N 미만인 모든 문자열은 2ᴺ−1개뿐입니다. 비둘기집 원리 때문에 모든 입력을 동시에 짧게 하는 단사 압축은 불가능합니다.

AI 체크포인트의 압축률도 가중치 본체뿐 아니라 코드북·스케일·인덱스·모형 버전을 포함해야 합니다. 손실 양자화는 정확한 원본 복원이라는 현재 조건을 바꿉니다. 전송 중 오류를 고치는 것은 별도의 오류정정부호 문제입니다.

## 외부 길이가 고정되면 prefix 해독도 가능하다

같은 N에서 raw는 길이 N+1로 고정이고, RLE와 첫 비트가 다릅니다. RLE에서는 w비트 토큰을 읽을 때마다 양의 run 길이가 늘어납니다. 합이 N에 도달하면 그 부호어가 끝납니다. 이미 유효한 RLE 뒤에 토큰을 더 붙이면 합이 N을 넘으므로 또 다른 유효 부호어가 될 수 없습니다. 따라서 같은 외부 N을 공유하는 유효 stream 집합은 prefix-free입니다. N=8과 16을 아무 표시 없이 섞어도 된다는 주장은 아닙니다.

Run이 r개인 N비트 원본은 첫 비트 두 선택과 N−1개 틈 중 r−1개 경계 선택으로 2·binom(N−1,r−1)개입니다. 모든 canonical RLE의 Kraft 합은 이항정리로 다음과 같습니다.

$$
S_{\mathrm{RLE}}=\sum_{r=1}^N2{N-1\choose r-1}2^{-(2+wr)}
=2^{-1-w}(1+2^{-w})^{N-1}.
$$

Raw의 Kraft 합은 1/2입니다. 더 짧은 방식만 선택하는 인코더는 두 유효 집합의 부분집합을 사용하므로 역시 prefix-free입니다. 서로 다른 원본의 출력은 복호 가능성 때문에 달라야 합니다.

## 균등 원본에서 절약하는 비율까지 세기

Kraft 부등식에서 길이가 N−s 이하인 선택 부호어 하나는 합에 최소 2⁻⁽ᴺ⁻ˢ⁾만큼 기여합니다. 따라서 그 개수는 2ᴺ⁻ˢ 이하이고, 균등 N비트 원본이 s비트 이상 줄어들 확률은 2⁻ˢ 이하입니다. 이는 실제 자료가 균등이라는 주장도, 모든 비균등 데이터의 압축률 한계도 아닙니다.

$$
P_{\mathrm{uniform}}[L(E(X))\le N-s]\le2^{-s}.
$$

| N | 최소 길이 | 균등 평균 길이 | 실제로 짧아진 비율 | 선택 코드 Kraft 합 |
|---|---|---|---|---|
| 8 | 5 | 8.914062 | .007812 | .585938 |
| 16 | 6 | 16.986847 | .003693 | .571518 |

표는 현재 raw/RLE 선택 규칙의 전수조사 결과이며 반올림했습니다. 코드가 256개와 65,536개 원본 모두를 복원하고, 길이순 prefix 검사·정확분수 Kraft 합을 확인합니다. 구조적인 상수열은 크게 줄어도 균등 평균에서는 프로토콜 비용이 남습니다. 소스의 분포나 실제 구조를 밝히고 압축률을 평가해야 하는 이유입니다.

**직접 풀기:** 균등 분포의 평균 코드 길이를 N보다 작게 만들 수 있나요? **해설:** 이 조건의 prefix 무손실 부호에서는 평균 길이≥H(X)=N입니다. 선택 코드가 더 짧은 몇 입력을 갖는 것과 평균 한계는 모순되지 않습니다.

```python
from collections import Counter
from fractions import Fraction as F
from itertools import groupby

for n in [8,16]:
    width=n.bit_length()-1
    codes={}
    for value in range(2**n):
        source=f'{value:0{n}b}'
        runs=[len(list(g)) for _,g in groupby(source)]
        raw='0'+source
        rle='1'+source[0]+''.join(f'{length-1:0{width}b}' for length in runs)
        encoded=rle if len(rle)<len(raw) else raw
        if encoded[0]=='0': restored=encoded[1:]
        else:
            bit=encoded[1];restored=''
            for i in range(2,len(encoded),width):
                restored+=bit*(int(encoded[i:i+width],2)+1)
                bit='1' if bit=='0' else '0'
        assert restored==source and encoded not in codes
        codes[encoded]=source
    # Increasing length makes any prefix conflict visible in the seen set.
    seen=set()
    for code in sorted(codes,key=len):
        assert not any(code[:i] in seen for i in range(len(code)))
        seen.add(code)
    histogram=Counter(map(len,codes))
    kraft=sum(F(count,2**length) for length,count in histogram.items())
    assert kraft<=1
    for saving in [1,2,3]:
        short=sum(count for length,count in histogram.items() if length<=n-saving)
        assert F(short,2**n)<=F(1,2**saving)
    average=sum(F(length*count,2**n) for length,count in histogram.items())
    short=sum(count for length,count in histogram.items() if length<n)
    print('N, shortest, longest, uniform mean length, shortened fraction, Kraft sum:',
          n,min(histogram),max(histogram),round(float(average),6),
          round(short/2**n,6),round(float(kraft),6))
```


## 확인 문제와 해설

1. N=8에서 상수열 길이는? **5비트입니다.** 헤더 1, 첫 비트 1, 길이 3비트입니다.
2. 헤더를 빼고 압축률을 비교해도 되나요? **아니요.** 복호기가 방식을 모르면 실제 독립 프로토콜이 아닙니다. 공유한 경우에는 그 공유 조건을 명시해야 합니다.
3. 압축된 stream 한 비트가 변하면 자동 복구되나요? **보장하지 않습니다.** 유효한 다른 길이나 다른 문자열이 될 수 있습니다.

실습의 문법과 복호기는 이 단원의 명세입니다. 추가 배경: [MIT 6.441, Source Coding and Channel Coding](https://ocw.mit.edu/courses/6-441-information-theory-spring-2010/pages/lecture-notes/).
