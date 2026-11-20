import type { BookText } from './interactive-book';
import type { FirstStepId } from '@/lib/learning-first-steps';
export interface FirstStepLesson {
  question: BookText; purpose: BookText; analogy: BookText; prediction: BookText; explanation: BookText;
  examples: BookText[]; stages: { title: BookText; body: BookText }[];
  worked: { title: BookText; body: BookText; tex: string }[];
  bridge: BookText;
}
export const firstStepLessons: Record<FirstStepId, FirstStepLesson> = {
  "inner-products": {
    "question": {
      "ko": "한 방향으로 설명하고, 남는 부분은 어떻게 찾을까요?",
      "en": "How much can one direction explain, and what remains?"
    },
    "purpose": {
      "ko": "데이터를 한 방향으로 요약하는 가장 작은 예입니다. 두 숫자로 된 벡터를 그림의 한 점으로 보고, 선 위의 가장 가까운 점을 찾아봅니다.",
      "en": "This is a small example of summarizing data along one direction. Read a pair of numbers as a point, then find the nearest point on a line."
    },
    "analogy": {
      "ko": "점에서 선으로 수직으로 내린 발을 생각하세요. 그 점까지의 벡터가 투영이고, 선에서 원래 점까지가 잔차입니다. 기준 화살표의 길이와 그 선 위의 위치는 서로 다릅니다.",
      "en": "Imagine dropping a perpendicular from the point to the line. The vector to that foot is the projection; the remaining displacement is the residual. The reference arrow and the projected arrow need not have the same length."
    },
    "prediction": {
      "ko": "기준선을 기울이면 투영은 어떻게 바뀔까요? 잔차는 새 기준선에도 수직일까요?",
      "en": "If you tilt the reference line, how will the projection change? Will its residual remain perpendicular to the new line?"
    },
    "explanation": {
      "ko": "기준선 위의 점을 움직이면 가까워지다가 다시 멀어집니다. 가장 가까운 점에서 선을 따라 조금 움직여도 거리를 더 줄일 수 없으므로 잔차가 수직입니다. 아래 손계산에서 내적이 영인지 확인하세요.",
      "en": "Moving along the line first brings you closer, then farther away. At the closest point, a small move along the line cannot reduce the distance, so the residual is perpendicular. Check its zero dot product in the calculation below."
    },
    "examples": [
      {
        "ko": "수평선에 투영",
        "en": "Project onto a horizontal line"
      },
      {
        "ko": "대각선에 투영",
        "en": "Project onto a diagonal line"
      },
      {
        "ko": "수직선에 투영",
        "en": "Project onto a vertical line"
      }
    ],
    "stages": [
      {
        "title": {
          "ko": "원래 점 보기",
          "en": "Locate the original point"
        },
        "body": {
          "ko": "$\\mathbf u=(3,2)$는 오른쪽으로 세 칸, 위로 두 칸입니다. 그림의 각 격자 간격은 한 칸입니다.",
          "en": "$\\mathbf u=(3,2)$ means three units right and two units up. Each grid spacing is one unit."
        }
      },
      {
        "title": {
          "ko": "기준 방향 정하기",
          "en": "Choose a reference direction"
        },
        "body": {
          "ko": "초록색 직선이 허용하는 방향입니다. 그 위에서만 원래 점을 설명해 보겠습니다.",
          "en": "The green line is the allowed direction. We will approximate the original point using a point on this line."
        }
      },
      {
        "title": {
          "ko": "가장 가까운 점 찾기",
          "en": "Find the nearest point"
        },
        "body": {
          "ko": "주황색 점이 투영입니다. 원래 점과 주황색 점 사이의 점선이 잔차입니다.",
          "en": "The orange point is the projection. The dashed segment from it to the original point is the residual."
        }
      },
      {
        "title": {
          "ko": "숫자로 확인하기",
          "en": "Check the numbers"
        },
        "body": {
          "ko": "$\\mathbf u=\\mathbf p+\\mathbf r$로 나누었습니다. 투영과 잔차의 역할을 말로 설명한 뒤 다음 계산을 확인하세요.",
          "en": "We have split $\\mathbf u=\\mathbf p+\\mathbf r$. Explain the roles of the projection and residual before checking the calculation."
        }
      }
    ],
    "worked": [
      {
        "title": {
          "ko": "수평 방향의 손계산",
          "en": "A horizontal direction by hand"
        },
        "body": {
          "ko": "기준 벡터가 오른쪽 한 칸이면 계수는 세 배입니다. 위쪽 두 칸은 그 방향으로 설명할 수 없어 잔차로 남습니다.",
          "en": "With a reference vector one unit right, the coefficient is three. The two upward units cannot be explained along that line and remain in the residual."
        },
        "tex": "\\begin{gathered}\\mathbf v=(1,0)\\\\\\alpha=\\frac{3\\cdot1+2\\cdot0}{1^2+0^2}=3\\\\\\mathbf p=(3,0)\\\\\\mathbf r=(0,2)\\end{gathered}"
      },
      {
        "title": {
          "ko": "대각선 방향의 손계산",
          "en": "A diagonal direction by hand"
        },
        "body": {
          "ko": "기준선이 기울어지면 두 좌표를 함께 사용합니다. 잔차의 두 성분은 서로 상쇄되어 기준 벡터와 내적이 영입니다.",
          "en": "Tilting the line uses both coordinates. The two residual components cancel when dotted with the reference vector."
        },
        "tex": "\\begin{gathered}\\mathbf v=(1,1)\\\\\\alpha=\\frac{3+2}{1+1}=\\frac52\\\\\\mathbf p=\\left(\\frac52,\\frac52\\right)\\\\\\mathbf v\\cdot\\mathbf r=\\frac12-\\frac12=0\\end{gathered}"
      }
    ],
    "bridge": {
      "ko": "AI에서는 투영으로 특징의 일부를 설명하고 잔차를 살펴봅니다. 이후 최소제곱과 PCA에서 여러 방향으로 확장합니다. 아래 본문은 각도·길이·일반 투영 공식과 적용 조건을 설명합니다.",
      "en": "In AI, projections explain part of a feature vector and expose a residual. Least squares and PCA extend this to several directions. The chapter below develops angles, lengths, the general formula and its assumptions."
    }
  },
  "gradients": {
    "question": {
      "ko": "내려가는 방향인데, 왜 한 번에 너무 멀리 가면 손실이 커질까요?",
      "en": "Why can a step in a downhill direction increase the loss?"
    },
    "purpose": {
      "ko": "신경망 학습의 가장 작은 모형으로, 현재 좌표를 조금 바꾸어 손실을 줄여봅니다. 그림의 점은 모형의 매개변수이며 물리적으로 굴러가는 공이 아닙니다.",
      "en": "Use a small model of training: change the current coordinates to reduce a loss. The point represents model parameters, rather than a physical rolling ball."
    },
    "analogy": {
      "ko": "등고선을 지도처럼 읽으세요. 안쪽일수록 손실이 작습니다. 기울기는 지금 위치에서의 변화율이고, 학습률은 그 정보를 이용해 얼마나 이동할지 정합니다.",
      "en": "Read the contours as a map. Inner contours have lower loss. The gradient describes change at the current location; the learning rate determines how far to move using that information."
    },
    "prediction": {
      "ko": "이동 폭을 크게 하면 항상 더 빨리 도착할까요? 같은 시작점에서 작은 폭과 큰 폭을 비교해보세요.",
      "en": "Does a larger step always arrive faster? Compare small and large steps from the same starting point."
    },
    "explanation": {
      "ko": "방향이 맞아도 한 번에 너무 멀리 이동하면 낮은 지역을 지나칠 수 있습니다. 이 예에서는 좌표별 배율의 절댓값이 모두 일보다 작아야 모든 시작점에서 수렴합니다. 일반 신경망에는 별도의 분석이 필요합니다.",
      "en": "A correct local direction can still overshoot a low region. For this example, both coordinate multipliers must have magnitude below one for convergence from every start. General neural networks require separate analysis."
    },
    "examples": [
      {
        "ko": "작은 학습률",
        "en": "Small learning rate"
      },
      {
        "ko": "왕복하며 수렴",
        "en": "Alternating but convergent"
      },
      {
        "ko": "너무 큰 학습률",
        "en": "Excessive learning rate"
      }
    ],
    "stages": [
      {
        "title": {
          "ko": "손실 지형 보기",
          "en": "Read the loss landscape"
        },
        "body": {
          "ko": "$L(x,y)=x^2+\\tfrac12 y^2$의 최솟값은 원점에 있습니다. 파란 점은 $(2,2)$에서 출발합니다. 아래첨자는 이동 횟수입니다.",
          "en": "$L(x,y)=x^2+\\tfrac12 y^2$ has its minimum at the origin. The blue point starts at $(2,2)$. Subscripts count the updates."
        }
      },
      {
        "title": {
          "ko": "한 번 이동하기",
          "en": "Take one update"
        },
        "body": {
          "ko": "현재 기울기는 $(2x,y)$입니다. 현재 좌표에서 학습률과 기울기의 곱을 뺍니다.",
          "en": "The current gradient is $(2x,y)$. Subtract the learning rate times that gradient from the current coordinates."
        }
      },
      {
        "title": {
          "ko": "기울기 다시 계산하기",
          "en": "Recompute the gradient"
        },
        "body": {
          "ko": "새 위치에서는 기울기도 달라집니다. 처음 기울기를 계속 재사용하지 않고 새 좌표에서 다시 계산합니다.",
          "en": "The gradient changes at the new location. Recompute it there instead of reusing the first gradient."
        }
      },
      {
        "title": {
          "ko": "경로 비교하기",
          "en": "Compare the trajectory"
        },
        "body": {
          "ko": "세 번의 이동을 비교하세요. 부호가 바뀌어도 손실이 줄 수 있고, 큰 폭에서는 손실이 증가할 수 있습니다.",
          "en": "Compare three updates. Sign changes can accompany decreasing loss, while excessive steps can increase the loss."
        }
      }
    ],
    "worked": [
      {
        "title": {
          "ko": "작은 폭으로 한 번",
          "en": "One small update"
        },
        "body": {
          "ko": "현재 기울기를 구하고, 이동 벡터를 뺀 뒤 새 위치에서 손실을 다시 계산합니다.",
          "en": "Compute the current gradient, subtract the update vector, then evaluate the loss at the new point."
        },
        "tex": "\\begin{gathered}\\nabla L(2,2)=(4,2)\\\\\\eta=\\frac14\\\\x_1=2-\\frac14\\cdot4=1\\\\y_1=2-\\frac14\\cdot2=\\frac32\\\\L\\left(1,\\frac32\\right)=\\frac{17}{8}\\end{gathered}"
      },
      {
        "title": {
          "ko": "같은 방향으로 너무 멀리",
          "en": "Too far in the same direction"
        },
        "body": {
          "ko": "시작점과 방향은 같지만 학습률을 크게 하면 원래 손실보다 더 큰 곳으로 갑니다.",
          "en": "The start and direction are unchanged, but an excessive learning rate moves to a point with higher loss."
        },
        "tex": "\\begin{gathered}\\eta=\\frac54\\\\x_1=2-\\frac54\\cdot4=-3\\\\y_1=2-\\frac54\\cdot2=-\\frac12\\\\L\\left(-3,-\\frac12\\right)=\\frac{73}{8}>6\\end{gathered}"
      }
    ],
    "bridge": {
      "ko": "실제 신경망에서는 역전파로 기울기를 구합니다. 아래 본문에서 학습률·곡률·수렴 조건의 관계를 더 살펴보고, 기존 실습에서 반복 횟수를 늘려보세요.",
      "en": "In neural networks, backpropagation computes the gradient. Continue below to connect learning rate, curvature and convergence, then extend the iterations in the full experiment."
    }
  },
  "bayesian-inference": {
    "question": {
      "ko": "관측 하나가 추가되면, 믿음은 얼마나 바뀔까요?",
      "en": "How much should one new observation change a belief?"
    },
    "purpose": {
      "ko": "두 동전 중 어떤 동전인지 추론해봅니다. 하나는 공정하고 다른 하나는 앞면이 더 잘 나옵니다. 처음에는 둘을 똑같이 가능하다고 생각합니다.",
      "en": "Infer which of two coins is being used. One is fair; the other favors heads. Initially, regard them as equally plausible."
    },
    "analogy": {
      "ko": "가설마다 관측을 설명하는 힘을 점수처럼 계산한 뒤 합이 하나가 되도록 나눕니다. 앞면을 더 잘 설명하는 동전은 앞면을 관측했을 때 상대적 비중이 커집니다.",
      "en": "Score each hypothesis by how well it explains the observation, then divide by the total to make probabilities sum to one. A coin that better predicts heads gains relative weight after heads."
    },
    "prediction": {
      "ko": "앞면이 두 번 나온 뒤 뒷면이 나오면, 앞면에 편향된 동전의 확률은 어떻게 바뀔까요?",
      "en": "After two heads, what will a tail do to the probability of the heads-biased coin?"
    },
    "explanation": {
      "ko": "관측이 모든 가설에 같은 영향을 주지는 않습니다. 뒷면은 공정한 동전에서 더 잘 설명되므로 앞면 편향 가설의 비중을 줄입니다. 확률이 영이 된다는 뜻은 아닙니다.",
      "en": "Evidence affects hypotheses differently. A tail is better explained by the fair coin, so it reduces the relative weight of the heads-biased hypothesis without making that probability zero."
    },
    "examples": [
      {
        "ko": "앞면·앞면·뒷면 관측",
        "en": "Observe heads, heads, tails"
      },
      {
        "ko": "뒷면을 세 번 관측",
        "en": "Observe three tails"
      }
    ],
    "stages": [
      {
        "title": {
          "ko": "같은 출발 확률",
          "en": "Start with equal probabilities"
        },
        "body": {
          "ko": "공정한 동전 $H_f$는 앞면 확률이 $1/2$, 편향 동전 $H_b$는 $3/4$입니다. 두 가설의 사전 확률은 각각 $1/2$입니다.",
          "en": "The fair coin $H_f$ has heads probability $1/2$, and the biased coin $H_b$ has $3/4$. Each hypothesis starts with prior probability $1/2$."
        }
      },
      {
        "title": {
          "ko": "첫 관측으로 갱신",
          "en": "Update after one observation"
        },
        "body": {
          "ko": "각 가설에서 이 관측이 얼마나 잘 나오는지 사전 확률에 곱한 뒤 정규화합니다.",
          "en": "Multiply the prior by how likely this observation is under each hypothesis, then normalize."
        }
      },
      {
        "title": {
          "ko": "누적 관측 반영",
          "en": "Include the next observation"
        },
        "body": {
          "ko": "같은 동전을 계속 쓴다고 가정합니다. 동전이 정해지면 각 시행은 독립이며 같은 앞면 확률을 사용합니다.",
          "en": "Assume the same coin is used throughout. Conditional on its identity, trials are independent with the same heads probability."
        }
      },
      {
        "title": {
          "ko": "반대 증거도 반영",
          "en": "Respond to contrary evidence"
        },
        "body": {
          "ko": "새 관측을 포함한 두 막대의 합은 항상 하나입니다. 막대는 동전 가설의 확률이며 앞면 자체의 빈도와 구별하세요.",
          "en": "The two updated bars always sum to one. They are probabilities of coin hypotheses, distinct from the observed heads frequency."
        }
      }
    ],
    "worked": [
      {
        "title": {
          "ko": "앞면 한 번 뒤",
          "en": "After one head"
        },
        "body": {
          "ko": "두 가설의 사전 확률이 같으므로 앞면을 설명하는 우도의 비로 계산할 수 있습니다.",
          "en": "Equal priors let us calculate directly from the two heads likelihoods."
        },
        "tex": "\\begin{gathered}P(H_f)P(H\\mid H_f)=\\frac14\\\\P(H_b)P(H\\mid H_b)=\\frac38\\\\P(H_b\\mid H)=\\frac{\\frac38}{\\frac14+\\frac38}=\\frac35\\end{gathered}"
      },
      {
        "title": {
          "ko": "앞면 둘, 뒷면 하나 뒤",
          "en": "After two heads and one tail"
        },
        "body": {
          "ko": "순서 있는 관측을 각각의 동전 모형으로 계산합니다. 앞면 두 번 직후보다 편향 동전의 확률이 낮아집니다.",
          "en": "Compute the ordered observations under each coin model. The biased-coin probability falls from its value after two heads."
        },
        "tex": "\\begin{gathered}P(HHT\\mid H_f)=\\left(\\frac12\\right)^3=\\frac18\\\\P(HHT\\mid H_b)=\\left(\\frac34\\right)^2\\frac14\\\\P(HHT\\mid H_b)=\\frac9{64}\\\\P(H_b\\mid HHT)=\\frac{9/64}{17/64}=\\frac9{17}\\\\\\frac9{17}<\\frac9{13}=P(H_b\\mid HH)\\end{gathered}"
      }
    ],
    "bridge": {
      "ko": "여기서는 동전 가설이 둘뿐입니다. 아래 본문과 기존 실습은 앞면 확률 자체를 연속적인 미지수로 보고 Beta 분포로 갱신합니다. 작은 유한 예제로 갱신 원리를 익힌 뒤 연속 모형으로 확장하세요.",
      "en": "This preparation has only two coin hypotheses. The chapter and full lab below treat the heads probability itself as a continuous unknown and update a beta distribution. Use this finite example to understand the update before moving to that model."
    }
  },
  "finite-qubit-circuit-order-interference": {
    "question": {
      "ko": "확률 막대가 같은데, 왜 마지막 결과가 달라질까요?",
      "en": "Why can identical probability bars lead to different final outcomes?"
    },
    "purpose": {
      "ko": "양자 상태는 확률뿐 아니라 진폭의 부호도 담습니다. Hadamard로 나누고 다시 합치는 작은 회로에서, 중간 부호 하나가 마지막 읽기를 바꾸는 과정을 봅니다.",
      "en": "A quantum state carries amplitude signs as well as probabilities. Split and recombine with Hadamards to see how one intermediate sign changes the final readout."
    },
    "analogy": {
      "ko": "같은 방향으로 흔들리는 두 파동은 더해지고 반대 방향이면 상쇄될 수 있습니다. 여기서는 복소 진폭의 실수 부호만 다루는 작은 예로 그 효과를 확인합니다.",
      "en": "Two waves in phase can add, while opposing phases can cancel. This small example uses only real amplitude signs to illustrate that effect."
    },
    "prediction": {
      "ko": "중간에 아래 진폭의 부호를 뒤집어도 두 확률은 같습니다. 마지막 Hadamard 뒤에도 같을까요?",
      "en": "Flipping the lower amplitude sign preserves both intermediate probabilities. Will they still agree after the final Hadamard?"
    },
    "explanation": {
      "ko": "중간 확률은 각 진폭의 절댓값 제곱이므로 부호를 잃습니다. 마지막 Hadamard는 진폭을 더하고 빼므로 그 부호가 다시 결과에 드러납니다. 단계 재생은 게이트 순서이며 물리적 시간 전개를 뜻하지 않습니다.",
      "en": "Squaring amplitude magnitudes hides the sign in the intermediate probabilities. The last Hadamard adds and subtracts amplitudes, making the sign observable. Playback shows gate order, rather than physical time evolution."
    },
    "examples": [
      {
        "ko": "중간 부호 뒤집기",
        "en": "Flip the intermediate sign"
      },
      {
        "ko": "중간 부호 유지하기",
        "en": "Keep the intermediate sign"
      }
    ],
    "stages": [
      {
        "title": {
          "ko": "한 상태에서 출발",
          "en": "Start in one basis state"
        },
        "body": {
          "ko": "$|0\\rangle$에서 출발합니다. 읽기 결과가 영일 확률은 하나입니다.",
          "en": "Start in $|0\\rangle$. The probability of reading zero is one."
        }
      },
      {
        "title": {
          "ko": "진폭 나누기",
          "en": "Split amplitudes"
        },
        "body": {
          "ko": "첫 Hadamard가 두 진폭을 같게 만듭니다. 각 확률은 $1/2$이며 두 진폭의 부호는 같습니다.",
          "en": "The first Hadamard creates equal amplitudes. Each probability is $1/2$, and the signs agree."
        }
      },
      {
        "title": {
          "ko": "상대 부호 비교",
          "en": "Compare the relative sign"
        },
        "body": {
          "ko": "부호를 뒤집는 $Z$와 그대로 두는 $I$를 비교하세요. 이 단계의 확률 막대만으로는 두 경우를 구별할 수 없습니다.",
          "en": "Compare a sign flip $Z$ with the identity $I$. Probability bars alone cannot distinguish them at this stage."
        }
      },
      {
        "title": {
          "ko": "다시 합쳐 읽기",
          "en": "Recombine and read"
        },
        "body": {
          "ko": "마지막 Hadamard 뒤 한 결과에서는 더해지고 다른 결과에서는 상쇄됩니다. 어느 진폭의 부호를 바꿨는지 마지막 확률에 나타납니다.",
          "en": "After the last Hadamard, amplitudes add for one outcome and cancel for the other. The final probabilities reveal the changed relative sign."
        }
      }
    ],
    "worked": [
      {
        "title": {
          "ko": "부호를 바꾼 경우",
          "en": "With the sign flip"
        },
        "body": {
          "ko": "두 번째 진폭이 음수가 되어 마지막 Hadamard 뒤 일 상태로 모입니다.",
          "en": "The second amplitude becomes negative, and the last Hadamard directs the state into outcome one."
        },
        "tex": "\\begin{gathered}H|0\\rangle=|+\\rangle=\\frac{|0\\rangle+|1\\rangle}{\\sqrt2}\\\\Z|+\\rangle=|-\\rangle=\\frac{|0\\rangle-|1\\rangle}{\\sqrt2}\\\\H|-\\rangle=|1\\rangle\\end{gathered}"
      },
      {
        "title": {
          "ko": "부호를 유지한 경우",
          "en": "Without the sign flip"
        },
        "body": {
          "ko": "중간 확률은 위 예제와 같지만 마지막 결과는 영 상태입니다.",
          "en": "The intermediate probabilities match the previous example, but the final state is zero."
        },
        "tex": "\\begin{gathered}H|0\\rangle=|+\\rangle=\\frac{|0\\rangle+|1\\rangle}{\\sqrt2}\\\\I|+\\rangle=|+\\rangle\\\\H|+\\rangle=|0\\rangle\\end{gathered}"
      }
    ],
    "bridge": {
      "ko": "기존 회로 실습에서 같은 두 예제를 비교한 뒤 제어 게이트와 얽힘으로 확장하세요. 아래 본문은 복소 위상·선 번호·전체 상태와 국소 상태, 연구 수준의 미분까지 연결합니다.",
      "en": "Compare these cases in the full circuit lab, then extend to controlled gates and entanglement. The chapter connects complex phases, wire order, joint and local states, and research-level derivatives."
    }
  }
};
