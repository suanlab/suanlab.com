export type BookLocale = 'ko' | 'en';
export type BookText = Record<BookLocale, string>;
export type LearningUnitId = 'vectors-and-forces' | 'inner-products' | 'linear-transforms' | 'least-squares' | 'eigenvalues' | 'gradients' | 'chain-rule' | 'jacobians' | 'backpropagation' | 'svd' | 'probability-distributions' | 'expectation-variance' | 'statistical-estimation' | 'bayesian-inference' | 'information-theory' | 'regularization' | 'stochastic-optimization' | 'differential-equations' | 'mechanics-energy' | 'thermodynamics' | 'stochastic-processes' | 'inverse-problems' | 'research-methods' | 'propositions-implications' | 'quantifiers-counterexamples' | 'sets-functions' | 'relations-orders' | 'proof-strategies' | 'mathematical-induction' | 'numbers-ratios-units' | 'real-complex-numbers' | 'equations-inequalities' | 'polynomial-rational-functions' | 'exponentials-logarithms' | 'trigonometry-coordinates' | 'composition-inverse-functions' | 'sequences-series' | 'asymptotic-growth' | 'counting-permutations' | 'binomial-inclusion-exclusion' | 'recurrences-generating-functions' | 'graphs-trees-dags' | 'paths-cycles-connectivity' | 'boolean-algebra' | 'automata-formal-languages' | 'posets-lattices' | 'finite-state-transitions' | 'vector-spaces-rank' | 'determinants-traces' | 'positive-semidefinite-matrices' | 'quadratic-forms-orders' | 'kronecker-products' | 'limits-continuity-completeness' | 'differentiability-mean-value' | 'integration-fundamental-theorem' | 'taylor-remainders' | 'series-convergence-tests' | 'pointwise-uniform-convergence' | 'compactness-extreme-values' | 'convexity-first-order' | 'limits-integration-interchange' | 'implicit-inverse-functions' | 'matrix-trace-logdet-derivatives' | 'vector-fields-divergence-curl' | 'line-surface-volume-integrals' | 'gauss-stokes-theorems' | 'higher-order-autodiff' | 'complex-polar-maps' | 'cauchy-riemann-holomorphic' | 'complex-contour-integrals' | 'residues-winding' | 'analytic-functions-singularities' | 'analytic-continuation-branches' | 'complex-linear-algebra' | 'wirtinger-gradients' | 'complex-signals-phase' | 'tensor-order-shapes' | 'tensor-indices-contractions' | 'multilinear-maps' | 'broadcasting-semantics' | 'tensor-cp-tucker' | 'tensor-trains' | 'tensor-networks-contraction' | 'tensor-low-rank-approximation' | 'tensor-symmetry-antisymmetry' | 'tensor-storage-cost' | 'groups-rings-fields' | 'homomorphisms-isomorphisms' | 'subgroups-quotients' | 'polynomials-finite-fields' | 'modules-linear-actions' | 'algebraic-symmetries' | 'polynomial-systems' | 'invariant-constructions' | 'algebraic-representations-scope' | 'probability-spaces-events' | 'conditional-probability-independence' | 'mass-density-cdf' | 'joint-marginal-conditional' | 'multivariate-gaussians' | 'random-variable-transformations' | 'laws-large-numbers-clt' | 'reparameterization-gradients';
export interface LearningQuestion { id: string; question: BookText; options: BookText[]; answer: number; explanation: BookText; }
export interface LearningUnit {
  id: LearningUnitId; prerequisiteIds: LearningUnitId[]; title: BookText; subtitle: BookText; prerequisites: BookText;
  eyebrow: string; formula: string; accent: string; questions: LearningQuestion[];
}
export const interactiveBook = {
  slug: 'physics-math-for-ai',
  title: { ko: 'AI를 위한 물리와 수학', en: 'Physics and Mathematics for AI' },
  description: { ko: '수식을 읽고, 직접 조작하고, 원리를 발견하는 인터랙티브 교재', en: 'Read the equations. Change the experiment. Discover the principle.' },
};
export const bookPath = '/book/online/physics-math-for-ai/';
export const learningUnits: LearningUnit[] = [
{
  "id": "propositions-implications",
  "title": {
    "ko": "명제·조건과 논리적 함의",
    "en": "Propositions, Conditions and Implication"
  },
  "subtitle": {
    "ko": "조건이 충분하다는 말과 필요하다는 말을 진리표로 구분합니다.",
    "en": "Use truth tables to distinguish sufficient conditions from necessary ones."
  },
  "prerequisiteIds": [],
  "prerequisites": {
    "ko": "기본 정수 계산과 연결된 앞 단원의 표기",
    "en": "Basic integer arithmetic and notation in the linked prerequisites"
  },
  "eyebrow": "LOGIC LAB",
  "formula": "P⇒Q ≡ ¬P∨Q ≡ ¬Q⇒¬P",
  "accent": "logic",
  "questions": [
    {
      "id": "implication-false",
      "question": {
        "ko": "P⇒Q가 거짓인 경우는?",
        "en": "When is P⇒Q false?"
      },
      "options": [
        {
          "ko": "P 참, Q 거짓",
          "en": "P true, Q false"
        },
        {
          "ko": "둘 다 거짓",
          "en": "Both false"
        },
        {
          "ko": "둘 다 참",
          "en": "Both true"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "조건을 만족하지만 결론이 실패한 경우입니다.",
        "en": "The premise holds while the conclusion fails."
      }
    },
    {
      "id": "sufficient",
      "question": {
        "ko": "P⇒Q에서 Q는 P의 무엇인가요?",
        "en": "In P⇒Q, what is Q relative to P?"
      },
      "options": [
        {
          "ko": "충분조건",
          "en": "A sufficient condition"
        },
        {
          "ko": "필요조건",
          "en": "A necessary condition"
        },
        {
          "ko": "항상 원인",
          "en": "Always a cause"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "P가 성립하면 Q가 필요합니다. 인과관계는 별도입니다.",
        "en": "Q must hold whenever P holds. Causality is a separate matter."
      }
    }
  ]
},
{
  "id": "quantifiers-counterexamples",
  "title": {
    "ko": "양화사·증인과 반례",
    "en": "Quantifiers, Witnesses and Counterexamples"
  },
  "subtitle": {
    "ko": "모든 입력마다 다른 답을 고를 수 있는 것과 공통 답 하나가 존재하는 것은 다릅니다.",
    "en": "Choosing a separate answer for every input differs from finding one answer that works for all inputs."
  },
  "prerequisiteIds": [
    "propositions-implications"
  ],
  "prerequisites": {
    "ko": "기본 정수 계산과 연결된 앞 단원의 표기",
    "en": "Basic integer arithmetic and notation in the linked prerequisites"
  },
  "eyebrow": "LOGIC LAB",
  "formula": "∀x∃y R(x,y) ≠ ∃y∀x R(x,y)",
  "accent": "logic",
  "questions": [
    {
      "id": "quantifier-order",
      "question": {
        "ko": "D={0,1,2}, R는 y=x일 때 참인 것은?",
        "en": "For D={0,1,2} and R meaning y=x, which is true?"
      },
      "options": [
        {
          "ko": "∀x∃y R",
          "en": "∀x∃y R"
        },
        {
          "ko": "∃y∀x R",
          "en": "∃y∀x R"
        },
        {
          "ko": "둘 다 거짓",
          "en": "Both false"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "각 x에 y=x를 고릅니다. 하나의 y를 모두 공유할 필요는 없습니다.",
        "en": "Choose y=x for each x; a common y is not required."
      }
    },
    {
      "id": "negation-quantifiers",
      "question": {
        "ko": "¬∀x P(x)는?",
        "en": "What is ¬∀x P(x)?"
      },
      "options": [
        {
          "ko": "∀x ¬P(x)",
          "en": "∀x ¬P(x)"
        },
        {
          "ko": "∃x ¬P(x)",
          "en": "∃x ¬P(x)"
        },
        {
          "ko": "∃x P(x)",
          "en": "∃x P(x)"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "적어도 한 반례가 존재한다는 뜻입니다.",
        "en": "It means at least one counterexample exists."
      }
    }
  ]
},
{
  "id": "sets-functions",
  "title": {
    "ko": "집합·함수와 역상",
    "en": "Sets, Functions and Preimages"
  },
  "subtitle": {
    "ko": "원소의 소속과 함수의 입력·출력 공간을 구분하여 데이터 변환을 읽습니다.",
    "en": "Separate set membership from function domains and codomains when reading data transformations."
  },
  "prerequisiteIds": [
    "quantifiers-counterexamples"
  ],
  "prerequisites": {
    "ko": "기본 정수 계산과 연결된 앞 단원의 표기",
    "en": "Basic integer arithmetic and notation in the linked prerequisites"
  },
  "eyebrow": "LOGIC LAB",
  "formula": "f:D→C · image(f) ⊆ C",
  "accent": "logic",
  "questions": [
    {
      "id": "preimage-not-inverse",
      "question": {
        "ko": "역상 f⁻¹(E)는 역함수가 있어야 하나요?",
        "en": "Does a preimage f⁻¹(E) require an inverse function?"
      },
      "options": [
        {
          "ko": "항상 필요",
          "en": "Always"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "E가 비면 필요",
          "en": "Required if E is empty"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "어떤 함수든 출력 집합의 역상을 정의할 수 있습니다.",
        "en": "Every function defines preimages of codomain subsets."
      }
    },
    {
      "id": "surjective-not-injective",
      "question": {
        "ko": "4개 입력을 2개 출력에 보내 모두 쓰면?",
        "en": "If four inputs map onto two outputs, using both, then?"
      },
      "options": [
        {
          "ko": "전사지만 단사가 아니다",
          "en": "Surjective but not injective"
        },
        {
          "ko": "반드시 전단사",
          "en": "Necessarily bijective"
        },
        {
          "ko": "함수가 아니다",
          "en": "Not a function"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "유한 집합의 개수 때문에 충돌이 생깁니다.",
        "en": "Finite cardinalities force a collision."
      }
    }
  ]
},
{
  "id": "relations-orders",
  "title": {
    "ko": "동치관계와 부분순서",
    "en": "Equivalence Relations and Partial Orders"
  },
  "subtitle": {
    "ko": "같다고 묶는 관계와 앞뒤를 비교하는 관계의 조건을 따로 검사합니다.",
    "en": "Test grouping-by-equivalence separately from ordering elements."
  },
  "prerequisiteIds": [
    "sets-functions"
  ],
  "prerequisites": {
    "ko": "기본 정수 계산과 연결된 앞 단원의 표기",
    "en": "Basic integer arithmetic and notation in the linked prerequisites"
  },
  "eyebrow": "LOGIC LAB",
  "formula": "equivalence: reflexive ∧ symmetric ∧ transitive",
  "accent": "logic",
  "questions": [
    {
      "id": "antisymmetry",
      "question": {
        "ko": "반대칭은 무엇인가요?",
        "en": "What does antisymmetry mean?"
      },
      "options": [
        {
          "ko": "xRy이면 항상 ¬yRx",
          "en": "xRy always implies not yRx"
        },
        {
          "ko": "xRy와 yRx이면 x=y",
          "en": "xRy and yRx imply x=y"
        },
        {
          "ko": "대칭이 아니기만 하면 된다",
          "en": "Merely not symmetric"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "같은 원소의 양방향 관계는 허용합니다.",
        "en": "Two-way relations are allowed for the same element."
      }
    },
    {
      "id": "divides-order",
      "question": {
        "ko": "{1,2,3,4}의 약수 관계는?",
        "en": "What is divisibility on {1,2,3,4}?"
      },
      "options": [
        {
          "ko": "전순서",
          "en": "A total order"
        },
        {
          "ko": "부분순서지만 전순서는 아님",
          "en": "Partial but not total order"
        },
        {
          "ko": "동치관계",
          "en": "An equivalence relation"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "2와 3이 비교되지 않습니다.",
        "en": "2 and 3 are incomparable."
      }
    }
  ]
},
{
  "id": "proof-strategies",
  "title": {
    "ko": "정의·가정과 증명 전략",
    "en": "Definitions, Assumptions and Proof Strategies"
  },
  "subtitle": {
    "ko": "유한 검사와 일반 증명을 구분하고 직접·대우·모순 논증을 연결합니다.",
    "en": "Distinguish finite checks from general proofs and connect direct, contrapositive and contradiction arguments."
  },
  "prerequisiteIds": [
    "propositions-implications",
    "quantifiers-counterexamples"
  ],
  "prerequisites": {
    "ko": "기본 정수 계산과 연결된 앞 단원의 표기",
    "en": "Basic integer arithmetic and notation in the linked prerequisites"
  },
  "eyebrow": "LOGIC LAB",
  "formula": "n² even ⇒ n even",
  "accent": "logic",
  "questions": [
    {
      "id": "finite-check",
      "question": {
        "ko": "유한 범위에서 반례가 없으면?",
        "en": "If no counterexample appears in a finite range?"
      },
      "options": [
        {
          "ko": "모든 정수에 증명 완료",
          "en": "Proved for all integers"
        },
        {
          "ko": "검사 범위에서만 확인",
          "en": "Only verified on the checked range"
        },
        {
          "ko": "정리가 거짓",
          "en": "The theorem is false"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "보편 정리에는 별도의 논증이 필요합니다.",
        "en": "A universal theorem needs a separate argument."
      }
    },
    {
      "id": "proof-contrapositive",
      "question": {
        "ko": "P⇒Q의 대우는?",
        "en": "What is the contrapositive of P⇒Q?"
      },
      "options": [
        {
          "ko": "Q⇒P",
          "en": "Q⇒P"
        },
        {
          "ko": "¬P⇒¬Q",
          "en": "¬P⇒¬Q"
        },
        {
          "ko": "¬Q⇒¬P",
          "en": "¬Q⇒¬P"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "두 식은 고전 논리에서 동치입니다.",
        "en": "These two statements are equivalent in classical logic."
      }
    }
  ]
},
{
  "id": "mathematical-induction",
  "title": {
    "ko": "수학적 귀납법과 유한 검사의 한계",
    "en": "Mathematical Induction and the Limits of Finite Tests"
  },
  "subtitle": {
    "ko": "시작점과 다음 단계의 연결로 무한한 명제를 증명하고 그럴듯한 패턴을 반례와 비교합니다.",
    "en": "Prove an infinite family through a base and a successor step, and compare plausible patterns with counterexamples."
  },
  "prerequisiteIds": [
    "proof-strategies"
  ],
  "prerequisites": {
    "ko": "기본 정수 계산과 연결된 앞 단원의 표기",
    "en": "Basic integer arithmetic and notation in the linked prerequisites"
  },
  "eyebrow": "LOGIC LAB",
  "formula": "P(0) ∧ ∀k≥0(P(k)⇒P(k+1)) ⇒ ∀n≥0 P(n)",
  "accent": "logic",
  "questions": [
    {
      "id": "induction-parts",
      "question": {
        "ko": "귀납법에 필요한 것은?",
        "en": "What does induction require?"
      },
      "options": [
        {
          "ko": "시작점만",
          "en": "Only a base case"
        },
        {
          "ko": "시작점과 임의 단계의 연결",
          "en": "A base and a successor link for arbitrary k"
        },
        {
          "ko": "앞의 100개 사례",
          "en": "The first 100 examples"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "무한 범위는 임의 단계 논증으로 연결됩니다.",
        "en": "The arbitrary-step argument connects the infinite family."
      }
    },
    {
      "id": "prime-pattern",
      "question": {
        "ko": "n=40의 n²+n+41은?",
        "en": "What is n²+n+41 at n=40?"
      },
      "options": [
        {
          "ko": "1681=41²",
          "en": "1681=41²"
        },
        {
          "ko": "항상 소수",
          "en": "Always prime"
        },
        {
          "ko": "41",
          "en": "41"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "41이 약수인 합성수입니다.",
        "en": "It is composite with divisor 41."
      }
    }
  ]
},

{
  "id": "numbers-ratios-units",
  "title": {
    "ko": "수·분수·비율과 단위",
    "en": "Numbers, Fractions, Ratios and Units"
  },
  "subtitle": {
    "ko": "같은 양의 다른 표현과 서로 다른 차원의 양을 구분합니다.",
    "en": "Distinguish different representations of the same quantity from quantities with different dimensions."
  },
  "prerequisiteIds": [
    "sets-functions"
  ],
  "prerequisites": {
    "ko": "연결한 앞 단원의 표기와 기본 사칙연산",
    "en": "Notation in the linked prerequisites and basic arithmetic"
  },
  "eyebrow": "ELEMENTARY MATH LAB",
  "formula": "a/b = (ka)/(kb), b≠0, k≠0",
  "accent": "elementary",
  "questions": [
    {
      "id": "fraction-equivalence",
      "question": {
        "ko": "2/6과 같은 수는?",
        "en": "Which equals 2/6?"
      },
      "options": [
        {
          "ko": "1/3",
          "en": "1/3"
        },
        {
          "ko": "2/3",
          "en": "2/3"
        },
        {
          "ko": "1/6",
          "en": "1/6"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "분자·분모를 공통의 0이 아닌 수로 나눕니다.",
        "en": "Divide both by the same nonzero common factor."
      }
    },
    {
      "id": "speed-units",
      "question": {
        "ko": "150 cm를 2 s에 이동한 평균 속력은?",
        "en": "What is average speed for 150 cm in 2 s?"
      },
      "options": [
        {
          "ko": "75 m/s",
          "en": "75 m/s"
        },
        {
          "ko": "0.75 m/s",
          "en": "0.75 m/s"
        },
        {
          "ko": "3 m/s",
          "en": "3 m/s"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "길이를 m로 변환한 뒤 시간으로 나눕니다.",
        "en": "Convert length to meters before dividing by time."
      }
    }
  ]
},
{
  "id": "real-complex-numbers",
  "title": {
    "ko": "실수와 복소수 평면",
    "en": "Real Numbers and the Complex Plane"
  },
  "subtitle": {
    "ko": "수의 범위를 넓히고 복소수 곱셈을 회전과 크기로 해석합니다.",
    "en": "Expand number systems and interpret complex multiplication through rotation and magnitude."
  },
  "prerequisiteIds": [
    "numbers-ratios-units"
  ],
  "prerequisites": {
    "ko": "연결한 앞 단원의 표기와 기본 사칙연산",
    "en": "Notation in the linked prerequisites and basic arithmetic"
  },
  "eyebrow": "ELEMENTARY MATH LAB",
  "formula": "z=a+bi · |z|=√(a²+b²)",
  "accent": "elementary",
  "questions": [
    {
      "id": "complex-square",
      "question": {
        "ko": "i²은?",
        "en": "What is i²?"
      },
      "options": [
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "−1",
          "en": "−1"
        },
        {
          "ko": "i",
          "en": "i"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "복소수 확장은 i²=−1을 만족하도록 정의합니다.",
        "en": "The complex extension defines i²=−1."
      }
    },
    {
      "id": "rotation-magnitude",
      "question": {
        "ko": "exp(iθ)를 곱하면 크기는?",
        "en": "What happens to magnitude under multiplication by exp(iθ)?"
      },
      "options": [
        {
          "ko": "항상 두 배",
          "en": "Always doubles"
        },
        {
          "ko": "보존된다",
          "en": "Is preserved"
        },
        {
          "ko": "항상 0",
          "en": "Always becomes zero"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "곱하는 수의 크기가 1입니다.",
        "en": "The multiplier has magnitude one."
      }
    }
  ]
},
{
  "id": "equations-inequalities",
  "title": {
    "ko": "방정식·부등식과 해의 집합",
    "en": "Equations, Inequalities and Solution Sets"
  },
  "subtitle": {
    "ko": "해를 하나의 숫자라고 가정하지 않고 없는 경우·여러 경우·구간으로 나눕니다.",
    "en": "Treat solutions as sets, including empty, multiple and interval cases."
  },
  "prerequisiteIds": [
    "real-complex-numbers",
    "propositions-implications"
  ],
  "prerequisites": {
    "ko": "연결한 앞 단원의 표기와 기본 사칙연산",
    "en": "Notation in the linked prerequisites and basic arithmetic"
  },
  "eyebrow": "ELEMENTARY MATH LAB",
  "formula": "ax=b · x²+ax+b≤0",
  "accent": "elementary",
  "questions": [
    {
      "id": "negative-inequality",
      "question": {
        "ko": "−2x<4를 풀면?",
        "en": "Solve −2x<4."
      },
      "options": [
        {
          "ko": "x<−2",
          "en": "x<−2"
        },
        {
          "ko": "x>−2",
          "en": "x>−2"
        },
        {
          "ko": "x=−2",
          "en": "x=−2"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "음수로 나누면 부등호 방향이 바뀝니다.",
        "en": "Division by a negative number reverses the inequality."
      }
    },
    {
      "id": "quadratic-interval",
      "question": {
        "ko": "x²−3x+2≤0의 해는?",
        "en": "Solve x²−3x+2≤0."
      },
      "options": [
        {
          "ko": "[1,2]",
          "en": "[1,2]"
        },
        {
          "ko": "실수 전체",
          "en": "All real numbers"
        },
        {
          "ko": "∅",
          "en": "∅"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "인수의 부호와 끝점을 함께 확인합니다.",
        "en": "Check factor signs and include the endpoints."
      }
    }
  ]
},
{
  "id": "polynomial-rational-functions",
  "title": {
    "ko": "다항함수·유리함수와 정의역",
    "en": "Polynomial and Rational Functions: Domains Matter"
  },
  "subtitle": {
    "ko": "식을 약분해도 사라지지 않는 정의역의 제한을 그림에서 확인합니다.",
    "en": "See why simplifying an expression does not erase domain restrictions."
  },
  "prerequisiteIds": [
    "equations-inequalities"
  ],
  "prerequisites": {
    "ko": "연결한 앞 단원의 표기와 기본 사칙연산",
    "en": "Notation in the linked prerequisites and basic arithmetic"
  },
  "eyebrow": "ELEMENTARY MATH LAB",
  "formula": "(x²−1)/(x−1)=x+1 only if x≠1",
  "accent": "elementary",
  "questions": [
    {
      "id": "hole-domain",
      "question": {
        "ko": "(x²−1)/(x−1)의 x=1 값은?",
        "en": "What is (x²−1)/(x−1) at x=1?"
      },
      "options": [
        {
          "ko": "2",
          "en": "2"
        },
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "정의되지 않음",
          "en": "Undefined"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "약분은 원래 분모의 제한을 없애지 않습니다.",
        "en": "Cancellation does not remove the original domain restriction."
      }
    },
    {
      "id": "limit-value",
      "question": {
        "ko": "극한이 존재하면 함수값도 항상 존재하나요?",
        "en": "Does a limit always imply an existing function value?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "극한이 0이면 항상",
          "en": "Always if the limit is zero"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "제거된 점이 반례입니다.",
        "en": "A removed point is a counterexample."
      }
    }
  ]
},
{
  "id": "exponentials-logarithms",
  "title": {
    "ko": "지수·로그와 척도",
    "en": "Exponentials, Logarithms and Scale"
  },
  "subtitle": {
    "ko": "곱셈을 덧셈으로 바꾸는 로그와 밑·정의역의 조건을 연결합니다.",
    "en": "Connect logarithms, multiplicative scale and the conditions on bases and domains."
  },
  "prerequisiteIds": [
    "polynomial-rational-functions"
  ],
  "prerequisites": {
    "ko": "연결한 앞 단원의 표기와 기본 사칙연산",
    "en": "Notation in the linked prerequisites and basic arithmetic"
  },
  "eyebrow": "ELEMENTARY MATH LAB",
  "formula": "logₐ y=x ⇔ aˣ=y, a>0, a≠1, y>0",
  "accent": "elementary",
  "questions": [
    {
      "id": "log-product",
      "question": {
        "ko": "양수 x,y에 대해 ln(xy)는?",
        "en": "For positive x,y, what is ln(xy)?"
      },
      "options": [
        {
          "ko": "ln x+ln y",
          "en": "ln x+ln y"
        },
        {
          "ko": "ln x·ln y",
          "en": "ln x·ln y"
        },
        {
          "ko": "ln(x+y)",
          "en": "ln(x+y)"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "곱셈이 덧셈으로 바뀝니다.",
        "en": "Logarithms turn multiplication into addition."
      }
    },
    {
      "id": "log-domain",
      "question": {
        "ko": "실수 ln 0은?",
        "en": "What is real ln 0?"
      },
      "options": [
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "정의되지 않음",
          "en": "Undefined"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "극한이 −∞라는 것과 유한 입력에서 값이 있다는 것은 다릅니다.",
        "en": "A limit of −∞ does not supply a real value at zero."
      }
    }
  ]
},
{
  "id": "trigonometry-coordinates",
  "title": {
    "ko": "삼각함수·좌표·거리와 각",
    "en": "Trigonometry, Coordinates, Distance and Angle"
  },
  "subtitle": {
    "ko": "각도를 좌표로 바꾸고 주기성과 거리 계산을 같은 그림에서 확인합니다.",
    "en": "Convert angles into coordinates and check periodicity and distance in the same diagram."
  },
  "prerequisiteIds": [
    "real-complex-numbers"
  ],
  "prerequisites": {
    "ko": "연결한 앞 단원의 표기와 기본 사칙연산",
    "en": "Notation in the linked prerequisites and basic arithmetic"
  },
  "eyebrow": "ELEMENTARY MATH LAB",
  "formula": "p=(r cosθ,r sinθ)",
  "accent": "elementary",
  "questions": [
    {
      "id": "degree-radian",
      "question": {
        "ko": "180도는 몇 라디안인가요?",
        "en": "How many radians are 180 degrees?"
      },
      "options": [
        {
          "ko": "π",
          "en": "π"
        },
        {
          "ko": "180",
          "en": "180"
        },
        {
          "ko": "2π",
          "en": "2π"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "한 바퀴의 절반입니다.",
        "en": "It is half a full turn."
      }
    },
    {
      "id": "distance-formula",
      "question": {
        "ko": "평면 두 점의 유클리드 거리는?",
        "en": "What is Euclidean distance between planar points?"
      },
      "options": [
        {
          "ko": "좌표 차이의 제곱합의 제곱근",
          "en": "Square root of summed squared coordinate differences"
        },
        {
          "ko": "좌표의 합",
          "en": "Sum of coordinates"
        },
        {
          "ko": "각도의 합",
          "en": "Sum of angles"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "같은 좌표 단위와 직교 축을 전제로 합니다.",
        "en": "This assumes compatible coordinate units and orthogonal axes."
      }
    }
  ]
},
{
  "id": "composition-inverse-functions",
  "title": {
    "ko": "함수 합성과 역함수의 조건",
    "en": "Function Composition and Conditions for Inversion"
  },
  "subtitle": {
    "ko": "계산 순서와 입력 범위가 역변환 가능성을 어떻게 바꾸는지 살펴봅니다.",
    "en": "Inspect how operation order and input restrictions determine invertibility."
  },
  "prerequisiteIds": [
    "sets-functions",
    "polynomial-rational-functions",
    "exponentials-logarithms"
  ],
  "prerequisites": {
    "ko": "연결한 앞 단원의 표기와 기본 사칙연산",
    "en": "Notation in the linked prerequisites and basic arithmetic"
  },
  "eyebrow": "ELEMENTARY MATH LAB",
  "formula": "h=f∘g · g(x)=ax+b · f(u)=u²",
  "accent": "elementary",
  "questions": [
    {
      "id": "composition-order",
      "question": {
        "ko": "f∘g는 어느 함수를 먼저 적용하나요?",
        "en": "Which function is applied first in f∘g?"
      },
      "options": [
        {
          "ko": "f",
          "en": "f"
        },
        {
          "ko": "g",
          "en": "g"
        },
        {
          "ko": "항상 동시에",
          "en": "Always both simultaneously"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "입력에 가까운 g부터입니다.",
        "en": "Start with g, nearest the input."
      }
    },
    {
      "id": "square-inverse",
      "question": {
        "ko": "실수 전체의 x²에 역함수가 있나요?",
        "en": "Does x² have an inverse function on all real inputs?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "아니요, 입력 가지 제한이 필요",
          "en": "No, an input-branch restriction is needed"
        },
        {
          "ko": "역상도 없다",
          "en": "Even preimages do not exist"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "x와 −x가 같은 값을 만듭니다.",
        "en": "x and −x produce the same output."
      }
    }
  ]
},
{
  "id": "sequences-series",
  "title": {
    "ko": "수열·부분합과 급수",
    "en": "Sequences, Partial Sums and Series"
  },
  "subtitle": {
    "ko": "항이 작아지는 것과 누적 합이 수렴하는 것을 구분합니다.",
    "en": "Distinguish shrinking terms from convergence of accumulated sums."
  },
  "prerequisiteIds": [
    "mathematical-induction",
    "exponentials-logarithms"
  ],
  "prerequisites": {
    "ko": "연결한 앞 단원의 표기와 기본 사칙연산",
    "en": "Notation in the linked prerequisites and basic arithmetic"
  },
  "eyebrow": "ELEMENTARY MATH LAB",
  "formula": "S_N=Σₙ₌₀ᴺ⁻¹ arⁿ",
  "accent": "elementary",
  "questions": [
    {
      "id": "term-limit",
      "question": {
        "ko": "급수가 수렴하면 그 항은?",
        "en": "If a series converges, its terms must?"
      },
      "options": [
        {
          "ko": "0으로 간다",
          "en": "Tend to zero"
        },
        {
          "ko": "항상 양수",
          "en": "Always be positive"
        },
        {
          "ko": "항상 1",
          "en": "Always equal one"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "필요조건이지만 충분조건은 아닙니다.",
        "en": "This is necessary but not sufficient."
      }
    },
    {
      "id": "geometric-sum",
      "question": {
        "ko": "1+1/2+1/4+…의 합은?",
        "en": "What is 1+1/2+1/4+…?"
      },
      "options": [
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "2",
          "en": "2"
        },
        {
          "ko": "무조건 발산",
          "en": "Always diverges"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "공비 1/2의 수렴 기하급수입니다.",
        "en": "It is a convergent geometric series with ratio 1/2."
      }
    }
  ]
},
{
  "id": "asymptotic-growth",
  "title": {
    "ko": "점근 표기와 성장률",
    "en": "Asymptotic Notation and Growth Rates"
  },
  "subtitle": {
    "ko": "유한 관측과 충분히 큰 입력에서의 상한을 구분합니다.",
    "en": "Separate finite observations from an upper bound for sufficiently large inputs."
  },
  "prerequisiteIds": [
    "sequences-series",
    "equations-inequalities"
  ],
  "prerequisites": {
    "ko": "연결한 앞 단원의 표기와 기본 사칙연산",
    "en": "Notation in the linked prerequisites and basic arithmetic"
  },
  "eyebrow": "ELEMENTARY MATH LAB",
  "formula": "f(n)=O(n) ⇔ ∃C>0,∃n₀,∀n≥n₀:f(n)≤Cn",
  "accent": "elementary",
  "questions": [
    {
      "id": "big-o-meaning",
      "question": {
        "ko": "Big-O는 무엇을 뜻하나요?",
        "en": "What does Big-O specify?"
      },
      "options": [
        {
          "ko": "충분히 큰 입력의 상한",
          "en": "An eventual upper bound"
        },
        {
          "ko": "정확한 실행 시간",
          "en": "Exact runtime"
        },
        {
          "ko": "항상 최악의 경우만",
          "en": "Always only a worst case"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "어떤 비용 함수를 정의했는지 별도로 명시해야 합니다.",
        "en": "Specify separately which cost function is being bounded."
      }
    },
    {
      "id": "quadratic-bound",
      "question": {
        "ko": "n²은 O(n)인가요?",
        "en": "Is n² in O(n)?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "n=1에서 같으니 예",
          "en": "Yes because they agree at n=1"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "어떤 고정 C에도 n>C를 고르면 상한이 깨집니다.",
        "en": "For every fixed C, choosing n>C violates the bound."
      }
    }
  ]
},
{
  "id": "counting-permutations",
  "title": {
    "ko": "세기·순열·조합",
    "en": "Counting, Permutations and Combinations"
  },
  "subtitle": {
    "ko": "순서와 중복 허용 여부를 먼저 정하면 셀 대상이 명확해집니다.",
    "en": "Define order and replacement before counting outcomes."
  },
  "prerequisiteIds": [
    "sets-functions",
    "numbers-ratios-units"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·정수 계산",
    "en": "Sets, functions and integer arithmetic in the linked prerequisites"
  },
  "eyebrow": "DISCRETE LAB",
  "formula": "P(n,k)=n!/(n−k)!; C(n,k)=P(n,k)/k!",
  "accent": "discrete",
  "questions": [
    {
      "id": "order",
      "question": {
        "ko": "5개 중 무중복·무순서로 3개를 고르는 수는?",
        "en": "How many unordered selections of three distinct items from five?"
      },
      "options": [
        {
          "ko": "60",
          "en": "60"
        },
        {
          "ko": "10",
          "en": "10"
        },
        {
          "ko": "125",
          "en": "125"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "60개의 순서를 3!로 나눕니다.",
        "en": "Divide 60 ordered outcomes by 3!."
      }
    },
    {
      "id": "empty",
      "question": {
        "ko": "n=0,k=0인 경우의 수는?",
        "en": "How many outcomes for n=0,k=0?"
      },
      "options": [
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "정의 불가",
          "en": "Undefined"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "빈 선택을 한 가지로 셉니다.",
        "en": "Count the empty selection once."
      }
    }
  ]
},
{
  "id": "binomial-inclusion-exclusion",
  "title": {
    "ko": "이항계수·포함배제",
    "en": "Binomial Coefficients and Inclusion–Exclusion"
  },
  "subtitle": {
    "ko": "겹친 원소를 두 번 셌다면 한 번 빼서 각 원소의 기여를 맞춥니다.",
    "en": "Correct double counting by tracking each element’s contribution."
  },
  "prerequisiteIds": [
    "counting-permutations"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·정수 계산",
    "en": "Sets, functions and integer arithmetic in the linked prerequisites"
  },
  "eyebrow": "DISCRETE LAB",
  "formula": "|A∪B|=|A|+|B|−|A∩B|",
  "accent": "discrete",
  "questions": [
    {
      "id": "union",
      "question": {
        "ko": "|A|=3,|B|=3,|A∩B|=1이면 |A∪B|는?",
        "en": "If |A|=3,|B|=3,|A∩B|=1, what is |A∪B|?"
      },
      "options": [
        {
          "ko": "6",
          "en": "6"
        },
        {
          "ko": "5",
          "en": "5"
        },
        {
          "ko": "7",
          "en": "7"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "교집합은 두 번 더해졌으므로 한 번 뺍니다.",
        "en": "Subtract the intersection once to correct double counting."
      }
    },
    {
      "id": "all-subsets",
      "question": {
        "ko": "6개 원소의 전체 부분집합 수는?",
        "en": "How many subsets does a six-element set have?"
      },
      "options": [
        {
          "ko": "6!",
          "en": "6!"
        },
        {
          "ko": "36",
          "en": "36"
        },
        {
          "ko": "64",
          "en": "64"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "각 원소마다 포함 여부 두 선택입니다.",
        "en": "Each element has two membership choices."
      }
    }
  ]
},
{
  "id": "recurrences-generating-functions",
  "title": {
    "ko": "점화식·생성함수",
    "en": "Recurrences and Generating Functions"
  },
  "subtitle": {
    "ko": "이전 상태를 잇는 규칙을 계수로 저장하면 무한 수열을 하나의 형식적 식으로 다룰 수 있습니다.",
    "en": "Store a recurrence as coefficients of a formal series, then distinguish algebra from convergence."
  },
  "prerequisiteIds": [
    "sequences-series",
    "mathematical-induction"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·정수 계산",
    "en": "Sets, functions and integer arithmetic in the linked prerequisites"
  },
  "eyebrow": "DISCRETE LAB",
  "formula": "aₙ=paₙ₋₁+qaₙ₋₂; G(x)=x/(1−px−qx²)",
  "accent": "discrete",
  "questions": [
    {
      "id": "fibonacci",
      "question": {
        "ko": "a₀=0,a₁=1,aₙ=aₙ₋₁+aₙ₋₂일 때 a₈은?",
        "en": "For a₀=0,a₁=1,aₙ=aₙ₋₁+aₙ₋₂, what is a₈?"
      },
      "options": [
        {
          "ko": "13",
          "en": "13"
        },
        {
          "ko": "21",
          "en": "21"
        },
        {
          "ko": "34",
          "en": "34"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "0,1,1,2,3,5,8,13,21입니다.",
        "en": "The sequence is 0,1,1,2,3,5,8,13,21."
      }
    },
    {
      "id": "formal",
      "question": {
        "ko": "유리식의 분모가 0이 아니면 급수도 반드시 수렴하나요?",
        "en": "Does a nonzero rational denominator guarantee power-series convergence?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "아니다",
          "en": "No"
        },
        {
          "ko": "유한 표가 증명한다",
          "en": "A finite table proves it"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "수렴 반지름과 별도로 확인해야 합니다.",
        "en": "Check the convergence radius separately."
      }
    }
  ]
},
{
  "id": "graphs-trees-dags",
  "title": {
    "ko": "그래프·트리·DAG",
    "en": "Graphs, Trees and DAGs"
  },
  "subtitle": {
    "ko": "방향을 지운 트리와 방향 순환이 없는 그래프는 서로 다른 조건입니다.",
    "en": "An underlying undirected tree and a directed acyclic graph satisfy different conditions."
  },
  "prerequisiteIds": [
    "relations-orders"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·정수 계산",
    "en": "Sets, functions and integer arithmetic in the linked prerequisites"
  },
  "eyebrow": "DISCRETE LAB",
  "formula": "DAG ⇔ a topological order exists",
  "accent": "discrete",
  "questions": [
    {
      "id": "dag",
      "question": {
        "ko": "방향 순환이 있는 그래프에 위상 순서가 있나요?",
        "en": "Can a graph with a directed cycle have a topological order?"
      },
      "options": [
        {
          "ko": "있다",
          "en": "Yes"
        },
        {
          "ko": "없다",
          "en": "No"
        },
        {
          "ko": "정점 수만 본다",
          "en": "Only vertex count matters"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "순환의 부등식이 자신보다 앞서야 한다는 모순을 만듭니다.",
        "en": "Following the cycle would require a vertex to precede itself."
      }
    },
    {
      "id": "tree",
      "question": {
        "ko": "기저 무방향 트리의 조건은?",
        "en": "What characterizes the underlying undirected tree?"
      },
      "options": [
        {
          "ko": "연결·무순환",
          "en": "Connected and acyclic"
        },
        {
          "ko": "간선이 많음",
          "en": "Many edges"
        },
        {
          "ko": "방향 왕복 허용만",
          "en": "Only reciprocal directions"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "유한 단순 그래프에서는 연결이고 간선 수 n−1인 조건과 같습니다.",
        "en": "For a finite simple graph, equivalently connected with n−1 edges."
      }
    }
  ]
},
{
  "id": "paths-cycles-connectivity",
  "title": {
    "ko": "연결성·경로·순환",
    "en": "Connectivity, Paths and Cycles"
  },
  "subtitle": {
    "ko": "도달 가능성, 최단 경로, 고정 길이 보행 수는 서로 다른 질문입니다.",
    "en": "Reachability, shortest paths and fixed-length walk counts answer different questions."
  },
  "prerequisiteIds": [
    "graphs-trees-dags"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·정수 계산",
    "en": "Sets, functions and integer arithmetic in the linked prerequisites"
  },
  "eyebrow": "DISCRETE LAB",
  "formula": "(Aᵏ)ₛₜ = number of length-k walks",
  "accent": "discrete",
  "questions": [
    {
      "id": "walk",
      "question": {
        "ko": "A²의 원소가 세는 것은?",
        "en": "What does an entry of A² count?"
      },
      "options": [
        {
          "ko": "길이 2 보행",
          "en": "Length-two walks"
        },
        {
          "ko": "항상 최단 경로",
          "en": "Always shortest paths"
        },
        {
          "ko": "연결 성분",
          "en": "Components"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "중간 정점을 합산하므로 반복 정점도 허용합니다.",
        "en": "Summing over intermediate vertices permits repeated vertices."
      }
    },
    {
      "id": "self",
      "question": {
        "ko": "길이 0 경로를 허용할 때 d(s,s)는?",
        "en": "With length-zero paths allowed, what is d(s,s)?"
      },
      "options": [
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "항상 무한",
          "en": "Always infinite"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "아무 간선도 지나지 않는 경로가 있습니다.",
        "en": "There is a path that traverses no edges."
      }
    }
  ]
},
{
  "id": "boolean-algebra",
  "title": {
    "ko": "Boolean 대수와 논리 표현",
    "en": "Boolean Algebra and Logical Representations"
  },
  "subtitle": {
    "ko": "진리표·논리식·GF(2) 다항식은 같은 함수를 다른 규칙으로 표현합니다.",
    "en": "Truth tables, logical formulas and GF(2) polynomials represent the same function using different operations."
  },
  "prerequisiteIds": [
    "propositions-implications",
    "binomial-inclusion-exclusion"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·정수 계산",
    "en": "Sets, functions and integer arithmetic in the linked prerequisites"
  },
  "eyebrow": "DISCRETE LAB",
  "formula": "¬(x∧y)=¬x∨¬y; F=⊕ a_S∏x_i",
  "accent": "discrete",
  "questions": [
    {
      "id": "xor",
      "question": {
        "ko": "GF(2)에서 1+1은?",
        "en": "What is 1+1 in GF(2)?"
      },
      "options": [
        {
          "ko": "2",
          "en": "2"
        },
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "1",
          "en": "1"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "덧셈은 modulo 2이므로 0입니다.",
        "en": "Addition modulo two gives zero."
      }
    },
    {
      "id": "demorgan",
      "question": {
        "ko": "¬(x∧y)와 같은 식은?",
        "en": "Which equals ¬(x∧y)?"
      },
      "options": [
        {
          "ko": "¬x∧¬y",
          "en": "¬x∧¬y"
        },
        {
          "ko": "¬x∨¬y",
          "en": "¬x∨¬y"
        },
        {
          "ko": "x∨y",
          "en": "x∨y"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "두 입력 중 하나라도 거짓이면 원래 AND의 부정이 참입니다.",
        "en": "If either input is false, the negated conjunction is true."
      }
    }
  ]
},
{
  "id": "automata-formal-languages",
  "title": {
    "ko": "오토마타·형식언어",
    "en": "Automata and Formal Languages"
  },
  "subtitle": {
    "ko": "전체 입력을 기억하는 대신 앞으로의 판단에 필요한 유한 상태만 남깁니다.",
    "en": "Retain only a finite state sufficient for future acceptance decisions."
  },
  "prerequisiteIds": [
    "boolean-algebra",
    "graphs-trees-dags"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·정수 계산",
    "en": "Sets, functions and integer arithmetic in the linked prerequisites"
  },
  "eyebrow": "DISCRETE LAB",
  "formula": "δ*(q,wa)=δ(δ*(q,w),a)",
  "accent": "discrete",
  "questions": [
    {
      "id": "epsilon",
      "question": {
        "ko": "빈 문자열의 길이는?",
        "en": "What is the length of the empty word?"
      },
      "options": [
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "정의 불가",
          "en": "Undefined"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "ε는 기호가 하나도 없는 문자열입니다.",
        "en": "ε contains no symbols."
      }
    },
    {
      "id": "parity",
      "question": {
        "ko": "짝수 개의 1 언어에서 101의 최종 상태는?",
        "en": "For even parity of ones, where does 101 finish?"
      },
      "options": [
        {
          "ko": "짝수 수용 상태",
          "en": "Even accepting state"
        },
        {
          "ko": "홀수 거부 상태",
          "en": "Odd rejecting state"
        },
        {
          "ko": "전이 불가",
          "en": "No transition"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "1을 두 번 읽어 처음 홀짝 상태로 돌아옵니다.",
        "en": "Two ones return the machine to the initial parity."
      }
    }
  ]
},
{
  "id": "posets-lattices",
  "title": {
    "ko": "부분순서·격자",
    "en": "Partial Orders and Lattices"
  },
  "subtitle": {
    "ko": "비교할 수 없는 두 집합도 공통 하한과 공통 상한을 가질 수 있습니다.",
    "en": "Two incomparable sets can still have a greatest lower bound and a least upper bound."
  },
  "prerequisiteIds": [
    "relations-orders",
    "binomial-inclusion-exclusion"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·정수 계산",
    "en": "Sets, functions and integer arithmetic in the linked prerequisites"
  },
  "eyebrow": "DISCRETE LAB",
  "formula": "A∧B=A∩B; A∨B=A∪B",
  "accent": "discrete",
  "questions": [
    {
      "id": "meet",
      "question": {
        "ko": "부분집합 격자의 meet는?",
        "en": "What is the meet in a subset lattice?"
      },
      "options": [
        {
          "ko": "교집합",
          "en": "Intersection"
        },
        {
          "ko": "합집합",
          "en": "Union"
        },
        {
          "ko": "항상 공집합",
          "en": "Always empty"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "모든 공통 하한을 포함하는 가장 큰 공통 하한입니다.",
        "en": "It is the greatest common lower bound."
      }
    },
    {
      "id": "total",
      "question": {
        "ko": "격자는 반드시 전순서인가요?",
        "en": "Must a lattice be totally ordered?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "아니다",
          "en": "No"
        },
        {
          "ko": "원소가 8개일 때만",
          "en": "Only with eight elements"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "meet와 join은 존재해도 원소끼리 비교 불가할 수 있습니다.",
        "en": "Meets and joins may exist for incomparable elements."
      }
    }
  ]
},
{
  "id": "finite-state-transitions",
  "title": {
    "ko": "유한 상태와 전이",
    "en": "Finite States and Transitions"
  },
  "subtitle": {
    "ko": "결정적 유한계에서는 같은 상태를 다시 만나면 이후 궤적이 반복됩니다.",
    "en": "In a deterministic finite system, revisiting a state forces the future orbit to repeat."
  },
  "prerequisiteIds": [
    "automata-formal-languages",
    "mathematical-induction"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·정수 계산",
    "en": "Sets, functions and integer arithmetic in the linked prerequisites"
  },
  "eyebrow": "DISCRETE LAB",
  "formula": "sₜ₊₁=T(sₜ); sₜ₊λ=sₜ for t≥μ",
  "accent": "discrete",
  "questions": [
    {
      "id": "period",
      "question": {
        "ko": "고정점의 최소 양의 주기는?",
        "en": "What is a fixed point’s least positive period?"
      },
      "options": [
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "3",
          "en": "3"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "한 번 전이하면 같은 상태로 돌아옵니다.",
        "en": "One transition returns to the same state."
      }
    },
    {
      "id": "repeat",
      "question": {
        "ko": "상태가 반복되면 이후도 반복되는 이유는?",
        "en": "Why does a repeated state force a repeated future?"
      },
      "options": [
        {
          "ko": "결정적 전이 함수가 같기 때문",
          "en": "The deterministic transition map is unchanged"
        },
        {
          "ko": "상태 번호가 커지기 때문",
          "en": "State labels increase"
        },
        {
          "ko": "항상 무작위이기 때문",
          "en": "The system is always random"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "같은 입력 상태에는 같은 다음 상태가 대응합니다.",
        "en": "The same state always has the same next state."
      }
    }
  ]
},
  { id: 'vectors-and-forces', prerequisiteIds: [], title: { ko: '벡터와 힘의 합', en: 'Vectors and the Sum of Forces' },
    subtitle: { ko: '방향이 다른 두 힘은 어떻게 더해질까요?', en: 'How do forces pointing in different directions add up?' },
    prerequisites: { ko: '좌표와 기본 사칙연산', en: 'Coordinates and basic arithmetic' },
    eyebrow: 'VECTOR LAB', formula: 'F₁ + F₂ = Fnet', accent: 'vector', questions: [
      { id: 'vector-magnitude', question: { ko: '기준 예제의 힘 (3,0,0) N과 (0,4,0) N을 더한 크기는?', en: 'In the reference example, what is the magnitude of (3,0,0) N + (0,4,0) N?' }, options: [{ ko: '7 N', en: '7 N' }, { ko: '5 N', en: '5 N' }, { ko: '1 N', en: '1 N' }], answer: 1, explanation: { ko: '성분을 더한 뒤 √(3²+4²)=5 N을 구합니다. 크기를 바로 더하는 것은 같은 방향일 때입니다.', en: 'Add components first: √(3²+4²)=5 N. Adding magnitudes directly works for forces in the same direction.' } },
      { id: 'zero-force', question: { ko: '합력이 0이고 초기 속도가 (1,0,0) m/s일 때, 다른 힘이 없다면?', en: 'With zero net force and initial velocity (1,0,0) m/s, what happens if there are no other forces?' }, options: [{ ko: '즉시 정지한다', en: 'It stops immediately' }, { ko: '같은 속도로 계속 움직인다', en: 'It keeps moving at the same velocity' }, { ko: '속력이 계속 증가한다', en: 'Its speed keeps increasing' }], answer: 1, explanation: { ko: '합력이 0이면 가속도가 0입니다. 초기 속도는 그대로 유지됩니다.', en: 'Zero net force means zero acceleration. The initial velocity stays unchanged.' } },
    ] },
{
  "id": "inner-products", prerequisiteIds: ['vectors-and-forces'],
  "title": {
    "ko": "내적과 직교투영",
    "en": "Inner Products and Orthogonal Projection"
  },
  "subtitle": {
    "ko": "한 벡터가 다른 방향을 얼마나 담고 있을까요?",
    "en": "How much of one vector lies along another direction?"
  },
  "prerequisites": {
    "ko": "벡터의 성분·크기·덧셈",
    "en": "Vector components, magnitude and addition"
  },
  "eyebrow": "PROJECTION LAB",
  "formula": "u = projᵥ u + r",
  "accent": "projection",
  "questions": [
    {
      "id": "projection-coefficient",
      "question": {
        "ko": "u=(3,2,0), v=(2,0,0)일 때 u의 v 방향 투영은?",
        "en": "For u=(3,2,0), v=(2,0,0), what is the projection of u onto v?"
      },
      "options": [
        {
          "ko": "(6,0,0)",
          "en": "(6,0,0)"
        },
        {
          "ko": "(3,0,0)",
          "en": "(3,0,0)"
        },
        {
          "ko": "(1.5,0,0)",
          "en": "(1.5,0,0)"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "계수는 (u·v)/(v·v)=6/4=1.5입니다. 여기에 v를 곱하면 (3,0,0)입니다.",
        "en": "The coefficient is (u·v)/(v·v)=6/4=1.5. Multiplying v gives (3,0,0)."
      }
    },
    {
      "id": "zero-reference",
      "question": {
        "ko": "투영의 기준 벡터 v가 0이면?",
        "en": "What if the reference vector v is zero?"
      },
      "options": [
        {
          "ko": "항상 u로 투영한다",
          "en": "Always project to u"
        },
        {
          "ko": "각도는 90도다",
          "en": "The angle is 90 degrees"
        },
        {
          "ko": "방향이 없어 이 공식의 투영은 정의되지 않는다",
          "en": "The direction and this projection formula are undefined"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "v·v=0으로 나눌 수 없습니다. 영벡터의 방향을 임의로 정하지 않습니다.",
        "en": "Division by v·v=0 is invalid. Do not invent a direction for the zero vector."
      }
    }
  ]
},
  { id: 'linear-transforms', prerequisiteIds: ['vectors-and-forces'], title: { ko: '행렬과 선형변환', en: 'Matrices and Linear Transformations' },
    subtitle: { ko: '행렬 하나가 공간 전체를 바꾸는 방법', en: 'How one matrix changes an entire space' },
    prerequisites: { ko: '벡터의 성분과 덧셈', en: 'Vector components and addition' },
    eyebrow: 'MATRIX LAB', formula: 'y = Ax', accent: 'matrix', questions: [
      { id: 'reflection-volume', question: { ko: '행렬식이 −1인 반사 변환의 부피 배율은?', en: 'What is the volume scale of a reflection with determinant −1?' }, options: [{ ko: '−1', en: '−1' }, { ko: '0', en: '0' }, { ko: '1', en: '1' }], answer: 2, explanation: { ko: '부피 배율은 |det(A)|=1입니다. 음의 부호는 방향 반전을 나타냅니다.', en: 'Volume scales by |det(A)|=1. The negative sign indicates an orientation reversal.' } },
      { id: 'singular-transform', question: { ko: 'det(A)=0이면 어떤 일이 생길까요?', en: 'What happens when det(A)=0?' }, options: [{ ko: '역변환이 항상 가능하다', en: 'The transformation always has an inverse' }, { ko: '일부 차원이 사라져 역행렬이 없다', en: 'Some dimensions collapse and no inverse exists' }, { ko: '항상 회전이다', en: 'It is always a rotation' }], answer: 1, explanation: { ko: '서로 다른 입력이 같은 점으로 갈 수 있으므로 원래 입력을 유일하게 복원할 수 없습니다.', en: 'Different inputs can map to the same point, so the original input cannot be recovered uniquely.' } },
    ] },
{
  "id": "least-squares", prerequisiteIds: ['inner-products', 'linear-transforms'],
  "title": {
    "ko": "선형방정식과 최소제곱",
    "en": "Linear Equations and Least Squares"
  },
  "subtitle": {
    "ko": "모든 점을 맞출 수 없을 때, 무엇을 가장 가깝다고 할까요?",
    "en": "When no line fits every point, what does closest mean?"
  },
  "prerequisites": {
    "ko": "행렬 곱, 내적과 직교투영",
    "en": "Matrix multiplication, inner products and orthogonal projection"
  },
  "eyebrow": "REGRESSION LAB",
  "formula": "xᵀ(y − a*x) = 0",
  "accent": "regression",
  "questions": [
    {
      "id": "regression-optimum",
      "question": {
        "ko": "x=(1,2,3), y=(1,2,2), y≈ax에서 최소제곱 기울기는?",
        "en": "For x=(1,2,3), y=(1,2,2), y≈ax, what is the least-squares slope?"
      },
      "options": [
        {
          "ko": "11/14",
          "en": "11/14"
        },
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "5/6",
          "en": "5/6"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "a*=(x·y)/(x·x)=(1+4+6)/(1+4+9)=11/14입니다.",
        "en": "a*=(x·y)/(x·x)=(1+4+6)/(1+4+9)=11/14."
      }
    },
    {
      "id": "regression-residual",
      "question": {
        "ko": "최소제곱 해에서 잔차 r=y−a*x는?",
        "en": "At the least-squares solution, residual r=y−a*x is:"
      },
      "options": [
        {
          "ko": "항상 0이다",
          "en": "Always zero"
        },
        {
          "ko": "x에 직교하지만 0이 아닐 수 있다",
          "en": "Orthogonal to x, but possibly nonzero"
        },
        {
          "ko": "관측값과 무관하다",
          "en": "Independent of the observations"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "정규 조건은 x·r=0입니다. 정확히 맞출 수 없는 관측에서는 잔차가 남습니다.",
        "en": "The normal condition is x·r=0. Inconsistent observations leave a nonzero residual."
      }
    }
  ]
},
{
  "id": "eigenvalues", prerequisiteIds: ['inner-products', 'linear-transforms'],
  "title": {
    "ko": "고유값과 주성분 분석",
    "en": "Eigenvalues and Principal Component Analysis"
  },
  "subtitle": {
    "ko": "변환에도 유지되는 방향으로 데이터의 퍼짐을 읽기",
    "en": "Read data spread through directions preserved by a transformation"
  },
  "prerequisites": {
    "ko": "행렬과 선형변환, 내적·투영, 평균의 뜻",
    "en": "Linear transformations, inner products, projection and means"
  },
  "eyebrow": "EIGEN LAB",
  "formula": "Aq = λq",
  "accent": "eigen",
  "questions": [
    {
      "id": "pca-variance",
      "question": {
        "ko": "공분산 [[2,1],[1,2]]에서 첫 주성분이 담는 분산 비율은?",
        "en": "For covariance [[2,1],[1,2]], what variance fraction does the first component retain?"
      },
      "options": [
        {
          "ko": "50%",
          "en": "50%"
        },
        {
          "ko": "100%",
          "en": "100%"
        },
        {
          "ko": "75%",
          "en": "75%"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "고유값은 3과 1입니다. 비율은 3/(3+1)=0.75이며 타원 반지름의 비율과 다릅니다.",
        "en": "Eigenvalues are 3 and 1, so the ratio is 3/(3+1)=0.75. It is not an ellipse-radius ratio."
      }
    },
    {
      "id": "negative-covariance",
      "question": {
        "ko": "대칭 행렬의 고유값 하나가 −1이라면 공분산으로 쓸 수 있을까요?",
        "en": "Can a symmetric matrix with an eigenvalue −1 serve as a covariance matrix?"
      },
      "options": [
        {
          "ko": "가능하다: 대칭이면 충분하다",
          "en": "Yes: symmetry is sufficient"
        },
        {
          "ko": "불가능하다: 어떤 방향의 분산이 음수가 된다",
          "en": "No: variance would be negative along some direction"
        },
        {
          "ko": "항상 영행렬이다",
          "en": "It is always the zero matrix"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "공분산은 양의 준정부호입니다. 모든 방향의 분산 vᵀCv가 0 이상이어야 합니다.",
        "en": "Covariance is positive semidefinite: directional variance vᵀCv must be nonnegative."
      }
    }
  ]
},
  { id: 'gradients', prerequisiteIds: ['vectors-and-forces', 'linear-transforms'], title: { ko: '그래디언트와 학습률', en: 'Gradients and Learning Rates' },
    subtitle: { ko: '작게 내려가는 한 걸음이 학습이 되려면', en: 'When does a downhill step actually learn?' },
    prerequisites: { ko: '함수, 좌표, 기울기의 기본 개념', en: 'Functions, coordinates and an introduction to slope' },
    eyebrow: 'GRADIENT LAB', formula: 'pnext = p − η∇L', accent: 'gradient', questions: [
      { id: 'gradient-direction', question: { ko: '그래디언트 ∇L은 국소적으로 어느 방향을 가리킬까요?', en: 'Which direction does the gradient ∇L point locally?' }, options: [{ ko: '함수가 가장 빠르게 증가하는 방향', en: 'The direction of steepest increase' }, { ko: '항상 최솟값을 바로 가리킨다', en: 'Always directly toward the minimum' }, { ko: '물리적 속도 방향', en: 'The direction of physical velocity' }], answer: 0, explanation: { ko: '유클리드 거리에서 그래디언트는 가장 가파른 증가 방향입니다. 경사하강법은 반대 방향으로 갱신합니다.', en: 'Under the Euclidean metric, the gradient gives steepest increase. Gradient descent updates in the opposite direction.' } },
      { id: 'rate-boundary', question: { ko: 'L=(2x²+y²)/2에서 모든 시작점의 수렴을 보장하는 학습률 범위는?', en: 'For L=(2x²+y²)/2, which learning-rate range ensures convergence from every starting point?' }, options: [{ ko: 'η > 1', en: 'η > 1' }, { ko: '0 < η < 1', en: '0 < η < 1' }, { ko: '모든 양수 η', en: 'Every positive η' }], answer: 1, explanation: { ko: '이 양의 이차함수에서는 0<η<2/max(2,1)=1입니다. 이 범위를 임의의 비선형 함수에 그대로 적용할 수는 없습니다.', en: 'For this positive quadratic, 0<η<2/max(2,1)=1. This interval is not a universal rule for arbitrary nonlinear functions.' } },
    ] },
{
  "id": "chain-rule",
  "prerequisiteIds": [
    "gradients"
  ],
  "title": {
    "ko": "연쇄법칙과 국소 변화",
    "en": "The Chain Rule and Local Change"
  },
  "subtitle": {
    "ko": "작은 변화는 함수의 각 단계를 어떻게 통과할까요?",
    "en": "How does a small change travel through a composition?"
  },
  "prerequisites": {
    "ko": "미분·기울기와 일차 근사",
    "en": "Derivatives, gradients and linear approximation"
  },
  "eyebrow": "CHAIN RULE LAB",
  "formula": "dy/dx = (dy/du)(du/dx)",
  "accent": "chain",
  "questions": [
    {
      "id": "chain-product",
      "question": {
        "ko": "u=2x+1, y=u², x=1에서 dy/dx는?",
        "en": "For u=2x+1, y=u² at x=1, what is dy/dx?"
      },
      "options": [
        {
          "ko": "6",
          "en": "6"
        },
        {
          "ko": "12",
          "en": "12"
        },
        {
          "ko": "9",
          "en": "9"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "u=3에서 바깥 미분은 6, 안쪽 미분은 2이므로 12입니다.",
        "en": "At u=3 the outer derivative is 6 and the inner derivative is 2, giving 12."
      }
    },
    {
      "id": "chain-zero",
      "question": {
        "ko": "x=−0.5에서 기울기가 0이면 주변 출력도 항상 같은가요?",
        "en": "Does zero slope at x=−0.5 mean the output stays constant nearby?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "아니다. 이차 변화가 남는다",
          "en": "No. A quadratic change remains"
        },
        {
          "ko": "미분이 정의되지 않는다",
          "en": "The derivative is undefined"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "이 점에서 일차항만 0입니다. Δx=0.1이면 Δy=0.04입니다.",
        "en": "Only the first-order term is zero here. With Δx=0.1, Δy=0.04."
      }
    }
  ]
},
{
  "id": "jacobians",
  "prerequisiteIds": [
    "chain-rule",
    "linear-transforms"
  ],
  "title": {
    "ko": "Jacobian과 계산 그래프",
    "en": "Jacobians and Computation Graphs"
  },
  "subtitle": {
    "ko": "여러 입력과 출력의 변화는 어떻게 모아질까요?",
    "en": "How do we collect changes across multiple inputs and outputs?"
  },
  "prerequisites": {
    "ko": "행렬 곱과 연쇄법칙",
    "en": "Matrix multiplication and the chain rule"
  },
  "eyebrow": "JACOBIAN LAB",
  "formula": "ΔF ≈ Jd · ∇(qᵀF) = Jᵀq",
  "accent": "jacobian",
  "questions": [
    {
      "id": "jacobian-order",
      "question": {
        "ko": "F(x,y)=(xy,x²+y)의 (2,3)에서 Jacobian은?",
        "en": "What is the Jacobian of F(x,y)=(xy,x²+y) at (2,3)?"
      },
      "options": [
        {
          "ko": "[[3,2],[4,1]]",
          "en": "[[3,2],[4,1]]"
        },
        {
          "ko": "[[3,4],[2,1]]",
          "en": "[[3,4],[2,1]]"
        },
        {
          "ko": "[[6,7],[2,3]]",
          "en": "[[6,7],[2,3]]"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "행은 출력, 열은 입력입니다. 각 성분을 편미분하여 모읍니다.",
        "en": "Rows index outputs; columns index inputs. Collect the partial derivatives of each output."
      }
    },
    {
      "id": "jacobian-adjoint",
      "question": {
        "ko": "출력 가중치 q로 L=qᵀF를 정의하면 입력 기울기는?",
        "en": "If L=qᵀF for output weights q, what is its input gradient?"
      },
      "options": [
        {
          "ko": "Jq",
          "en": "Jq"
        },
        {
          "ko": "Jᵀq",
          "en": "Jᵀq"
        },
        {
          "ko": "항상 0",
          "en": "Always zero"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "각 입력에 도착하는 모든 출력 경로의 민감도를 합산합니다.",
        "en": "Sum sensitivities over all output paths reaching each input."
      }
    }
  ]
},
{
  "id": "backpropagation",
  "prerequisiteIds": [
    "chain-rule",
    "jacobians",
    "gradients"
  ],
  "title": {
    "ko": "역전파와 자동미분",
    "en": "Backpropagation and Automatic Differentiation"
  },
  "subtitle": {
    "ko": "하나의 손실에서 모든 매개변수의 기울기를 구해봅니다.",
    "en": "Compute every parameter gradient from one scalar loss."
  },
  "prerequisites": {
    "ko": "기울기·연쇄법칙·Jacobian의 전치 곱",
    "en": "Gradients, the chain rule and transposed Jacobian products"
  },
  "eyebrow": "BACKPROP LAB",
  "formula": "θnew = θ − η∇θL",
  "accent": "backprop",
  "questions": [
    {
      "id": "backprop-simultaneous",
      "question": {
        "ko": "한 번의 경사하강에서 매개변수는 어떻게 바꾸나요?",
        "en": "How should parameters change in one gradient-descent step?"
      },
      "options": [
        {
          "ko": "앞서 바꾼 값으로 다음 미분을 계산한다",
          "en": "Recompute each derivative using earlier changes"
        },
        {
          "ko": "모두 같은 이전 상태의 기울기로 갱신한다",
          "en": "Update all using gradients at the same old state"
        },
        {
          "ko": "기울기를 매개변수에 더한다",
          "en": "Add the gradient to each parameter"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "동시에 같은 점에서 평가한 기울기를 사용해야 표시된 벡터 갱신과 일치합니다.",
        "en": "Using gradients evaluated at the same point implements the stated vector update."
      }
    },
    {
      "id": "relu-kink",
      "question": {
        "ko": "ReLU의 입력 z=0에서 이 실험의 미분값 0은 무엇인가요?",
        "en": "What does the displayed ReLU slope 0 mean at z=0?"
      },
      "options": [
        {
          "ko": "유일한 고전적 도함수",
          "en": "The unique classical derivative"
        },
        {
          "ko": "명시적으로 선택한 계산 규약",
          "en": "An explicitly chosen computational convention"
        },
        {
          "ko": "중앙차분이 항상 반환하는 값",
          "en": "The value always returned by central differences"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "고전적 도함수는 없습니다. 실험은 0을 선택하며 중앙차분 불일치를 구분해서 표시합니다.",
        "en": "There is no classical derivative. This lab chooses 0 and identifies central-difference disagreement."
      }
    }
  ]
},
{
  "id": "svd",
  "title": {
    "ko": "SVD와 저랭크 근사",
    "en": "SVD and Low-Rank Approximation"
  },
  "subtitle": {
    "ko": "직사각형 선형변환의 늘어남을 분해하고 버릴 정보의 크기를 계산합니다.",
    "en": "Decompose stretching in a linear map and quantify the information discarded by approximation."
  },
  "prerequisiteIds": [
    "eigenvalues",
    "linear-transforms"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "SVD LAB",
  "formula": "A = UΣVᵀ · ‖A−A₁‖F = σ₂",
  "accent": "foundation",
  "questions": [
    {
      "id": "svd-0",
      "question": {
        "ko": "특잇값은 음수가 될 수 있나요?",
        "en": "Can a singular value be negative?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "정사각형일 때만",
          "en": "Only for square matrices"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "특잇값은 AᵀA의 고유값의 음이 아닌 제곱근입니다.",
        "en": "Singular values are nonnegative square roots of eigenvalues of AᵀA."
      }
    },
    {
      "id": "svd-1",
      "question": {
        "ko": "diag(3,1)의 최적 랭크 1 Frobenius 오차는?",
        "en": "What is the best rank-one Frobenius error for diag(3,1)?"
      },
      "options": [
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "3",
          "en": "3"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "버린 특잇값 1이 오차입니다.",
        "en": "The discarded singular value 1 is the error."
      }
    }
  ]
},
{
  "id": "vector-spaces-rank",
  "title": {
    "ko": "벡터공간·기저·랭크·영공간",
    "en": "Vector Spaces, Bases, Rank and Nullspace"
  },
  "subtitle": {
    "ko": "독립인 방향의 수와 사라지는 입력의 차원을 함께 계산합니다.",
    "en": "Count independent output directions and input directions lost by a linear map."
  },
  "prerequisiteIds": [
    "linear-transforms",
    "inner-products"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 선형사상·내적·고유값",
    "en": "Linear maps, inner products and eigenvalues in the linked prerequisites"
  },
  "eyebrow": "MATRIX LAB",
  "formula": "dim ker A + rank A = 2",
  "accent": "matrix",
  "questions": [
    {
      "id": "rank",
      "question": {
        "ko": "(1,2),(2,4)의 생성 공간 차원은?",
        "en": "What is the span dimension of (1,2),(2,4)?"
      },
      "options": [
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "2",
          "en": "2"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "두 벡터가 같은 직선의 방향입니다.",
        "en": "They span the same line."
      }
    },
    {
      "id": "nullity",
      "question": {
        "ko": "2차원 입력의 랭크 0 사상에서 영공간 차원은?",
        "en": "For a rank-zero map from R², what is nullity?"
      },
      "options": [
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "2",
          "en": "2"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "모든 입력이 영벡터로 갑니다.",
        "en": "Every input maps to zero."
      }
    }
  ]
},
{
  "id": "determinants-traces",
  "title": {
    "ko": "행렬식·trace와 기저 불변량",
    "en": "Determinants, Traces and Basis Invariants"
  },
  "subtitle": {
    "ko": "면적의 부호 있는 변화와 대각합의 기저 불변성을 구분합니다.",
    "en": "Distinguish oriented area scaling from the basis-invariant diagonal sum."
  },
  "prerequisiteIds": [
    "vector-spaces-rank"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 선형사상·내적·고유값",
    "en": "Linear maps, inner products and eigenvalues in the linked prerequisites"
  },
  "eyebrow": "MATRIX LAB",
  "formula": "det A=ad−bc; tr A=a+d",
  "accent": "matrix",
  "questions": [
    {
      "id": "area",
      "question": {
        "ko": "행렬식 −2인 변환의 면적 배율은?",
        "en": "What is the area magnitude factor for determinant −2?"
      },
      "options": [
        {
          "ko": "−2",
          "en": "−2"
        },
        {
          "ko": "2",
          "en": "2"
        },
        {
          "ko": "0",
          "en": "0"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "부호 없는 면적은 행렬식의 절댓값입니다.",
        "en": "Unsigned area uses the absolute determinant."
      }
    },
    {
      "id": "trace",
      "question": {
        "ko": "행렬식 1,trace 0이면 실수 고유값이 반드시 있나요?",
        "en": "With determinant one and trace zero, must real eigenvalues exist?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "아니다",
          "en": "No"
        },
        {
          "ko": "항등행렬만 가능",
          "en": "Only the identity"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "특성다항식은 λ²+1입니다.",
        "en": "The characteristic polynomial is λ²+1."
      }
    }
  ]
},
{
  "id": "positive-semidefinite-matrices",
  "title": {
    "ko": "대칭·양의 준정부호 행렬",
    "en": "Symmetric and Positive Semidefinite Matrices"
  },
  "subtitle": {
    "ko": "대각 원소가 양수인 것과 모든 방향의 이차형식이 음이 아닌 것은 다릅니다.",
    "en": "Positive diagonal entries alone do not guarantee nonnegative quadratic values in every direction."
  },
  "prerequisiteIds": [
    "eigenvalues",
    "vector-spaces-rank"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 선형사상·내적·고유값",
    "en": "Linear maps, inner products and eigenvalues in the linked prerequisites"
  },
  "eyebrow": "MATRIX LAB",
  "formula": "A⪰0 ⇔ vᵀAv≥0 for every v",
  "accent": "matrix",
  "questions": [
    {
      "id": "diag",
      "question": {
        "ko": "양의 대각 원소만으로 PSD인가요?",
        "en": "Do positive diagonal entries alone imply PSD?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "아니다",
          "en": "No"
        },
        {
          "ko": "trace만 확인",
          "en": "Check only trace"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "[[1,2],[2,1]]이 반례입니다.",
        "en": "[[1,2],[2,1]] is a counterexample."
      }
    },
    {
      "id": "zero-eigen",
      "question": {
        "ko": "PSD는 0 고유값을 허용하나요?",
        "en": "Can a PSD matrix have a zero eigenvalue?"
      },
      "options": [
        {
          "ko": "허용한다",
          "en": "Yes"
        },
        {
          "ko": "허용하지 않는다",
          "en": "No"
        },
        {
          "ko": "항상 0행렬만",
          "en": "Only the zero matrix"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "음수는 금지하지만 0은 허용합니다.",
        "en": "Nonnegative includes zero."
      }
    }
  ]
},
{
  "id": "quadratic-forms-orders",
  "title": {
    "ko": "이차형식·행렬 부등식",
    "en": "Quadratic Forms and Matrix Inequalities"
  },
  "subtitle": {
    "ko": "한 방향의 상한과 모든 방향에서의 행렬 상한은 다른 주장입니다.",
    "en": "A bound in one direction differs from a matrix bound valid in all directions."
  },
  "prerequisiteIds": [
    "positive-semidefinite-matrices"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 선형사상·내적·고유값",
    "en": "Linear maps, inner products and eigenvalues in the linked prerequisites"
  },
  "eyebrow": "MATRIX LAB",
  "formula": "A⪯CI ⇔ λmax(A)≤C",
  "accent": "matrix",
  "questions": [
    {
      "id": "loewner",
      "question": {
        "ko": "A⪯B의 뜻은?",
        "en": "What does A⪯B mean?"
      },
      "options": [
        {
          "ko": "모든 원소가 작음",
          "en": "Entrywise less or equal"
        },
        {
          "ko": "B−A가 PSD",
          "en": "B−A is PSD"
        },
        {
          "ko": "trace만 작음",
          "en": "Only smaller trace"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "모든 벡터 v에서 vᵀ(B−A)v≥0입니다.",
        "en": "For every v, vᵀ(B−A)v≥0."
      }
    },
    {
      "id": "probe",
      "question": {
        "ko": "한 방향의 부등식이 모든 방향을 보장하나요?",
        "en": "Does one directional test establish the bound in all directions?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "아니다",
          "en": "No"
        },
        {
          "ko": "각도 0이면 보장",
          "en": "Yes at zero degrees"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "일반적으로 다른 방향이 반례가 될 수 있습니다.",
        "en": "Another direction can be a counterexample."
      }
    }
  ]
},
{
  "id": "kronecker-products",
  "title": {
    "ko": "Kronecker 곱·벡터화",
    "en": "Kronecker Products and Vectorization"
  },
  "subtitle": {
    "ko": "행렬을 벡터로 펼칠 때 열 우선 순서와 곱의 순서를 함께 고정합니다.",
    "en": "Fix the vectorization convention before choosing the Kronecker factor order."
  },
  "prerequisiteIds": [
    "linear-transforms",
    "determinants-traces"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 선형사상·내적·고유값",
    "en": "Linear maps, inner products and eigenvalues in the linked prerequisites"
  },
  "eyebrow": "MATRIX LAB",
  "formula": "vec(AXBᵀ)=(B⊗A)vec(X)",
  "accent": "matrix",
  "questions": [
    {
      "id": "vec",
      "question": {
        "ko": "열 우선 vec([[1,2],[3,4]])는?",
        "en": "What is column-major vec([[1,2],[3,4]])?"
      },
      "options": [
        {
          "ko": "(1,2,3,4)",
          "en": "(1,2,3,4)"
        },
        {
          "ko": "(1,3,2,4)",
          "en": "(1,3,2,4)"
        },
        {
          "ko": "(4,3,2,1)",
          "en": "(4,3,2,1)"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "첫 열 다음 둘째 열을 쌓습니다.",
        "en": "Stack the first column, then the second."
      }
    },
    {
      "id": "factor",
      "question": {
        "ko": "vec(AXBᵀ)의 인자는?",
        "en": "Which factor acts on vec(X) for vec(AXBᵀ)?"
      },
      "options": [
        {
          "ko": "A⊗B",
          "en": "A⊗B"
        },
        {
          "ko": "B⊗A",
          "en": "B⊗A"
        },
        {
          "ko": "A+B",
          "en": "A+B"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "열 우선 규약의 성분 전개로 유도됩니다.",
        "en": "Expand component indices under column-major vectorization."
      }
    }
  ]
},
{
  "id": "limits-continuity-completeness",
  "title": {
    "ko": "극한·연속·실수의 완비성",
    "en": "Limits, Continuity and Completeness"
  },
  "subtitle": {
    "ko": "점의 함수값과 주변에서 접근하는 값을 분리하고 실수의 빈틈 없음이 쓰이는 곳을 찾습니다.",
    "en": "Separate a value at a point from nearby limits and identify where completeness matters."
  },
  "prerequisiteIds": [
    "quantifiers-counterexamples",
    "polynomial-rational-functions"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "limₓ→₁f(x)=f(1) ⇔ continuity at 1",
  "accent": "analysis",
  "questions": [
    {
      "id": "value-limit",
      "question": {
        "ko": "f(1)=0이면 극한도 반드시 0인가요?",
        "en": "Does f(1)=0 force the limit to be zero?"
      },
      "options": [
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "근방의 값이 극한을 결정합니다.",
        "en": "The limit depends on nearby values."
      }
    },
    {
      "id": "complete",
      "question": {
        "ko": "유리수의 빈틈을 채우는 실수의 성질은?",
        "en": "Which real-number property fills rational gaps?"
      },
      "options": [
        {
          "ko": "유한성",
          "en": "Finiteness"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "완비성",
          "en": "Completeness"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "유계 비어 있지 않은 집합의 실수 상한이 존재합니다.",
        "en": "Every nonempty real set bounded above has a real supremum."
      }
    }
  ]
},
{
  "id": "differentiability-mean-value",
  "title": {
    "ko": "미분 가능성과 평균값정리",
    "en": "Differentiability and the Mean Value Theorem"
  },
  "subtitle": {
    "ko": "접선의 존재 조건과 구간 평균 변화율이 한 점에서 실현되는 조건을 구분합니다.",
    "en": "Distinguish the existence of a tangent from the hypotheses ensuring a mean slope is attained."
  },
  "prerequisiteIds": [
    "limits-continuity-completeness"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "f′(c)=(f(b)−f(a))/(b−a)",
  "accent": "analysis",
  "questions": [
    {
      "id": "corner",
      "question": {
        "ko": "|x|는 0에서 미분 가능한가요?",
        "en": "Is |x| differentiable at zero?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "좌우 기울기가 −1과 1입니다.",
        "en": "The one-sided slopes are minus one and one."
      }
    },
    {
      "id": "mvt",
      "question": {
        "ko": "x²의 [−1,1] 평균 기울기와 같은 도함수 위치는?",
        "en": "Where does x² attain its mean slope on [−1,1]?"
      },
      "options": [
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "1",
          "en": "1"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "평균 기울기 0이고 f′(0)=0입니다.",
        "en": "The mean slope is zero and f′(0)=0."
      }
    }
  ]
},
{
  "id": "integration-fundamental-theorem",
  "title": {
    "ko": "적분·누적과 미적분 기본정리",
    "en": "Integration, Accumulation and the Fundamental Theorem"
  },
  "subtitle": {
    "ko": "작은 면적의 합과 누적 함수의 도함수를 연결하고 유한 합의 오차를 계산합니다.",
    "en": "Connect small-area sums to derivatives of accumulation and quantify finite-sum error."
  },
  "prerequisiteIds": [
    "differentiability-mean-value"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "F(x)=∫₀ˣt²dt=x³/3; F′(x)=x²",
  "accent": "analysis",
  "questions": [
    {
      "id": "ftc",
      "question": {
        "ko": "연속 f의 F(x)=∫₀ˣf(t)dt에서 F′는?",
        "en": "For continuous f and F(x)=∫₀ˣf(t)dt, what is F′?"
      },
      "options": [
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "f(x)",
          "en": "f(x)"
        },
        {
          "ko": "F(x)",
          "en": "F(x)"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "연속성 조건의 미적분 기본정리입니다.",
        "en": "This is the fundamental theorem under continuity."
      }
    },
    {
      "id": "sum",
      "question": {
        "ko": "왼쪽 합 N=4는 정확한 적분인가요?",
        "en": "Is the N=4 left sum the exact integral?"
      },
      "options": [
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "유한 근사와 극한을 구분합니다.",
        "en": "Distinguish a finite approximation from its limit."
      }
    }
  ]
},
{
  "id": "taylor-remainders",
  "title": {
    "ko": "Taylor 전개와 나머지",
    "en": "Taylor Expansions and Remainders"
  },
  "subtitle": {
    "ko": "다항 근사와 실제 오차, 오차를 보장하는 상한을 함께 읽습니다.",
    "en": "Read the polynomial approximation, actual error and a justified error bound together."
  },
  "prerequisiteIds": [
    "integration-fundamental-theorem"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "exp(x)=Σₖ₌₀ⁿxᵏ/k!+Rₙ(x)",
  "accent": "analysis",
  "questions": [
    {
      "id": "remainder",
      "question": {
        "ko": "Taylor 다항식만으로 근사 정확도를 보장하나요?",
        "en": "Does writing a Taylor polynomial alone guarantee its accuracy?"
      },
      "options": [
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "나머지와 미분 조건을 확인해야 합니다.",
        "en": "Check the remainder and differentiability assumptions."
      }
    },
    {
      "id": "bound",
      "question": {
        "ko": "나머지 상한은 실제 오차와 같아야 하나요?",
        "en": "Must a remainder bound equal the actual error?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "상한은 더 클 수 있습니다.",
        "en": "A valid bound may be larger."
      }
    }
  ]
},
{
  "id": "series-convergence-tests",
  "title": {
    "ko": "수열·급수의 수렴 판정",
    "en": "Sequences, Series and Convergence Tests"
  },
  "subtitle": {
    "ko": "항이 0으로 가는 것과 부분합이 수렴하는 것은 다른 조건입니다.",
    "en": "A vanishing term does not by itself make the sequence of partial sums converge."
  },
  "prerequisiteIds": [
    "sequences-series",
    "limits-continuity-completeness"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "Σ1/n diverges; Σ(−1)ⁿ⁺¹/n=ln2; Σ1/n²=π²/6",
  "accent": "analysis",
  "questions": [
    {
      "id": "term",
      "question": {
        "ko": "aₙ→0은 급수 수렴의 충분조건인가요?",
        "en": "Is aₙ→0 sufficient for series convergence?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "조화급수가 반례입니다.",
        "en": "The harmonic series is a counterexample."
      }
    },
    {
      "id": "absolute",
      "question": {
        "ko": "교대 조화급수는 절대 수렴하나요?",
        "en": "Is the alternating harmonic series absolutely convergent?"
      },
      "options": [
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "절댓값을 취하면 발산하는 조화급수입니다.",
        "en": "Its absolute-value series is harmonic and divergent."
      }
    }
  ]
},
{
  "id": "pointwise-uniform-convergence",
  "title": {
    "ko": "점별 수렴과 균등 수렴",
    "en": "Pointwise and Uniform Convergence"
  },
  "subtitle": {
    "ko": "점마다 다른 대기 시간과 전체 구간에 공통인 대기 시간을 구분합니다.",
    "en": "Distinguish a waiting time for each point from one waiting time valid across the whole domain."
  },
  "prerequisiteIds": [
    "series-convergence-tests"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "∀x∀ε∃N … versus ∀ε∃N∀x …",
  "accent": "analysis",
  "questions": [
    {
      "id": "uniform",
      "question": {
        "ko": "xⁿ은 [0,1]에서 균등 수렴하나요?",
        "en": "Does xⁿ converge uniformly on [0,1]?"
      },
      "options": [
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "오차 supremum이 1로 유지됩니다.",
        "en": "The error supremum stays one."
      }
    },
    {
      "id": "quantifiers",
      "question": {
        "ko": "균등 수렴의 N은 무엇에 독립적인가요?",
        "en": "In uniform convergence, N is independent of what?"
      },
      "options": [
        {
          "ko": "점 x",
          "en": "The point x"
        },
        {
          "ko": "오차 ε",
          "en": "The tolerance ε"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "ε에는 의존하되 x 전체에 공통입니다.",
        "en": "It may depend on ε, but is shared by all x."
      }
    }
  ]
},
{
  "id": "compactness-extreme-values",
  "title": {
    "ko": "Compactness·연속성과 극값",
    "en": "Compactness, Continuity and Extreme Values"
  },
  "subtitle": {
    "ko": "상한이 존재하는 것과 함수가 그 값을 실제로 얻는 것은 다릅니다.",
    "en": "Having a supremum differs from attaining a maximum."
  },
  "prerequisiteIds": [
    "limits-continuity-completeness"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "continuous f on compact K ⇒ extrema attained",
  "accent": "analysis",
  "questions": [
    {
      "id": "sup-max",
      "question": {
        "ko": "상한은 반드시 최대값인가요?",
        "en": "Must a supremum be a maximum?"
      },
      "options": [
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "최대값은 실제 함수값으로 달성되어야 합니다.",
        "en": "A maximum must be attained."
      }
    },
    {
      "id": "compact",
      "question": {
        "ko": "Rⁿ에서 compact의 동치 조건은?",
        "en": "In Rⁿ, what characterizes compactness?"
      },
      "options": [
        {
          "ko": "열림과 유계",
          "en": "Open and bounded"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "닫힘과 유계",
          "en": "Closed and bounded"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "유한차원 유클리드 공간의 Heine–Borel 정리입니다.",
        "en": "This is the finite-dimensional Heine–Borel theorem."
      }
    }
  ]
},
{
  "id": "convexity-first-order",
  "title": {
    "ko": "볼록성과 일차 지지 부등식",
    "en": "Convexity and First-Order Supporting Inequalities"
  },
  "subtitle": {
    "ko": "접선이 아래에서 함수를 받치는지 보고 한 점 검사와 전역 성질을 구분합니다.",
    "en": "Compare a tangent with the function and separate a probe from a global property."
  },
  "prerequisiteIds": [
    "differentiability-mean-value"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "f(y)≥f(x)+f′(x)(y−x)",
  "accent": "analysis",
  "questions": [
    {
      "id": "curvature",
      "question": {
        "ko": "ax²가 볼록인 조건은?",
        "en": "When is ax² convex?"
      },
      "options": [
        {
          "ko": "a<0",
          "en": "a<0"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "a≥0",
          "en": "a≥0"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "이차 도함수는 2a입니다.",
        "en": "Its second derivative is 2a."
      }
    },
    {
      "id": "global",
      "question": {
        "ko": "한 점쌍의 부등식으로 전역 볼록성을 증명하나요?",
        "en": "Does one pair certify global convexity?"
      },
      "options": [
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "정의역의 모든 점쌍에 대한 주장입니다.",
        "en": "The property quantifies over all pairs in the domain."
      }
    }
  ]
},
{
  "id": "limits-integration-interchange",
  "title": {
    "ko": "극한과 적분의 교환 조건",
    "en": "When Limits and Integrals Commute"
  },
  "subtitle": {
    "ko": "높아지고 좁아지는 봉우리가 점별 극한에서 사라져도 면적은 남을 수 있습니다.",
    "en": "A taller, narrower spike can vanish pointwise while retaining its integral."
  },
  "prerequisiteIds": [
    "integration-fundamental-theorem",
    "pointwise-uniform-convergence"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "lim∫fₙ ≠ ∫lim fₙ without suitable conditions",
  "accent": "analysis",
  "questions": [
    {
      "id": "swap",
      "question": {
        "ko": "점별 수렴만으로 적분을 교환하나요?",
        "en": "Does pointwise convergence alone justify interchanging integrals?"
      },
      "options": [
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "적분 가능한 공통 지배 함수 등 추가 조건이 필요합니다.",
        "en": "Additional conditions such as integrable domination are needed."
      }
    },
    {
      "id": "spike",
      "question": {
        "ko": "봉우리 모형의 lim∫fₙ은?",
        "en": "What is lim∫fₙ for the spike model?"
      },
      "options": [
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "삼각형 면적은 모든 n에서 1입니다.",
        "en": "The triangular area is one for every n."
      }
    }
  ]
},
{
  "id": "implicit-inverse-functions",
  "title": {
    "ko": "암묵함수·역함수 정리",
    "en": "Implicit and Inverse Function Theorems"
  },
  "subtitle": {
    "ko": "해가 존재하는 것과 미분 가능한 국소 해를 얻는 조건을 구분합니다.",
    "en": "Distinguish existence of an inverse or branch from hypotheses giving a differentiable local one."
  },
  "prerequisiteIds": [
    "differentiability-mean-value",
    "jacobians"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "F_y≠0 ⇒ dy/dx=−F_x/F_y",
  "accent": "analysis",
  "questions": [
    {
      "id": "ift",
      "question": {
        "ko": "암묵 미분의 분모 조건은?",
        "en": "What denominator condition supports implicit differentiation?"
      },
      "options": [
        {
          "ko": "F_y≠0",
          "en": "F_y≠0"
        },
        {
          "ko": "F_y=0",
          "en": "F_y=0"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "국소적으로 y를 x의 미분 가능한 함수로 풉니다.",
        "en": "It gives a local differentiable y as a function of x."
      }
    },
    {
      "id": "inverse",
      "question": {
        "ko": "f′(0)=0이면 항상 역함수가 없나요?",
        "en": "Does f′(0)=0 always rule out an inverse?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "x³이 반례입니다.",
        "en": "The cube function is a counterexample."
      }
    }
  ]
},
{
  "id": "matrix-trace-logdet-derivatives",
  "title": {
    "ko": "행렬·trace·logdet 미분",
    "en": "Matrix, Trace and Log-Determinant Derivatives"
  },
  "subtitle": {
    "ko": "행렬 미분의 형상과 양의 정부호 정의역을 명시하고 차분으로 검산합니다.",
    "en": "Specify derivative shapes and the SPD domain, then check with finite differences."
  },
  "prerequisiteIds": [
    "determinants-traces",
    "positive-semidefinite-matrices",
    "chain-rule"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "d log det A=tr(A⁻¹dA)",
  "accent": "analysis",
  "questions": [
    {
      "id": "shape",
      "question": {
        "ko": "행렬 Frobenius 내적에서 logdet의 gradient는?",
        "en": "What is the logdet gradient under the Frobenius inner product?"
      },
      "options": [
        {
          "ko": "A",
          "en": "A"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "A⁻ᵀ",
          "en": "A⁻ᵀ"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "differential=tr(A⁻¹dA)입니다.",
        "en": "Its differential is tr(A⁻¹dA)."
      }
    },
    {
      "id": "shared",
      "question": {
        "ko": "tr(AᵀA)에서 공유 b의 도함수는?",
        "en": "For tr(AᵀA), what is the derivative with respect to shared b?"
      },
      "options": [
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "4b",
          "en": "4b"
        },
        {
          "ko": "2b",
          "en": "2b"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "b² 항이 두 개입니다.",
        "en": "There are two b² terms."
      }
    }
  ]
},
{
  "id": "vector-fields-divergence-curl",
  "title": {
    "ko": "벡터장·발산·회전",
    "en": "Vector Fields, Divergence and Curl"
  },
  "subtitle": {
    "ko": "위치마다 주어진 벡터와 국소 팽창·회전을 구분합니다.",
    "en": "Separate vectors assigned to positions from local expansion and rotation."
  },
  "prerequisiteIds": [
    "jacobians",
    "determinants-traces"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "div F=a+d; curl_z F=c−b",
  "accent": "analysis",
  "questions": [
    {
      "id": "div",
      "question": {
        "ko": "Jacobian의 어떤 값이 발산인가요?",
        "en": "Which Jacobian quantity is divergence?"
      },
      "options": [
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "trace",
          "en": "Trace"
        },
        {
          "ko": "determinant",
          "en": "Determinant"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "대각 편미분을 더합니다.",
        "en": "Sum the diagonal partial derivatives."
      }
    },
    {
      "id": "trajectory",
      "question": {
        "ko": "그림의 화살표 선분은 입자의 궤적인가요?",
        "en": "Are drawn arrow segments particle trajectories?"
      },
      "options": [
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "현재 위치에서의 장 값입니다.",
        "en": "They show the field at each location."
      }
    }
  ]
},
{
  "id": "line-surface-volume-integrals",
  "title": {
    "ko": "선·면·체적 적분의 의미",
    "en": "Line, Surface and Volume Integrals"
  },
  "subtitle": {
    "ko": "같은 곡선에서도 접선 방향의 일과 법선 방향의 유출은 다릅니다.",
    "en": "Work along a tangent and outward flux use the same curve with different directions."
  },
  "prerequisiteIds": [
    "integration-fundamental-theorem",
    "vector-fields-divergence-curl"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "∮F·dr versus ∮F·n ds",
  "accent": "analysis",
  "questions": [
    {
      "id": "orientation",
      "question": {
        "ko": "선적분 방향을 뒤집으면?",
        "en": "What happens to circulation when orientation reverses?"
      },
      "options": [
        {
          "ko": "부호 반전",
          "en": "Sign reversal"
        },
        {
          "ko": "항상 같음",
          "en": "Always unchanged"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "dr의 방향이 바뀝니다.",
        "en": "The direction of dr reverses."
      }
    },
    {
      "id": "measure",
      "question": {
        "ko": "원판 면적 요소는 극좌표에서?",
        "en": "What is the polar area element?"
      },
      "options": [
        {
          "ko": "dr dθ",
          "en": "dr dθ"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "r dr dθ",
          "en": "r dr dθ"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "Jacobian r을 포함합니다.",
        "en": "Include the Jacobian factor r."
      }
    }
  ]
},
{
  "id": "gauss-stokes-theorems",
  "title": {
    "ko": "Gauss·Stokes 정리와 방향",
    "en": "Gauss, Stokes and Orientation"
  },
  "subtitle": {
    "ko": "경계의 적분과 내부의 미분 연산을 연결하되 매끄러움과 방향 조건을 확인합니다.",
    "en": "Connect boundary integrals with interior derivatives while checking smoothness and orientation."
  },
  "prerequisiteIds": [
    "line-surface-volume-integrals"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "∯F·n dS=∭div F dV; ∮F·dr=∬curl F·n dS",
  "accent": "analysis",
  "questions": [
    {
      "id": "gauss",
      "question": {
        "ko": "Gauss 정리의 구면 법선은?",
        "en": "Which sphere normal is used in the divergence theorem?"
      },
      "options": [
        {
          "ko": "임의 혼합 방향",
          "en": "Arbitrarily mixed"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "바깥 방향",
          "en": "Outward"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "닫힌 영역의 경계에 일관된 바깥 법선을 씁니다.",
        "en": "Use the consistent outward boundary normal."
      }
    },
    {
      "id": "singular",
      "question": {
        "ko": "원점에 특이점이 있는 장에도 조건 없이 적용하나요?",
        "en": "Can the theorem be applied without checking a singularity at the origin?"
      },
      "options": [
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "영역에서의 정칙성과 특이점 처리가 필요합니다.",
        "en": "Check regularity on the domain and handle singularities."
      }
    }
  ]
},
{
  "id": "higher-order-autodiff",
  "title": {
    "ko": "고차 미분·Hessian과 방향 곱",
    "en": "Higher Derivatives, Hessians and Directional Products"
  },
  "subtitle": {
    "ko": "Hessian 전체와 특정 방향 곱을 구분하고 이차 근사의 나머지를 확인합니다.",
    "en": "Separate the full Hessian from its action on a direction and inspect a quadratic approximation."
  },
  "prerequisiteIds": [
    "backpropagation",
    "taylor-remainders",
    "quadratic-forms-orders"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 극한·함수·미분·선형사상",
    "en": "Limits, functions, derivatives and linear maps in the linked prerequisites"
  },
  "eyebrow": "ANALYSIS LAB",
  "formula": "H v = D(∇f)[v]; d²f(x+tv)/dt²=vᵀHv",
  "accent": "analysis",
  "questions": [
    {
      "id": "hvp",
      "question": {
        "ko": "Hv의 형상은?",
        "en": "What is the shape of Hv?"
      },
      "options": [
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "입력과 같은 벡터",
          "en": "A vector shaped like the input"
        },
        {
          "ko": "항상 스칼라",
          "en": "Always a scalar"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "스칼라는 vᵀHv입니다.",
        "en": "The scalar is vᵀHv."
      }
    },
    {
      "id": "symmetry",
      "question": {
        "ko": "매끄러운 스칼라 함수의 Hessian은?",
        "en": "What property does the Hessian of a smooth scalar function have?"
      },
      "options": [
        {
          "ko": "대칭",
          "en": "Symmetry"
        },
        {
          "ko": "항상 양의 정부호",
          "en": "Always positive definite"
        },
        {
          "ko": "판정할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "혼합 편미분 교환 조건에서 대칭입니다.",
        "en": "It is symmetric under mixed-partial interchange conditions."
      }
    }
  ]
},
{
  "id": "complex-polar-maps",
  "title": {
    "ko": "복소평면·극형식과 거듭제곱",
    "en": "Complex Polar Form and Power Maps"
  },
  "subtitle": {
    "ko": "거듭제곱은 크기를 거듭제곱하고 각도를 배수로 바꾸지만 주값 각도는 감깁니다.",
    "en": "A power raises the magnitude and multiplies the angle, while the principal argument wraps."
  },
  "prerequisiteIds": [
    "real-complex-numbers",
    "trigonometry-coordinates"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 복소수·미분·적분·선형대수",
    "en": "Complex numbers, differentiation, integration and linear algebra in the linked prerequisites"
  },
  "eyebrow": "COMPLEX LAB",
  "formula": "z=r exp(iθ) ⇒ zᵏ=rᵏ exp(ikθ)",
  "accent": "complex",
  "questions": [
    {
      "id": "power",
      "question": {
        "ko": "r=2,k=2의 출력 크기는?",
        "en": "What is output magnitude for r=2,k=2?"
      },
      "options": [
        {
          "ko": "4",
          "en": "4"
        },
        {
          "ko": "2",
          "en": "2"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "크기는 rᵏ입니다.",
        "en": "Magnitude is rᵏ."
      }
    },
    {
      "id": "argument",
      "question": {
        "ko": "arg(0)은?",
        "en": "What is arg(0)?"
      },
      "options": [
        {
          "ko": "항상 0",
          "en": "Always zero"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "정의되지 않음",
          "en": "Undefined"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "영점은 방향을 정하지 않습니다.",
        "en": "Zero does not select a direction."
      }
    }
  ]
},
{
  "id": "cauchy-riemann-holomorphic",
  "title": {
    "ko": "정칙성과 Cauchy–Riemann 조건",
    "en": "Holomorphicity and Cauchy–Riemann Conditions"
  },
  "subtitle": {
    "ko": "한 점의 미분 가능성과 열린 근방의 정칙성을 구분합니다.",
    "en": "Separate differentiability at one point from holomorphicity on a neighborhood."
  },
  "prerequisiteIds": [
    "complex-polar-maps",
    "differentiability-mean-value"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 복소수·미분·적분·선형대수",
    "en": "Complex numbers, differentiation, integration and linear algebra in the linked prerequisites"
  },
  "eyebrow": "COMPLEX LAB",
  "formula": "uₓ=vᵧ, uᵧ=−vₓ",
  "accent": "complex",
  "questions": [
    {
      "id": "holo",
      "question": {
        "ko": "정칙성은 어떤 범위의 조건인가요?",
        "en": "Where is holomorphicity required?"
      },
      "options": [
        {
          "ko": "한 점만",
          "en": "Only one point"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "열린 근방",
          "en": "An open neighborhood"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "근방의 각 점에서 복소 미분 가능해야 합니다.",
        "en": "Complex differentiability must hold throughout a neighborhood."
      }
    },
    {
      "id": "conj",
      "question": {
        "ko": "켤레 함수는 정칙인가요?",
        "en": "Is conjugation holomorphic?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "두 방향의 몫 극한이 다릅니다.",
        "en": "Its directional quotient limits disagree."
      }
    }
  ]
},
{
  "id": "complex-contour-integrals",
  "title": {
    "ko": "복소 경로 적분과 Cauchy 정리",
    "en": "Complex Contour Integrals and Cauchy’s Theorem"
  },
  "subtitle": {
    "ko": "경로의 매개화·방향·내부 특이점을 함께 확인합니다.",
    "en": "Check parametrization, orientation and interior singularities together."
  },
  "prerequisiteIds": [
    "cauchy-riemann-holomorphic",
    "line-surface-volume-integrals"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 복소수·미분·적분·선형대수",
    "en": "Complex numbers, differentiation, integration and linear algebra in the linked prerequisites"
  },
  "eyebrow": "COMPLEX LAB",
  "formula": "∮z dz=0; ∮dz/z=2πi for one CCW turn",
  "accent": "complex",
  "questions": [
    {
      "id": "dz",
      "question": {
        "ko": "경로 적분에서 필요한 인자는?",
        "en": "What factor is required in a parametrized contour integral?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "z′(t)",
          "en": "z′(t)"
        },
        {
          "ko": "함수값만",
          "en": "Only function values"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "dz=z′dt입니다.",
        "en": "Use dz=z′dt."
      }
    },
    {
      "id": "pole",
      "question": {
        "ko": "원 내부의 1/z에 정칙 함수의 Cauchy 정리를 적용하나요?",
        "en": "Does the holomorphic version of Cauchy’s theorem apply across the pole of 1/z?"
      },
      "options": [
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "내부 정칙성 가정이 깨집니다.",
        "en": "Interior holomorphicity fails."
      }
    }
  ]
},
{
  "id": "residues-winding",
  "title": {
    "ko": "유수·감김 수와 특이점",
    "en": "Residues, Winding and Poles"
  },
  "subtitle": {
    "ko": "극의 차수와 유수는 다르며 경로 위의 극은 적분을 정의하지 못하게 합니다.",
    "en": "Pole order differs from residue, and a pole on the contour invalidates the ordinary integral."
  },
  "prerequisiteIds": [
    "complex-contour-integrals"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 복소수·미분·적분·선형대수",
    "en": "Complex numbers, differentiation, integration and linear algebra in the linked prerequisites"
  },
  "eyebrow": "COMPLEX LAB",
  "formula": "∮f dz=2πi Σ winding·residue",
  "accent": "complex",
  "questions": [
    {
      "id": "residue",
      "question": {
        "ko": "1/(z−a)²의 유수는?",
        "en": "What is the residue of 1/(z−a)²?"
      },
      "options": [
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "2",
          "en": "2"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "−1차 항이 없습니다.",
        "en": "There is no power-minus-one term."
      }
    },
    {
      "id": "boundary",
      "question": {
        "ko": "극이 경로 위에 있으면?",
        "en": "What if the pole lies on the contour?"
      },
      "options": [
        {
          "ko": "항상 2πi",
          "en": "Always 2πi"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "보통 적분 정의 불가",
          "en": "Ordinary integral undefined"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "별도의 우회·주값 규약 없이 정의하지 않습니다.",
        "en": "No ordinary value exists without a separate prescription."
      }
    }
  ]
},
{
  "id": "analytic-functions-singularities",
  "title": {
    "ko": "해석 함수와 고립 특이점",
    "en": "Analytic Functions and Isolated Singularities"
  },
  "subtitle": {
    "ko": "제거 가능·극·본질적 특이점은 서로 다른 Laurent 구조를 가집니다.",
    "en": "Removable singularities, poles and essential singularities have different Laurent structures."
  },
  "prerequisiteIds": [
    "residues-winding",
    "taylor-remainders"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 복소수·미분·적분·선형대수",
    "en": "Complex numbers, differentiation, integration and linear algebra in the linked prerequisites"
  },
  "eyebrow": "COMPLEX LAB",
  "formula": "sin z/z →1; 1/z has a pole; exp(1/z) is essential",
  "accent": "complex",
  "questions": [
    {
      "id": "essential",
      "question": {
        "ko": "exp(1/z)의 0 특이점은?",
        "en": "What type is the singularity of exp(1/z) at zero?"
      },
      "options": [
        {
          "ko": "제거 가능",
          "en": "Removable"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "본질적",
          "en": "Essential"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "음수 차수 Laurent 항이 무한합니다.",
        "en": "There are infinitely many negative Laurent powers."
      }
    },
    {
      "id": "overflow",
      "question": {
        "ko": "유한한 수학 값의 기계 범위 초과는?",
        "en": "How should a finite value exceeding machine range be interpreted?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "표현 범위 문제",
          "en": "A representation limit"
        },
        {
          "ko": "수학적 무한으로 확정",
          "en": "Proof of mathematical infinity"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "정의역 미정의와 다릅니다.",
        "en": "This differs from domain failure."
      }
    }
  ]
},
{
  "id": "analytic-continuation-branches",
  "title": {
    "ko": "해석적 연속·로그와 가지",
    "en": "Analytic Continuation, Logarithms and Branches"
  },
  "subtitle": {
    "ko": "같은 종점이라도 따라온 경로의 감김 수에 따라 로그와 제곱근의 연속값이 달라집니다.",
    "en": "The same endpoint can have different continued logarithm and square-root values after different windings."
  },
  "prerequisiteIds": [
    "analytic-functions-singularities"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 복소수·미분·적분·선형대수",
    "en": "Complex numbers, differentiation, integration and linear algebra in the linked prerequisites"
  },
  "eyebrow": "COMPLEX LAB",
  "formula": "Log z=ln r+iθ; continued log=ln r+i(θ+2πk)",
  "accent": "complex",
  "questions": [
    {
      "id": "branch",
      "question": {
        "ko": "원점 주위를 한 바퀴 돌면 연속 제곱근은?",
        "en": "After one turn around zero, the continued square root does what?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "부호 반전",
          "en": "Changes sign"
        },
        {
          "ko": "항상 원래 값",
          "en": "Always returns unchanged"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "누적 각도의 절반이 π만큼 변합니다.",
        "en": "Half the lifted angle changes by π."
      }
    },
    {
      "id": "global",
      "question": {
        "ko": "C\\{0} 전체에 연속 단일값 로그가 있나요?",
        "en": "Is there a continuous single-valued logarithm on all of C\\{0}?"
      },
      "options": [
        {
          "ko": "없음",
          "en": "No"
        },
        {
          "ko": "있음",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "닫힌 한 바퀴에서 로그 변화가 2πi입니다.",
        "en": "A closed turn changes the continued log by 2πi."
      }
    }
  ]
},
{
  "id": "complex-linear-algebra",
  "title": {
    "ko": "복소 선형대수·켤레 내적",
    "en": "Complex Linear Algebra and Hermitian Products"
  },
  "subtitle": {
    "ko": "복소 벡터의 길이는 켤레전치로 정의하며 전치만 쓰면 양성이 사라집니다.",
    "en": "Use conjugate transpose for lengths; plain transpose loses positivity."
  },
  "prerequisiteIds": [
    "complex-polar-maps",
    "inner-products"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 복소수·미분·적분·선형대수",
    "en": "Complex numbers, differentiation, integration and linear algebra in the linked prerequisites"
  },
  "eyebrow": "COMPLEX LAB",
  "formula": "⟨v,w⟩=v* w; ‖v‖²=v* v",
  "accent": "complex",
  "questions": [
    {
      "id": "conjugate",
      "question": {
        "ko": "복소 길이 제곱에 필요한 것은?",
        "en": "What defines squared complex-vector norm?"
      },
      "options": [
        {
          "ko": "켤레전치",
          "en": "Conjugate transpose"
        },
        {
          "ko": "전치만",
          "en": "Plain transpose"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "양성을 보장합니다.",
        "en": "It preserves positivity."
      }
    },
    {
      "id": "zero-norm",
      "question": {
        "ko": "v=(1,i)의 vᵀv=0이면 v=0인가요?",
        "en": "Does vᵀv=0 for v=(1,i) imply v=0?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "vᵀv는 Hermitian 노름이 아닙니다.",
        "en": "The bilinear form is not the Hermitian norm."
      }
    }
  ]
},
{
  "id": "wirtinger-gradients",
  "title": {
    "ko": "Wirtinger 미분과 복소 손실 최적화",
    "en": "Wirtinger Derivatives and Complex Loss Optimization"
  },
  "subtitle": {
    "ko": "실수값 목적함수는 보통 정칙이 아니어도 두 실수 변수로 미분하고 최적화할 수 있습니다.",
    "en": "A real-valued objective can be optimized through two real variables without being holomorphic."
  },
  "prerequisiteIds": [
    "cauchy-riemann-holomorphic",
    "gradients"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 복소수·미분·적분·선형대수",
    "en": "Complex numbers, differentiation, integration and linear algebra in the linked prerequisites"
  },
  "eyebrow": "COMPLEX LAB",
  "formula": "∂z=½(∂x−i∂y); ∂z̄=½(∂x+i∂y)",
  "accent": "complex",
  "questions": [
    {
      "id": "gradient",
      "question": {
        "ko": "유클리드 gradient의 복소 표현은?",
        "en": "What is the complex encoding of the Euclidean gradient?"
      },
      "options": [
        {
          "ko": "∂L/∂z만",
          "en": "Only ∂L/∂z"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "2∂L/∂z̄",
          "en": "2∂L/∂z̄"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "여기 정의한 1/2 규약에서 계수 2가 필요합니다.",
        "en": "The stated half-factor convention needs the factor two."
      }
    },
    {
      "id": "holo-loss",
      "question": {
        "ko": "실수값 손실이 정칙이 아니면 최적화 불가인가요?",
        "en": "Does nonholomorphicity prevent optimizing a real loss?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "R² 미분을 사용합니다.",
        "en": "Use differentiation on R²."
      }
    }
  ]
},
{
  "id": "complex-signals-phase",
  "title": {
    "ko": "복소 신호·위상과 간섭",
    "en": "Complex Signals, Phase and Interference"
  },
  "subtitle": {
    "ko": "위상 차이는 같은 크기의 신호를 강화하거나 상쇄하며 영 진폭에서는 위상을 정의하지 않습니다.",
    "en": "Phase differences reinforce or cancel signals; zero amplitude has no phase."
  },
  "prerequisiteIds": [
    "complex-polar-maps",
    "trigonometry-coordinates"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 복소수·미분·적분·선형대수",
    "en": "Complex numbers, differentiation, integration and linear algebra in the linked prerequisites"
  },
  "eyebrow": "COMPLEX LAB",
  "formula": "A exp(iφ)=exp(iφ₁)+A₂exp(iφ₂)",
  "accent": "complex",
  "questions": [
    {
      "id": "phase",
      "question": {
        "ko": "진폭 0의 위상은?",
        "en": "What is the phase at zero amplitude?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "정의되지 않음",
          "en": "Undefined"
        },
        {
          "ko": "항상 0",
          "en": "Always zero"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "방향이 없습니다.",
        "en": "There is no direction."
      }
    },
    {
      "id": "amplitude",
      "question": {
        "ko": "동위상 단위 신호 둘의 합 진폭은?",
        "en": "What is the amplitude of two in-phase unit signals?"
      },
      "options": [
        {
          "ko": "2",
          "en": "2"
        },
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "복소 진폭을 먼저 더합니다.",
        "en": "Add complex amplitudes first."
      }
    }
  ]
},
{
  "id": "tensor-order-shapes",
  "title": {
    "ko": "텐서의 차수·축·shape",
    "en": "Tensor Order, Axes and Shapes"
  },
  "subtitle": {
    "ko": "배열 차수와 행렬 랭크를 구분하고 축 순서를 바꾸면 성분 대응이 어떻게 달라지는지 확인합니다.",
    "en": "Separate array order from matrix rank and track entries when axes are permuted."
  },
  "prerequisiteIds": [
    "vector-spaces-rank",
    "kronecker-products"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기저·선형사상·내적·분해",
    "en": "Bases, linear maps, inner products and decompositions in the linked prerequisites"
  },
  "eyebrow": "TENSOR LAB",
  "formula": "X[b,t,c] → X̃[b,c,t]",
  "accent": "tensor",
  "questions": [
    {
      "id": "order",
      "question": {
        "ko": "shape (1,2,3)의 배열 차수는?",
        "en": "What is array order for shape (1,2,3)?"
      },
      "options": [
        {
          "ko": "3",
          "en": "3"
        },
        {
          "ko": "6",
          "en": "6"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "축 개수입니다.",
        "en": "Count axes."
      }
    },
    {
      "id": "reshape",
      "question": {
        "ko": "reshape와 축 순열은 항상 같은가요?",
        "en": "Are reshape and axis permutation always identical?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "좌표·저장 순서의 의미가 다릅니다.",
        "en": "Their coordinate and storage semantics differ."
      }
    }
  ]
},
{
  "id": "tensor-indices-contractions",
  "title": {
    "ko": "지표·자유축과 수축",
    "en": "Indices, Free Axes and Contractions"
  },
  "subtitle": {
    "ko": "합산할 지표와 남길 지표를 먼저 정한 뒤 출력 shape와 계산값을 확인합니다.",
    "en": "Name summed and free indices before checking output shape and values."
  },
  "prerequisiteIds": [
    "tensor-order-shapes",
    "inner-products"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기저·선형사상·내적·분해",
    "en": "Bases, linear maps, inner products and decompositions in the linked prerequisites"
  },
  "eyebrow": "TENSOR LAB",
  "formula": "Yᵢⱼ=ΣₖXᵢⱼₖvₖ",
  "accent": "tensor",
  "questions": [
    {
      "id": "free",
      "question": {
        "ko": "Yᵢⱼ=ΣₖXᵢⱼₖvₖ의 자유 지표는?",
        "en": "Which indices are free in Yᵢⱼ=ΣₖXᵢⱼₖvₖ?"
      },
      "options": [
        {
          "ko": "k",
          "en": "k"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "i,j",
          "en": "i,j"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "합산하지 않은 지표입니다.",
        "en": "The unsummed indices remain."
      }
    },
    {
      "id": "dot",
      "question": {
        "ko": "수축과 원소별 곱은 같은가요?",
        "en": "Is contraction the same as elementwise multiplication?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "수축은 지정 축의 합산을 포함합니다.",
        "en": "Contraction includes summing designated axes."
      }
    }
  ]
},
{
  "id": "multilinear-maps",
  "title": {
    "ko": "다중선형사상과 텐서 계수",
    "en": "Multilinear Maps and Tensor Coefficients"
  },
  "subtitle": {
    "ko": "각 입력을 따로 고정했을 때의 선형성과 모든 입력을 함께 묶은 선형성을 구분합니다.",
    "en": "Separate linearity in each slot from joint linearity in concatenated inputs."
  },
  "prerequisiteIds": [
    "tensor-indices-contractions"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기저·선형사상·내적·분해",
    "en": "Bases, linear maps, inner products and decompositions in the linked prerequisites"
  },
  "eyebrow": "TENSOR LAB",
  "formula": "F(u,v)ₖ=ΣᵢⱼXᵢⱼₖuᵢvⱼ",
  "accent": "tensor",
  "questions": [
    {
      "id": "scale",
      "question": {
        "ko": "두 슬롯을 동시에 두 배로 하면?",
        "en": "What happens when both slots are doubled?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "네 배",
          "en": "Four times"
        },
        {
          "ko": "두 배",
          "en": "Twice"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "각 슬롯의 선형성이 한 번씩 적용됩니다.",
        "en": "Linearity applies once per slot."
      }
    },
    {
      "id": "zero",
      "question": {
        "ko": "다중선형사상의 한 슬롯이 0이면?",
        "en": "What if one slot of a multilinear map is zero?"
      },
      "options": [
        {
          "ko": "출력 0",
          "en": "Zero output"
        },
        {
          "ko": "항상 출력 1",
          "en": "Always one"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "그 슬롯의 선형성 때문입니다.",
        "en": "By linearity in that slot."
      }
    }
  ]
},
{
  "id": "broadcasting-semantics",
  "title": {
    "ko": "Broadcasting과 연산의 의미",
    "en": "Broadcasting and Operation Semantics"
  },
  "subtitle": {
    "ko": "shape 호환성과 계산의 뜻을 따로 확인하고 원소별 곱을 행렬곱으로 읽지 않습니다.",
    "en": "Check both shape compatibility and operation meaning; elementwise multiplication is not matrix multiplication."
  },
  "prerequisiteIds": [
    "tensor-order-shapes"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기저·선형사상·내적·분해",
    "en": "Bases, linear maps, inner products and decompositions in the linked prerequisites"
  },
  "eyebrow": "TENSOR LAB",
  "formula": "trailing axes compatible ⇔ equal or one axis is 1",
  "accent": "tensor",
  "questions": [
    {
      "id": "alignment",
      "question": {
        "ko": "Broadcasting은 어느 쪽 축부터 맞추나요?",
        "en": "Which axes align first in broadcasting?"
      },
      "options": [
        {
          "ko": "오른쪽 끝",
          "en": "Trailing/rightmost"
        },
        {
          "ko": "왼쪽 끝",
          "en": "Leading/leftmost"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "누락된 선두 축은 1로 봅니다.",
        "en": "Missing leading axes behave as one."
      }
    },
    {
      "id": "invalid",
      "question": {
        "ko": "호환 불가이면 출력은?",
        "en": "What is the output for incompatible shapes?"
      },
      "options": [
        {
          "ko": "항상 영 텐서",
          "en": "Always a zero tensor"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "정의되지 않음",
          "en": "No defined output"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "shape 오류를 정상 값으로 숨기지 않습니다.",
        "en": "A shape error is not a numeric tensor value."
      }
    }
  ]
},
{
  "id": "tensor-cp-tucker",
  "title": {
    "ko": "CP·Tucker 분해와 복원",
    "en": "CP and Tucker Decompositions"
  },
  "subtitle": {
    "ko": "rank-one 항의 합과 작은 core의 다중선형 변환을 실제 원소별로 복원합니다.",
    "en": "Reconstruct entries from rank-one sums and from a core with mode factors."
  },
  "prerequisiteIds": [
    "multilinear-maps",
    "svd"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기저·선형사상·내적·분해",
    "en": "Bases, linear maps, inner products and decompositions in the linked prerequisites"
  },
  "eyebrow": "TENSOR LAB",
  "formula": "X≈Σᵣλᵣaᵣ⊗bᵣ⊗cᵣ = G ×₀U₀ ×₁U₁ ×₂U₂",
  "accent": "tensor",
  "questions": [
    {
      "id": "cp",
      "question": {
        "ko": "CP는 무엇의 합인가요?",
        "en": "CP is a sum of what?"
      },
      "options": [
        {
          "ko": "항상 정방행렬",
          "en": "Always square matrices"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "Rank-one 바깥 곱",
          "en": "Rank-one outer products"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "각 항은 축별 벡터의 바깥 곱입니다.",
        "en": "Each term is an outer product of mode vectors."
      }
    },
    {
      "id": "storage",
      "question": {
        "ko": "분해하면 항상 저장량이 줄어드나요?",
        "en": "Does decomposition always reduce storage?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "차원과 선택 랭크에 따라 다릅니다.",
        "en": "It depends on dimensions and chosen ranks."
      }
    }
  ]
},
{
  "id": "tensor-trains",
  "title": {
    "ko": "Tensor Train과 bond 차원",
    "en": "Tensor Trains and Bond Dimensions"
  },
  "subtitle": {
    "ko": "연결된 작은 core를 합산해 텐서를 복원하고 gauge 변화와 실제 랭크를 구분합니다.",
    "en": "Contract linked cores and separate gauge choices from actual unfolding rank."
  },
  "prerequisiteIds": [
    "tensor-cp-tucker"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기저·선형사상·내적·분해",
    "en": "Bases, linear maps, inner products and decompositions in the linked prerequisites"
  },
  "eyebrow": "TENSOR LAB",
  "formula": "Xᵢⱼₖ=ΣₐᵦG₀[1,i,a]G₁[a,j,b]G₂[b,k,1]",
  "accent": "tensor",
  "questions": [
    {
      "id": "bond",
      "question": {
        "ko": "최소 TT bond는 어떤 랭크와 관련 있나요?",
        "en": "Minimal TT bonds relate to which ranks?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "순서 있는 unfolding 랭크",
          "en": "Ordered unfolding ranks"
        },
        {
          "ko": "배열 차수만",
          "en": "Only array order"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "축 순서에 따른 행렬화의 랭크입니다.",
        "en": "They are ranks of ordered matricizations."
      }
    },
    {
      "id": "gauge",
      "question": {
        "ko": "Gauge core가 달라지면 표현 텐서가 반드시 달라지나요?",
        "en": "Must changing gauge cores change the represented tensor?"
      },
      "options": [
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "가역 변환과 역변환을 이웃 bond에 넣을 수 있습니다.",
        "en": "Insert an invertible transform and its inverse on a bond."
      }
    }
  ]
},
{
  "id": "tensor-networks-contraction",
  "title": {
    "ko": "Tensor Network와 수축 순서",
    "en": "Tensor Networks and Contraction Order"
  },
  "subtitle": {
    "ko": "같은 최종 값도 중간 텐서의 크기와 곱셈 수는 달라질 수 있습니다.",
    "en": "The same final value can require different intermediate sizes and multiplication counts."
  },
  "prerequisiteIds": [
    "tensor-trains",
    "tensor-indices-contractions"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기저·선형사상·내적·분해",
    "en": "Bases, linear maps, inner products and decompositions in the linked prerequisites"
  },
  "eyebrow": "TENSOR LAB",
  "formula": "(AB)C=A(BC); costs need not agree",
  "accent": "tensor",
  "questions": [
    {
      "id": "order",
      "question": {
        "ko": "수축 순서가 바뀌면 이 정수 예제의 결과는?",
        "en": "What changes in the exact result when contraction order changes here?"
      },
      "options": [
        {
          "ko": "같은 결과",
          "en": "Same result"
        },
        {
          "ko": "항상 다른 결과",
          "en": "Always a different result"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "행렬곱 결합법칙입니다.",
        "en": "Matrix multiplication is associative."
      }
    },
    {
      "id": "runtime",
      "question": {
        "ko": "곱셈 수만으로 실제 속도를 확정하나요?",
        "en": "Do multiplication counts determine measured speed?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "메모리·구현·하드웨어 비용도 있습니다.",
        "en": "Memory, implementation and hardware matter."
      }
    }
  ]
},
{
  "id": "tensor-low-rank-approximation",
  "title": {
    "ko": "텐서 저랭크 근사와 오차",
    "en": "Tensor Low-Rank Approximation and Error"
  },
  "subtitle": {
    "ko": "명시적 직교 예제의 최적 절단과 일반 텐서 근사의 어려움을 구분합니다.",
    "en": "Separate optimal truncation in an explicit orthogonal example from general tensor-approximation difficulties."
  },
  "prerequisiteIds": [
    "tensor-cp-tucker",
    "svd"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기저·선형사상·내적·분해",
    "en": "Bases, linear maps, inner products and decompositions in the linked prerequisites"
  },
  "eyebrow": "TENSOR LAB",
  "formula": "‖X−X̂‖F²=Σ omitted |λ|² for this orthogonal example",
  "accent": "tensor",
  "questions": [
    {
      "id": "optimal",
      "question": {
        "ko": "행렬 SVD 최적성이 모든 CP 근사에 그대로 적용되나요?",
        "en": "Does matrix SVD optimality transfer to every CP tensor approximation?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "분해 제약과 닫힘 성질이 다릅니다.",
        "en": "Decomposition constraints and closure properties differ."
      }
    },
    {
      "id": "zero",
      "question": {
        "ko": "영 텐서의 상대 에너지 비율은?",
        "en": "What is a zero tensor’s relative energy fraction?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "정의되지 않음",
          "en": "Undefined"
        },
        {
          "ko": "항상 1",
          "en": "Always one"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "분모가 0입니다.",
        "en": "The denominator is zero."
      }
    }
  ]
},
{
  "id": "tensor-symmetry-antisymmetry",
  "title": {
    "ko": "대칭·반대칭 텐서와 분해",
    "en": "Symmetric and Antisymmetric Tensors"
  },
  "subtitle": {
    "ko": "지표 교환의 부호와 직교 분해를 실제 원소로 확인합니다.",
    "en": "Inspect index-swap signs and orthogonal decomposition entry by entry."
  },
  "prerequisiteIds": [
    "tensor-order-shapes",
    "inner-products"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기저·선형사상·내적·분해",
    "en": "Bases, linear maps, inner products and decompositions in the linked prerequisites"
  },
  "eyebrow": "TENSOR LAB",
  "formula": "A=½(A+Aᵀ)+½(A−Aᵀ)=S+K",
  "accent": "tensor",
  "questions": [
    {
      "id": "skew",
      "question": {
        "ko": "실수 반대칭 K의 vᵀKv는?",
        "en": "For real skew-symmetric K, what is vᵀKv?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "항상 양수",
          "en": "Always positive"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "전치하면 자신의 음수인 스칼라입니다.",
        "en": "Transposition makes the scalar its own negative."
      }
    },
    {
      "id": "complex",
      "question": {
        "ko": "복소 Hermitian 분해에서 쓰는 것은?",
        "en": "What is used in a complex Hermitian decomposition?"
      },
      "options": [
        {
          "ko": "켤레전치",
          "en": "Conjugate transpose"
        },
        {
          "ko": "전치만",
          "en": "Plain transpose"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "대칭/반대칭과 Hermitian/skew-Hermitian은 구분됩니다.",
        "en": "Distinguish symmetric/skew from Hermitian/skew-Hermitian."
      }
    }
  ]
},
{
  "id": "tensor-storage-cost",
  "title": {
    "ko": "텐서 저장량·계산 복잡도",
    "en": "Tensor Storage and Computational Complexity"
  },
  "subtitle": {
    "ko": "같은 랭크 기호를 써도 표현 형식의 제약과 저장량은 다르며 실제 시간 측정과도 다릅니다.",
    "en": "The same rank symbol means different constraints across formats; storage counts are not runtime measurements."
  },
  "prerequisiteIds": [
    "tensor-trains",
    "tensor-networks-contraction",
    "tensor-cp-tucker"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기저·선형사상·내적·분해",
    "en": "Bases, linear maps, inner products and decompositions in the linked prerequisites"
  },
  "eyebrow": "TENSOR LAB",
  "formula": "dense nᵈ; TT 2nr+(d−2)nr²",
  "accent": "tensor",
  "questions": [
    {
      "id": "tt-count",
      "question": {
        "ko": "d=6,n=4,r=2의 TT 원소 수는?",
        "en": "What is TT scalar count for d=6,n=4,r=2?"
      },
      "options": [
        {
          "ko": "80",
          "en": "80"
        },
        {
          "ko": "4096",
          "en": "4096"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "2nr+(d−2)nr²입니다.",
        "en": "Use 2nr+(d−2)nr²."
      }
    },
    {
      "id": "rank",
      "question": {
        "ko": "CP·Tucker·TT의 같은 r은 같은 표현력을 뜻하나요?",
        "en": "Does the same r in CP, Tucker and TT mean equal expressiveness?"
      },
      "options": [
        {
          "ko": "그렇다",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "각 형식의 제약이 다릅니다.",
        "en": "The formats impose different constraints."
      }
    }
  ]
},
{
  "id": "groups-rings-fields",
  "title": {
    "ko": "군·환·체와 모듈러 연산",
    "en": "Groups, Rings, Fields and Modular Arithmetic"
  },
  "subtitle": {
    "ko": "덧셈 역원과 곱셈 역원을 구분하고 체가 되는 조건을 실제 표에서 확인합니다.",
    "en": "Distinguish additive and multiplicative inverses and check when residues form a field."
  },
  "prerequisiteIds": [
    "sets-functions",
    "numbers-ratios-units"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·모듈러 연산·선형사상",
    "en": "Sets, functions, modular arithmetic and linear maps in the linked prerequisites"
  },
  "eyebrow": "ABSTRACT ALGEBRA LAB",
  "formula": "a+(−a)=0",
  "accent": "abstract-algebra",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "법 5에서 2의 역원은?",
        "en": "What is the inverse of two modulo five?"
      },
      "options": [
        {
          "ko": "3",
          "en": "3"
        },
        {
          "ko": "2",
          "en": "2"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "2·3≡1 mod 5입니다.",
        "en": "Two times three reduces to one modulo five."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "법 6의 잉여류는 체인가요?",
        "en": "Do residues modulo six form a field?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "2·3≡0 mod 6인 비영 영인자가 있습니다.",
        "en": "Two times three is zero modulo six, with both factors nonzero."
      }
    }
  ]
},
{
  "id": "homomorphisms-isomorphisms",
  "title": {
    "ko": "준동형·동형과 구조 보존",
    "en": "Homomorphisms, Isomorphisms and Structure"
  },
  "subtitle": {
    "ko": "공식이 몫집합 위에서 잘 정의되는지 확인한 뒤 핵·상·동형을 판단합니다.",
    "en": "Check representative independence before discussing kernels, images or isomorphisms."
  },
  "prerequisiteIds": [
    "groups-rings-fields"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·모듈러 연산·선형사상",
    "en": "Sets, functions, modular arithmetic and linear maps in the linked prerequisites"
  },
  "eyebrow": "ABSTRACT ALGEBRA LAB",
  "formula": "f([x])=[kx]",
  "accent": "abstract-algebra",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "준동형을 논하기 전 필요한 것은?",
        "en": "What must precede a homomorphism claim?"
      },
      "options": [
        {
          "ko": "값 몇 개만 계산",
          "en": "Only a few values"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "잘 정의된 사상",
          "en": "A well-defined map"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "동치인 대표원의 출력이 같아야 합니다.",
        "en": "Equivalent representatives must yield the same output."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "Z5에서 두 배 사상은?",
        "en": "What is doubling on Z5 as an additive group?"
      },
      "options": [
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "동형",
          "en": "An isomorphism"
        },
        {
          "ko": "항상 영 사상",
          "en": "Always zero"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "2는 법 5에서 단원이므로 덧셈군의 전단사 준동형입니다.",
        "en": "Two is a unit modulo five, so doubling is a bijective additive homomorphism."
      }
    }
  ]
},
{
  "id": "subgroups-quotients",
  "title": {
    "ko": "부분군·잉여류와 몫구조",
    "en": "Subgroups, Cosets and Quotients"
  },
  "subtitle": {
    "ko": "원소를 부분군만큼 다른 동치류로 묶고 몫 연산이 대표원과 무관한지 확인합니다.",
    "en": "Group elements into cosets and establish representative-independent quotient operations."
  },
  "prerequisiteIds": [
    "homomorphisms-isomorphisms"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·모듈러 연산·선형사상",
    "en": "Sets, functions, modular arithmetic and linear maps in the linked prerequisites"
  },
  "eyebrow": "ABSTRACT ALGEBRA LAB",
  "formula": "⟨2⟩={0,2,4} in Z6",
  "accent": "abstract-algebra",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "Z6/⟨2⟩의 크기는?",
        "en": "What is the size of Z6/⟨2⟩?"
      },
      "options": [
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "2",
          "en": "2"
        },
        {
          "ko": "3",
          "en": "3"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "부분군은 세 원소이므로 몫은 6/3=2개입니다.",
        "en": "The subgroup has three elements, leaving two cosets."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "일반 군의 몫군에 필요한 조건은?",
        "en": "What condition is needed for a quotient group?"
      },
      "options": [
        {
          "ko": "정규부분군",
          "en": "A normal subgroup"
        },
        {
          "ko": "임의의 부분집합",
          "en": "Any subset"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "정규성이 대표원과 무관한 몫군 연산을 보장합니다.",
        "en": "Normality ensures representative-independent quotient-group multiplication."
      }
    }
  ]
},
{
  "id": "polynomials-finite-fields",
  "title": {
    "ko": "다항식 몫환과 유한체",
    "en": "Polynomial Quotients and Finite Fields"
  },
  "subtitle": {
    "ko": "다항식 계수를 비트로 표현하고 기약 다항식으로 나눌 때 체가 되는 이유를 확인합니다.",
    "en": "Encode polynomial coefficients as bits and contrast irreducible and reducible quotients."
  },
  "prerequisiteIds": [
    "groups-rings-fields",
    "subgroups-quotients"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·모듈러 연산·선형사상",
    "en": "Sets, functions, modular arithmetic and linear maps in the linked prerequisites"
  },
  "eyebrow": "ABSTRACT ALGEBRA LAB",
  "formula": "2=t, 4=t²",
  "accent": "abstract-algebra",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "F8의 덧셈은 비트 표현에서?",
        "en": "How is addition encoded in F8?"
      },
      "options": [
        {
          "ko": "XOR",
          "en": "XOR"
        },
        {
          "ko": "정수 덧셈 mod 8",
          "en": "Integer addition mod 8"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "F2에서 같은 항은 상쇄됩니다.",
        "en": "Matching terms cancel over F2."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "기약 다항식으로 나눈 몫은?",
        "en": "What is a quotient by an irreducible polynomial over a field?"
      },
      "options": [
        {
          "ko": "항상 정수환",
          "en": "Always the integers"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "체",
          "en": "A field"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "기약성이 모든 비영 잉여류의 곱셈 역원을 보장합니다.",
        "en": "Irreducibility guarantees inverses of all nonzero residue classes."
      }
    }
  ]
},
{
  "id": "modules-linear-actions",
  "title": {
    "ko": "모듈과 환 위의 선형사상",
    "en": "Modules and Linear Maps over Rings"
  },
  "subtitle": {
    "ko": "체 위 벡터공간의 성질을 일반 환에 옮길 때 어떤 나눗셈이 실패하는지 살펴봅니다.",
    "en": "See which vector-space arguments fail when scalars form a ring rather than a field."
  },
  "prerequisiteIds": [
    "groups-rings-fields",
    "vector-spaces-rank"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·모듈러 연산·선형사상",
    "en": "Sets, functions, modular arithmetic and linear maps in the linked prerequisites"
  },
  "eyebrow": "ABSTRACT ALGEBRA LAB",
  "formula": "M=(Z6)²",
  "accent": "abstract-algebra",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "Z6에서 2x=0의 비영 해는?",
        "en": "What nonzero x solves 2x=0 modulo six?"
      },
      "options": [
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "3",
          "en": "3"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "2·3=6≡0입니다.",
        "en": "Two times three is six, which reduces to zero."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "모든 환 위 모듈이 기저를 가지나요?",
        "en": "Does every module over a ring have a basis?"
      },
      "options": [
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "예",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "일반 모듈은 자유 모듈일 필요가 없습니다.",
        "en": "A general module need not be free."
      }
    }
  ]
},
{
  "id": "algebraic-symmetries",
  "title": {
    "ko": "대수적 대칭과 군 작용",
    "en": "Algebraic Symmetries and Group Actions"
  },
  "subtitle": {
    "ko": "회전·반사 합성의 순서를 정하고 궤도와 안정자를 계산합니다.",
    "en": "Fix composition order for rotations/reflections and compute orbits and stabilizers."
  },
  "prerequisiteIds": [
    "groups-rings-fields",
    "linear-transforms"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·모듈러 연산·선형사상",
    "en": "Sets, functions, modular arithmetic and linear maps in the linked prerequisites"
  },
  "eyebrow": "ABSTRACT ALGEBRA LAB",
  "formula": "R(x,y)=(-y,x), S(x,y)=(x,-y)",
  "accent": "abstract-algebra",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "중심의 안정자 크기는?",
        "en": "What is the center stabilizer size?"
      },
      "options": [
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "8",
          "en": "8"
        },
        {
          "ko": "1",
          "en": "1"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "여덟 대칭 모두 중심을 고정합니다.",
        "en": "All eight symmetries fix the center."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "회전과 반사는 항상 교환하나요?",
        "en": "Do rotations and reflections always commute?"
      },
      "options": [
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "SR과 RS가 같은 점을 서로 다른 위치로 보낼 수 있습니다.",
        "en": "SR and RS can send the same point to different locations."
      }
    }
  ]
},
{
  "id": "polynomial-systems",
  "title": {
    "ko": "유한체 위 다항식 시스템",
    "en": "Polynomial Systems over Finite Fields"
  },
  "subtitle": {
    "ko": "계수체를 먼저 고르고 유한 격자의 모든 해와 잔차를 확인합니다.",
    "en": "Choose the coefficient field first, then enumerate solutions and residuals on a finite grid."
  },
  "prerequisiteIds": [
    "polynomials-finite-fields"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·모듈러 연산·선형사상",
    "en": "Sets, functions, modular arithmetic and linear maps in the linked prerequisites"
  },
  "eyebrow": "ABSTRACT ALGEBRA LAB",
  "formula": "x+y=3, xy=2 over F5",
  "accent": "abstract-algebra",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "계수체를 바꾸면 해가 달라질 수 있나요?",
        "en": "Can changing the coefficient field change solutions?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "근의 존재는 계수체에 의존합니다.",
        "en": "Existence of roots depends on the coefficient field."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "작은 격자 열거가 일반 소거 알고리즘인가요?",
        "en": "Is a small grid enumeration a general elimination algorithm?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "열거 비용은 체 크기의 변수 개수 승으로 증가합니다.",
        "en": "Enumeration grows as field size raised to the number of variables."
      }
    }
  ]
},
{
  "id": "invariant-constructions",
  "title": {
    "ko": "군 평균으로 불변량 구성하기",
    "en": "Constructing Invariants by Group Averaging"
  },
  "subtitle": {
    "ko": "궤도에서 특징을 평균해 불변량을 만들고 그 과정에서 사라지는 정보를 확인합니다.",
    "en": "Average features over an orbit and inspect the information that averaging removes."
  },
  "prerequisiteIds": [
    "algebraic-symmetries"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·모듈러 연산·선형사상",
    "en": "Sets, functions, modular arithmetic and linear maps in the linked prerequisites"
  },
  "eyebrow": "ABSTRACT ALGEBRA LAB",
  "formula": "(1,2),(-2,1),(-1,-2),(2,-1)",
  "accent": "abstract-algebra",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "불변량은 궤도를 항상 구분하나요?",
        "en": "Must an invariant separate all orbits?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "서로 다른 궤도도 같은 특징 평균을 가질 수 있습니다.",
        "en": "Different orbits can have the same averaged feature."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "실수에서 유한군 평균에 쓰는 가중치는?",
        "en": "What weights define the finite-group average over reals?"
      },
      "options": [
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "1/|G|",
          "en": "1/|G|"
        },
        {
          "ko": "항상 1/2",
          "en": "Always 1/2"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "각 군 원소에 같은 가중치를 줍니다.",
        "en": "Every group element receives equal weight."
      }
    }
  ]
},
{
  "id": "algebraic-representations-scope",
  "title": {
    "ko": "대수적 표현·등변성과 적용 범위",
    "en": "Representations, Equivariance and Their Scope"
  },
  "subtitle": {
    "ko": "대칭을 행렬로 표현하고 군에 따라 달라지는 등변 선형층의 제약을 계산합니다.",
    "en": "Represent symmetry by matrices and compute how equivariant-layer constraints depend on the group."
  },
  "prerequisiteIds": [
    "algebraic-symmetries",
    "invariant-constructions",
    "matrix-trace-logdet-derivatives"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 집합·함수·모듈러 연산·선형사상",
    "en": "Sets, functions, modular arithmetic and linear maps in the linked prerequisites"
  },
  "eyebrow": "ABSTRACT ALGEBRA LAB",
  "formula": "ρ(gh)=ρ(g)ρ(h)",
  "accent": "abstract-algebra",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "등변성과 불변성은 같은가요?",
        "en": "Are equivariance and invariance identical?"
      },
      "options": [
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "예",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "등변 출력은 입력 변환에 따라 변환되고 불변 출력은 같습니다.",
        "en": "An equivariant output transforms with the input; an invariant output stays fixed."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "선택한 대칭이 작업에 맞는지 확인해야 하나요?",
        "en": "Must the chosen symmetry match the task?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "판단할 수 없음",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "잘못된 대칭은 작업에 필요한 정보를 제거할 수 있습니다.",
        "en": "An inappropriate symmetry can remove task-relevant information."
      }
    }
  ]
},
{
  "id": "probability-spaces-events",
  "title": {
    "ko": "확률공간·사건과 확률공리",
    "en": "Probability Spaces, Events and Axioms"
  },
  "subtitle": {
    "ko": "표본 결과와 사건을 구분하고 가중치의 합으로 확률을 계산합니다.",
    "en": "Separate outcomes from events and calculate probabilities by summing weights."
  },
  "prerequisiteIds": [
    "sets-functions",
    "binomial-inclusion-exclusion"
  ],
  "prerequisites": {
    "ko": "연결된 선수 절의 사건·분포·기대값·선형대수·미분",
    "en": "Events, distributions, expectation, linear algebra and differentiation in the linked prerequisites"
  },
  "eyebrow": "PROBABILITY LAB",
  "formula": "Ω={0,1,2,3}, A={0,1}",
  "accent": "probability",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "확률의 전체 합은?",
        "en": "What is total probability?"
      },
      "options": [
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "결과 개수",
          "en": "Number of outcomes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "전체 공간의 확률은 1입니다.",
        "en": "The whole space has probability one."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "확률 0인 사건은 언제나 공집합인가요?",
        "en": "Must a zero-probability event be empty?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "확률 0인 원소를 포함할 수 있습니다.",
        "en": "It can contain outcomes of zero mass."
      }
    }
  ]
},
{
  "id": "conditional-probability-independence",
  "title": {
    "ko": "조건부확률·독립성과 Bayes",
    "en": "Conditional Probability, Independence and Bayes"
  },
  "subtitle": {
    "ko": "어떤 사건으로 모집단을 제한하는지 확인하고 조건을 뒤집는 계산을 합니다.",
    "en": "Identify the conditioning population and calculate the reversed conditional."
  },
  "prerequisiteIds": [
    "probability-spaces-events"
  ],
  "prerequisites": {
    "ko": "연결된 선수 절의 사건·분포·기대값·선형대수·미분",
    "en": "Events, distributions, expectation, linear algebra and differentiation in the linked prerequisites"
  },
  "eyebrow": "PROBABILITY LAB",
  "formula": "P(A|B)=P(A∩B)/P(B)",
  "accent": "probability",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "P(A|B)와 P(B|A)는 항상 같은가요?",
        "en": "Are P(A|B) and P(B|A) always equal?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "분모가 서로 다릅니다.",
        "en": "Their denominators differ."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "독립의 기본 조건은?",
        "en": "What is the basic independence condition?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "P(A∩B)=P(A)P(B)",
          "en": "P(A∩B)=P(A)P(B)"
        },
        {
          "ko": "P(A∩B)=0",
          "en": "P(A∩B)=0"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "서로소와 독립은 다른 조건입니다.",
        "en": "Disjointness and independence differ."
      }
    }
  ]
},
{
  "id": "probability-distributions",
  "title": {
    "ko": "확률분포와 표본",
    "en": "Probability Distributions and Samples"
  },
  "subtitle": {
    "ko": "가능한 결과에 확률을 배정하고 반복 관측과 모형을 구분합니다.",
    "en": "Assign probabilities to outcomes and distinguish a model from repeated observations."
  },
  "prerequisiteIds": [],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "PROBABILITY DISTRIBUTIONS LAB",
  "formula": "K∼Binomial(n,p)",
  "accent": "foundation",
  "questions": [
    {
      "id": "probability-distributions-0",
      "question": {
        "ko": "확률질량의 총합은?",
        "en": "What is total probability mass?"
      },
      "options": [
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "n",
          "en": "n"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "가능한 결과 전체의 확률은 1입니다.",
        "en": "The whole outcome space has probability one."
      }
    },
    {
      "id": "probability-distributions-1",
      "question": {
        "ko": "이항 모형은 어떤 시행을 가정하나요?",
        "en": "Which trial assumption defines this binomial model?"
      },
      "options": [
        {
          "ko": "독립이고 같은 p",
          "en": "Independent with common p"
        },
        {
          "ko": "항상 성공",
          "en": "Always successful"
        },
        {
          "ko": "항상 서로 종속",
          "en": "Always dependent"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "독립성과 동일 성공 확률을 확인해야 합니다.",
        "en": "Check independence and a common success probability."
      }
    }
  ]
},
{
  "id": "mass-density-cdf",
  "title": {
    "ko": "확률질량·밀도·누적분포",
    "en": "Probability Mass, Density and CDFs"
  },
  "subtitle": {
    "ko": "한 점의 확률과 단위 길이당 밀도를 구분하고 CDF의 점프를 읽습니다.",
    "en": "Separate point mass from density per unit length and read CDF jumps."
  },
  "prerequisiteIds": [
    "probability-distributions"
  ],
  "prerequisites": {
    "ko": "연결된 선수 절의 사건·분포·기대값·선형대수·미분",
    "en": "Events, distributions, expectation, linear algebra and differentiation in the linked prerequisites"
  },
  "eyebrow": "PROBABILITY LAB",
  "formula": "P(X=x)",
  "accent": "probability",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "밀도가 2이면 잘못된 확률모형인가요?",
        "en": "Does density two invalidate a probability model?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "예",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "밀도의 적분이 1이어야 합니다.",
        "en": "The density must integrate to one."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "CDF의 점프 크기가 뜻하는 것은?",
        "en": "What does a CDF jump measure?"
      },
      "options": [
        {
          "ko": "점 질량",
          "en": "Point mass"
        },
        {
          "ko": "표본 크기",
          "en": "Sample size"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "우연속 CDF의 좌극한과의 차이입니다.",
        "en": "It is the difference from the left limit."
      }
    }
  ]
},
{
  "id": "expectation-variance",
  "title": {
    "ko": "기댓값·분산과 불확실성",
    "en": "Expectation, Variance and Uncertainty"
  },
  "subtitle": {
    "ko": "가능한 값의 가중 평균과 평균 주변의 퍼짐을 따로 계산합니다.",
    "en": "Calculate a weighted mean and spread around that mean as distinct quantities."
  },
  "prerequisiteIds": [
    "probability-distributions"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "EXPECTATION VARIANCE LAB",
  "formula": "Var(X)=E[X²]−E[X]²",
  "accent": "foundation",
  "questions": [
    {
      "id": "expectation-variance-0",
      "question": {
        "ko": "분산의 단위는?",
        "en": "What units does variance have?"
      },
      "options": [
        {
          "ko": "X와 같다",
          "en": "Same as X"
        },
        {
          "ko": "X 단위의 제곱",
          "en": "Squared units of X"
        },
        {
          "ko": "항상 없다",
          "en": "Always unitless"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "제곱 편차의 평균이므로 제곱 단위입니다.",
        "en": "It averages squared deviations."
      }
    },
    {
      "id": "expectation-variance-1",
      "question": {
        "ko": "평균을 제곱하면 항상 E[X²]인가요?",
        "en": "Is squaring the mean always E[X²]?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "음수일 때만",
          "en": "Only for negative X"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "차이는 분산입니다.",
        "en": "Their difference is variance."
      }
    }
  ]
},
{
  "id": "joint-marginal-conditional",
  "title": {
    "ko": "결합·주변·조건부 분포와 공분산",
    "en": "Joint, Marginal, Conditional Laws and Covariance"
  },
  "subtitle": {
    "ko": "결합 표를 합하거나 정규화해 주변·조건부 분포를 얻고 의존성을 계산합니다.",
    "en": "Sum or normalize a joint table to obtain marginal/conditional laws and dependence."
  },
  "prerequisiteIds": [
    "conditional-probability-independence",
    "expectation-variance"
  ],
  "prerequisites": {
    "ko": "연결된 선수 절의 사건·분포·기대값·선형대수·미분",
    "en": "Events, distributions, expectation, linear algebra and differentiation in the linked prerequisites"
  },
  "eyebrow": "PROBABILITY LAB",
  "formula": "p00=p11=3/8",
  "accent": "probability",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "주변 분포는 어떻게 구하나요?",
        "en": "How is a marginal obtained?"
      },
      "options": [
        {
          "ko": "다른 축을 합산",
          "en": "Sum the other axis"
        },
        {
          "ko": "최댓값 선택",
          "en": "Take the maximum"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "서로소 경우의 확률을 더합니다.",
        "en": "Add probabilities of disjoint alternatives."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "일반적으로 공분산 0이면 독립인가요?",
        "en": "Does zero covariance imply independence in general?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "이진 두 변수의 특수성은 일반 분포로 확대되지 않습니다.",
        "en": "The special binary case does not extend to every distribution."
      }
    }
  ]
},
{
  "id": "multivariate-gaussians",
  "title": {
    "ko": "다변량 Gaussian과 조건부 기하",
    "en": "Multivariate Gaussians and Conditional Geometry"
  },
  "subtitle": {
    "ko": "공분산 타원·조건부 평균·분산을 연결하고 특이 Gaussian을 구분합니다.",
    "en": "Connect covariance ellipses to conditional means/variances and distinguish singular Gaussians."
  },
  "prerequisiteIds": [
    "joint-marginal-conditional",
    "positive-semidefinite-matrices"
  ],
  "prerequisites": {
    "ko": "연결된 선수 절의 사건·분포·기대값·선형대수·미분",
    "en": "Events, distributions, expectation, linear algebra and differentiation in the linked prerequisites"
  },
  "eyebrow": "PROBABILITY LAB",
  "formula": "Σ=[1,.5;.5,1]",
  "accent": "probability",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "단위 Mahalanobis 타원 안 확률은 약?",
        "en": "What mass lies inside a unit Mahalanobis ellipse in 2D?"
      },
      "options": [
        {
          "ko": "68%",
          "en": "68%"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "39.3%",
          "en": "39.3%"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "이차원에서는 1−exp(−1/2)입니다.",
        "en": "In two dimensions it is 1−exp(−1/2)."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "특이 Gaussian은 무효 확률분포인가요?",
        "en": "Is a singular Gaussian an invalid probability law?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "예",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "낮은 차원에 지지된 확률법칙입니다.",
        "en": "It is supported on a lower-dimensional set."
      }
    }
  ]
},
{
  "id": "random-variable-transformations",
  "title": {
    "ko": "확률변수 변환과 Jacobian",
    "en": "Random-Variable Transformations and Jacobians"
  },
  "subtitle": {
    "ko": "역상과 절댓값 Jacobian으로 밀도를 바꾸고 여러 역상을 합산합니다.",
    "en": "Transform densities using preimages and absolute Jacobians, adding multiple branches."
  },
  "prerequisiteIds": [
    "mass-density-cdf",
    "chain-rule"
  ],
  "prerequisites": {
    "ko": "연결된 선수 절의 사건·분포·기대값·선형대수·미분",
    "en": "Events, distributions, expectation, linear algebra and differentiation in the linked prerequisites"
  },
  "eyebrow": "PROBABILITY LAB",
  "formula": "P(Y≤y)=P(g(X)≤y)",
  "accent": "probability",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "밀도 변환에서 Jacobian은?",
        "en": "Which Jacobian enters density transformation?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "절댓값",
          "en": "Absolute value"
        },
        {
          "ko": "부호 그대로",
          "en": "Signed determinant"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "확률은 방향 반전에 음수가 되지 않습니다.",
        "en": "Orientation reversal cannot make probability negative."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "여러 역상이 있으면?",
        "en": "What if several preimages exist?"
      },
      "options": [
        {
          "ko": "각 기여를 합산",
          "en": "Sum contributions"
        },
        {
          "ko": "하나만 선택",
          "en": "Choose only one"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "서로소 입력 분기들의 확률을 더합니다.",
        "en": "Add the probabilities of disjoint input branches."
      }
    }
  ]
},
{
  "id": "laws-large-numbers-clt",
  "title": {
    "ko": "대수의 법칙·중심극한정리",
    "en": "Laws of Large Numbers and the Central Limit Theorem"
  },
  "subtitle": {
    "ko": "평균의 집중과 표준화된 요동의 근사를 분리해 읽습니다.",
    "en": "Separate concentration of the mean from approximation of standardized fluctuations."
  },
  "prerequisiteIds": [
    "expectation-variance",
    "mass-density-cdf"
  ],
  "prerequisites": {
    "ko": "연결된 선수 절의 사건·분포·기대값·선형대수·미분",
    "en": "Events, distributions, expectation, linear algebra and differentiation in the linked prerequisites"
  },
  "eyebrow": "PROBABILITY LAB",
  "formula": "Xi iid Bernoulli(p)",
  "accent": "probability",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "대수의 법칙은 모든 표본평균이 p와 같다는 뜻인가요?",
        "en": "Does the LLN make every finite sample mean equal p?"
      },
      "options": [
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "고정 오차를 넘을 확률이 줄어드는 극한 주장입니다.",
        "en": "It concerns the limiting probability of a fixed deviation."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "CLT가 분산 0에도 같은 식으로 적용되나요?",
        "en": "Can the same standardized CLT expression be used at zero variance?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "표준편차로 나눌 수 없습니다.",
        "en": "Division by the standard deviation is undefined."
      }
    }
  ]
},
{
  "id": "reparameterization-gradients",
  "title": {
    "ko": "재매개화와 확률적 경사",
    "en": "Reparameterization and Pathwise Gradients"
  },
  "subtitle": {
    "ko": "고정한 잡음을 통해 표본 손실을 미분하고 기대 경사와 구분합니다.",
    "en": "Differentiate a sample loss through fixed noise and distinguish its expected gradient."
  },
  "prerequisiteIds": [
    "random-variable-transformations",
    "expectation-variance"
  ],
  "prerequisites": {
    "ko": "연결된 선수 절의 사건·분포·기대값·선형대수·미분",
    "en": "Events, distributions, expectation, linear algebra and differentiation in the linked prerequisites"
  },
  "eyebrow": "PROBABILITY LAB",
  "formula": "z=μ+σε, ε~N(0,1)",
  "accent": "probability",
  "questions": [
    {
      "id": "check-1",
      "question": {
        "ko": "경사를 계산할 때 기초 잡음은?",
        "en": "How is base noise treated during differentiation?"
      },
      "options": [
        {
          "ko": "매번 다른 값",
          "en": "Changed every time"
        },
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "고정",
          "en": "Held fixed"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "같은 경로의 매개변수 의존성을 미분합니다.",
        "en": "Differentiate parameter dependence along the same noise path."
      }
    },
    {
      "id": "check-2",
      "question": {
        "ko": "±1 두 점은 Gaussian 표본인가요?",
        "en": "Are the two nodes ±1 Gaussian samples?"
      },
      "options": [
        {
          "ko": "판정 불가",
          "en": "Cannot decide"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "예",
          "en": "Yes"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "이차 예제의 모멘트 검산용 결정론적 노드입니다.",
        "en": "They are deterministic nodes for this quadratic moment check."
      }
    }
  ]
},
{
  "id": "statistical-estimation",
  "title": {
    "ko": "통계적 추정과 가능도",
    "en": "Statistical Estimation and Likelihood"
  },
  "subtitle": {
    "ko": "관측을 고정하고 매개변수를 바꾸며 어떤 모형이 데이터를 잘 설명하는지 비교합니다.",
    "en": "Hold observations fixed and compare how parameter values explain the data."
  },
  "prerequisiteIds": [
    "expectation-variance",
    "gradients"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "STATISTICAL ESTIMATION LAB",
  "formula": "p̂ = k/n",
  "accent": "foundation",
  "questions": [
    {
      "id": "statistical-estimation-0",
      "question": {
        "ko": "가능도 계산에서 고정하는 것은?",
        "en": "What stays fixed when computing likelihood?"
      },
      "options": [
        {
          "ko": "관측 데이터",
          "en": "Observed data"
        },
        {
          "ko": "후보 매개변수만",
          "en": "Only the candidate parameter"
        },
        {
          "ko": "아무것도 없다",
          "en": "Nothing"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "같은 관측을 서로 다른 매개변수로 평가합니다.",
        "en": "Evaluate the same observations under different parameters."
      }
    },
    {
      "id": "statistical-estimation-1",
      "question": {
        "ko": "7/10의 Bernoulli MLE는?",
        "en": "What is the Bernoulli MLE for 7/10 successes?"
      },
      "options": [
        {
          "ko": "0.3",
          "en": "0.3"
        },
        {
          "ko": "0.5",
          "en": "0.5"
        },
        {
          "ko": "0.7",
          "en": "0.7"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "성공 수를 전체 시행 수로 나눕니다.",
        "en": "Divide successes by trials."
      }
    }
  ]
},
{
  "id": "bayesian-inference",
  "title": {
    "ko": "Bayes 추론과 사후예측",
    "en": "Bayesian Inference and Posterior Prediction"
  },
  "subtitle": {
    "ko": "사전 지식과 관측 근거를 결합하고 매개변수 불확실성을 예측에 반영합니다.",
    "en": "Combine prior knowledge with evidence and carry parameter uncertainty into prediction."
  },
  "prerequisiteIds": [
    "statistical-estimation"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "BAYESIAN INFERENCE LAB",
  "formula": "Beta(α,β) → Beta(α+7,β+3)",
  "accent": "foundation",
  "questions": [
    {
      "id": "bayesian-inference-0",
      "question": {
        "ko": "연속 밀도의 한 점 높이는?",
        "en": "What is the height of a continuous density at one point?"
      },
      "options": [
        {
          "ko": "그 점의 확률",
          "en": "The probability of that point"
        },
        {
          "ko": "확률밀도 값",
          "en": "A density value"
        },
        {
          "ko": "항상 1 이하",
          "en": "Always at most one"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "확률은 구간에서 밀도를 적분하여 구합니다.",
        "en": "Probabilities are integrals of density over intervals."
      }
    },
    {
      "id": "bayesian-inference-1",
      "question": {
        "ko": "Beta(1,1)에 성공 7·실패 3을 관측하면?",
        "en": "After seven successes and three failures with Beta(1,1), what is the posterior?"
      },
      "options": [
        {
          "ko": "Beta(7,3)",
          "en": "Beta(7,3)"
        },
        {
          "ko": "Beta(8,4)",
          "en": "Beta(8,4)"
        },
        {
          "ko": "Beta(1,1)",
          "en": "Beta(1,1)"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "모양 매개변수에 관측 횟수를 더합니다.",
        "en": "Add observed counts to the shape parameters."
      }
    }
  ]
},
{
  "id": "information-theory",
  "title": {
    "ko": "엔트로피·교차엔트로피·KL",
    "en": "Entropy, Cross-Entropy and KL Divergence"
  },
  "subtitle": {
    "ko": "불확실성과 잘못된 확률 모형의 추가 비용을 구분합니다.",
    "en": "Distinguish uncertainty from the extra cost of using the wrong probability model."
  },
  "prerequisiteIds": [
    "expectation-variance"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "INFORMATION THEORY LAB",
  "formula": "H(P,Q)=H(P)+D_KL(P‖Q)",
  "accent": "foundation",
  "questions": [
    {
      "id": "information-theory-0",
      "question": {
        "ko": "P=Q이면 KL은?",
        "en": "What is KL when P=Q?"
      },
      "options": [
        {
          "ko": "0",
          "en": "0"
        },
        {
          "ko": "1",
          "en": "1"
        },
        {
          "ko": "H(P)",
          "en": "H(P)"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "같은 분포 사이 추가 비용은 없습니다.",
        "en": "There is no extra cost between identical distributions."
      }
    },
    {
      "id": "information-theory-1",
      "question": {
        "ko": "KL은 항상 대칭인가요?",
        "en": "Is KL always symmetric?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "분산이 같으면 항상",
          "en": "Always if variances match"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "기댓값을 취하는 기준 분포가 다릅니다.",
        "en": "Its expectation uses a different reference distribution in each direction."
      }
    }
  ]
},
{
  "id": "regularization",
  "title": {
    "ko": "정규화와 편향·안정성",
    "en": "Regularization, Bias and Stability"
  },
  "subtitle": {
    "ko": "약한 관측 방향의 증폭을 줄이는 대신 어떤 편향이 생기는지 살펴봅니다.",
    "en": "Inspect the bias introduced when stabilizing weakly observed directions."
  },
  "prerequisiteIds": [
    "svd",
    "least-squares"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "REGULARIZATION LAB",
  "formula": "wλ=(XᵀX+λI)⁻¹Xᵀy",
  "accent": "foundation",
  "questions": [
    {
      "id": "regularization-0",
      "question": {
        "ko": "ridge의 λ를 키우면 일반적으로?",
        "en": "What does increasing ridge λ generally do?"
      },
      "options": [
        {
          "ko": "계수 크기를 줄인다",
          "en": "Shrinks coefficients"
        },
        {
          "ko": "항상 무편향이다",
          "en": "Always makes estimates unbiased"
        },
        {
          "ko": "관측을 새로 만든다",
          "en": "Creates observations"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "축소는 편향과 안정성의 절충입니다.",
        "en": "Shrinkage trades bias against stability."
      }
    },
    {
      "id": "regularization-1",
      "question": {
        "ko": "s=λ=0의 둘째 계수는?",
        "en": "What happens to the second coefficient at s=λ=0?"
      },
      "options": [
        {
          "ko": "유일하게 0",
          "en": "Uniquely zero"
        },
        {
          "ko": "데이터만으로 식별되지 않는다",
          "en": "Not identified by the data"
        },
        {
          "ko": "유일하게 1",
          "en": "Uniquely one"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "최소 노름 선택과 식별 가능성은 다릅니다.",
        "en": "A minimum-norm choice differs from identifiability."
      }
    }
  ]
},
{
  "id": "stochastic-optimization",
  "title": {
    "ko": "확률적 경사와 학습 잡음",
    "en": "Stochastic Gradients and Learning Noise"
  },
  "subtitle": {
    "ko": "전체 평균 기울기와 무작위 표본 하나의 기울기를 비교합니다.",
    "en": "Compare the full mean gradient with the gradient of one randomly sampled example."
  },
  "prerequisiteIds": [
    "backpropagation",
    "expectation-variance"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "STOCHASTIC OPTIMIZATION LAB",
  "formula": "E[gᵢ(w)]=∇L(w)",
  "accent": "foundation",
  "questions": [
    {
      "id": "stochastic-optimization-0",
      "question": {
        "ko": "불편 경사 추정은 무엇인가요?",
        "en": "What defines an unbiased gradient estimate?"
      },
      "options": [
        {
          "ko": "매번 정확한 경사",
          "en": "Exact gradient every time"
        },
        {
          "ko": "기댓값이 전체 경사",
          "en": "Its expectation equals the full gradient"
        },
        {
          "ko": "항상 손실 감소",
          "en": "Always decreases loss"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "기댓값의 성질이지 개별 경로 보장이 아닙니다.",
        "en": "It is a property of expectation, not a per-path guarantee."
      }
    },
    {
      "id": "stochastic-optimization-1",
      "question": {
        "ko": "상수 학습률 SGD가 최적점 근처에서 흔들리는 이유는?",
        "en": "Why can constant-rate SGD fluctuate near the optimum?"
      },
      "options": [
        {
          "ko": "표본 기울기의 분산",
          "en": "Sample-gradient variance"
        },
        {
          "ko": "목적함수가 없다",
          "en": "There is no objective"
        },
        {
          "ko": "미분이 항상 틀린다",
          "en": "Derivatives are always wrong"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "전체 경사가 0이어도 표본 경사는 다를 수 있습니다.",
        "en": "Individual gradients can differ even when the mean gradient is zero."
      }
    }
  ]
},
{
  "id": "differential-equations",
  "title": {
    "ko": "미분방정식과 시간 이산화",
    "en": "Differential Equations and Time Discretization"
  },
  "subtitle": {
    "ko": "연속 시간의 감쇠와 유한 보폭 수치해법을 분리해 이해합니다.",
    "en": "Separate continuous-time decay from its finite-step numerical approximation."
  },
  "prerequisiteIds": [
    "chain-rule",
    "gradients"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "DIFFERENTIAL EQUATIONS LAB",
  "formula": "x′=−kx · x(t)=x₀e⁻ᵏᵗ",
  "accent": "foundation",
  "questions": [
    {
      "id": "differential-equations-0",
      "question": {
        "ko": "전진 Euler의 안정 조건은?",
        "en": "What is the forward-Euler stability condition here?"
      },
      "options": [
        {
          "ko": "0<kΔt<2",
          "en": "0<kΔt<2"
        },
        {
          "ko": "kΔt>2",
          "en": "kΔt>2"
        },
        {
          "ko": "항상 안정",
          "en": "Always stable"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "한 단계 배율의 절댓값이 1보다 작아야 합니다.",
        "en": "The one-step multiplier must have absolute value below one."
      }
    },
    {
      "id": "differential-equations-1",
      "question": {
        "ko": "초기 조건은 무엇을 정하나요?",
        "en": "What does an initial condition select?"
      },
      "options": [
        {
          "ko": "해의 궤적",
          "en": "A solution trajectory"
        },
        {
          "ko": "미분 기호",
          "en": "Derivative notation"
        },
        {
          "ko": "항상 최적해",
          "en": "Always an optimum"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "같은 법칙에도 초기값별 해가 있습니다.",
        "en": "The same evolution law has solutions for different initial states."
      }
    }
  ]
},
{
  "id": "mechanics-energy",
  "title": {
    "ko": "역학·에너지와 구조 보존",
    "en": "Mechanics, Energy and Structure Preservation"
  },
  "subtitle": {
    "ko": "조화진동자의 힘·운동·에너지를 연결하고 수치 오차를 관찰합니다.",
    "en": "Connect force, motion and energy in an oscillator and inspect numerical error."
  },
  "prerequisiteIds": [
    "differential-equations",
    "vectors-and-forces"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "MECHANICS ENERGY LAB",
  "formula": "m q″=−kq · E=½mv²+½kq²",
  "accent": "foundation",
  "questions": [
    {
      "id": "mechanics-energy-0",
      "question": {
        "ko": "마찰 없는 실제 진동자의 총에너지는?",
        "en": "What happens to total energy in the frictionless exact oscillator?"
      },
      "options": [
        {
          "ko": "보존된다",
          "en": "It is conserved"
        },
        {
          "ko": "항상 감소",
          "en": "Always decreases"
        },
        {
          "ko": "항상 증가",
          "en": "Always increases"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "외부 일과 마찰을 포함하지 않은 모형입니다.",
        "en": "The model has no external work or friction."
      }
    },
    {
      "id": "mechanics-energy-1",
      "question": {
        "ko": "symplectic Euler는 정확한 E를 항상 보존하나요?",
        "en": "Does symplectic Euler always conserve exact E?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "질량이 1이면 항상",
          "en": "Always if mass is one"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "구조 보존과 정확한 에너지 보존은 다릅니다.",
        "en": "Structure preservation differs from exact energy conservation."
      }
    }
  ]
},
{
  "id": "thermodynamics",
  "title": {
    "ko": "통계역학과 분배함수",
    "en": "Statistical Mechanics and the Partition Function"
  },
  "subtitle": {
    "ko": "미시 상태의 확률에서 에너지·자유에너지·엔트로피를 계산합니다.",
    "en": "Calculate energy, free energy and entropy from microscopic state probabilities."
  },
  "prerequisiteIds": [
    "information-theory",
    "mechanics-energy"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "THERMODYNAMICS LAB",
  "formula": "Z=1+g e⁻Δ/T · F=−T log Z",
  "accent": "foundation",
  "questions": [
    {
      "id": "thermodynamics-0",
      "question": {
        "ko": "분배함수의 역할은?",
        "en": "What does the partition function do?"
      },
      "options": [
        {
          "ko": "정규화한다",
          "en": "Normalizes weights"
        },
        {
          "ko": "온도를 없앤다",
          "en": "Removes temperature"
        },
        {
          "ko": "항상 1이다",
          "en": "Is always one"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "가중치 합으로 나눠 확률을 만듭니다.",
        "en": "Divide by total weight to form probabilities."
      }
    },
    {
      "id": "thermodynamics-1",
      "question": {
        "ko": "g>1에서 이진 점유 엔트로피가 전체 엔트로피인가요?",
        "en": "For g>1, is binary occupancy entropy the full entropy?"
      },
      "options": [
        {
          "ko": "항상 같다",
          "en": "Always"
        },
        {
          "ko": "아니다",
          "en": "No"
        },
        {
          "ko": "온도와 무관하게 0",
          "en": "Always zero"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "에너지 수준 내부의 미시 상태도 세야 합니다.",
        "en": "Count microstates within each energy level too."
      }
    }
  ]
},
{
  "id": "stochastic-processes",
  "title": {
    "ko": "Markov 과정과 평형",
    "en": "Markov Processes and Equilibrium"
  },
  "subtitle": {
    "ko": "한 번의 분포가 아니라 시간에 따라 이동하는 확률을 추적합니다.",
    "en": "Track probability evolving over time rather than only one static distribution."
  },
  "prerequisiteIds": [
    "probability-distributions",
    "differential-equations"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "STOCHASTIC PROCESSES LAB",
  "formula": "pₜ₊₁=a+(1−a−b)pₜ",
  "accent": "foundation",
  "questions": [
    {
      "id": "stochastic-processes-0",
      "question": {
        "ko": "a=b=1에서 초기 p=1의 경로는?",
        "en": "At a=b=1 and initial p=1, what is the path?"
      },
      "options": [
        {
          "ko": "1,0,1,0,…",
          "en": "1,0,1,0,…"
        },
        {
          "ko": "항상 0.5",
          "en": "Always 0.5"
        },
        {
          "ko": "항상 0",
          "en": "Always 0"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "주기적 전이는 정상분포로 수렴하지 않을 수 있습니다.",
        "en": "Periodic transitions need not converge to stationarity."
      }
    },
    {
      "id": "stochastic-processes-1",
      "question": {
        "ko": "a=b=0의 정상분포는?",
        "en": "What is stationary when a=b=0?"
      },
      "options": [
        {
          "ko": "유일하게 0.5",
          "en": "Only 0.5"
        },
        {
          "ko": "모든 분포",
          "en": "Every distribution"
        },
        {
          "ko": "없다",
          "en": "None"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "아무 전이도 없으므로 모든 분포가 유지됩니다.",
        "en": "No transitions means every distribution stays unchanged."
      }
    }
  ]
},
{
  "id": "inverse-problems",
  "title": {
    "ko": "역문제·식별성과 관측 설계",
    "en": "Inverse Problems, Identifiability and Observation Design"
  },
  "subtitle": {
    "ko": "관측에서 원인을 복원할 때 정보 부족과 잡음 증폭을 구분합니다.",
    "en": "Separate missing information from noise amplification when reconstructing causes from observations."
  },
  "prerequisiteIds": [
    "regularization",
    "bayesian-inference"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "INVERSE PROBLEMS LAB",
  "formula": "y=Xθ+ε · weak directions amplify noise",
  "accent": "foundation",
  "questions": [
    {
      "id": "inverse-problems-0",
      "question": {
        "ko": "정규화는 관측 정보 자체를 추가하나요?",
        "en": "Does regularization itself add observations?"
      },
      "options": [
        {
          "ko": "예",
          "en": "Yes"
        },
        {
          "ko": "아니요",
          "en": "No"
        },
        {
          "ko": "항상 두 배",
          "en": "Always doubles them"
        }
      ],
      "answer": 1,
      "explanation": {
        "ko": "선호를 추가하는 것과 실제 관측을 늘리는 것은 다릅니다.",
        "en": "Adding a preference differs from adding observations."
      }
    },
    {
      "id": "inverse-problems-1",
      "question": {
        "ko": "작은 특잇값 방향의 역변환은?",
        "en": "What does inversion do along a small-singular-value direction?"
      },
      "options": [
        {
          "ko": "잡음을 증폭할 수 있다",
          "en": "Can amplify noise"
        },
        {
          "ko": "항상 잡음을 없앤다",
          "en": "Always removes noise"
        },
        {
          "ko": "항상 유일하지 않다",
          "en": "Is always nonunique"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "불안정성과 정확한 랭크 부족은 구분합니다.",
        "en": "Distinguish instability from exact rank deficiency."
      }
    }
  ]
},
{
  "id": "research-methods",
  "title": {
    "ko": "재현·불확실성과 연구 주장",
    "en": "Reproducibility, Uncertainty and Research Claims"
  },
  "subtitle": {
    "ko": "평균 개선값만 보고 결론 내리지 않고 대응 실험과 가정·오차를 함께 검토합니다.",
    "en": "Assess paired experiments, assumptions and uncertainty rather than concluding from mean improvement alone."
  },
  "prerequisiteIds": [
    "statistical-estimation",
    "inverse-problems"
  ],
  "prerequisites": {
    "ko": "연결된 선수 단원의 기호와 기준 예제를 먼저 확인하세요.",
    "en": "Review notation and reference examples in the linked prerequisites."
  },
  "eyebrow": "RESEARCH METHODS LAB",
  "formula": "paired differences → mean ± t SE",
  "accent": "foundation",
  "questions": [
    {
      "id": "research-methods-0",
      "question": {
        "ko": "서로 다른 seed 4개는 무엇을 보장하나요?",
        "en": "What do four different seeds guarantee?"
      },
      "options": [
        {
          "ko": "전역 일반화",
          "en": "Global generalization"
        },
        {
          "ko": "자동으로 독립인 데이터",
          "en": "Automatically independent data"
        },
        {
          "ko": "그 자체로 어느 것도 보장하지 않는다",
          "en": "Neither guarantee follows by itself"
        }
      ],
      "answer": 2,
      "explanation": {
        "ko": "seed 수와 표본 독립성·데이터 다양성은 다릅니다.",
        "en": "Seed count differs from independence and data diversity."
      }
    },
    {
      "id": "research-methods-1",
      "question": {
        "ko": "가정이 있는 신뢰구간은?",
        "en": "What is an assumption-dependent confidence interval?"
      },
      "options": [
        {
          "ko": "가정과 절차의 장기적 성질",
          "en": "A long-run property of a procedure under assumptions"
        },
        {
          "ko": "항상 참값이 들어간다",
          "en": "Always contains truth"
        },
        {
          "ko": "모든 편향을 제거한다",
          "en": "Removes every bias"
        }
      ],
      "answer": 0,
      "explanation": {
        "ko": "반복 표집에서의 포함률 의미를 구분하세요.",
        "en": "Distinguish coverage under repeated sampling."
      }
    }
  ]
},
];

export interface ConceptStep { title: BookText; explanation: BookText; formula: string; diagram: BookText; }
export interface GuidedExperiment {
  title: BookText; predict: BookText; action: BookText; explain: BookText;
  preset: number; time?: number; steps?: number;
}
export interface LearningGuide { intuition: BookText; steps: ConceptStep[]; experiments: GuidedExperiment[]; }
export const learningGuides: Record<LearningUnitId, LearningGuide> = {
"probability-spaces-events":{
  "intuition": {
    "ko": "표본 결과와 사건을 구분하고 가중치의 합으로 확률을 계산합니다.",
    "en": "Separate outcomes from events and calculate probabilities by summing weights."
  },
  "steps": [
    {
      "title": {
        "ko": "결과와 사건",
        "en": "Outcomes and events"
      },
      "explanation": {
        "ko": "사건은 결과의 집합입니다.",
        "en": "An event is a set of outcomes."
      },
      "formula": "Ω={0,1,2,3}, A={0,1}",
      "diagram": {
        "ko": "확률공간·사건과 확률공리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Probability Spaces, Events and Axioms. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "정규화",
        "en": "Normalize"
      },
      "explanation": {
        "ko": "가중치를 전체 합으로 나눕니다.",
        "en": "Divide weights by their total."
      },
      "formula": "p=(.1,.2,.3,.4)",
      "diagram": {
        "ko": "확률공간·사건과 확률공리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Probability Spaces, Events and Axioms. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "합집합",
        "en": "Union"
      },
      "explanation": {
        "ko": "교집합을 중복 세지 않습니다.",
        "en": "Correct for the shared outcomes."
      },
      "formula": "P(A∪B)=.3+.5−.2=.6",
      "diagram": {
        "ko": "확률공간·사건과 확률공리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Probability Spaces, Events and Axioms. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "영 사건",
        "en": "Null event"
      },
      "explanation": {
        "ko": "가능한 결과 목록과 양의 확률을 구분합니다.",
        "en": "Distinguish membership in the space from positive probability."
      },
      "formula": "P({2})=0 may hold",
      "diagram": {
        "ko": "확률공간·사건과 확률공리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Probability Spaces, Events and Axioms. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "가중 사건",
        "en": "Weighted events"
      },
      "predict": {
        "ko": "합집합 확률은?",
        "en": "What is the union probability?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0.6입니다.",
        "en": "0.6."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "서로소 사건",
        "en": "Disjoint events"
      },
      "predict": {
        "ko": "교집합이 비면?",
        "en": "What if the events are disjoint?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "이 예제의 합집합 확률은 1입니다.",
        "en": "The union probability is one here."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "확률 0인 비공집합",
        "en": "Nonempty null events"
      },
      "predict": {
        "ko": "비공집합이면 확률이 양수인가요?",
        "en": "Must a nonempty event have positive probability?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 두 사건 모두 0입니다.",
        "en": "No: both events here have probability zero."
      },
      "preset": 2
    }
  ]
},
"conditional-probability-independence":{
  "intuition": {
    "ko": "어떤 사건으로 모집단을 제한하는지 확인하고 조건을 뒤집는 계산을 합니다.",
    "en": "Identify the conditioning population and calculate the reversed conditional."
  },
  "steps": [
    {
      "title": {
        "ko": "제한",
        "en": "Restrict"
      },
      "explanation": {
        "ko": "B에 속한 결과만 분모에 남깁니다.",
        "en": "Keep only B outcomes in the denominator."
      },
      "formula": "P(A|B)=P(A∩B)/P(B)",
      "diagram": {
        "ko": "조건부확률·독립성과 Bayes의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Conditional Probability, Independence and Bayes. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "방향",
        "en": "Direction"
      },
      "explanation": {
        "ko": "조건 순서를 구분합니다.",
        "en": "Distinguish the two conditioning directions."
      },
      "formula": "P(A|B)=1/3 ≠ P(B|A)=1/2",
      "diagram": {
        "ko": "조건부확률·독립성과 Bayes의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Conditional Probability, Independence and Bayes. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "Bayes",
        "en": "Bayes"
      },
      "explanation": {
        "ko": "결합확률을 두 방식으로 씁니다.",
        "en": "Write the joint probability in two ways."
      },
      "formula": "P(A|B)=P(B|A)P(A)/P(B)",
      "diagram": {
        "ko": "조건부확률·독립성과 Bayes의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Conditional Probability, Independence and Bayes. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "독립",
        "en": "Independence"
      },
      "explanation": {
        "ko": "확률 0에서도 곱 조건은 의미가 있습니다.",
        "en": "The product condition also makes sense for null events."
      },
      "formula": "P(A∩B)=P(A)P(B)",
      "diagram": {
        "ko": "조건부확률·독립성과 Bayes의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Conditional Probability, Independence and Bayes. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "다른 조건부",
        "en": "Different conditionals"
      },
      "predict": {
        "ko": "A|B는 얼마인가요?",
        "en": "What is A given B?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1/3입니다.",
        "en": "One third."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "독립 사건",
        "en": "Independent events"
      },
      "predict": {
        "ko": "B를 알면 A가 바뀌나요?",
        "en": "Does learning B change A?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 독립입니다.",
        "en": "No: these events are independent."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "조건 사건 확률 0",
        "en": "Zero-probability condition"
      },
      "predict": {
        "ko": "B가 발생하지 않는 모형이면?",
        "en": "What if the model assigns zero probability to B?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "초등 비율의 조건부확률은 정의되지 않습니다.",
        "en": "The elementary ratio is undefined."
      },
      "preset": 2
    }
  ]
},
"mass-density-cdf":{
  "intuition": {
    "ko": "한 점의 확률과 단위 길이당 밀도를 구분하고 CDF의 점프를 읽습니다.",
    "en": "Separate point mass from density per unit length and read CDF jumps."
  },
  "steps": [
    {
      "title": {
        "ko": "질량",
        "en": "Mass"
      },
      "explanation": {
        "ko": "한 점에 배정된 확률입니다.",
        "en": "Probability assigned to one point."
      },
      "formula": "P(X=x)",
      "diagram": {
        "ko": "확률질량·밀도·누적분포의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Probability Mass, Density and CDFs. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "밀도",
        "en": "Density"
      },
      "explanation": {
        "ko": "밀도는 1보다 클 수 있습니다.",
        "en": "Density may exceed one."
      },
      "formula": "f(x)=2 on [0,.5]",
      "diagram": {
        "ko": "확률질량·밀도·누적분포의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Probability Mass, Density and CDFs. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "누적",
        "en": "Cumulative"
      },
      "explanation": {
        "ko": "밀도를 적분해 확률을 얻습니다.",
        "en": "Integrate density to obtain probability."
      },
      "formula": "F(.25)=∫₀^.25 2 dx=.5",
      "diagram": {
        "ko": "확률질량·밀도·누적분포의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Probability Mass, Density and CDFs. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "점프",
        "en": "Jump"
      },
      "explanation": {
        "ko": "CDF 점프가 점 질량입니다.",
        "en": "A CDF jump equals point mass."
      },
      "formula": "ΔF(1)=.25 for Bernoulli(.25)",
      "diagram": {
        "ko": "확률질량·밀도·누적분포의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Probability Mass, Density and CDFs. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "높은 밀도",
        "en": "Density above one"
      },
      "predict": {
        "ko": "밀도와 점 확률은?",
        "en": "What are density and point mass?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2와 0입니다.",
        "en": "Two and zero."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "점프가 있는 분포",
        "en": "Distribution with jumps"
      },
      "predict": {
        "ko": "x=1의 점프 크기는?",
        "en": "What is the jump size at one?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0.25입니다.",
        "en": "0.25."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "지지집합 밖",
        "en": "Outside support"
      },
      "predict": {
        "ko": "지지집합 밖 CDF는?",
        "en": "What is the CDF above the support?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1이고 밀도는 0입니다.",
        "en": "One; density is zero."
      },
      "preset": 2
    }
  ]
},
"joint-marginal-conditional":{
  "intuition": {
    "ko": "결합 표를 합하거나 정규화해 주변·조건부 분포를 얻고 의존성을 계산합니다.",
    "en": "Sum or normalize a joint table to obtain marginal/conditional laws and dependence."
  },
  "steps": [
    {
      "title": {
        "ko": "결합",
        "en": "Joint"
      },
      "explanation": {
        "ko": "동시에 일어나는 값을 셉니다.",
        "en": "Count simultaneous values."
      },
      "formula": "p00=p11=3/8",
      "diagram": {
        "ko": "결합·주변·조건부 분포와 공분산의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Joint, Marginal, Conditional Laws and Covariance. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "주변",
        "en": "Marginal"
      },
      "explanation": {
        "ko": "다른 변수를 합산합니다.",
        "en": "Sum over the other variable."
      },
      "formula": "P(X=1)=P(Y=1)=1/2",
      "diagram": {
        "ko": "결합·주변·조건부 분포와 공분산의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Joint, Marginal, Conditional Laws and Covariance. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "조건부",
        "en": "Conditional"
      },
      "explanation": {
        "ko": "행을 그 합으로 나눕니다.",
        "en": "Divide the row by its sum."
      },
      "formula": "P(Y=1|X=1)=3/4",
      "diagram": {
        "ko": "결합·주변·조건부 분포와 공분산의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Joint, Marginal, Conditional Laws and Covariance. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "공분산",
        "en": "Covariance"
      },
      "explanation": {
        "ko": "공동 변화와 독립을 구분합니다.",
        "en": "Separate joint variation from independence."
      },
      "formula": "Cov(X,Y)=3/8−1/4=1/8",
      "diagram": {
        "ko": "결합·주변·조건부 분포와 공분산의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Joint, Marginal, Conditional Laws and Covariance. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "양의 공분산",
        "en": "Positive covariance"
      },
      "predict": {
        "ko": "조건부확률은?",
        "en": "What is the conditional probability?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0.75입니다.",
        "en": "0.75."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "독립 결합",
        "en": "Independent joint law"
      },
      "predict": {
        "ko": "공분산은?",
        "en": "What is the covariance?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0이며 독립입니다.",
        "en": "Zero, with independence here."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "비어 있는 조건 행",
        "en": "Empty conditioning row"
      },
      "predict": {
        "ko": "X=1이 없으면?",
        "en": "What if X=1 has zero mass?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "조건부확률과 상관계수는 정의되지 않습니다.",
        "en": "That conditional and the correlation are undefined."
      },
      "preset": 2
    }
  ]
},
"multivariate-gaussians":{
  "intuition": {
    "ko": "공분산 타원·조건부 평균·분산을 연결하고 특이 Gaussian을 구분합니다.",
    "en": "Connect covariance ellipses to conditional means/variances and distinguish singular Gaussians."
  },
  "steps": [
    {
      "title": {
        "ko": "공분산",
        "en": "Covariance"
      },
      "explanation": {
        "ko": "스케일과 상관을 분리합니다.",
        "en": "Separate scales from correlation."
      },
      "formula": "Σ=[1,.5;.5,1]",
      "diagram": {
        "ko": "다변량 Gaussian과 조건부 기하의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Multivariate Gaussians and Conditional Geometry. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "기하",
        "en": "Geometry"
      },
      "explanation": {
        "ko": "단위원을 L로 변환합니다.",
        "en": "Transform the unit circle by L."
      },
      "formula": "Σ=LLᵀ",
      "diagram": {
        "ko": "다변량 Gaussian과 조건부 기하의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Multivariate Gaussians and Conditional Geometry. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "조건부",
        "en": "Conditional"
      },
      "explanation": {
        "ko": "조건부 평균은 선형입니다.",
        "en": "The conditional mean is linear."
      },
      "formula": "E[Y|X=1]=.5, Var=.75",
      "diagram": {
        "ko": "다변량 Gaussian과 조건부 기하의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Multivariate Gaussians and Conditional Geometry. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "퇴화",
        "en": "Degeneracy"
      },
      "explanation": {
        "ko": "확률법칙은 있지만 2차원 밀도는 없습니다.",
        "en": "The law exists but has no two-dimensional density."
      },
      "formula": "ρ=±1 ⇒ det Σ=0",
      "diagram": {
        "ko": "다변량 Gaussian과 조건부 기하의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Multivariate Gaussians and Conditional Geometry. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "상관 Gaussian",
        "en": "Correlated Gaussian"
      },
      "predict": {
        "ko": "조건부 평균은?",
        "en": "What is the conditional mean?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0.5입니다.",
        "en": "0.5."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "독립 Gaussian",
        "en": "Independent Gaussian"
      },
      "predict": {
        "ko": "상관이 0이면?",
        "en": "What if correlation is zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "이 공동 Gaussian에서는 독립입니다.",
        "en": "These jointly Gaussian variables are independent."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "특이 Gaussian",
        "en": "Singular Gaussian"
      },
      "predict": {
        "ko": "ρ=1이면?",
        "en": "What happens at ρ=1?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "조건부 분산 0, 이차원 밀도는 없습니다.",
        "en": "Conditional variance zero; no 2D density."
      },
      "preset": 2
    }
  ]
},
"random-variable-transformations":{
  "intuition": {
    "ko": "역상과 절댓값 Jacobian으로 밀도를 바꾸고 여러 역상을 합산합니다.",
    "en": "Transform densities using preimages and absolute Jacobians, adding multiple branches."
  },
  "steps": [
    {
      "title": {
        "ko": "역상",
        "en": "Preimage"
      },
      "explanation": {
        "ko": "출력 사건을 입력으로 되돌립니다.",
        "en": "Pull the output event back to the input."
      },
      "formula": "P(Y≤y)=P(g(X)≤y)",
      "diagram": {
        "ko": "확률변수 변환과 Jacobian의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Random-Variable Transformations and Jacobians. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "Jacobian",
        "en": "Jacobian"
      },
      "explanation": {
        "ko": "확대하면 같은 질량이 넓게 퍼집니다.",
        "en": "Scaling spreads the same mass over more length."
      },
      "formula": "Y=2X ⇒ fY=1/2",
      "diagram": {
        "ko": "확률변수 변환과 Jacobian의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Random-Variable Transformations and Jacobians. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "여러 분기",
        "en": "Multiple branches"
      },
      "explanation": {
        "ko": "두 역상의 기여를 더합니다.",
        "en": "Add both preimage contributions."
      },
      "formula": "X²=.25 ⇒ X=±.5",
      "diagram": {
        "ko": "확률변수 변환과 Jacobian의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Random-Variable Transformations and Jacobians. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "붕괴",
        "en": "Collapse"
      },
      "explanation": {
        "ko": "역함수 공식 대신 점 질량을 씁니다.",
        "en": "Use a point mass instead of an inverse formula."
      },
      "formula": "a=0 ⇒ P(Y=0)=1",
      "diagram": {
        "ko": "확률변수 변환과 Jacobian의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Random-Variable Transformations and Jacobians. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "두 배 확대",
        "en": "Double the scale"
      },
      "predict": {
        "ko": "밀도와 CDF는?",
        "en": "What are density and CDF?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "둘 다 0.5입니다.",
        "en": "Both are 0.5."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "두 역상 제곱",
        "en": "Two-branch square"
      },
      "predict": {
        "ko": "y=.25의 밀도는?",
        "en": "What is density at y=.25?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "두 분기를 더해 1입니다.",
        "en": "One after adding both branches."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "한 점으로 붕괴",
        "en": "Collapse to one point"
      },
      "predict": {
        "ko": "배율 0이면?",
        "en": "What if scale is zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "점 질량 1이고 Lebesgue 밀도는 없습니다.",
        "en": "Point mass one; no Lebesgue density."
      },
      "preset": 2
    }
  ]
},
"laws-large-numbers-clt":{
  "intuition": {
    "ko": "평균의 집중과 표준화된 요동의 근사를 분리해 읽습니다.",
    "en": "Separate concentration of the mean from approximation of standardized fluctuations."
  },
  "steps": [
    {
      "title": {
        "ko": "가정",
        "en": "Assumptions"
      },
      "explanation": {
        "ko": "독립·동일 분포를 명시합니다.",
        "en": "Assume independent identically distributed trials."
      },
      "formula": "Xi iid Bernoulli(p)",
      "diagram": {
        "ko": "대수의 법칙·중심극한정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Laws of Large Numbers and the Central Limit Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "집중",
        "en": "Concentration"
      },
      "explanation": {
        "ko": "평균의 분산이 줄어듭니다.",
        "en": "The mean variance shrinks."
      },
      "formula": "Var(mean)=p(1−p)/n",
      "diagram": {
        "ko": "대수의 법칙·중심극한정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Laws of Large Numbers and the Central Limit Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "표준화",
        "en": "Standardization"
      },
      "explanation": {
        "ko": "요동을 자체 표준편차 단위로 읽습니다.",
        "en": "Measure fluctuations in their standard-deviation units."
      },
      "formula": "Z=(S−np)/√(np(1−p))",
      "diagram": {
        "ko": "대수의 법칙·중심극한정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Laws of Large Numbers and the Central Limit Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "근사",
        "en": "Approximation"
      },
      "explanation": {
        "ko": "밀도가 아닌 같은 구간의 질량을 비교합니다.",
        "en": "Compare masses in matching bins, not density heights."
      },
      "formula": "P(S=k)≈Φ(z+)-Φ(z−)",
      "diagram": {
        "ko": "대수의 법칙·중심극한정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Laws of Large Numbers and the Central Limit Theorem. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "열 표본",
        "en": "Ten observations"
      },
      "predict": {
        "ko": "표본평균의 분산은?",
        "en": "What is the mean variance?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0.025입니다.",
        "en": "0.025."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "서른 표본",
        "en": "Thirty observations"
      },
      "predict": {
        "ko": "n=30이면 분산은?",
        "en": "What is the variance for n=30?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1/120입니다.",
        "en": "One over 120."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "분산 0",
        "en": "Zero variance"
      },
      "predict": {
        "ko": "p=0이면 표준화는?",
        "en": "What happens to standardization at p=0?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "분산 0으로 정의되지 않습니다.",
        "en": "It is undefined because variance is zero."
      },
      "preset": 2
    }
  ]
},
"reparameterization-gradients":{
  "intuition": {
    "ko": "고정한 잡음을 통해 표본 손실을 미분하고 기대 경사와 구분합니다.",
    "en": "Differentiate a sample loss through fixed noise and distinguish its expected gradient."
  },
  "steps": [
    {
      "title": {
        "ko": "분리",
        "en": "Separate"
      },
      "explanation": {
        "ko": "분포의 매개변수와 기초 잡음을 분리합니다.",
        "en": "Separate distribution parameters from base noise."
      },
      "formula": "z=μ+σε, ε~N(0,1)",
      "diagram": {
        "ko": "재매개화와 확률적 경사의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Reparameterization and Pathwise Gradients. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "표본 경사",
        "en": "Sample gradient"
      },
      "explanation": {
        "ko": "미분 중 ε는 고정합니다.",
        "en": "Hold ε fixed while differentiating."
      },
      "formula": "∂μℓ=z−1, ∂σℓ=(z−1)ε",
      "diagram": {
        "ko": "재매개화와 확률적 경사의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Reparameterization and Pathwise Gradients. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "기대 경사",
        "en": "Expected gradient"
      },
      "explanation": {
        "ko": "한 잡음의 경사와 다릅니다.",
        "en": "It differs from one noise value’s gradient."
      },
      "formula": "∇Eℓ=(μ−1,σ)",
      "diagram": {
        "ko": "재매개화와 확률적 경사의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Reparameterization and Pathwise Gradients. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "검산",
        "en": "Moment check"
      },
      "explanation": {
        "ko": "이차 손실의 첫 두 모멘트를 정확히 맞춥니다.",
        "en": "These nodes exactly match the moments needed for this quadratic."
      },
      "formula": "ε=±1 with equal weights",
      "diagram": {
        "ko": "재매개화와 확률적 경사의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Reparameterization and Pathwise Gradients. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "중앙 잡음",
        "en": "Central noise"
      },
      "predict": {
        "ko": "ε=0의 경사와 기대 경사는?",
        "en": "What are sample and expected gradients at ε=0?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "(−1,0)과 (−1,1)입니다.",
        "en": "(−1,0) and (−1,1)."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "한 표본의 최적값",
        "en": "One sample at target"
      },
      "predict": {
        "ko": "표본 손실 0이면 기대 손실도 0인가요?",
        "en": "Does zero sample loss imply zero expected loss?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 기대 손실은 1입니다.",
        "en": "No: expected loss is one."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "척도 0",
        "en": "Zero scale"
      },
      "predict": {
        "ko": "σ=0이면?",
        "en": "What happens at σ=0?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "확률법칙은 한 점으로 붕괴합니다.",
        "en": "The law collapses to a point."
      },
      "preset": 2
    }
  ]
},

"groups-rings-fields":{
  "intuition": {
    "ko": "덧셈 역원과 곱셈 역원을 구분하고 체가 되는 조건을 실제 표에서 확인합니다.",
    "en": "Distinguish additive and multiplicative inverses and check when residues form a field."
  },
  "steps": [
    {
      "title": {
        "ko": "덧셈군",
        "en": "Additive group"
      },
      "explanation": {
        "ko": "덧셈 역원은 모든 원소에 있습니다.",
        "en": "Every element has an additive inverse."
      },
      "formula": "a+(−a)=0",
      "diagram": {
        "ko": "군·환·체와 모듈러 연산의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Groups, Rings, Fields and Modular Arithmetic. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "환",
        "en": "Ring"
      },
      "explanation": {
        "ko": "곱셈과 분배법칙을 추가합니다.",
        "en": "Add multiplication and distributivity."
      },
      "formula": "a(b+c)=ab+ac",
      "diagram": {
        "ko": "군·환·체와 모듈러 연산의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Groups, Rings, Fields and Modular Arithmetic. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "단원",
        "en": "Unit"
      },
      "explanation": {
        "ko": "0이 아닌 원소의 역원을 검사합니다.",
        "en": "Check inverses of nonzero elements."
      },
      "formula": "2·3=1 mod 5",
      "diagram": {
        "ko": "군·환·체와 모듈러 연산의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Groups, Rings, Fields and Modular Arithmetic. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "체",
        "en": "Field"
      },
      "explanation": {
        "ko": "합성수에서는 영인자가 생깁니다.",
        "en": "Composite moduli introduce zero divisors."
      },
      "formula": "Z/pZ is a field ⇔ p prime",
      "diagram": {
        "ko": "군·환·체와 모듈러 연산의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Groups, Rings, Fields and Modular Arithmetic. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "소수 계수",
        "en": "Prime modulus"
      },
      "predict": {
        "ko": "2의 곱셈 역원은?",
        "en": "What is the inverse of two?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "3입니다.",
        "en": "Three."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "영인자",
        "en": "Zero divisor"
      },
      "predict": {
        "ko": "법 6에서도 2를 나눌 수 있나요?",
        "en": "Can you divide by two modulo six?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 2·3=0이고 역원이 없습니다.",
        "en": "No: 2·3=0 and two has no inverse."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "영 원소",
        "en": "Zero element"
      },
      "predict": {
        "ko": "0의 곱셈 역원은?",
        "en": "What is the inverse of zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "정의되지 않습니다.",
        "en": "Undefined."
      },
      "preset": 2
    }
  ]
},
"homomorphisms-isomorphisms":{
  "intuition": {
    "ko": "공식이 몫집합 위에서 잘 정의되는지 확인한 뒤 핵·상·동형을 판단합니다.",
    "en": "Check representative independence before discussing kernels, images or isomorphisms."
  },
  "steps": [
    {
      "title": {
        "ko": "대표원",
        "en": "Representative"
      },
      "explanation": {
        "ko": "같은 잉여류의 대표원이 같은 상으로 가야 합니다.",
        "en": "Equivalent representatives must have the same image."
      },
      "formula": "f([x])=[kx]",
      "diagram": {
        "ko": "준동형·동형과 구조 보존의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Homomorphisms, Isomorphisms and Structure. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "조건",
        "en": "Condition"
      },
      "explanation": {
        "ko": "x와 x+m의 상 차이는 km입니다.",
        "en": "The images of x and x+m differ by km."
      },
      "formula": "n divides km",
      "diagram": {
        "ko": "준동형·동형과 구조 보존의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Homomorphisms, Isomorphisms and Structure. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "핵과 상",
        "en": "Kernel and image"
      },
      "explanation": {
        "ko": "영으로 가는 원소와 도달 원소를 구분합니다.",
        "en": "Separate elements mapped to zero from reached elements."
      },
      "formula": "ker f={0,3}, im f={0,1,2}",
      "diagram": {
        "ko": "준동형·동형과 구조 보존의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Homomorphisms, Isomorphisms and Structure. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "동형",
        "en": "Isomorphism"
      },
      "explanation": {
        "ko": "구조 보존과 전단사가 모두 필요합니다.",
        "en": "Both structure preservation and bijectivity are required."
      },
      "formula": "bijective + homomorphism",
      "diagram": {
        "ko": "준동형·동형과 구조 보존의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Homomorphisms, Isomorphisms and Structure. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "자연스러운 사상",
        "en": "Natural map"
      },
      "predict": {
        "ko": "핵의 크기는?",
        "en": "What is the kernel size?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2입니다.",
        "en": "Two."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "대표원 불일치",
        "en": "Inconsistent representatives"
      },
      "predict": {
        "ko": "공식만 있으면 사상인가요?",
        "en": "Does a formula guarantee a map?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 0과 3의 상이 다릅니다.",
        "en": "No: zero and three have different images."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "동형",
        "en": "Isomorphism"
      },
      "predict": {
        "ko": "법 5의 두 배 사상은?",
        "en": "What about doubling modulo five?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "덧셈군 동형입니다.",
        "en": "An additive-group isomorphism."
      },
      "preset": 2
    }
  ]
},
"subgroups-quotients":{
  "intuition": {
    "ko": "원소를 부분군만큼 다른 동치류로 묶고 몫 연산이 대표원과 무관한지 확인합니다.",
    "en": "Group elements into cosets and establish representative-independent quotient operations."
  },
  "steps": [
    {
      "title": {
        "ko": "생성",
        "en": "Generate"
      },
      "explanation": {
        "ko": "반복 덧셈으로 닫힌 부분집합을 만듭니다.",
        "en": "Repeated addition generates a closed subset."
      },
      "formula": "⟨2⟩={0,2,4} in Z6",
      "diagram": {
        "ko": "부분군·잉여류와 몫구조의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Subgroups, Cosets and Quotients. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "동치",
        "en": "Equivalence"
      },
      "explanation": {
        "ko": "차이가 부분군에 속하면 같은 류입니다.",
        "en": "A difference in H identifies the same coset."
      },
      "formula": "x~y ⇔ x−y∈H",
      "diagram": {
        "ko": "부분군·잉여류와 몫구조의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Subgroups, Cosets and Quotients. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "분할",
        "en": "Partition"
      },
      "explanation": {
        "ko": "잉여류는 겹치지 않거나 같습니다.",
        "en": "Cosets are disjoint or identical."
      },
      "formula": "G/H={H,1+H}",
      "diagram": {
        "ko": "부분군·잉여류와 몫구조의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Subgroups, Cosets and Quotients. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "크기",
        "en": "Cardinality"
      },
      "explanation": {
        "ko": "부분군 크기와 몫 크기를 곱합니다.",
        "en": "Multiply subgroup and quotient sizes."
      },
      "formula": "3·2=6",
      "diagram": {
        "ko": "부분군·잉여류와 몫구조의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Subgroups, Cosets and Quotients. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "짝수 부분군",
        "en": "Even subgroup"
      },
      "predict": {
        "ko": "몫 원소는 몇 개인가요?",
        "en": "How many quotient elements?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2개입니다.",
        "en": "Two."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "영 부분군",
        "en": "Zero subgroup"
      },
      "predict": {
        "ko": "H={0}이면?",
        "en": "What if H={0}?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "몫에 6개 원소가 남습니다.",
        "en": "Six quotient elements remain."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "전체 부분군",
        "en": "Whole subgroup"
      },
      "predict": {
        "ko": "H=G이면?",
        "en": "What if H=G?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "몫은 한 원소입니다.",
        "en": "The quotient has one element."
      },
      "preset": 2
    }
  ]
},
"polynomials-finite-fields":{
  "intuition": {
    "ko": "다항식 계수를 비트로 표현하고 기약 다항식으로 나눌 때 체가 되는 이유를 확인합니다.",
    "en": "Encode polynomial coefficients as bits and contrast irreducible and reducible quotients."
  },
  "steps": [
    {
      "title": {
        "ko": "계수",
        "en": "Coefficients"
      },
      "explanation": {
        "ko": "정수 비트는 F2 계수입니다.",
        "en": "Integer bits encode coefficients in F2."
      },
      "formula": "2=t, 4=t²",
      "diagram": {
        "ko": "다항식 몫환과 유한체의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Polynomial Quotients and Finite Fields. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "곱셈",
        "en": "Multiplication"
      },
      "explanation": {
        "ko": "다항식 곱셈 후 나머지를 구합니다.",
        "en": "Multiply polynomials and reduce the remainder."
      },
      "formula": "t·t²=t³",
      "diagram": {
        "ko": "다항식 몫환과 유한체의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Polynomial Quotients and Finite Fields. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "축약",
        "en": "Reduction"
      },
      "explanation": {
        "ko": "기준 곱은 비트 3입니다.",
        "en": "The default product has bit encoding three."
      },
      "formula": "t³=t+1 mod (t³+t+1)",
      "diagram": {
        "ko": "다항식 몫환과 유한체의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Polynomial Quotients and Finite Fields. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "기약성",
        "en": "Irreducibility"
      },
      "explanation": {
        "ko": "가약 법에서는 영인자가 생깁니다.",
        "en": "A reducible modulus creates zero divisors."
      },
      "formula": "t³+t=t(t+1)²",
      "diagram": {
        "ko": "다항식 몫환과 유한체의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Polynomial Quotients and Finite Fields. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "기약 다항식",
        "en": "Irreducible modulus"
      },
      "predict": {
        "ko": "t 곱하기 t²는?",
        "en": "What is t times t²?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "기준 체에서 t+1, 비트 3입니다.",
        "en": "t+1, encoded three, in the default field."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "가약 다항식",
        "en": "Reducible modulus"
      },
      "predict": {
        "ko": "가약 법에서 곱은?",
        "en": "What is the product with the reducible modulus?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "t(t²+1)=0입니다.",
        "en": "t(t²+1)=0."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "영 다항식",
        "en": "Zero polynomial"
      },
      "predict": {
        "ko": "0을 나눌 수 있나요?",
        "en": "Can zero be inverted?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "없습니다.",
        "en": "No."
      },
      "preset": 2
    }
  ]
},
"modules-linear-actions":{
  "intuition": {
    "ko": "체 위 벡터공간의 성질을 일반 환에 옮길 때 어떤 나눗셈이 실패하는지 살펴봅니다.",
    "en": "See which vector-space arguments fail when scalars form a ring rather than a field."
  },
  "steps": [
    {
      "title": {
        "ko": "스칼라",
        "en": "Scalars"
      },
      "explanation": {
        "ko": "환이 가법군에 작용합니다.",
        "en": "A ring acts on an additive group."
      },
      "formula": "M=(Z6)²",
      "diagram": {
        "ko": "모듈과 환 위의 선형사상의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Modules and Linear Maps over Rings. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "사상",
        "en": "Map"
      },
      "explanation": {
        "ko": "각 좌표에 스칼라를 곱합니다.",
        "en": "Multiply each coordinate by a scalar."
      },
      "formula": "T(x,y)=(2x,3y)",
      "diagram": {
        "ko": "모듈과 환 위의 선형사상의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Modules and Linear Maps over Rings. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "핵",
        "en": "Kernel"
      },
      "explanation": {
        "ko": "영인자 때문에 비영 입력이 사라집니다.",
        "en": "Zero divisors can annihilate nonzero inputs."
      },
      "formula": "2x=0, 3y=0 mod 6",
      "diagram": {
        "ko": "모듈과 환 위의 선형사상의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Modules and Linear Maps over Rings. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "범위",
        "en": "Scope"
      },
      "explanation": {
        "ko": "개수와 벡터공간 차원을 구분합니다.",
        "en": "Distinguish cardinality from vector-space dimension."
      },
      "formula": "|ker T|=6, |im T|=6",
      "diagram": {
        "ko": "모듈과 환 위의 선형사상의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Modules and Linear Maps over Rings. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "환의 영인자",
        "en": "Ring zero divisors"
      },
      "predict": {
        "ko": "핵에 몇 원소가 있나요?",
        "en": "How many kernel elements?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "6개입니다.",
        "en": "Six."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "체 위 선형사상",
        "en": "Linear map over a field"
      },
      "predict": {
        "ko": "법 5로 바꾸면?",
        "en": "What changes modulo five?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "두 계수가 단원이므로 가역입니다.",
        "en": "Both coefficients are units, so the map is invertible."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "영 사상",
        "en": "Zero map"
      },
      "predict": {
        "ko": "영 사상의 핵 크기는?",
        "en": "What is the kernel size of the zero map?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "25입니다.",
        "en": "Twenty-five."
      },
      "preset": 2
    }
  ]
},
"algebraic-symmetries":{
  "intuition": {
    "ko": "회전·반사 합성의 순서를 정하고 궤도와 안정자를 계산합니다.",
    "en": "Fix composition order for rotations/reflections and compute orbits and stabilizers."
  },
  "steps": [
    {
      "title": {
        "ko": "작용",
        "en": "Action"
      },
      "explanation": {
        "ko": "변환을 점에 적용합니다.",
        "en": "Apply transformations to points."
      },
      "formula": "R(x,y)=(-y,x), S(x,y)=(x,-y)",
      "diagram": {
        "ko": "대수적 대칭과 군 작용의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Algebraic Symmetries and Group Actions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "순서",
        "en": "Order"
      },
      "explanation": {
        "ko": "곱의 오른쪽부터 적용합니다.",
        "en": "Apply the rightmost factor first."
      },
      "formula": "RʳSᶠ: S first",
      "diagram": {
        "ko": "대수적 대칭과 군 작용의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Algebraic Symmetries and Group Actions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "궤도",
        "en": "Orbit"
      },
      "explanation": {
        "ko": "모든 군 원소의 도달점을 모읍니다.",
        "en": "Collect images under all group elements."
      },
      "formula": "Orb(1,1)={(±1,±1)}",
      "diagram": {
        "ko": "대수적 대칭과 군 작용의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Algebraic Symmetries and Group Actions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "안정자",
        "en": "Stabilizer"
      },
      "explanation": {
        "ko": "제자리에 두는 변환을 셉니다.",
        "en": "Count transformations fixing the point."
      },
      "formula": "|Orb(v)|·|Stab(v)|=8",
      "diagram": {
        "ko": "대수적 대칭과 군 작용의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Algebraic Symmetries and Group Actions. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "꼭짓점",
        "en": "Corner"
      },
      "predict": {
        "ko": "꼭짓점의 궤도 크기는?",
        "en": "What is the corner orbit size?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "4입니다.",
        "en": "Four."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "축 위 점",
        "en": "Axis point"
      },
      "predict": {
        "ko": "축 위 점의 안정자 크기는?",
        "en": "What is the axis-point stabilizer size?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2입니다.",
        "en": "Two."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "중심",
        "en": "Center"
      },
      "predict": {
        "ko": "중심은 어떻게 되나요?",
        "en": "What happens to the center?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "궤도 1, 안정자 8입니다.",
        "en": "Orbit one, stabilizer eight."
      },
      "preset": 2
    }
  ]
},
"polynomial-systems":{
  "intuition": {
    "ko": "계수체를 먼저 고르고 유한 격자의 모든 해와 잔차를 확인합니다.",
    "en": "Choose the coefficient field first, then enumerate solutions and residuals on a finite grid."
  },
  "steps": [
    {
      "title": {
        "ko": "제약",
        "en": "Constraints"
      },
      "explanation": {
        "ko": "두 조건을 동시에 만족해야 합니다.",
        "en": "Both constraints must hold."
      },
      "formula": "x+y=3, xy=2 over F5",
      "diagram": {
        "ko": "유한체 위 다항식 시스템의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Polynomial Systems over Finite Fields. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "소거",
        "en": "Eliminate"
      },
      "explanation": {
        "ko": "y=s−x를 대입합니다.",
        "en": "Substitute y=s−x."
      },
      "formula": "x²−3x+2=0",
      "diagram": {
        "ko": "유한체 위 다항식 시스템의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Polynomial Systems over Finite Fields. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "해",
        "en": "Solutions"
      },
      "explanation": {
        "ko": "순서쌍을 구분합니다.",
        "en": "Keep ordered pairs distinct."
      },
      "formula": "(x,y)=(1,2),(2,1)",
      "diagram": {
        "ko": "유한체 위 다항식 시스템의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Polynomial Systems over Finite Fields. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "계수체",
        "en": "Coefficient field"
      },
      "explanation": {
        "ko": "실수의 부호 판정과 다릅니다.",
        "en": "This is not the real discriminant sign test."
      },
      "formula": "Δ=s²−4q mod p",
      "diagram": {
        "ko": "유한체 위 다항식 시스템의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Polynomial Systems over Finite Fields. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "두 순서 해",
        "en": "Two ordered solutions"
      },
      "predict": {
        "ko": "해는 몇 개인가요?",
        "en": "How many solutions?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "순서쌍 두 개입니다.",
        "en": "Two ordered pairs."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "중근",
        "en": "Repeated root"
      },
      "predict": {
        "ko": "중근이면?",
        "en": "What if the root repeats?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "(1,1) 한 순서쌍입니다.",
        "en": "One ordered pair: (1,1)."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "해 없음",
        "en": "No solution"
      },
      "predict": {
        "ko": "법 5에서 x²=3이면?",
        "en": "Does x²=3 have a root modulo five?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "제곱 잉여가 아니라 해가 없습니다.",
        "en": "No: three is not a square residue."
      },
      "preset": 2
    }
  ]
},
"invariant-constructions":{
  "intuition": {
    "ko": "궤도에서 특징을 평균해 불변량을 만들고 그 과정에서 사라지는 정보를 확인합니다.",
    "en": "Average features over an orbit and inspect the information that averaging removes."
  },
  "steps": [
    {
      "title": {
        "ko": "궤도",
        "en": "Orbit"
      },
      "explanation": {
        "ko": "네 회전을 적용합니다.",
        "en": "Apply all four rotations."
      },
      "formula": "(1,2),(-2,1),(-1,-2),(2,-1)",
      "diagram": {
        "ko": "군 평균으로 불변량 구성하기의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Constructing Invariants by Group Averaging. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "특징",
        "en": "Feature"
      },
      "explanation": {
        "ko": "좌표 대신 특징 값을 읽습니다.",
        "en": "Evaluate the feature rather than the coordinates."
      },
      "formula": "φ(x,y)=x²",
      "diagram": {
        "ko": "군 평균으로 불변량 구성하기의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Constructing Invariants by Group Averaging. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "평균",
        "en": "Average"
      },
      "explanation": {
        "ko": "입력을 회전해도 같은 합입니다.",
        "en": "Rotating the input permutes the sum."
      },
      "formula": "Pφ=(1+4+1+4)/4=2.5",
      "diagram": {
        "ko": "군 평균으로 불변량 구성하기의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Constructing Invariants by Group Averaging. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "정보",
        "en": "Information"
      },
      "explanation": {
        "ko": "불변화가 정보를 지울 수 있습니다.",
        "en": "Invariantization can erase information."
      },
      "formula": "P(x)=P(xy)=0",
      "diagram": {
        "ko": "군 평균으로 불변량 구성하기의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Constructing Invariants by Group Averaging. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "제곱 특징",
        "en": "Squared feature"
      },
      "predict": {
        "ko": "제곱 특징의 평균은?",
        "en": "What is the averaged squared feature?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2.5입니다.",
        "en": "2.5."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "선형 특징",
        "en": "Linear feature"
      },
      "predict": {
        "ko": "x의 평균은?",
        "en": "What is the mean of x?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0입니다.",
        "en": "Zero."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "교차 특징",
        "en": "Cross feature"
      },
      "predict": {
        "ko": "xy의 평균은?",
        "en": "What is the mean of xy?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0입니다.",
        "en": "Zero."
      },
      "preset": 2
    }
  ]
},
"algebraic-representations-scope":{
  "intuition": {
    "ko": "대칭을 행렬로 표현하고 군에 따라 달라지는 등변 선형층의 제약을 계산합니다.",
    "en": "Represent symmetry by matrices and compute how equivariant-layer constraints depend on the group."
  },
  "steps": [
    {
      "title": {
        "ko": "표현",
        "en": "Representation"
      },
      "explanation": {
        "ko": "군 합성을 행렬곱으로 보존합니다.",
        "en": "Preserve group composition with matrix multiplication."
      },
      "formula": "ρ(gh)=ρ(g)ρ(h)",
      "diagram": {
        "ko": "대수적 표현·등변성과 적용 범위의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Representations, Equivariance and Their Scope. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "등변",
        "en": "Equivariance"
      },
      "explanation": {
        "ko": "선형층과 회전이 교환해야 합니다.",
        "en": "The layer must commute with rotation."
      },
      "formula": "MJ=JM",
      "diagram": {
        "ko": "대수적 표현·등변성과 적용 범위의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Representations, Equivariance and Their Scope. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "사영",
        "en": "Projection"
      },
      "explanation": {
        "ko": "켤레 작용을 평균합니다.",
        "en": "Average conjugation actions."
      },
      "formula": "P(M)=¼Σ J⁻ᵏMJᵏ",
      "diagram": {
        "ko": "대수적 표현·등변성과 적용 범위의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Representations, Equivariance and Their Scope. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "추가 대칭",
        "en": "Additional symmetry"
      },
      "explanation": {
        "ko": "회전 조건만으로 반사 등변이 되지 않습니다.",
        "en": "Rotation equivariance does not imply reflection equivariance."
      },
      "formula": "MS=SM",
      "diagram": {
        "ko": "대수적 표현·등변성과 적용 범위의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Representations, Equivariance and Their Scope. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "일반 행렬",
        "en": "General matrix"
      },
      "predict": {
        "ko": "사영 후 회전 잔차는?",
        "en": "What is the rotation residual after projection?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0입니다.",
        "en": "Zero."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "회전 등변",
        "en": "Rotation equivariant"
      },
      "predict": {
        "ko": "반사도 보존하나요?",
        "en": "Does it preserve reflections too?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "이 예제에서는 아닙니다.",
        "en": "Not in this example."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "모든 대칭 등변",
        "en": "All square symmetries"
      },
      "predict": {
        "ko": "항등행렬은?",
        "en": "What about the identity matrix?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "회전과 반사 모두 교환합니다.",
        "en": "It commutes with rotations and reflections."
      },
      "preset": 2
    }
  ]
},

"tensor-order-shapes":{
  "intuition": {
    "ko": "배열 차수와 행렬 랭크를 구분하고 축 순서를 바꾸면 성분 대응이 어떻게 달라지는지 확인합니다.",
    "en": "Separate array order from matrix rank and track entries when axes are permuted."
  },
  "steps": [
    {
      "title": {
        "ko": "축의 의미",
        "en": "Axis meaning"
      },
      "explanation": {
        "ko": "기준 shape는 배치 1, 길이 2, 채널 3입니다.",
        "en": "The reference shape has batch one, length two and three channels."
      },
      "formula": "shape=(1,2,3)",
      "diagram": {
        "ko": "텐서의 차수·축·shape의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Order, Axes and Shapes. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "원소 수",
        "en": "Entry count"
      },
      "explanation": {
        "ko": "크기를 곱하면 원소 수입니다.",
        "en": "Multiply axis sizes for entry count."
      },
      "formula": "1·2·3=6",
      "diagram": {
        "ko": "텐서의 차수·축·shape의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Order, Axes and Shapes. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "좌표와 값",
        "en": "Coordinates and values"
      },
      "explanation": {
        "ko": "X[0,1,2]=12입니다.",
        "en": "X[0,1,2]=12."
      },
      "formula": "X[b,t,c]=100b+10t+c",
      "diagram": {
        "ko": "텐서의 차수·축·shape의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Order, Axes and Shapes. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "순열",
        "en": "Permutation"
      },
      "explanation": {
        "ko": "축 교환은 같은 값을 새 좌표로 옮깁니다.",
        "en": "Permutation reassigns the same values to new coordinates."
      },
      "formula": "X̃[b,c,t]=X[b,t,c]",
      "diagram": {
        "ko": "텐서의 차수·축·shape의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Order, Axes and Shapes. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "원래 축",
        "en": "Original axes"
      },
      "predict": {
        "ko": "차수와 원소 수는?",
        "en": "What are order and entry count?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "3과 6입니다.",
        "en": "Three and six."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "축 교환",
        "en": "Swap axes"
      },
      "predict": {
        "ko": "축 교환 뒤 평탄 값은?",
        "en": "What are the flattened values after swapping?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "(0,10,1,11,2,12)입니다.",
        "en": "They are (0,10,1,11,2,12)."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "배치 추가",
        "en": "Add a batch"
      },
      "predict": {
        "ko": "배치를 두 개로 하면?",
        "en": "What changes with two batches?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "차수는 3으로 같고 원소는 12개입니다.",
        "en": "Order stays three; entry count becomes twelve."
      },
      "preset": 2
    }
  ]
},
"tensor-indices-contractions":{
  "intuition": {
    "ko": "합산할 지표와 남길 지표를 먼저 정한 뒤 출력 shape와 계산값을 확인합니다.",
    "en": "Name summed and free indices before checking output shape and values."
  },
  "steps": [
    {
      "title": {
        "ko": "입력",
        "en": "Input"
      },
      "explanation": {
        "ko": "X의 shape는 (2,2,2)입니다.",
        "en": "X has shape (2,2,2)."
      },
      "formula": "Xᵢⱼₖ=1+4i+2j+k",
      "diagram": {
        "ko": "지표·자유축과 수축의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Indices, Free Axes and Contractions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "합산축",
        "en": "Summed axis"
      },
      "explanation": {
        "ko": "k를 길이 2 벡터와 짝지어 합합니다.",
        "en": "Pair and sum k with a length-two covector."
      },
      "formula": "v=(1,2)",
      "diagram": {
        "ko": "지표·자유축과 수축의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Indices, Free Axes and Contractions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "한 원소",
        "en": "One entry"
      },
      "explanation": {
        "ko": "Y₀₀=1·1+2·2=5입니다.",
        "en": "Y₀₀=1·1+2·2=5."
      },
      "formula": "Y₀₀=5",
      "diagram": {
        "ko": "지표·자유축과 수축의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Indices, Free Axes and Contractions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "자유축",
        "en": "Free axes"
      },
      "explanation": {
        "ko": "i,j가 남아 shape (2,2)가 됩니다.",
        "en": "i,j remain, giving shape (2,2)."
      },
      "formula": "Y=(5,11,17,23) row-major",
      "diagram": {
        "ko": "지표·자유축과 수축의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Indices, Free Axes and Contractions. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "마지막 축 수축",
        "en": "Contract the last axis"
      },
      "predict": {
        "ko": "출력 네 값은?",
        "en": "What are the four outputs?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "5,11,17,23입니다.",
        "en": "5,11,17,23."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "첫 축 수축",
        "en": "Contract the first axis"
      },
      "predict": {
        "ko": "첫 축을 합하면 같은 결과인가요?",
        "en": "Does summing the first axis give the same result?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 11,14,17,20입니다.",
        "en": "No: 11,14,17,20."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "영 공변벡터",
        "en": "Zero covector"
      },
      "predict": {
        "ko": "영벡터를 수축하면?",
        "en": "What happens with a zero covector?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "shape는 유지되고 모든 출력은 0입니다.",
        "en": "The output shape stays fixed and all values are zero."
      },
      "preset": 2
    }
  ]
},
"multilinear-maps":{
  "intuition": {
    "ko": "각 입력을 따로 고정했을 때의 선형성과 모든 입력을 함께 묶은 선형성을 구분합니다.",
    "en": "Separate linearity in each slot from joint linearity in concatenated inputs."
  },
  "steps": [
    {
      "title": {
        "ko": "계수",
        "en": "Coefficients"
      },
      "explanation": {
        "ko": "두 입력을 텐서 계수로 결합합니다.",
        "en": "Combine two inputs with tensor coefficients."
      },
      "formula": "Xᵢⱼₖ=1+4i+2j+k",
      "diagram": {
        "ko": "다중선형사상과 텐서 계수의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Multilinear Maps and Tensor Coefficients. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "기저 선택",
        "en": "Basis selection"
      },
      "explanation": {
        "ko": "u=e₀,v=e₁이면 계수 X₀₁ₖ를 고릅니다.",
        "en": "Basis inputs select X₀₁ₖ."
      },
      "formula": "F(e₀,e₁)=(3,4)",
      "diagram": {
        "ko": "다중선형사상과 텐서 계수의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Multilinear Maps and Tensor Coefficients. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "슬롯 선형성",
        "en": "Slotwise linearity"
      },
      "explanation": {
        "ko": "u만 두 배로 하면 (6,8)입니다.",
        "en": "Doubling only u gives (6,8)."
      },
      "formula": "F(2u,v)=2F(u,v)",
      "diagram": {
        "ko": "다중선형사상과 텐서 계수의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Multilinear Maps and Tensor Coefficients. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "동시 확대",
        "en": "Simultaneous scaling"
      },
      "explanation": {
        "ko": "두 슬롯을 두 배로 하면 네 배입니다.",
        "en": "Doubling both slots gives four times the output."
      },
      "formula": "F(2u,2v)=4F(u,v)",
      "diagram": {
        "ko": "다중선형사상과 텐서 계수의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Multilinear Maps and Tensor Coefficients. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "기저 입력",
        "en": "Basis inputs"
      },
      "predict": {
        "ko": "출력은?",
        "en": "What is the output?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "(3,4)입니다.",
        "en": "(3,4)."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "모든 성분 1",
        "en": "All-one inputs"
      },
      "predict": {
        "ko": "둘 다 (1,1)이면?",
        "en": "What if both inputs are (1,1)?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "(16,20)입니다.",
        "en": "(16,20)."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "한 슬롯 0",
        "en": "One zero slot"
      },
      "predict": {
        "ko": "한 슬롯이 0이면?",
        "en": "What if one slot is zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "출력은 (0,0)입니다.",
        "en": "The output is (0,0)."
      },
      "preset": 2
    }
  ]
},
"broadcasting-semantics":{
  "intuition": {
    "ko": "shape 호환성과 계산의 뜻을 따로 확인하고 원소별 곱을 행렬곱으로 읽지 않습니다.",
    "en": "Check both shape compatibility and operation meaning; elementwise multiplication is not matrix multiplication."
  },
  "steps": [
    {
      "title": {
        "ko": "입력 shape",
        "en": "Input shapes"
      },
      "explanation": {
        "ko": "(2,3)과 (3,)을 맞춥니다.",
        "en": "Align (2,3) with (3,)."
      },
      "formula": "(2,3)+(1,3)",
      "diagram": {
        "ko": "Broadcasting과 연산의 의미의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Broadcasting and Operation Semantics. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "확장",
        "en": "Expansion"
      },
      "explanation": {
        "ko": "빠진 선두 축은 크기 1처럼 취급합니다.",
        "en": "A missing leading axis behaves like size one."
      },
      "formula": "B[j] used for every i",
      "diagram": {
        "ko": "Broadcasting과 연산의 의미의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Broadcasting and Operation Semantics. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "결과",
        "en": "Result"
      },
      "explanation": {
        "ko": "출력 shape는 (2,3)입니다.",
        "en": "The output shape is (2,3)."
      },
      "formula": "output=(2,4,6,5,7,9)",
      "diagram": {
        "ko": "Broadcasting과 연산의 의미의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Broadcasting and Operation Semantics. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "의미",
        "en": "Semantics"
      },
      "explanation": {
        "ko": "이 규칙은 합산 지표를 새로 만들지 않습니다.",
        "en": "Broadcasting does not introduce a contraction sum."
      },
      "formula": "elementwise × ≠ matrix product",
      "diagram": {
        "ko": "Broadcasting과 연산의 의미의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Broadcasting and Operation Semantics. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "채널 편향",
        "en": "Channel bias"
      },
      "predict": {
        "ko": "채널 편향 뒤 값은?",
        "en": "What are channel-bias outputs?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2,4,6,5,7,9입니다.",
        "en": "2,4,6,5,7,9."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "바깥 곱 모양",
        "en": "Outer-product layout"
      },
      "predict": {
        "ko": "(2,1)과 (1,3)의 곱은?",
        "en": "What does multiplying (2,1) by (1,3) produce?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "바깥 곱 모양의 (2,3) 배열입니다.",
        "en": "A (2,3) outer-product layout."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "호환 불가",
        "en": "Incompatible shapes"
      },
      "predict": {
        "ko": "(2,3)과 (2,)는 호환되나요?",
        "en": "Are (2,3) and (2,) compatible?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "마지막 축 3과 2가 달라 불가합니다.",
        "en": "No: the trailing dimensions three and two conflict."
      },
      "preset": 2
    }
  ]
},
"tensor-cp-tucker":{
  "intuition": {
    "ko": "rank-one 항의 합과 작은 core의 다중선형 변환을 실제 원소별로 복원합니다.",
    "en": "Reconstruct entries from rank-one sums and from a core with mode factors."
  },
  "steps": [
    {
      "title": {
        "ko": "원본",
        "en": "Original"
      },
      "explanation": {
        "ko": "비영 성분은 X₀₀₀=2, X₁₁₁=1입니다.",
        "en": "Only X₀₀₀=2 and X₁₁₁=1 are nonzero."
      },
      "formula": "X=2e₀⊗³+e₁⊗³",
      "diagram": {
        "ko": "CP·Tucker 분해와 복원의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for CP and Tucker Decompositions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "CP",
        "en": "CP"
      },
      "explanation": {
        "ko": "두 rank-one 항의 합으로 정확히 복원합니다.",
        "en": "Two rank-one terms reconstruct it exactly."
      },
      "formula": "CP rank=2",
      "diagram": {
        "ko": "CP·Tucker 분해와 복원의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for CP and Tucker Decompositions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "Tucker",
        "en": "Tucker"
      },
      "explanation": {
        "ko": "좌표 인자와 대각 core로 같은 값을 만듭니다.",
        "en": "Coordinate factors and a diagonal core give the same entries."
      },
      "formula": "G ×₀U₀ ×₁U₁ ×₂U₂",
      "diagram": {
        "ko": "CP·Tucker 분해와 복원의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for CP and Tucker Decompositions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "비용",
        "en": "Cost"
      },
      "explanation": {
        "ko": "작은 예제의 완전 Tucker는 20개를 저장합니다.",
        "en": "A full Tucker form stores twenty entries here."
      },
      "formula": "8 core + 12 factors =20",
      "diagram": {
        "ko": "CP·Tucker 분해와 복원의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for CP and Tucker Decompositions. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "정확한 두 항",
        "en": "Exact two-term model"
      },
      "predict": {
        "ko": "두 표현이 같은가요?",
        "en": "Do both representations agree?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "복원 차이는 0입니다.",
        "en": "The reconstruction difference is zero."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "한 항 근사",
        "en": "One-term approximation"
      },
      "predict": {
        "ko": "한 항만 남기면 오차는?",
        "en": "What error remains with one term?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "작은 가중치 1을 버려 오차 1입니다.",
        "en": "Dropping weight one gives error one."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "항 없음",
        "en": "No retained terms"
      },
      "predict": {
        "ko": "아무 항도 남기지 않으면?",
        "en": "What if no terms are kept?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "영 근사이며 출력 shape는 (2,2,2)입니다.",
        "en": "The approximation is zero with output shape (2,2,2)."
      },
      "preset": 2
    }
  ]
},
"tensor-trains":{
  "intuition": {
    "ko": "연결된 작은 core를 합산해 텐서를 복원하고 gauge 변화와 실제 랭크를 구분합니다.",
    "en": "Contract linked cores and separate gauge choices from actual unfolding rank."
  },
  "steps": [
    {
      "title": {
        "ko": "Core shape",
        "en": "Core shapes"
      },
      "explanation": {
        "ko": "경계 차원 1과 내부 차원 2입니다.",
        "en": "Boundary dimensions are one; internal dimensions are two."
      },
      "formula": "(1,2,2),(2,2,2),(2,2,1)",
      "diagram": {
        "ko": "Tensor Train과 bond 차원의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Trains and Bond Dimensions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "합산",
        "en": "Contraction"
      },
      "explanation": {
        "ko": "내부 지표 두 개를 합합니다.",
        "en": "Sum the two internal indices."
      },
      "formula": "r₀=r₃=1",
      "diagram": {
        "ko": "Tensor Train과 bond 차원의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Trains and Bond Dimensions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "Gauge",
        "en": "Gauge"
      },
      "explanation": {
        "ko": "한 core의 g와 이웃의 1/g가 상쇄합니다.",
        "en": "A factor g cancels against 1/g in its neighbor."
      },
      "formula": "G₀M · M⁻¹G₁ unchanged",
      "diagram": {
        "ko": "Tensor Train과 bond 차원의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Trains and Bond Dimensions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "랭크 제한",
        "en": "Rank restriction"
      },
      "explanation": {
        "ko": "bond 1은 두 독립 성분을 모두 보존하지 못합니다.",
        "en": "Bond one cannot retain both independent components."
      },
      "formula": "truncation error=|t|",
      "diagram": {
        "ko": "Tensor Train과 bond 차원의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Trains and Bond Dimensions. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "정확한 TT",
        "en": "Exact TT"
      },
      "predict": {
        "ko": "정확한 복원 오차는?",
        "en": "What is the reconstruction error?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0입니다.",
        "en": "Zero."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "Gauge 변경",
        "en": "Gauge change"
      },
      "predict": {
        "ko": "Gauge를 두 배로 하면 텐서도 바뀌나요?",
        "en": "Does doubling the gauge change the tensor?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "이웃 역배율 때문에 변하지 않습니다.",
        "en": "No; the inverse neighboring factor cancels."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "Bond 제한",
        "en": "Restricted bond"
      },
      "predict": {
        "ko": "bond 1의 오차는?",
        "en": "What is the bond-one error?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "둘째 성분 t=1을 버려 1입니다.",
        "en": "Dropping the second component gives one."
      },
      "preset": 2
    }
  ]
},
"tensor-networks-contraction":{
  "intuition": {
    "ko": "같은 최종 값도 중간 텐서의 크기와 곱셈 수는 달라질 수 있습니다.",
    "en": "The same final value can require different intermediate sizes and multiplication counts."
  },
  "steps": [
    {
      "title": {
        "ko": "연결",
        "en": "Connections"
      },
      "explanation": {
        "ko": "m,p를 합하고 n,q를 남깁니다.",
        "en": "Contract m,p and leave n,q free."
      },
      "formula": "A(n,m) B(m,p) C(p,q)",
      "diagram": {
        "ko": "Tensor Network와 수축 순서의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Networks and Contraction Order. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "왼쪽 먼저",
        "en": "Left first"
      },
      "explanation": {
        "ko": "AB의 shape는 (2,4)입니다.",
        "en": "AB has shape (2,4)."
      },
      "formula": "left cost=24+16=40",
      "diagram": {
        "ko": "Tensor Network와 수축 순서의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Networks and Contraction Order. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "오른쪽 먼저",
        "en": "Right first"
      },
      "explanation": {
        "ko": "BC의 shape는 (3,2)입니다.",
        "en": "BC has shape (3,2)."
      },
      "formula": "right cost=24+12=36",
      "diagram": {
        "ko": "Tensor Network와 수축 순서의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Networks and Contraction Order. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "같은 값",
        "en": "Same value"
      },
      "explanation": {
        "ko": "두 별도 계산의 결과를 비교합니다.",
        "en": "Compare independently computed results."
      },
      "formula": "association residual=0",
      "diagram": {
        "ko": "Tensor Network와 수축 순서의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Networks and Contraction Order. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "오른쪽 먼저가 유리",
        "en": "Right-first is cheaper"
      },
      "predict": {
        "ko": "곱셈이 더 적은 순서는?",
        "en": "Which order uses fewer multiplications?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "오른쪽 먼저가 36 대 40입니다.",
        "en": "Right-first: 36 versus 40."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "왼쪽 먼저가 유리",
        "en": "Left-first is cheaper"
      },
      "predict": {
        "ko": "차원을 바꾸면?",
        "en": "What if dimensions change?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "왼쪽 먼저가 24 대 40으로 유리해집니다.",
        "en": "Left-first becomes cheaper: 24 versus 40."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "비용 같음",
        "en": "Equal costs"
      },
      "predict": {
        "ko": "모두 차원 1이면?",
        "en": "What if all dimensions equal one?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "두 순서 모두 곱셈 2회입니다.",
        "en": "Both use two multiplications."
      },
      "preset": 2
    }
  ]
},
"tensor-low-rank-approximation":{
  "intuition": {
    "ko": "명시적 직교 예제의 최적 절단과 일반 텐서 근사의 어려움을 구분합니다.",
    "en": "Separate optimal truncation in an explicit orthogonal example from general tensor-approximation difficulties."
  },
  "steps": [
    {
      "title": {
        "ko": "직교 항",
        "en": "Orthogonal terms"
      },
      "explanation": {
        "ko": "두 성분의 Frobenius 내적은 0입니다.",
        "en": "The two components are Frobenius-orthogonal."
      },
      "formula": "X=3e₀⊗³+e₁⊗³",
      "diagram": {
        "ko": "텐서 저랭크 근사와 오차의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Low-Rank Approximation and Error. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "절단",
        "en": "Truncate"
      },
      "explanation": {
        "ko": "큰 크기 가중치 3을 유지합니다.",
        "en": "Keep the larger magnitude, three."
      },
      "formula": "R=1",
      "diagram": {
        "ko": "텐서 저랭크 근사와 오차의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Low-Rank Approximation and Error. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "오차",
        "en": "Error"
      },
      "explanation": {
        "ko": "버린 크기 1이 오차입니다.",
        "en": "The omitted magnitude one is the error."
      },
      "formula": "error²=1",
      "diagram": {
        "ko": "텐서 저랭크 근사와 오차의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Low-Rank Approximation and Error. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "정규화",
        "en": "Normalize"
      },
      "explanation": {
        "ko": "원본 에너지 10 중 9를 유지합니다.",
        "en": "Nine of ten energy units are retained."
      },
      "formula": "retained fraction=0.9",
      "diagram": {
        "ko": "텐서 저랭크 근사와 오차의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Low-Rank Approximation and Error. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "한 항 유지",
        "en": "Keep one term"
      },
      "predict": {
        "ko": "오차와 유지 에너지는?",
        "en": "What are error and retained fraction?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1과 0.9입니다.",
        "en": "One and 0.9."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "모두 유지",
        "en": "Keep both terms"
      },
      "predict": {
        "ko": "두 항이면?",
        "en": "What if both terms are kept?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "오차 0, 유지 비율 1입니다.",
        "en": "Error zero, retained fraction one."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "영 텐서",
        "en": "Zero tensor"
      },
      "predict": {
        "ko": "영 텐서의 정규화는?",
        "en": "What is normalized energy for zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0/0이라 —이며 오차는 0입니다.",
        "en": "Undefined (0/0), while error is zero."
      },
      "preset": 2
    }
  ]
},
"tensor-symmetry-antisymmetry":{
  "intuition": {
    "ko": "지표 교환의 부호와 직교 분해를 실제 원소로 확인합니다.",
    "en": "Inspect index-swap signs and orthogonal decomposition entry by entry."
  },
  "steps": [
    {
      "title": {
        "ko": "교환",
        "en": "Swap"
      },
      "explanation": {
        "ko": "지표 교환과 전치를 대응합니다.",
        "en": "Relate swapping two indices to transposition."
      },
      "formula": "Aᵀᵢⱼ=Aⱼᵢ",
      "diagram": {
        "ko": "대칭·반대칭 텐서와 분해의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Symmetric and Antisymmetric Tensors. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "대칭 부분",
        "en": "Symmetric part"
      },
      "explanation": {
        "ko": "평균으로 S를 만듭니다.",
        "en": "Average to obtain S."
      },
      "formula": "S=[1,1;1,3]",
      "diagram": {
        "ko": "대칭·반대칭 텐서와 분해의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Symmetric and Antisymmetric Tensors. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "반대칭 부분",
        "en": "Antisymmetric part"
      },
      "explanation": {
        "ko": "차이의 절반으로 K를 만듭니다.",
        "en": "Take half the difference for K."
      },
      "formula": "K=[0,1;−1,0]",
      "diagram": {
        "ko": "대칭·반대칭 텐서와 분해의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Symmetric and Antisymmetric Tensors. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "직교성",
        "en": "Orthogonality"
      },
      "explanation": {
        "ko": "두 부분의 Frobenius 내적은 0입니다.",
        "en": "Their Frobenius inner product is zero."
      },
      "formula": "⟨S,K⟩F=0",
      "diagram": {
        "ko": "대칭·반대칭 텐서와 분해의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Symmetric and Antisymmetric Tensors. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "혼합 성분",
        "en": "Mixed components"
      },
      "predict": {
        "ko": "원본을 되찾나요?",
        "en": "Does the sum recover the input?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "S+K=A이고 잔차는 0입니다.",
        "en": "S+K=A, residual zero."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "대칭",
        "en": "Symmetric"
      },
      "predict": {
        "ko": "이미 대칭이면?",
        "en": "What if the input is symmetric?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "K=0입니다.",
        "en": "K is zero."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "반대칭",
        "en": "Antisymmetric"
      },
      "predict": {
        "ko": "이미 반대칭이면?",
        "en": "What if it is antisymmetric?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "S=0이며 vᵀKv=0입니다.",
        "en": "S is zero and vᵀKv=0."
      },
      "preset": 2
    }
  ]
},
"tensor-storage-cost":{
  "intuition": {
    "ko": "같은 랭크 기호를 써도 표현 형식의 제약과 저장량은 다르며 실제 시간 측정과도 다릅니다.",
    "en": "The same rank symbol means different constraints across formats; storage counts are not runtime measurements."
  },
  "steps": [
    {
      "title": {
        "ko": "원본",
        "en": "Dense form"
      },
      "explanation": {
        "ko": "4⁶=4096개 원소입니다.",
        "en": "The dense tensor has 4⁶=4096 scalars."
      },
      "formula": "dense=4096",
      "diagram": {
        "ko": "텐서 저장량·계산 복잡도의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Storage and Computational Complexity. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "TT",
        "en": "TT cores"
      },
      "explanation": {
        "ko": "경계와 내부 core를 합합니다.",
        "en": "Count boundary and interior cores."
      },
      "formula": "TT=16+64=80",
      "diagram": {
        "ko": "텐서 저장량·계산 복잡도의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Storage and Computational Complexity. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "바이트",
        "en": "Bytes"
      },
      "explanation": {
        "ko": "4바이트면 16384와 320바이트입니다.",
        "en": "Four-byte storage gives 16384 and 320 bytes."
      },
      "formula": "bytes=entries·bytes_per_scalar",
      "diagram": {
        "ko": "텐서 저장량·계산 복잡도의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Storage and Computational Complexity. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "해석의 한계",
        "en": "Limits of the count"
      },
      "explanation": {
        "ko": "압축률만으로 정확도·속도를 보장하지 않습니다.",
        "en": "Storage ratio does not certify accuracy or speed."
      },
      "formula": "representation ≠ benchmark",
      "diagram": {
        "ko": "텐서 저장량·계산 복잡도의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Tensor Storage and Computational Complexity. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "고차 예제",
        "en": "Higher-order example"
      },
      "predict": {
        "ko": "TT 저장 원소 수는?",
        "en": "How many TT scalars are stored?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "80개입니다.",
        "en": "Eighty."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "작은 원본",
        "en": "Small dense tensor"
      },
      "predict": {
        "ko": "작은 원본보다 분해가 클 수 있나요?",
        "en": "Can factor storage exceed a small dense tensor?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "예. 이 경우 TT는 16, 원본은 4입니다.",
        "en": "Yes. Here TT stores sixteen versus four dense scalars."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "2바이트 저장",
        "en": "Two-byte storage"
      },
      "predict": {
        "ko": "2바이트면 저장량은?",
        "en": "What changes with two-byte storage?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "원소 수는 같고 바이트만 절반입니다.",
        "en": "Scalar counts stay the same; bytes halve."
      },
      "preset": 2
    }
  ]
},

"complex-polar-maps":{
  "intuition": {
    "ko": "거듭제곱은 크기를 거듭제곱하고 각도를 배수로 바꾸지만 주값 각도는 감깁니다.",
    "en": "A power raises the magnitude and multiplies the angle, while the principal argument wraps."
  },
  "steps": [
    {
      "title": {
        "ko": "극형식",
        "en": "Polar form"
      },
      "explanation": {
        "ko": "기준 z는 크기 1, 각도 45도입니다.",
        "en": "The reference has magnitude one and angle 45 degrees."
      },
      "formula": "z=exp(iπ/4)",
      "diagram": {
        "ko": "복소평면·극형식과 거듭제곱의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Polar Form and Power Maps. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "곱셈",
        "en": "Multiplication"
      },
      "explanation": {
        "ko": "두 각도가 더해집니다.",
        "en": "Multiplication adds arguments."
      },
      "formula": "z²=exp(iπ/2)",
      "diagram": {
        "ko": "복소평면·극형식과 거듭제곱의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Polar Form and Power Maps. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "크기",
        "en": "Magnitude"
      },
      "explanation": {
        "ko": "r=2이면 제곱 크기는 4입니다.",
        "en": "At r=2 the squared magnitude is four."
      },
      "formula": "|z²|=r²",
      "diagram": {
        "ko": "복소평면·극형식과 거듭제곱의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Polar Form and Power Maps. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "정의 조건",
        "en": "Domain of argument"
      },
      "explanation": {
        "ko": "영점에는 고유한 방향이 없습니다.",
        "en": "Zero has no unique direction."
      },
      "formula": "arg(0) undefined",
      "diagram": {
        "ko": "복소평면·극형식과 거듭제곱의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Polar Form and Power Maps. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "두 배 각도",
        "en": "Double angle"
      },
      "predict": {
        "ko": "제곱 뒤 크기와 각도는?",
        "en": "What are squared magnitude and angle?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "크기 1, 각도 90도입니다.",
        "en": "Magnitude one and angle 90 degrees."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "크기 확대",
        "en": "Magnitude expansion"
      },
      "predict": {
        "ko": "반지름 두 배면 제곱 크기도 두 배인가요?",
        "en": "Does doubling radius double the squared magnitude?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 네 배인 4입니다.",
        "en": "No. It becomes four."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "영점",
        "en": "Zero"
      },
      "predict": {
        "ko": "영점 각도는 0인가요?",
        "en": "Is zero’s argument zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "정의되지 않아 —입니다.",
        "en": "It is undefined, shown as —."
      },
      "preset": 2
    }
  ]
},
"cauchy-riemann-holomorphic":{
  "intuition": {
    "ko": "한 점의 미분 가능성과 열린 근방의 정칙성을 구분합니다.",
    "en": "Separate differentiability at one point from holomorphicity on a neighborhood."
  },
  "steps": [
    {
      "title": {
        "ko": "실수 분해",
        "en": "Real decomposition"
      },
      "explanation": {
        "ko": "z²의 실수부와 허수부를 분리합니다.",
        "en": "Split z² into real and imaginary components."
      },
      "formula": "u=x²−y², v=2xy",
      "diagram": {
        "ko": "정칙성과 Cauchy–Riemann 조건의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Holomorphicity and Cauchy–Riemann Conditions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "방향 일치",
        "en": "Direction agreement"
      },
      "explanation": {
        "ko": "실수축과 허수축 몫이 일치해야 합니다.",
        "en": "Real- and imaginary-direction quotients must agree."
      },
      "formula": "fₓ=fᵧ/i",
      "diagram": {
        "ko": "정칙성과 Cauchy–Riemann 조건의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Holomorphicity and Cauchy–Riemann Conditions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "CR 조건",
        "en": "CR conditions"
      },
      "explanation": {
        "ko": "기준 함수는 모든 점에서 조건을 만족합니다.",
        "en": "The reference satisfies CR everywhere."
      },
      "formula": "f′(z)=2z",
      "diagram": {
        "ko": "정칙성과 Cauchy–Riemann 조건의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Holomorphicity and Cauchy–Riemann Conditions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "근방 조건",
        "en": "Neighborhood condition"
      },
      "explanation": {
        "ko": "한 점의 CR 성립만으로 정칙 근방을 얻지 않습니다.",
        "en": "CR at one point alone does not supply a holomorphic neighborhood."
      },
      "formula": "|z|²: f′(0)=0, not holomorphic",
      "diagram": {
        "ko": "정칙성과 Cauchy–Riemann 조건의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Holomorphicity and Cauchy–Riemann Conditions. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "정칙 제곱",
        "en": "Holomorphic square"
      },
      "predict": {
        "ko": "z=1의 복소 도함수는?",
        "en": "What is the derivative at z=1?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2입니다.",
        "en": "Two."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "켤레 함수",
        "en": "Conjugation"
      },
      "predict": {
        "ko": "켤레 함수의 두 방향 극한은?",
        "en": "What are the two quotient limits for conjugation?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1과 −1로 다릅니다.",
        "en": "One and minus one."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "한 점에서만 미분 가능",
        "en": "Differentiable only at one point"
      },
      "predict": {
        "ko": "원점에서 도함수가 있으면 정칙인가요?",
        "en": "Does a derivative at zero imply holomorphicity?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. |z|²가 반례입니다.",
        "en": "No. |z|² is a counterexample."
      },
      "preset": 2
    }
  ]
},
"complex-contour-integrals":{
  "intuition": {
    "ko": "경로의 매개화·방향·내부 특이점을 함께 확인합니다.",
    "en": "Check parametrization, orientation and interior singularities together."
  },
  "steps": [
    {
      "title": {
        "ko": "매개화",
        "en": "Parametrize"
      },
      "explanation": {
        "ko": "단위 원을 반시계로 돕니다.",
        "en": "Traverse the unit circle counterclockwise."
      },
      "formula": "z(t)=exp(it)",
      "diagram": {
        "ko": "복소 경로 적분과 Cauchy 정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Contour Integrals and Cauchy’s Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "dz 포함",
        "en": "Include dz"
      },
      "explanation": {
        "ko": "함수값만 더하지 않고 z′를 곱합니다.",
        "en": "Multiply the integrand by z′ before summing."
      },
      "formula": "∫f(z(t))z′(t)dt",
      "diagram": {
        "ko": "복소 경로 적분과 Cauchy 정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Contour Integrals and Cauchy’s Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "정칙성",
        "en": "Holomorphic domain"
      },
      "explanation": {
        "ko": "z는 전체 평면에서 원시함수를 가집니다.",
        "en": "z has a primitive on the whole plane."
      },
      "formula": "∮z dz=0",
      "diagram": {
        "ko": "복소 경로 적분과 Cauchy 정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Contour Integrals and Cauchy’s Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "특이점",
        "en": "Singularity"
      },
      "explanation": {
        "ko": "1/z는 원 내부 원점에서 정칙이 아닙니다.",
        "en": "1/z fails holomorphicity at the enclosed origin."
      },
      "formula": "∮dz/z=2πi",
      "diagram": {
        "ko": "복소 경로 적분과 Cauchy 정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Contour Integrals and Cauchy’s Theorem. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "정칙 함수",
        "en": "Holomorphic integrand"
      },
      "predict": {
        "ko": "정칙 z의 닫힌 적분은?",
        "en": "What is the closed integral of z?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0이며 수치 잔차와 구분합니다.",
        "en": "Zero, distinct from numerical residual."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "원점 특이점",
        "en": "Pole at zero"
      },
      "predict": {
        "ko": "1/z도 Cauchy로 0인가요?",
        "en": "Does Cauchy make the integral of 1/z zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 내부 원점 때문에 2πi입니다.",
        "en": "No. The enclosed pole gives 2πi."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "방향 반전",
        "en": "Reversed orientation"
      },
      "predict": {
        "ko": "방향을 뒤집으면?",
        "en": "What changes when orientation reverses?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "−2πi입니다.",
        "en": "It becomes −2πi."
      },
      "preset": 2
    }
  ]
},
"residues-winding":{
  "intuition": {
    "ko": "극의 차수와 유수는 다르며 경로 위의 극은 적분을 정의하지 못하게 합니다.",
    "en": "Pole order differs from residue, and a pole on the contour invalidates the ordinary integral."
  },
  "steps": [
    {
      "title": {
        "ko": "Laurent 계수",
        "en": "Laurent coefficient"
      },
      "explanation": {
        "ko": "유수는 (z−a)⁻¹의 계수입니다.",
        "en": "Residue is the coefficient of (z−a)⁻¹."
      },
      "formula": "Res=coefficient of (z−a)⁻¹",
      "diagram": {
        "ko": "유수·감김 수와 특이점의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Residues, Winding and Poles. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "위치",
        "en": "Location"
      },
      "explanation": {
        "ko": "a=0은 단위 원 안입니다.",
        "en": "The origin lies inside the unit circle."
      },
      "formula": "winding=1",
      "diagram": {
        "ko": "유수·감김 수와 특이점의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Residues, Winding and Poles. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "유수 정리",
        "en": "Residue theorem"
      },
      "explanation": {
        "ko": "단순극의 유수 1을 곱합니다.",
        "en": "Multiply winding by residue one."
      },
      "formula": "integral=2πi",
      "diagram": {
        "ko": "유수·감김 수와 특이점의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Residues, Winding and Poles. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "경계의 실패",
        "en": "Boundary failure"
      },
      "explanation": {
        "ko": "경로가 극을 통과하면 보통 적분이 없습니다.",
        "en": "An ordinary integral fails when the contour passes through the pole."
      },
      "formula": "a=R: undefined",
      "diagram": {
        "ko": "유수·감김 수와 특이점의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Residues, Winding and Poles. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "내부 단순극",
        "en": "Enclosed simple pole"
      },
      "predict": {
        "ko": "내부 단순극의 적분은?",
        "en": "What is the enclosed simple-pole integral?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2πi입니다.",
        "en": "2πi."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "외부 극",
        "en": "Pole outside"
      },
      "predict": {
        "ko": "외부 극의 감김 수는?",
        "en": "What is winding about the outside pole?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0이므로 적분도 0입니다.",
        "en": "Zero, so the integral is zero."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "경로 위 극",
        "en": "Pole on contour"
      },
      "predict": {
        "ko": "경로 위에서도 중점 합이 유한하면 되나요?",
        "en": "Does a finite midpoint sum rescue an on-contour pole?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 적분은 —입니다.",
        "en": "No. The ordinary integral is undefined."
      },
      "preset": 2
    }
  ]
},
"analytic-functions-singularities":{
  "intuition": {
    "ko": "제거 가능·극·본질적 특이점은 서로 다른 Laurent 구조를 가집니다.",
    "en": "Removable singularities, poles and essential singularities have different Laurent structures."
  },
  "steps": [
    {
      "title": {
        "ko": "Taylor와 Laurent",
        "en": "Taylor and Laurent"
      },
      "explanation": {
        "ko": "해석 함수는 국소 멱급수를 가집니다.",
        "en": "An analytic function has a local power series."
      },
      "formula": "f=Σaₙ(z−a)ⁿ",
      "diagram": {
        "ko": "해석 함수와 고립 특이점의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Analytic Functions and Isolated Singularities. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "제거",
        "en": "Removal"
      },
      "explanation": {
        "ko": "sin z/z의 0값을 1로 채울 수 있습니다.",
        "en": "sin z/z extends to one at zero."
      },
      "formula": "1−z²/3!+⋯",
      "diagram": {
        "ko": "해석 함수와 고립 특이점의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Analytic Functions and Isolated Singularities. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "극",
        "en": "Pole"
      },
      "explanation": {
        "ko": "1/z에는 유한 개 음수 차수 항이 있습니다.",
        "en": "1/z has finitely many negative-power terms."
      },
      "formula": "pole order=1",
      "diagram": {
        "ko": "해석 함수와 고립 특이점의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Analytic Functions and Isolated Singularities. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "본질적 특이점",
        "en": "Essential point"
      },
      "explanation": {
        "ko": "exp(1/z)는 음수 차수 항이 무한히 많습니다.",
        "en": "exp(1/z) has infinitely many negative powers."
      },
      "formula": "1+1/z+1/(2z²)+⋯",
      "diagram": {
        "ko": "해석 함수와 고립 특이점의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Analytic Functions and Isolated Singularities. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "제거 가능한 점",
        "en": "Removable point"
      },
      "predict": {
        "ko": "제거 뒤 0값은?",
        "en": "What is the extended value at zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1입니다.",
        "en": "One."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "단순극",
        "en": "Simple pole"
      },
      "predict": {
        "ko": "r=0.5의 1/z는?",
        "en": "What is 1/z at radius 0.5 on the positive ray?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2입니다.",
        "en": "Two."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "본질적 특이점",
        "en": "Essential singularity"
      },
      "predict": {
        "ko": "r=0.5의 exp(1/z)는?",
        "en": "What is exp(1/z) at radius 0.5 on the positive ray?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "e²≈7.389입니다. 다른 접근 방향도 비교하세요.",
        "en": "e²≈7.389; compare other approach directions."
      },
      "preset": 2
    }
  ]
},
"analytic-continuation-branches":{
  "intuition": {
    "ko": "같은 종점이라도 따라온 경로의 감김 수에 따라 로그와 제곱근의 연속값이 달라집니다.",
    "en": "The same endpoint can have different continued logarithm and square-root values after different windings."
  },
  "steps": [
    {
      "title": {
        "ko": "주값",
        "en": "Principal value"
      },
      "explanation": {
        "ko": "종점 i의 주각은 π/2입니다.",
        "en": "The principal argument at i is π/2."
      },
      "formula": "Log i=iπ/2",
      "diagram": {
        "ko": "해석적 연속·로그와 가지의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Analytic Continuation, Logarithms and Branches. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "경로 기억",
        "en": "Path memory"
      },
      "explanation": {
        "ko": "한 바퀴마다 로그 허수부가 2π 늘어납니다.",
        "en": "Each turn adds 2π to the continued imaginary part."
      },
      "formula": "log increment=2πik",
      "diagram": {
        "ko": "해석적 연속·로그와 가지의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Analytic Continuation, Logarithms and Branches. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "제곱근",
        "en": "Square root"
      },
      "explanation": {
        "ko": "각도를 반으로 나누어 연속시킵니다.",
        "en": "Continue a square root by halving the lifted angle."
      },
      "formula": "sqrt=√r exp(iΘ/2)",
      "diagram": {
        "ko": "해석적 연속·로그와 가지의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Analytic Continuation, Logarithms and Branches. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "단일값성",
        "en": "Single-valuedness"
      },
      "explanation": {
        "ko": "한 바퀴면 제곱근 부호가 바뀝니다.",
        "en": "One turn changes the square-root sign."
      },
      "formula": "k odd ⇒ root sign flips",
      "diagram": {
        "ko": "해석적 연속·로그와 가지의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Analytic Continuation, Logarithms and Branches. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "감김 없음",
        "en": "No extra winding"
      },
      "predict": {
        "ko": "주값 로그 허수부는?",
        "en": "What is the principal log imaginary part?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "π/2입니다.",
        "en": "π/2."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "한 바퀴 추가",
        "en": "One extra turn"
      },
      "predict": {
        "ko": "한 바퀴 뒤 종점이 같으면 제곱근도 같나요?",
        "en": "Does the same endpoint after one turn give the same root?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 부호가 바뀝니다.",
        "en": "No. Its sign flips."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "두 바퀴 추가",
        "en": "Two extra turns"
      },
      "predict": {
        "ko": "두 바퀴 뒤에는?",
        "en": "What happens after two turns?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "제곱근은 돌아오지만 로그는 4πi 차이가 납니다.",
        "en": "The root returns, while the logarithm differs by 4πi."
      },
      "preset": 2
    }
  ]
},
"complex-linear-algebra":{
  "intuition": {
    "ko": "복소 벡터의 길이는 켤레전치로 정의하며 전치만 쓰면 양성이 사라집니다.",
    "en": "Use conjugate transpose for lengths; plain transpose loses positivity."
  },
  "steps": [
    {
      "title": {
        "ko": "벡터",
        "en": "Vector"
      },
      "explanation": {
        "ko": "v는 두 복소 성분을 가집니다.",
        "en": "v has two complex components."
      },
      "formula": "v=(1,i)",
      "diagram": {
        "ko": "복소 선형대수·켤레 내적의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Linear Algebra and Hermitian Products. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "켤레전치",
        "en": "Conjugate transpose"
      },
      "explanation": {
        "ko": "켤레를 쓰면 길이 제곱은 2입니다.",
        "en": "Conjugation gives squared norm two."
      },
      "formula": "v* v=1+(−i)i=2",
      "diagram": {
        "ko": "복소 선형대수·켤레 내적의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Linear Algebra and Hermitian Products. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "전치의 함정",
        "en": "Plain transpose trap"
      },
      "explanation": {
        "ko": "전치만 쓰면 0입니다.",
        "en": "Plain transpose instead gives zero."
      },
      "formula": "vᵀv=1+i²=0",
      "diagram": {
        "ko": "복소 선형대수·켤레 내적의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Linear Algebra and Hermitian Products. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "내적 성질",
        "en": "Inner-product properties"
      },
      "explanation": {
        "ko": "길이와 직교성을 Hermitian 내적으로 정의합니다.",
        "en": "Use the Hermitian product for norm and orthogonality."
      },
      "formula": "|v* w|≤‖v‖‖w‖",
      "diagram": {
        "ko": "복소 선형대수·켤레 내적의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Linear Algebra and Hermitian Products. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "같은 벡터",
        "en": "Matching vectors"
      },
      "predict": {
        "ko": "v와 같은 w의 내적은?",
        "en": "What is the product for w matching v?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2이며 정규화 크기는 1입니다.",
        "en": "Two, with normalized magnitude one."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "직교 벡터",
        "en": "Orthogonal vectors"
      },
      "predict": {
        "ko": "w=(1,−i)이면?",
        "en": "What if w=(1,−i)?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "Hermitian 내적은 0으로 직교입니다.",
        "en": "The Hermitian product is zero: orthogonal."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "영벡터",
        "en": "Zero vector"
      },
      "predict": {
        "ko": "w=0의 정규화는?",
        "en": "What is the normalized value for w=0?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "분모가 0이라 —입니다.",
        "en": "Undefined because the denominator is zero."
      },
      "preset": 2
    }
  ]
},
"wirtinger-gradients":{
  "intuition": {
    "ko": "실수값 목적함수는 보통 정칙이 아니어도 두 실수 변수로 미분하고 최적화할 수 있습니다.",
    "en": "A real-valued objective can be optimized through two real variables without being holomorphic."
  },
  "steps": [
    {
      "title": {
        "ko": "실수 손실",
        "en": "Real objective"
      },
      "explanation": {
        "ko": "L=|z|²은 x²+y²입니다.",
        "en": "L=|z|² is x²+y²."
      },
      "formula": "L=x²+y²",
      "diagram": {
        "ko": "Wirtinger 미분과 복소 손실 최적화의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Wirtinger Derivatives and Complex Loss Optimization. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "두 미분",
        "en": "Two derivatives"
      },
      "explanation": {
        "ko": "두 Wirtinger 미분은 서로 켤레입니다.",
        "en": "The two Wirtinger derivatives are conjugates."
      },
      "formula": "∂L/∂z=z̄; ∂L/∂z̄=z",
      "diagram": {
        "ko": "Wirtinger 미분과 복소 손실 최적화의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Wirtinger Derivatives and Complex Loss Optimization. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "계수 2",
        "en": "Factor two"
      },
      "explanation": {
        "ko": "실수 gradient는 (2x,2y)입니다.",
        "en": "The real gradient is (2x,2y)."
      },
      "formula": "z_next=z−2ηz",
      "diagram": {
        "ko": "Wirtinger 미분과 복소 손실 최적화의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Wirtinger Derivatives and Complex Loss Optimization. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "안정성",
        "en": "Stability"
      },
      "explanation": {
        "ko": "η=0.25면 오차가 절반으로 줄어듭니다.",
        "en": "At η=0.25 the error halves each step."
      },
      "formula": "error_next=(1−2η)error",
      "diagram": {
        "ko": "Wirtinger 미분과 복소 손실 최적화의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Wirtinger Derivatives and Complex Loss Optimization. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "수축하는 갱신",
        "en": "Contracting update"
      },
      "predict": {
        "ko": "한 번 뒤 z와 손실은?",
        "en": "What are z and loss after one step?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "z=(0.5,0.5), 손실 0.5입니다.",
        "en": "z=(0.5,0.5), loss 0.5."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "진동하는 갱신",
        "en": "Oscillating update"
      },
      "predict": {
        "ko": "η=1이면 수렴하나요?",
        "en": "Does η=1 converge?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "진동하며 손실은 유지됩니다.",
        "en": "It oscillates with unchanged loss."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "이동된 목표",
        "en": "Shifted target"
      },
      "predict": {
        "ko": "목표를 1+i로 옮기면?",
        "en": "What changes when the target is 1+i?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "gradient가 z−(1+i)를 기준으로 정해집니다.",
        "en": "The gradient uses displacement from 1+i."
      },
      "preset": 2
    }
  ]
},
"complex-signals-phase":{
  "intuition": {
    "ko": "위상 차이는 같은 크기의 신호를 강화하거나 상쇄하며 영 진폭에서는 위상을 정의하지 않습니다.",
    "en": "Phase differences reinforce or cancel signals; zero amplitude has no phase."
  },
  "steps": [
    {
      "title": {
        "ko": "복소 진폭",
        "en": "Complex amplitude"
      },
      "explanation": {
        "ko": "위상 0의 두 단위 phasor를 더합니다.",
        "en": "Add two unit phasors at phase zero."
      },
      "formula": "1+1=2",
      "diagram": {
        "ko": "복소 신호·위상과 간섭의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Signals, Phase and Interference. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "실수 신호",
        "en": "Real signal"
      },
      "explanation": {
        "ko": "실제 표시는 복소 회전의 실수부입니다.",
        "en": "The plotted signal is the real part of complex rotation."
      },
      "formula": "s(t)=Re[A exp(it)]",
      "diagram": {
        "ko": "복소 신호·위상과 간섭의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Signals, Phase and Interference. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "위상 차이",
        "en": "Phase difference"
      },
      "explanation": {
        "ko": "180도 차이는 같은 크기를 상쇄합니다.",
        "en": "A 180-degree difference cancels equal magnitudes."
      },
      "formula": "1+exp(iπ)=0",
      "diagram": {
        "ko": "복소 신호·위상과 간섭의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Signals, Phase and Interference. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "위상 정의",
        "en": "Phase domain"
      },
      "explanation": {
        "ko": "상쇄된 신호에는 고유한 위상이 없습니다.",
        "en": "A canceled signal has no unique phase."
      },
      "formula": "arg(0) undefined",
      "diagram": {
        "ko": "복소 신호·위상과 간섭의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Complex Signals, Phase and Interference. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "보강 간섭",
        "en": "Constructive interference"
      },
      "predict": {
        "ko": "합의 진폭은?",
        "en": "What is the combined amplitude?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2입니다.",
        "en": "Two."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "상쇄 간섭",
        "en": "Destructive interference"
      },
      "predict": {
        "ko": "상쇄 뒤 위상은?",
        "en": "What phase remains after cancellation?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "진폭 0이며 위상은 —입니다.",
        "en": "Amplitude zero, phase undefined."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "직교 위상",
        "en": "Quadrature phases"
      },
      "predict": {
        "ko": "90도 차이에서 진폭은?",
        "en": "What is amplitude at a 90-degree difference?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "√2, 합의 위상은 45도입니다.",
        "en": "√2, with combined phase 45 degrees."
      },
      "preset": 2
    }
  ]
},

"limits-continuity-completeness":{
  "intuition": {
    "ko": "점의 함수값과 주변에서 접근하는 값을 분리하고 실수의 빈틈 없음이 쓰이는 곳을 찾습니다.",
    "en": "Separate a value at a point from nearby limits and identify where completeness matters."
  },
  "steps": [
    {
      "title": {
        "ko": "점과 근방",
        "en": "Point and neighborhood"
      },
      "explanation": {
        "ko": "f(1)=0이나 근방은 2로 접근합니다.",
        "en": "The assigned value is zero while nearby values approach two."
      },
      "formula": "f(1)=0, L=2",
      "diagram": {
        "ko": "극한·연속·실수의 완비성의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Limits, Continuity and Completeness. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "양쪽 극한",
        "en": "Two sides"
      },
      "explanation": {
        "ko": "양쪽이 같은 값에 접근해야 합니다.",
        "en": "Both sides must approach the same value."
      },
      "formula": "L_left=L_right=2",
      "diagram": {
        "ko": "극한·연속·실수의 완비성의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Limits, Continuity and Completeness. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "연속 확장",
        "en": "Continuous extension"
      },
      "explanation": {
        "ko": "점의 값만 2로 바꾸면 연속입니다.",
        "en": "Changing only the assigned value to two makes it continuous."
      },
      "formula": "f(1)=L",
      "diagram": {
        "ko": "극한·연속·실수의 완비성의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Limits, Continuity and Completeness. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "완비성",
        "en": "Completeness"
      },
      "explanation": {
        "ko": "상한 공리는 극한 존재 논증을 뒷받침합니다.",
        "en": "The least-upper-bound property supports existence proofs."
      },
      "formula": "bounded monotone ⇒ convergent in R",
      "diagram": {
        "ko": "극한·연속·실수의 완비성의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Limits, Continuity and Completeness. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "제거 가능한 불연속",
        "en": "Removable discontinuity"
      },
      "predict": {
        "ko": "f(1)이 극한을 결정하나요?",
        "en": "Does f(1) determine the limit?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 점의 값은 0, 극한은 2입니다.",
        "en": "No. The assigned value is zero and the limit is two."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "연속으로 채우기",
        "en": "Continuous extension"
      },
      "predict": {
        "ko": "연속으로 만드는 값은?",
        "en": "Which assigned value restores continuity?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2이며 양쪽 극한과 같아집니다.",
        "en": "Two, matching the two-sided limit."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "점프",
        "en": "Jump"
      },
      "predict": {
        "ko": "점프도 점 하나로 고칠 수 있나요?",
        "en": "Can changing one point repair a jump?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 좌우 극한 0과 1이 다릅니다.",
        "en": "No. The one-sided limits are zero and one."
      },
      "preset": 2
    }
  ]
},
"differentiability-mean-value":{
  "intuition": {
    "ko": "접선의 존재 조건과 구간 평균 변화율이 한 점에서 실현되는 조건을 구분합니다.",
    "en": "Distinguish the existence of a tangent from the hypotheses ensuring a mean slope is attained."
  },
  "steps": [
    {
      "title": {
        "ko": "차분",
        "en": "Difference quotient"
      },
      "explanation": {
        "ko": "제곱 함수의 전진 차분은 2x+h입니다.",
        "en": "The forward quotient for x² is 2x+h."
      },
      "formula": "D_h=2x+h",
      "diagram": {
        "ko": "미분 가능성과 평균값정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Differentiability and the Mean Value Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "미분",
        "en": "Derivative"
      },
      "explanation": {
        "ko": "h→0에서 2x를 얻습니다.",
        "en": "As h tends to zero the quotient tends to 2x."
      },
      "formula": "f′(0.5)=1",
      "diagram": {
        "ko": "미분 가능성과 평균값정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Differentiability and the Mean Value Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "평균 기울기",
        "en": "Mean slope"
      },
      "explanation": {
        "ko": "[−1,1] 양 끝의 함수값은 같습니다.",
        "en": "Endpoint values on [−1,1] agree."
      },
      "formula": "mean slope=0",
      "diagram": {
        "ko": "미분 가능성과 평균값정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Differentiability and the Mean Value Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "가정 확인",
        "en": "Check assumptions"
      },
      "explanation": {
        "ko": "x²은 c=0이 증인이지만 |x|는 모서리가 있습니다.",
        "en": "For x², c=0 witnesses the MVT; |x| has a corner."
      },
      "formula": "continuity + differentiability",
      "diagram": {
        "ko": "미분 가능성과 평균값정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Differentiability and the Mean Value Theorem. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "매끄러운 함수",
        "en": "Smooth function"
      },
      "predict": {
        "ko": "차분은 정확히 1인가요?",
        "en": "Is the forward quotient exactly one?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "h=0.1이면 1.1이며 도함수 1과 다릅니다.",
        "en": "At h=0.1 it is 1.1, distinct from derivative one."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "모서리",
        "en": "Corner"
      },
      "predict": {
        "ko": "연속이면 항상 미분 가능한가요?",
        "en": "Does continuity imply differentiability?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. |x|의 0이 반례입니다.",
        "en": "No. |x| at zero is a counterexample."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "폭 0",
        "en": "Zero step"
      },
      "predict": {
        "ko": "h=0을 넣으면 미분인가요?",
        "en": "Does substituting h=0 calculate the derivative?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "차분은 —이고 도함수는 극한으로 정의합니다.",
        "en": "The quotient is undefined; the derivative is defined by a limit."
      },
      "preset": 2
    }
  ]
},
"integration-fundamental-theorem":{
  "intuition": {
    "ko": "작은 면적의 합과 누적 함수의 도함수를 연결하고 유한 합의 오차를 계산합니다.",
    "en": "Connect small-area sums to derivatives of accumulation and quantify finite-sum error."
  },
  "steps": [
    {
      "title": {
        "ko": "분할",
        "en": "Partition"
      },
      "explanation": {
        "ko": "[0,1]을 네 구간으로 나눕니다.",
        "en": "Split [0,1] into four intervals."
      },
      "formula": "Δx=1/4",
      "diagram": {
        "ko": "적분·누적과 미적분 기본정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Integration, Accumulation and the Fundamental Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "합",
        "en": "Sum"
      },
      "explanation": {
        "ko": "왼쪽 끝점 높이를 사용합니다.",
        "en": "Use the left-endpoint heights."
      },
      "formula": "L₄=7/32",
      "diagram": {
        "ko": "적분·누적과 미적분 기본정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Integration, Accumulation and the Fundamental Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "극한",
        "en": "Limit"
      },
      "explanation": {
        "ko": "정확한 적분은 1/3입니다.",
        "en": "The exact integral is one third."
      },
      "formula": "∫₀¹t²dt=1/3",
      "diagram": {
        "ko": "적분·누적과 미적분 기본정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Integration, Accumulation and the Fundamental Theorem. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "기본정리",
        "en": "Fundamental theorem"
      },
      "explanation": {
        "ko": "누적의 순간 변화는 현재 함수값입니다.",
        "en": "The accumulation rate equals the current integrand."
      },
      "formula": "F′(1)=1",
      "diagram": {
        "ko": "적분·누적과 미적분 기본정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Integration, Accumulation and the Fundamental Theorem. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "네 직사각형",
        "en": "Four rectangles"
      },
      "predict": {
        "ko": "네 사각형의 합은?",
        "en": "What is the four-rectangle sum?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0.21875이고 정확한 1/3보다 작습니다.",
        "en": "It is 0.21875, below one third."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "분할 세분",
        "en": "Refined partition"
      },
      "predict": {
        "ko": "분할을 늘리면 무엇을 확인하나요?",
        "en": "What should refinement demonstrate?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "오차 감소와 해석적 상한을 비교합니다.",
        "en": "Compare decreasing error with the analytic bound."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "빈 구간",
        "en": "Zero-width interval"
      },
      "predict": {
        "ko": "길이 0 구간의 적분은?",
        "en": "What is the integral over a zero-width interval?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0이며 함수가 사라진다는 뜻은 아닙니다.",
        "en": "Zero; the integrand need not vanish."
      },
      "preset": 2
    }
  ]
},
"taylor-remainders":{
  "intuition": {
    "ko": "다항 근사와 실제 오차, 오차를 보장하는 상한을 함께 읽습니다.",
    "en": "Read the polynomial approximation, actual error and a justified error bound together."
  },
  "steps": [
    {
      "title": {
        "ko": "값 맞추기",
        "en": "Match value"
      },
      "explanation": {
        "ko": "원점 함수값은 1입니다.",
        "en": "The value at zero is one."
      },
      "formula": "T₀=1",
      "diagram": {
        "ko": "Taylor 전개와 나머지의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Taylor Expansions and Remainders. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "기울기 맞추기",
        "en": "Match slope"
      },
      "explanation": {
        "ko": "지수의 도함수도 원점에서 1입니다.",
        "en": "The derivative at zero is also one."
      },
      "formula": "T₁=1+x",
      "diagram": {
        "ko": "Taylor 전개와 나머지의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Taylor Expansions and Remainders. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "곡률 맞추기",
        "en": "Match curvature"
      },
      "explanation": {
        "ko": "두 번째 도함수를 2!로 나눕니다.",
        "en": "Divide the second derivative coefficient by 2!."
      },
      "formula": "T₂=1+x+x²/2",
      "diagram": {
        "ko": "Taylor 전개와 나머지의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Taylor Expansions and Remainders. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "나머지 제어",
        "en": "Control remainder"
      },
      "explanation": {
        "ko": "근사에는 조건 있는 오차 상한이 필요합니다.",
        "en": "Approximation needs a condition-dependent remainder bound."
      },
      "formula": "|R₂(1)|≤e/6",
      "diagram": {
        "ko": "Taylor 전개와 나머지의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Taylor Expansions and Remainders. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "이차 근사",
        "en": "Quadratic approximation"
      },
      "predict": {
        "ko": "T₂(1)은?",
        "en": "What is T₂(1)?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2.5이고 실제 e와 약 0.218 차이입니다.",
        "en": "It is 2.5, about 0.218 below e."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "고차 근사",
        "en": "Higher-degree approximation"
      },
      "predict": {
        "ko": "차수를 높이면 이 예제의 오차는?",
        "en": "What happens to error as degree increases here?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "작아지며 차수 8에서는 약 3×10⁻⁶입니다.",
        "en": "It decreases; at degree eight it is about 3×10⁻⁶."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "원점",
        "en": "Expansion center"
      },
      "predict": {
        "ko": "차수 0으로도 정확한 곳은?",
        "en": "Where is even degree zero exact?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "전개 중심 x=0입니다.",
        "en": "At the expansion center x=0."
      },
      "preset": 2
    }
  ]
},
"series-convergence-tests":{
  "intuition": {
    "ko": "항이 0으로 가는 것과 부분합이 수렴하는 것은 다른 조건입니다.",
    "en": "A vanishing term does not by itself make the sequence of partial sums converge."
  },
  "steps": [
    {
      "title": {
        "ko": "필요 조건",
        "en": "Necessary condition"
      },
      "explanation": {
        "ko": "조화급수 항은 0으로 갑니다.",
        "en": "Harmonic terms tend to zero."
      },
      "formula": "1/n→0",
      "diagram": {
        "ko": "수열·급수의 수렴 판정의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Sequences, Series and Convergence Tests. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "충분하지 않음",
        "en": "Not sufficient"
      },
      "explanation": {
        "ko": "지수 길이 블록마다 적어도 1/2을 더합니다.",
        "en": "Each dyadic block adds at least one half."
      },
      "formula": "Σ over n=2^k+1,…,2^(k+1): 1/n≥1/2",
      "diagram": {
        "ko": "수열·급수의 수렴 판정의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Sequences, Series and Convergence Tests. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "교대의 효과",
        "en": "Alternation"
      },
      "explanation": {
        "ko": "감소하는 항에 교대 부호를 주면 조건부 수렴합니다.",
        "en": "Alternating decreasing terms gives conditional convergence."
      },
      "formula": "Σ(−1)ⁿ⁺¹/n=ln2",
      "diagram": {
        "ko": "수열·급수의 수렴 판정의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Sequences, Series and Convergence Tests. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "절대 수렴",
        "en": "Absolute convergence"
      },
      "explanation": {
        "ko": "1/n²은 적분 비교로 수렴합니다.",
        "en": "Integral comparison proves convergence of 1/n²."
      },
      "formula": "tail≤∫ᴺ∞x⁻²dx=1/N",
      "diagram": {
        "ko": "수열·급수의 수렴 판정의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Sequences, Series and Convergence Tests. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "조화급수",
        "en": "Harmonic series"
      },
      "predict": {
        "ko": "항이 작아지면 수렴할까요?",
        "en": "Do smaller terms guarantee convergence?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "조화급수는 그렇지 않다는 반례입니다.",
        "en": "The harmonic series is a counterexample."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "교대 조화급수",
        "en": "Alternating harmonic series"
      },
      "predict": {
        "ko": "교대합의 오차 보장은?",
        "en": "What bounds the alternating error?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "다음 항 1/9 이하입니다.",
        "en": "At most the next term, one ninth."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "제곱 역수 급수",
        "en": "Reciprocal-square series"
      },
      "predict": {
        "ko": "N=8의 제곱 역수 꼬리 상한은?",
        "en": "What bounds the reciprocal-square tail at N=8?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1/8입니다.",
        "en": "One eighth."
      },
      "preset": 2
    }
  ]
},
"pointwise-uniform-convergence":{
  "intuition": {
    "ko": "점마다 다른 대기 시간과 전체 구간에 공통인 대기 시간을 구분합니다.",
    "en": "Distinguish a waiting time for each point from one waiting time valid across the whole domain."
  },
  "steps": [
    {
      "title": {
        "ko": "고정된 점",
        "en": "Fixed point"
      },
      "explanation": {
        "ko": "x=1/2에서는 (1/2)ⁿ→0입니다.",
        "en": "At x=1/2, (1/2)ⁿ tends to zero."
      },
      "formula": "f₈(1/2)=1/256",
      "diagram": {
        "ko": "점별 수렴과 균등 수렴의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Pointwise and Uniform Convergence. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "끝점",
        "en": "Endpoint"
      },
      "explanation": {
        "ko": "x=1에서는 항상 1입니다.",
        "en": "At x=1 the value is always one."
      },
      "formula": "limit(1)=1",
      "diagram": {
        "ko": "점별 수렴과 균등 수렴의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Pointwise and Uniform Convergence. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "공통 제어 실패",
        "en": "No uniform control"
      },
      "explanation": {
        "ko": "x<1을 1에 가깝게 고르면 오차는 1에 접근합니다.",
        "en": "Choosing x<1 close enough to one makes error approach one."
      },
      "formula": "sup error=1",
      "diagram": {
        "ko": "점별 수렴과 균등 수렴의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Pointwise and Uniform Convergence. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "균등 반대 예",
        "en": "Uniform comparison"
      },
      "explanation": {
        "ko": "x/n은 전체 구간에서 1/n 이하입니다.",
        "en": "x/n is bounded by 1/n across the entire interval."
      },
      "formula": "sup |x/n|=1/n",
      "diagram": {
        "ko": "점별 수렴과 균등 수렴의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Pointwise and Uniform Convergence. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "점별 수렴",
        "en": "Pointwise convergence"
      },
      "predict": {
        "ko": "x=0.5의 오차는?",
        "en": "What is the error at x=0.5?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1/256입니다. 그러나 전 구간 오차는 작지 않습니다.",
        "en": "One over 256; the whole-domain error is still not small."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "끝점 근처",
        "en": "Near the endpoint"
      },
      "predict": {
        "ko": "n을 늘리면 균등해질까요?",
        "en": "Does increasing n establish uniform convergence?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. supremum은 여전히 1입니다.",
        "en": "No; the supremum remains one."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "균등 수렴",
        "en": "Uniform convergence"
      },
      "predict": {
        "ko": "x/n의 공통 오차는?",
        "en": "What is the common error bound for x/n?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1/8입니다.",
        "en": "One eighth."
      },
      "preset": 2
    }
  ]
},
"compactness-extreme-values":{
  "intuition": {
    "ko": "상한이 존재하는 것과 함수가 그 값을 실제로 얻는 것은 다릅니다.",
    "en": "Having a supremum differs from attaining a maximum."
  },
  "steps": [
    {
      "title": {
        "ko": "닫힘과 유계",
        "en": "Closed and bounded"
      },
      "explanation": {
        "ko": "[0,1]은 R에서 compact입니다.",
        "en": "[0,1] is compact in R."
      },
      "formula": "K=[0,1]",
      "diagram": {
        "ko": "Compactness·연속성과 극값의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Compactness, Continuity and Extreme Values. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "연속 함수",
        "en": "Continuous map"
      },
      "explanation": {
        "ko": "f(x)=x는 연속입니다.",
        "en": "The identity function is continuous."
      },
      "formula": "f(K)=[0,1]",
      "diagram": {
        "ko": "Compactness·연속성과 극값의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Compactness, Continuity and Extreme Values. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "최대값",
        "en": "Attained maximum"
      },
      "explanation": {
        "ko": "x=1이 실제 정의역에 들어 있습니다.",
        "en": "The endpoint one belongs to the domain."
      },
      "formula": "max f=1",
      "diagram": {
        "ko": "Compactness·연속성과 극값의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Compactness, Continuity and Extreme Values. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "가정 제거",
        "en": "Remove a hypothesis"
      },
      "explanation": {
        "ko": "(0,1)에서는 같은 상한을 달성하지 못합니다.",
        "en": "On (0,1) the same supremum is not attained."
      },
      "formula": "sup f=1, no max",
      "diagram": {
        "ko": "Compactness·연속성과 극값의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Compactness, Continuity and Extreme Values. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "닫힌 구간",
        "en": "Closed interval"
      },
      "predict": {
        "ko": "최대값 1을 얻나요?",
        "en": "Is the maximum one attained?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "닫힌 구간에서는 x=1에서 얻습니다.",
        "en": "On the closed interval, yes at x=1."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "열린 구간",
        "en": "Open interval"
      },
      "predict": {
        "ko": "열린 구간에도 상한은 있나요?",
        "en": "Does the open interval still have a supremum?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1이지만 최대값은 없습니다.",
        "en": "Yes, one, but there is no maximum."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "제외된 끝점",
        "en": "Excluded endpoint"
      },
      "predict": {
        "ko": "제외된 점의 함수값은?",
        "en": "What is the value at an excluded endpoint?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "정의되지 않아 —입니다.",
        "en": "Undefined, displayed as —."
      },
      "preset": 2
    }
  ]
},
"convexity-first-order":{
  "intuition": {
    "ko": "접선이 아래에서 함수를 받치는지 보고 한 점 검사와 전역 성질을 구분합니다.",
    "en": "Compare a tangent with the function and separate a probe from a global property."
  },
  "steps": [
    {
      "title": {
        "ko": "일차 모형",
        "en": "First-order model"
      },
      "explanation": {
        "ko": "x=0의 접선은 0입니다.",
        "en": "The tangent at zero is zero."
      },
      "formula": "f(0)+f′(0)y=0",
      "diagram": {
        "ko": "볼록성과 일차 지지 부등식의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Convexity and First-Order Supporting Inequalities. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "차이",
        "en": "Gap"
      },
      "explanation": {
        "ko": "y=1에서 이차식은 1입니다.",
        "en": "At y=1 the quadratic equals one."
      },
      "formula": "gap=1",
      "diagram": {
        "ko": "볼록성과 일차 지지 부등식의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Convexity and First-Order Supporting Inequalities. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "전역 인증",
        "en": "Global certificate"
      },
      "explanation": {
        "ko": "간격은 a(y−x)²입니다.",
        "en": "The exact gap is a(y−x)²."
      },
      "formula": "a≥0 ⇒ gap≥0",
      "diagram": {
        "ko": "볼록성과 일차 지지 부등식의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Convexity and First-Order Supporting Inequalities. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "프로브 한계",
        "en": "Probe limitation"
      },
      "explanation": {
        "ko": "x=y에서는 오목함수도 간격 0입니다.",
        "en": "At x=y even a concave function has zero gap."
      },
      "formula": "one pair ≠ all pairs",
      "diagram": {
        "ko": "볼록성과 일차 지지 부등식의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Convexity and First-Order Supporting Inequalities. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "볼록 이차식",
        "en": "Convex quadratic"
      },
      "predict": {
        "ko": "지지 부등식이 맞나요?",
        "en": "Does the supporting inequality hold?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "a=1이면 모든 쌍에서 맞습니다.",
        "en": "For a=1 it holds for every pair."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "오목 이차식",
        "en": "Concave quadratic"
      },
      "predict": {
        "ko": "a=−1이면?",
        "en": "What if a=−1?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "서로 다른 점에서 간격이 음수입니다.",
        "en": "Distinct points give a negative gap."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "같은 점",
        "en": "Same point"
      },
      "predict": {
        "ko": "간격 0이면 볼록인가요?",
        "en": "Does a zero gap certify convexity?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 같은 점만 검사했습니다.",
        "en": "No; this probe tests the same point twice."
      },
      "preset": 2
    }
  ]
},
"limits-integration-interchange":{
  "intuition": {
    "ko": "높아지고 좁아지는 봉우리가 점별 극한에서 사라져도 면적은 남을 수 있습니다.",
    "en": "A taller, narrower spike can vanish pointwise while retaining its integral."
  },
  "steps": [
    {
      "title": {
        "ko": "유한 면적",
        "en": "Finite area"
      },
      "explanation": {
        "ko": "폭 2/n, 높이 n인 삼각형입니다.",
        "en": "The triangle has width 2/n and height n."
      },
      "formula": "area=1",
      "diagram": {
        "ko": "극한과 적분의 교환 조건의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for When Limits and Integrals Commute. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "점별 소멸",
        "en": "Pointwise disappearance"
      },
      "explanation": {
        "ko": "고정 x>0은 결국 지지 밖에 있습니다.",
        "en": "Every fixed x>0 eventually lies outside the support."
      },
      "formula": "fₙ(x)→0",
      "diagram": {
        "ko": "극한과 적분의 교환 조건의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for When Limits and Integrals Commute. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "교환 실패",
        "en": "Interchange failure"
      },
      "explanation": {
        "ko": "적분 극한은 1, 극한의 적분은 0입니다.",
        "en": "The limit of integrals is one; the integral of the limit is zero."
      },
      "formula": "1≠0",
      "diagram": {
        "ko": "극한과 적분의 교환 조건의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for When Limits and Integrals Commute. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "충분 조건",
        "en": "Sufficient condition"
      },
      "explanation": {
        "ko": "x/n은 적분 가능한 1에 지배됩니다.",
        "en": "x/n is dominated by the integrable function one."
      },
      "formula": "∫x/n=1/(2n)→0",
      "diagram": {
        "ko": "극한과 적분의 교환 조건의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for When Limits and Integrals Commute. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "움직이는 봉우리",
        "en": "Shrinking spike"
      },
      "predict": {
        "ko": "봉우리 면적은?",
        "en": "What is the spike area?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "항상 1입니다.",
        "en": "Always one."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "더 좁은 봉우리",
        "en": "Narrower spike"
      },
      "predict": {
        "ko": "더 좁히면 면적이 사라지나요?",
        "en": "Does narrowing remove the mass?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 높이가 함께 증가합니다.",
        "en": "No; height increases with narrowing."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "지배되는 함수열",
        "en": "Dominated sequence"
      },
      "predict": {
        "ko": "지배되는 예제의 적분은?",
        "en": "What is the dominated example’s integral?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1/16이며 극한은 0입니다.",
        "en": "One sixteenth, tending to zero."
      },
      "preset": 2
    }
  ]
},
"implicit-inverse-functions":{
  "intuition": {
    "ko": "해가 존재하는 것과 미분 가능한 국소 해를 얻는 조건을 구분합니다.",
    "en": "Distinguish existence of an inverse or branch from hypotheses giving a differentiable local one."
  },
  "steps": [
    {
      "title": {
        "ko": "제약",
        "en": "Constraint"
      },
      "explanation": {
        "ko": "x²+y²=1 위의 점 (0.6,0.8)을 봅니다.",
        "en": "Use point (0.6,0.8) on x²+y²=1."
      },
      "formula": "F=0",
      "diagram": {
        "ko": "암묵함수·역함수 정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Implicit and Inverse Function Theorems. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "정칙성",
        "en": "Regularity"
      },
      "explanation": {
        "ko": "F_y=1.6으로 0이 아닙니다.",
        "en": "F_y=1.6 is nonzero."
      },
      "formula": "F_y≠0",
      "diagram": {
        "ko": "암묵함수·역함수 정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Implicit and Inverse Function Theorems. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "암묵 미분",
        "en": "Implicit derivative"
      },
      "explanation": {
        "ko": "2x+2yy′=0을 풉니다.",
        "en": "Solve 2x+2yy′=0."
      },
      "formula": "y′=−0.75",
      "diagram": {
        "ko": "암묵함수·역함수 정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Implicit and Inverse Function Theorems. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "국소성",
        "en": "Locality"
      },
      "explanation": {
        "ko": "끝점에서는 분모가 0이라 같은 정리를 쓸 수 없습니다.",
        "en": "At the endpoint the denominator vanishes."
      },
      "formula": "F_y=0: theorem inapplicable",
      "diagram": {
        "ko": "암묵함수·역함수 정리의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Implicit and Inverse Function Theorems. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "정칙 원의 가지",
        "en": "Regular circle branch"
      },
      "predict": {
        "ko": "기울기는?",
        "en": "What is the slope?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "−0.75입니다.",
        "en": "Minus 0.75."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "수직 접선",
        "en": "Vertical tangent"
      },
      "predict": {
        "ko": "끝점 기울기는 0인가요?",
        "en": "Is the endpoint slope zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 유한 도함수가 없어 —입니다.",
        "en": "No; no finite derivative is available."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "역함수는 있지만",
        "en": "Inverse without regular derivative"
      },
      "predict": {
        "ko": "도함수 0이면 역함수도 없나요?",
        "en": "Does a zero derivative rule out an inverse?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. x³의 역은 존재하지만 원점에서 미분 가능하지 않습니다.",
        "en": "No. The cube has an inverse, but the inverse is not differentiable at zero."
      },
      "preset": 2
    }
  ]
},
"matrix-trace-logdet-derivatives":{
  "intuition": {
    "ko": "행렬 미분의 형상과 양의 정부호 정의역을 명시하고 차분으로 검산합니다.",
    "en": "Specify derivative shapes and the SPD domain, then check with finite differences."
  },
  "steps": [
    {
      "title": {
        "ko": "미분 형상",
        "en": "Derivative shape"
      },
      "explanation": {
        "ko": "dA는 행렬 방향이고 결과는 스칼라입니다.",
        "en": "dA is a matrix direction; the differential is scalar."
      },
      "formula": "D logdet[A](E)=tr(A⁻¹E)",
      "diagram": {
        "ko": "행렬·trace·logdet 미분의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Matrix, Trace and Log-Determinant Derivatives. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "원소 방향",
        "en": "Entry direction"
      },
      "explanation": {
        "ko": "E₀₀ 방향은 역행렬의 첫 대각을 고릅니다.",
        "en": "Direction E₀₀ selects the first inverse diagonal entry."
      },
      "formula": "∂/∂a=c/(ac−b²)",
      "diagram": {
        "ko": "행렬·trace·logdet 미분의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Matrix, Trace and Log-Determinant Derivatives. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "숫자 검산",
        "en": "Numeric check"
      },
      "explanation": {
        "ko": "기준 행렬에서는 2/3입니다.",
        "en": "The reference derivative is two thirds."
      },
      "formula": "2/(4−1)=2/3",
      "diagram": {
        "ko": "행렬·trace·logdet 미분의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Matrix, Trace and Log-Determinant Derivatives. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "공유 매개변수",
        "en": "Shared parameter"
      },
      "explanation": {
        "ko": "b는 두 대칭 원소를 함께 바꿉니다.",
        "en": "Changing b changes both symmetric entries."
      },
      "formula": "d tr(AᵀA)/db=4b",
      "diagram": {
        "ko": "행렬·trace·logdet 미분의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Matrix, Trace and Log-Determinant Derivatives. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "SPD 내부",
        "en": "Inside SPD domain"
      },
      "predict": {
        "ko": "해석 도함수는?",
        "en": "What is the analytic derivative?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2/3이며 중앙차분과 비교할 수 있습니다.",
        "en": "Two thirds, checked against the central difference."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "특이 경계",
        "en": "Singular boundary"
      },
      "predict": {
        "ko": "특이점의 logdet 도함수는?",
        "en": "What derivative is available at singularity?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "SPD 정의역 밖이라 —입니다.",
        "en": "None in the SPD domain."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "분리된 성분",
        "en": "Diagonal example"
      },
      "predict": {
        "ko": "대각 예제의 도함수는?",
        "en": "What is the derivative in the diagonal example?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1/2입니다.",
        "en": "One half."
      },
      "preset": 2
    }
  ]
},
"vector-fields-divergence-curl":{
  "intuition": {
    "ko": "위치마다 주어진 벡터와 국소 팽창·회전을 구분합니다.",
    "en": "Separate vectors assigned to positions from local expansion and rotation."
  },
  "steps": [
    {
      "title": {
        "ko": "장",
        "en": "Field"
      },
      "explanation": {
        "ko": "회전장은 위치에 수직입니다.",
        "en": "The rotational field is perpendicular to position."
      },
      "formula": "F=(−y,x)",
      "diagram": {
        "ko": "벡터장·발산·회전의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Vector Fields, Divergence and Curl. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "Jacobian",
        "en": "Jacobian"
      },
      "explanation": {
        "ko": "미분 행렬은 상수입니다.",
        "en": "The derivative matrix is constant."
      },
      "formula": "J=[0,−1;1,0]",
      "diagram": {
        "ko": "벡터장·발산·회전의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Vector Fields, Divergence and Curl. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "발산",
        "en": "Divergence"
      },
      "explanation": {
        "ko": "대각합이 0이라 국소 면적 팽창률은 0입니다.",
        "en": "Trace zero gives zero local area expansion rate."
      },
      "formula": "div F=0",
      "diagram": {
        "ko": "벡터장·발산·회전의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Vector Fields, Divergence and Curl. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "회전",
        "en": "Curl"
      },
      "explanation": {
        "ko": "교차 미분의 차이는 2입니다.",
        "en": "The cross-derivative difference is two."
      },
      "formula": "curl_z=1−(−1)=2",
      "diagram": {
        "ko": "벡터장·발산·회전의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Vector Fields, Divergence and Curl. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "회전장",
        "en": "Rotational field"
      },
      "predict": {
        "ko": "도는 장이면 발산도 있나요?",
        "en": "Does a rotating field necessarily diverge?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 회전장의 발산은 0입니다.",
        "en": "No. This rotational field has zero divergence."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "방사장",
        "en": "Radial field"
      },
      "predict": {
        "ko": "방사장의 curl은?",
        "en": "What is the radial field’s curl?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0이며 발산은 2입니다.",
        "en": "Zero; its divergence is two."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "전단장",
        "en": "Shear field"
      },
      "predict": {
        "ko": "전단도 회전 성분을 가질까요?",
        "en": "Can shear contain local rotation?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "그렇습니다. 이 예제의 curl_z는 −1입니다.",
        "en": "Yes. This shear has curl_z=−1."
      },
      "preset": 2
    }
  ]
},
"line-surface-volume-integrals":{
  "intuition": {
    "ko": "같은 곡선에서도 접선 방향의 일과 법선 방향의 유출은 다릅니다.",
    "en": "Work along a tangent and outward flux use the same curve with different directions."
  },
  "steps": [
    {
      "title": {
        "ko": "매개화",
        "en": "Parametrize"
      },
      "explanation": {
        "ko": "반지름 1 원을 반시계로 돕니다.",
        "en": "Parametrize the unit circle counterclockwise."
      },
      "formula": "r(t)=(cos t,sin t)",
      "diagram": {
        "ko": "선·면·체적 적분의 의미의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Line, Surface and Volume Integrals. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "접선",
        "en": "Tangent"
      },
      "explanation": {
        "ko": "회전장은 접선과 같습니다.",
        "en": "The rotational field equals the tangent."
      },
      "formula": "F(r)=r′",
      "diagram": {
        "ko": "선·면·체적 적분의 의미의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Line, Surface and Volume Integrals. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "법선",
        "en": "Normal"
      },
      "explanation": {
        "ko": "회전장은 바깥 법선과 직교합니다.",
        "en": "The rotational field is orthogonal to the outward normal."
      },
      "formula": "F·n=0",
      "diagram": {
        "ko": "선·면·체적 적분의 의미의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Line, Surface and Volume Integrals. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "누적",
        "en": "Accumulate"
      },
      "explanation": {
        "ko": "접선 일은 2π이고 유출은 0입니다.",
        "en": "Circulation is 2π and flux zero."
      },
      "formula": "∮F·dr=2π",
      "diagram": {
        "ko": "선·면·체적 적분의 의미의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Line, Surface and Volume Integrals. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "원 위의 회전",
        "en": "Rotation on a circle"
      },
      "predict": {
        "ko": "접선 적분과 유출은?",
        "en": "What are circulation and flux?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2π와 0입니다.",
        "en": "2π and zero."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "원 밖으로 유출",
        "en": "Outward radial flux"
      },
      "predict": {
        "ko": "방사장에서는?",
        "en": "What about the radial field?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "접선 적분 0, 유출 2π입니다.",
        "en": "Circulation zero, flux 2π."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "퇴화 원",
        "en": "Zero-radius limit"
      },
      "predict": {
        "ko": "반지름이 0이면?",
        "en": "What happens at radius zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "퇴화 극한에서 모든 적분이 0입니다.",
        "en": "All displayed integrals vanish in the degenerate limit."
      },
      "preset": 2
    }
  ]
},
"gauss-stokes-theorems":{
  "intuition": {
    "ko": "경계의 적분과 내부의 미분 연산을 연결하되 매끄러움과 방향 조건을 확인합니다.",
    "en": "Connect boundary integrals with interior derivatives while checking smoothness and orientation."
  },
  "steps": [
    {
      "title": {
        "ko": "장과 영역",
        "en": "Field and domain"
      },
      "explanation": {
        "ko": "구와 적도 원판에 매끄러운 장을 둡니다.",
        "en": "Use a smooth field on the ball and equatorial disk."
      },
      "formula": "F=(x−y,y+x,z)",
      "diagram": {
        "ko": "Gauss·Stokes 정리와 방향의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Gauss, Stokes and Orientation. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "Gauss",
        "en": "Gauss"
      },
      "explanation": {
        "ko": "발산 3을 구의 체적에 적분합니다.",
        "en": "Integrate divergence three over the ball."
      },
      "formula": "3·4π/3=4π",
      "diagram": {
        "ko": "Gauss·Stokes 정리와 방향의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Gauss, Stokes and Orientation. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "Stokes",
        "en": "Stokes"
      },
      "explanation": {
        "ko": "curl의 z 성분 2를 원판에 적분합니다.",
        "en": "Integrate curl_z=2 over the disk."
      },
      "formula": "2·π=2π",
      "diagram": {
        "ko": "Gauss·Stokes 정리와 방향의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Gauss, Stokes and Orientation. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "방향과 가정",
        "en": "Orientation and assumptions"
      },
      "explanation": {
        "ko": "구는 바깥 법선, 원판은 +z 방향입니다.",
        "en": "Use the outward sphere normal and +z disk normal."
      },
      "formula": "boundary orientation matters",
      "diagram": {
        "ko": "Gauss·Stokes 정리와 방향의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Gauss, Stokes and Orientation. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "유출과 순환",
        "en": "Flux and circulation"
      },
      "predict": {
        "ko": "두 내부 적분은?",
        "en": "What are the two interior integrals?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "4π와 2π입니다. 경계 수치 적분과 비교하세요.",
        "en": "4π and 2π; compare against boundary quadratures."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "방향 반전 장",
        "en": "Reversed field"
      },
      "predict": {
        "ko": "장 계수의 부호를 바꾸면?",
        "en": "What happens when field coefficients reverse?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "두 적분의 부호도 바뀝니다.",
        "en": "Both integral signs reverse."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "퇴화 극한",
        "en": "Degenerate limit"
      },
      "predict": {
        "ko": "R=0에도 정리의 표면 가정이 맞나요?",
        "en": "Is R=0 a regular surface satisfying the theorem assumptions?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 표시한 0은 퇴화 극한값입니다.",
        "en": "No. The displayed zeros are degenerate limit values."
      },
      "preset": 2
    }
  ]
},
"higher-order-autodiff":{
  "intuition": {
    "ko": "Hessian 전체와 특정 방향 곱을 구분하고 이차 근사의 나머지를 확인합니다.",
    "en": "Separate the full Hessian from its action on a direction and inspect a quadratic approximation."
  },
  "steps": [
    {
      "title": {
        "ko": "일차 미분",
        "en": "First derivative"
      },
      "explanation": {
        "ko": "gradient는 (xy,x²/2+cos y)입니다.",
        "en": "The gradient is (xy,x²/2+cos y)."
      },
      "formula": "∇f(1,0)=(0,1.5)",
      "diagram": {
        "ko": "고차 미분·Hessian과 방향 곱의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Higher Derivatives, Hessians and Directional Products. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "Hessian",
        "en": "Hessian"
      },
      "explanation": {
        "ko": "혼합 편미분은 x로 서로 같습니다.",
        "en": "The mixed partials both equal x."
      },
      "formula": "H(1,0)=[0,1;1,0]",
      "diagram": {
        "ko": "고차 미분·Hessian과 방향 곱의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Higher Derivatives, Hessians and Directional Products. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "방향 곱",
        "en": "Directional product"
      },
      "explanation": {
        "ko": "v=(1,1)에서 Hv=(1,1)입니다.",
        "en": "For v=(1,1), Hv=(1,1)."
      },
      "formula": "vᵀHv=2",
      "diagram": {
        "ko": "고차 미분·Hessian과 방향 곱의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Higher Derivatives, Hessians and Directional Products. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "이차 근사",
        "en": "Quadratic approximation"
      },
      "explanation": {
        "ko": "경로의 이차 근사는 1.5t+t²입니다.",
        "en": "The path approximation is 1.5t+t²."
      },
      "formula": "f(x+tv)≈f+t∇f·v+½t²vᵀHv",
      "diagram": {
        "ko": "고차 미분·Hessian과 방향 곱의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Higher Derivatives, Hessians and Directional Products. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "혼합 방향",
        "en": "Mixed direction"
      },
      "predict": {
        "ko": "방향 곡률은?",
        "en": "What is the directional curvature?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "2이며 Hessian 고유값과는 다른 양입니다.",
        "en": "Two, a directional quadratic value rather than an eigenvalue."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "x 방향",
        "en": "Along x"
      },
      "predict": {
        "ko": "x 방향 곡률은?",
        "en": "What is curvature along x?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "이 기준점에서는 0입니다.",
        "en": "Zero at this base point."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "영 방향",
        "en": "Zero direction"
      },
      "predict": {
        "ko": "영 방향의 변화는?",
        "en": "What changes along the zero direction?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "값과 모든 방향 변화가 일정합니다.",
        "en": "The value is constant and directional changes vanish."
      },
      "preset": 2
    }
  ]
},

"vector-spaces-rank":{
  "intuition": {
    "ko": "독립인 방향의 수와 사라지는 입력의 차원을 함께 계산합니다.",
    "en": "Count independent output directions and input directions lost by a linear map."
  },
  "steps": [
    {
      "title": {
        "ko": "생성 공간",
        "en": "Span"
      },
      "explanation": {
        "ko": "두 열은 (1,2),(2,4)로 한 직선 위에 있습니다.",
        "en": "Columns (1,2) and (2,4) lie on one line."
      },
      "formula": "v₁=2v₀",
      "diagram": {
        "ko": "벡터공간·기저·랭크·영공간의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Vector Spaces, Bases, Rank and Nullspace. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "기저 선택",
        "en": "Choose a basis"
      },
      "explanation": {
        "ko": "첫 열 하나로 같은 열공간을 생성합니다.",
        "en": "The first column alone spans the column space."
      },
      "formula": "basis={v₀}",
      "diagram": {
        "ko": "벡터공간·기저·랭크·영공간의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Vector Spaces, Bases, Rank and Nullspace. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "사라지는 방향",
        "en": "Lost direction"
      },
      "explanation": {
        "ko": "입력 (−2,1)은 출력 0으로 갑니다.",
        "en": "Input (−2,1) maps to zero."
      },
      "formula": "A(−2,1)ᵀ=0",
      "diagram": {
        "ko": "벡터공간·기저·랭크·영공간의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Vector Spaces, Bases, Rank and Nullspace. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "차원 보존식",
        "en": "Dimension accounting"
      },
      "explanation": {
        "ko": "출력 차원 1과 영공간 차원 1의 합은 입력 차원 2입니다.",
        "en": "Rank one plus nullity one equals input dimension two."
      },
      "formula": "rank+nullity=2",
      "diagram": {
        "ko": "벡터공간·기저·랭크·영공간의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Vector Spaces, Bases, Rank and Nullspace. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "랭크 1",
        "en": "Rank one"
      },
      "predict": {
        "ko": "두 열이 서로 다른데 독립일까요?",
        "en": "Are distinct columns necessarily independent?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 둘째 열이 첫 열의 두 배입니다.",
        "en": "No: the second is twice the first."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "랭크 2",
        "en": "Rank two"
      },
      "predict": {
        "ko": "항등행렬의 사라지는 방향은?",
        "en": "What nonzero input does the identity lose?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "없습니다. 영공간은 {0}, 차원은 0입니다.",
        "en": "None; its nullspace is {0}, dimension zero."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "영 사상",
        "en": "Zero map"
      },
      "predict": {
        "ko": "모든 출력이 0이면 입력 방향도 없는가요?",
        "en": "If every output is zero, are there no input directions?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "입력 공간은 그대로이며 모두 영공간에 속합니다.",
        "en": "The whole input space remains and lies in the nullspace."
      },
      "preset": 2
    }
  ]
},
"determinants-traces":{
  "intuition": {
    "ko": "면적의 부호 있는 변화와 대각합의 기저 불변성을 구분합니다.",
    "en": "Distinguish oriented area scaling from the basis-invariant diagonal sum."
  },
  "steps": [
    {
      "title": {
        "ko": "열의 면적",
        "en": "Column area"
      },
      "explanation": {
        "ko": "회전한 두 열은 여전히 면적 1을 만듭니다.",
        "en": "The rotated columns still enclose unit area."
      },
      "formula": "det A=1",
      "diagram": {
        "ko": "행렬식·trace와 기저 불변량의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Determinants, Traces and Basis Invariants. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "방향의 부호",
        "en": "Orientation sign"
      },
      "explanation": {
        "ko": "반사는 면적 크기는 유지하고 방향 부호를 바꿉니다.",
        "en": "Reflection preserves area magnitude but reverses orientation."
      },
      "formula": "det diag(1,−1)=−1",
      "diagram": {
        "ko": "행렬식·trace와 기저 불변량의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Determinants, Traces and Basis Invariants. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "대각합",
        "en": "Diagonal sum"
      },
      "explanation": {
        "ko": "90도 회전의 trace는 0입니다.",
        "en": "A quarter turn has trace zero."
      },
      "formula": "tr A=0",
      "diagram": {
        "ko": "행렬식·trace와 기저 불변량의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Determinants, Traces and Basis Invariants. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "고유값의 영역",
        "en": "Eigenvalue field"
      },
      "explanation": {
        "ko": "실수 고유방향 없이 복소 고유값 ±i를 가집니다.",
        "en": "It has complex eigenvalues ±i and no real eigenvectors."
      },
      "formula": "λ²+1=0",
      "diagram": {
        "ko": "행렬식·trace와 기저 불변량의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Determinants, Traces and Basis Invariants. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "90도 회전",
        "en": "Quarter turn"
      },
      "predict": {
        "ko": "면적을 보존하면 항등행렬인가요?",
        "en": "Does preserving area imply the identity?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 90도 회전이 반례입니다.",
        "en": "No: a quarter turn is a counterexample."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "반사",
        "en": "Reflection"
      },
      "predict": {
        "ko": "행렬식 −1이면 면적도 음수인가요?",
        "en": "Does determinant −1 mean negative physical area?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "면적 크기는 1이며 부호는 방향입니다.",
        "en": "Area magnitude is one; the sign records orientation."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "면적 붕괴",
        "en": "Area collapse"
      },
      "predict": {
        "ko": "면적 0이면 모든 벡터가 0이 되나요?",
        "en": "Does zero area force all outputs to zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 랭크 1로 직선에 남을 수 있습니다.",
        "en": "No: rank one can retain a line."
      },
      "preset": 2
    }
  ]
},
"positive-semidefinite-matrices":{
  "intuition": {
    "ko": "대각 원소가 양수인 것과 모든 방향의 이차형식이 음이 아닌 것은 다릅니다.",
    "en": "Positive diagonal entries alone do not guarantee nonnegative quadratic values in every direction."
  },
  "steps": [
    {
      "title": {
        "ko": "모든 방향",
        "en": "All directions"
      },
      "explanation": {
        "ko": "단위 원의 모든 방향에서 q를 생각합니다.",
        "en": "Consider q in every unit direction."
      },
      "formula": "q(v)=vᵀAv",
      "diagram": {
        "ko": "대칭·양의 준정부호 행렬의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Symmetric and Positive Semidefinite Matrices. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "직교 분해",
        "en": "Orthogonal decomposition"
      },
      "explanation": {
        "ko": "대칭행렬은 직교 고유기저에서 대각화됩니다.",
        "en": "A symmetric matrix diagonalizes in an orthonormal eigenbasis."
      },
      "formula": "q=λ₁z₁²+λ₂z₂²",
      "diagram": {
        "ko": "대칭·양의 준정부호 행렬의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Symmetric and Positive Semidefinite Matrices. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "정부호 조건",
        "en": "Definiteness condition"
      },
      "explanation": {
        "ko": "기준 행렬의 고유값은 3,1로 모두 양수입니다.",
        "en": "The reference eigenvalues are three and one, both positive."
      },
      "formula": "λmin=1>0",
      "diagram": {
        "ko": "대칭·양의 준정부호 행렬의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Symmetric and Positive Semidefinite Matrices. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "특이성 구분",
        "en": "Distinguish singularity"
      },
      "explanation": {
        "ko": "0 고유값은 PSD에 허용되지만 PD에는 허용되지 않습니다.",
        "en": "A zero eigenvalue is allowed for PSD but not PD."
      },
      "formula": "PSD ≠ PD",
      "diagram": {
        "ko": "대칭·양의 준정부호 행렬의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Symmetric and Positive Semidefinite Matrices. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "양의 정부호",
        "en": "Positive definite"
      },
      "predict": {
        "ko": "q의 최솟값은?",
        "en": "What is the minimum quadratic value on the unit circle?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "최소 고유값 1입니다.",
        "en": "It equals the minimum eigenvalue, one."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "준정부호·특이",
        "en": "Semidefinite and singular"
      },
      "predict": {
        "ko": "고유값 0이면 PSD가 아닌가요?",
        "en": "Does a zero eigenvalue rule out PSD?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. (1,−1) 방향에서 0이지만 다른 방향은 음이 아닙니다.",
        "en": "No: it is zero along (1,−1), and nonnegative elsewhere."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "양의 대각·부정부호",
        "en": "Positive diagonal, indefinite"
      },
      "predict": {
        "ko": "대각 원소가 양수면 충분할까요?",
        "en": "Are positive diagonal entries sufficient?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 최소 고유값 −1이라 음수 방향이 있습니다.",
        "en": "No: minimum eigenvalue −1 supplies a negative direction."
      },
      "preset": 2
    }
  ]
},
"quadratic-forms-orders":{
  "intuition": {
    "ko": "한 방향의 상한과 모든 방향에서의 행렬 상한은 다른 주장입니다.",
    "en": "A bound in one direction differs from a matrix bound valid in all directions."
  },
  "steps": [
    {
      "title": {
        "ko": "방향별 값",
        "en": "Directional value"
      },
      "explanation": {
        "ko": "θ=45도는 최대 고유방향입니다.",
        "en": "The 45-degree probe is a maximum-eigenvalue direction."
      },
      "formula": "vᵀAv=3",
      "diagram": {
        "ko": "이차형식·행렬 부등식의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Quadratic Forms and Matrix Inequalities. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "Rayleigh 경계",
        "en": "Rayleigh bounds"
      },
      "explanation": {
        "ko": "모든 단위 방향의 값은 고유값 사이입니다.",
        "en": "Every unit-direction value lies between the extreme eigenvalues."
      },
      "formula": "1≤vᵀAv≤3",
      "diagram": {
        "ko": "이차형식·행렬 부등식의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Quadratic Forms and Matrix Inequalities. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "행렬 상한",
        "en": "Matrix bound"
      },
      "explanation": {
        "ko": "C=3이면 모든 방향의 상한입니다.",
        "en": "C=3 bounds all directions."
      },
      "formula": "3I−A⪰0",
      "diagram": {
        "ko": "이차형식·행렬 부등식의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Quadratic Forms and Matrix Inequalities. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "유한 검사 한계",
        "en": "Limits of a probe"
      },
      "explanation": {
        "ko": "θ=0에서 C=2를 통과해도 45도에서 실패합니다.",
        "en": "Passing C=2 at zero degrees still fails at 45 degrees."
      },
      "formula": "∀v is stronger than one probe",
      "diagram": {
        "ko": "이차형식·행렬 부등식의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Quadratic Forms and Matrix Inequalities. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "성립하는 상한",
        "en": "Valid bound"
      },
      "predict": {
        "ko": "C=3은 모든 방향에서 맞나요?",
        "en": "Does C=3 hold in every direction?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "그렇습니다. λmax=3입니다.",
        "en": "Yes: λmax=3."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "한 방향만 통과",
        "en": "One direction passes"
      },
      "predict": {
        "ko": "검사 방향에서 통과하면 증명인가요?",
        "en": "Does passing one probe prove the matrix bound?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. θ=0에서 2지만 45도에서 3입니다.",
        "en": "No: the value is two at zero degrees and three at 45 degrees."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "영 행렬",
        "en": "Zero matrix"
      },
      "predict": {
        "ko": "0⪯0은 성립하나요?",
        "en": "Does 0⪯0 hold?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "차이가 영행렬이므로 성립합니다.",
        "en": "Yes: the difference is the PSD zero matrix."
      },
      "preset": 2
    }
  ]
},
"kronecker-products":{
  "intuition": {
    "ko": "행렬을 벡터로 펼칠 때 열 우선 순서와 곱의 순서를 함께 고정합니다.",
    "en": "Fix the vectorization convention before choosing the Kronecker factor order."
  },
  "steps": [
    {
      "title": {
        "ko": "블록 곱",
        "en": "Block construction"
      },
      "explanation": {
        "ko": "A의 원소 하나마다 B 전체를 곱한 블록을 놓습니다.",
        "en": "Each entry of A scales an entire B block."
      },
      "formula": "(A⊗B)_(2i+k,2j+l)=A_ij B_kl",
      "diagram": {
        "ko": "Kronecker 곱·벡터화의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Kronecker Products and Vectorization. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "벡터 순서",
        "en": "Vector order"
      },
      "explanation": {
        "ko": "열을 차례로 쌓아 (1,3,2,4)를 얻습니다.",
        "en": "Stack columns to obtain (1,3,2,4)."
      },
      "formula": "vec(X)=(1,3,2,4)",
      "diagram": {
        "ko": "Kronecker 곱·벡터화의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Kronecker Products and Vectorization. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "인자 순서",
        "en": "Factor order"
      },
      "explanation": {
        "ko": "왼쪽 작용 A와 오른쪽 작용 Bᵀ는 B⊗A로 바뀝니다.",
        "en": "Left action A and right action Bᵀ give B⊗A."
      },
      "formula": "vec(AXBᵀ)=(8,6,6,4)",
      "diagram": {
        "ko": "Kronecker 곱·벡터화의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Kronecker Products and Vectorization. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "독립 검산",
        "en": "Independent check"
      },
      "explanation": {
        "ko": "직접 행렬 곱과 Kronecker 작용의 차이를 계산합니다.",
        "en": "Compare direct matrix multiplication with the Kronecker action."
      },
      "formula": "residual=0",
      "diagram": {
        "ko": "Kronecker 곱·벡터화의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Kronecker Products and Vectorization. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "열 우선 벡터화",
        "en": "Column-major vectorization"
      },
      "predict": {
        "ko": "A⊗B와 B⊗A가 같을까요?",
        "en": "Are A⊗B and B⊗A equal?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "일반적으로 다릅니다. 잘못된 순서는 (6,7,4,4)를 냅니다.",
        "en": "Generally not: the wrong order gives (6,7,4,4)."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "항등 인자",
        "en": "Identity factors"
      },
      "predict": {
        "ko": "두 인자가 항등이면?",
        "en": "What if both factors are identities?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "4차원 항등 작용으로 vec(X)가 유지됩니다.",
        "en": "The four-dimensional identity preserves vec(X)."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "특이 인자",
        "en": "Singular factor"
      },
      "predict": {
        "ko": "인자 하나가 특이면 전체도 특이한가요?",
        "en": "Does a singular factor make the product singular?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "예. 이 예제에서는 랭크가 2입니다.",
        "en": "Yes; here the rank is two."
      },
      "preset": 2
    }
  ]
},

"counting-permutations":{
  "intuition": {
    "ko": "순서와 중복 허용 여부를 먼저 정하면 셀 대상이 명확해집니다.",
    "en": "Define order and replacement before counting outcomes."
  },
  "steps": [
    {
      "title": {
        "ko": "대상 정의",
        "en": "Define outcomes"
      },
      "explanation": {
        "ko": "5개 중 순서 있게 3개를 고릅니다.",
        "en": "Choose an ordered triple from five distinct items."
      },
      "formula": "n=5, k=3",
      "diagram": {
        "ko": "세기·순열·조합의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Counting, Permutations and Combinations. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "곱의 법칙",
        "en": "Product rule"
      },
      "explanation": {
        "ko": "매번 선택지가 하나씩 줄어듭니다.",
        "en": "Each choice removes one available item."
      },
      "formula": "5·4·3=60",
      "diagram": {
        "ko": "세기·순열·조합의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Counting, Permutations and Combinations. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "중복 세기 제거",
        "en": "Remove repeated counts"
      },
      "explanation": {
        "ko": "무순서 집합마다 3!개 순서가 있습니다.",
        "en": "Each unordered subset has 3! orderings."
      },
      "formula": "60/3!=10",
      "diagram": {
        "ko": "세기·순열·조합의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Counting, Permutations and Combinations. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "빈 대상",
        "en": "Empty object"
      },
      "explanation": {
        "ko": "아무것도 고르지 않는 결과도 하나의 대상입니다.",
        "en": "The empty selection is itself one object."
      },
      "formula": "C(0,0)=1",
      "diagram": {
        "ko": "세기·순열·조합의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Counting, Permutations and Combinations. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "순서 있는 선택",
        "en": "Ordered selection"
      },
      "predict": {
        "ko": "60과 10 중 어느 값일까요?",
        "en": "Will the count be 60 or 10?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "순서를 구분하므로 60입니다.",
        "en": "Order matters, giving 60."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "순서 없는 선택",
        "en": "Unordered selection"
      },
      "predict": {
        "ko": "순서를 지우면 몇 배 줄까요?",
        "en": "By what factor does removing order reduce this count?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "3!=6배 줄어 10입니다.",
        "en": "It falls by 3!=6, giving 10."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "빈 선택",
        "en": "Empty selection"
      },
      "predict": {
        "ko": "빈 집합에서 0개를 뽑을 수 있나요?",
        "en": "Can you choose zero items from an empty set?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "빈 선택 하나가 있으므로 1입니다.",
        "en": "Yes: exactly one empty selection."
      },
      "preset": 2
    }
  ]
},
"binomial-inclusion-exclusion":{
  "intuition": {
    "ko": "겹친 원소를 두 번 셌다면 한 번 빼서 각 원소의 기여를 맞춥니다.",
    "en": "Correct double counting by tracking each element’s contribution."
  },
  "steps": [
    {
      "title": {
        "ko": "소속 확인",
        "en": "Check membership"
      },
      "explanation": {
        "ko": "A={0,1,2}, B={2,3,4}입니다.",
        "en": "A={0,1,2} and B={2,3,4}."
      },
      "formula": "A mask=7, B mask=28",
      "diagram": {
        "ko": "이항계수·포함배제의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Binomial Coefficients and Inclusion–Exclusion. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "중복 보정",
        "en": "Correct overlap"
      },
      "explanation": {
        "ko": "원소 2가 두 번 세어졌습니다.",
        "en": "Element 2 was counted twice."
      },
      "formula": "3+3−1=5",
      "diagram": {
        "ko": "이항계수·포함배제의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Binomial Coefficients and Inclusion–Exclusion. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "부분집합 분류",
        "en": "Classify subsets"
      },
      "explanation": {
        "ko": "크기 3인 부분집합은 20개입니다.",
        "en": "There are 20 subsets of size three."
      },
      "formula": "C(6,3)=20",
      "diagram": {
        "ko": "이항계수·포함배제의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Binomial Coefficients and Inclusion–Exclusion. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "모든 크기 합",
        "en": "Sum all sizes"
      },
      "explanation": {
        "ko": "각 원소는 넣거나 빼는 두 선택을 가집니다.",
        "en": "Each element has two membership choices."
      },
      "formula": "ΣₖC(6,k)=2⁶=64",
      "diagram": {
        "ko": "이항계수·포함배제의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Binomial Coefficients and Inclusion–Exclusion. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "한 원소 중첩",
        "en": "One shared element"
      },
      "predict": {
        "ko": "합집합 크기는 6일까요?",
        "en": "Is the union size six?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "겹친 원소 하나를 빼면 5입니다.",
        "en": "Subtract the shared element: five."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "같은 집합",
        "en": "Identical sets"
      },
      "predict": {
        "ko": "같은 집합을 합하면 두 배가 되나요?",
        "en": "Does a set union itself double its size?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 교집합도 같은 집합이므로 크기는 3입니다.",
        "en": "No: the intersection is the same set, so the union size is three."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "전체와 공집합",
        "en": "Full and empty sets"
      },
      "predict": {
        "ko": "전체 집합의 크기 6인 부분집합 수는?",
        "en": "How many size-six subsets does the universe have?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "전체 집합 하나입니다.",
        "en": "Exactly one: the universe itself."
      },
      "preset": 2
    }
  ]
},
"recurrences-generating-functions":{
  "intuition": {
    "ko": "이전 상태를 잇는 규칙을 계수로 저장하면 무한 수열을 하나의 형식적 식으로 다룰 수 있습니다.",
    "en": "Store a recurrence as coefficients of a formal series, then distinguish algebra from convergence."
  },
  "steps": [
    {
      "title": {
        "ko": "초깃값",
        "en": "Initial values"
      },
      "explanation": {
        "ko": "Fibonacci 계수는 0,1에서 시작합니다.",
        "en": "The Fibonacci coefficients start at zero and one."
      },
      "formula": "a₀=0, a₁=1",
      "diagram": {
        "ko": "점화식·생성함수의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Recurrences and Generating Functions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "국소 규칙",
        "en": "Local rule"
      },
      "explanation": {
        "ko": "이전 두 계수의 합으로 다음 값을 만듭니다.",
        "en": "Add the preceding two coefficients."
      },
      "formula": "a₈=13+8=21",
      "diagram": {
        "ko": "점화식·생성함수의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Recurrences and Generating Functions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "형식적 정리",
        "en": "Formal rearrangement"
      },
      "explanation": {
        "ko": "지수 이동으로 점화식을 한 식에 모읍니다.",
        "en": "Shift exponents to collect the recurrence."
      },
      "formula": "(1−x−x²)G=x",
      "diagram": {
        "ko": "점화식·생성함수의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Recurrences and Generating Functions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "해석적 조건",
        "en": "Analytic condition"
      },
      "explanation": {
        "ko": "x=0.5는 수렴 반지름 약 0.618 안입니다.",
        "en": "x=0.5 lies inside radius about 0.618."
      },
      "formula": "|x|<1/φ",
      "diagram": {
        "ko": "점화식·생성함수의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Recurrences and Generating Functions. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "수렴 범위 안",
        "en": "Inside convergence radius"
      },
      "predict": {
        "ko": "a₈은 얼마일까요?",
        "en": "What is a₈?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "21입니다. 유한 부분합은 2보다 작습니다.",
        "en": "It is 21; the finite weighted partial sum is below two."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "수렴 범위 밖",
        "en": "Outside convergence radius"
      },
      "predict": {
        "ko": "유리식 값이 음수면 급수도 음수일까요?",
        "en": "If the rational expression is negative, is the series negative?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "양의 항 급수는 여기서 발산합니다. 유리식의 연장값과 다릅니다.",
        "en": "The positive-term series diverges here; it does not equal the continued rational value."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "유한 다항식",
        "en": "Finite polynomial"
      },
      "predict": {
        "ko": "p=q=0이면 무한히 많은 비영 계수가 있나요?",
        "en": "Are infinitely many coefficients nonzero when p=q=0?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. G=x로 끝납니다.",
        "en": "No. G=x is a polynomial."
      },
      "preset": 2
    }
  ]
},
"graphs-trees-dags":{
  "intuition": {
    "ko": "방향을 지운 트리와 방향 순환이 없는 그래프는 서로 다른 조건입니다.",
    "en": "An underlying undirected tree and a directed acyclic graph satisfy different conditions."
  },
  "steps": [
    {
      "title": {
        "ko": "간선 지정",
        "en": "Specify edges"
      },
      "explanation": {
        "ko": "마스크 9는 0→1과 1→2를 켭니다.",
        "en": "Mask nine enables 0→1 and 1→2."
      },
      "formula": "9=2⁰+2³",
      "diagram": {
        "ko": "그래프·트리·DAG의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Graphs, Trees and DAGs. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "방향 제거",
        "en": "Forget direction"
      },
      "explanation": {
        "ko": "기저 그래프는 연결되어 있고 간선이 두 개입니다.",
        "en": "The underlying graph is connected with two edges."
      },
      "formula": "|E_undirected|=|V|−1",
      "diagram": {
        "ko": "그래프·트리·DAG의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Graphs, Trees and DAGs. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "순서 만들기",
        "en": "Build an order"
      },
      "explanation": {
        "ko": "모든 간선이 앞에서 뒤로 향합니다.",
        "en": "Every edge points forward in the order."
      },
      "formula": "0≺1≺2",
      "diagram": {
        "ko": "그래프·트리·DAG의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Graphs, Trees and DAGs. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "순환 반례",
        "en": "Cycle obstruction"
      },
      "explanation": {
        "ko": "2→0을 더하면 선형 순서를 만들 수 없습니다.",
        "en": "Adding 2→0 prevents any topological order."
      },
      "formula": "0→1→2→0",
      "diagram": {
        "ko": "그래프·트리·DAG의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Graphs, Trees and DAGs. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "사슬 DAG",
        "en": "Chain DAG"
      },
      "predict": {
        "ko": "위상 순서는 무엇인가요?",
        "en": "What is the topological order?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "(0,1,2)이고 기저 트리입니다.",
        "en": "It is (0,1,2), and the underlying graph is a tree."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "방향 순환",
        "en": "Directed cycle"
      },
      "predict": {
        "ko": "간선을 하나 추가하면 DAG가 유지될까요?",
        "en": "Does adding one edge preserve the DAG?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "마스크 25는 3-순환이므로 위상 순서가 없습니다.",
        "en": "Mask 25 is a three-cycle and has no topological order."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "왕복 간선 트리",
        "en": "Tree with reciprocal edges"
      },
      "predict": {
        "ko": "기저 그래프가 트리이면 항상 DAG인가요?",
        "en": "Does an underlying tree always imply a DAG?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0↔1 때문에 2-순환이 있어 아닙니다.",
        "en": "No: the reciprocal 0↔1 pair forms a two-cycle."
      },
      "preset": 2
    }
  ]
},
"paths-cycles-connectivity":{
  "intuition": {
    "ko": "도달 가능성, 최단 경로, 고정 길이 보행 수는 서로 다른 질문입니다.",
    "en": "Reachability, shortest paths and fixed-length walk counts answer different questions."
  },
  "steps": [
    {
      "title": {
        "ko": "대상 지정",
        "en": "Name endpoints"
      },
      "explanation": {
        "ko": "출발점 0에서 도착점 2를 찾습니다.",
        "en": "Search from source zero to target two."
      },
      "formula": "s=0, t=2",
      "diagram": {
        "ko": "연결성·경로·순환의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Connectivity, Paths and Cycles. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "너비 우선 탐색",
        "en": "Breadth-first search"
      },
      "explanation": {
        "ko": "방문 깊이 순서로 확장하여 간선 수를 최소화합니다.",
        "en": "Expand by depth to minimize edge count."
      },
      "formula": "0→1→2; d=2",
      "diagram": {
        "ko": "연결성·경로·순환의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Connectivity, Paths and Cycles. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "보행 합성",
        "en": "Compose walks"
      },
      "explanation": {
        "ko": "중간 정점별 두 간선의 곱을 합합니다.",
        "en": "Sum products over the intermediate vertex."
      },
      "formula": "(A²)₀₂=ΣᵥA₀ᵥAᵥ₂=1",
      "diagram": {
        "ko": "연결성·경로·순환의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Connectivity, Paths and Cycles. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "연결 방향",
        "en": "Direction matters"
      },
      "explanation": {
        "ko": "0에서 2로 가지만 2에서 0으로 돌아오지 못합니다.",
        "en": "Zero reaches two, but two does not reach zero."
      },
      "formula": "SCC count=3",
      "diagram": {
        "ko": "연결성·경로·순환의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Connectivity, Paths and Cycles. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "두 단계 경로",
        "en": "Two-edge path"
      },
      "predict": {
        "ko": "거리는 몇인가요?",
        "en": "What is the distance?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "경로 (0,1,2), 거리 2, 길이 2 보행 수 1입니다.",
        "en": "Path (0,1,2), distance two, one length-two walk."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "강연결 순환",
        "en": "Strongly connected cycle"
      },
      "predict": {
        "ko": "순환에서 자기 자신까지 거리는 3인가요?",
        "en": "Is distance to oneself three in the cycle?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "길이 0 경로를 허용하므로 거리는 0입니다. 양의 길이 순환은 별도로 있습니다.",
        "en": "Distance is zero because an empty path is allowed; a positive-length cycle also exists."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "도달 불가",
        "en": "Unreachable target"
      },
      "predict": {
        "ko": "도달 불가의 거리는 0인가요?",
        "en": "Is an unreachable distance zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. —로 표시하며 자기 자신까지 거리 0과 구분합니다.",
        "en": "No: it is —, distinct from zero distance to oneself."
      },
      "preset": 2
    }
  ]
},
"boolean-algebra":{
  "intuition": {
    "ko": "진리표·논리식·GF(2) 다항식은 같은 함수를 다른 규칙으로 표현합니다.",
    "en": "Truth tables, logical formulas and GF(2) polynomials represent the same function using different operations."
  },
  "steps": [
    {
      "title": {
        "ko": "입출력 표",
        "en": "Input-output table"
      },
      "explanation": {
        "ko": "마스크 6은 행 01,10에서만 참입니다.",
        "en": "Mask six is true only on rows 01 and 10."
      },
      "formula": "6=0110₂",
      "diagram": {
        "ko": "Boolean 대수와 논리 표현의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Boolean Algebra and Logical Representations. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "최소항",
        "en": "Minterms"
      },
      "explanation": {
        "ko": "참인 각 행을 분리하여 OR로 합칩니다.",
        "en": "Isolate each true row and combine with OR."
      },
      "formula": "F=(¬x∧y)∨(x∧¬y)",
      "diagram": {
        "ko": "Boolean 대수와 논리 표현의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Boolean Algebra and Logical Representations. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "다른 대수",
        "en": "Another algebra"
      },
      "explanation": {
        "ko": "같은 함수는 GF(2)에서 x+y입니다.",
        "en": "The same function is x+y over GF(2)."
      },
      "formula": "F=x⊕y",
      "diagram": {
        "ko": "Boolean 대수와 논리 표현의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Boolean Algebra and Logical Representations. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "동치 검증",
        "en": "Verify equivalence"
      },
      "explanation": {
        "ko": "네 입력 모두 비교해야 진리표 동치를 보입니다.",
        "en": "Compare all four inputs for truth-table equivalence."
      },
      "formula": "x=y=1 ⇒ F=0",
      "diagram": {
        "ko": "Boolean 대수와 논리 표현의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Boolean Algebra and Logical Representations. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "XOR",
        "en": "XOR"
      },
      "predict": {
        "ko": "두 입력이 참이면 XOR도 참인가요?",
        "en": "Is XOR true when both inputs are true?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 1⊕1=0입니다.",
        "en": "No: 1⊕1=0."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "NAND",
        "en": "NAND"
      },
      "predict": {
        "ko": "NAND와 XOR의 차이는 어느 행인가요?",
        "en": "Which row distinguishes NAND from XOR?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "00 행입니다. NAND는 1, XOR는 0입니다.",
        "en": "Row 00: NAND is one and XOR is zero."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "상수 거짓",
        "en": "Constant false"
      },
      "predict": {
        "ko": "참인 최소항이 없으면 출력은?",
        "en": "What is the output with no true minterms?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "항상 0입니다. 빈 OR은 거짓입니다.",
        "en": "Always zero: an empty OR is false."
      },
      "preset": 2
    }
  ]
},
"automata-formal-languages":{
  "intuition": {
    "ko": "전체 입력을 기억하는 대신 앞으로의 판단에 필요한 유한 상태만 남깁니다.",
    "en": "Retain only a finite state sufficient for future acceptance decisions."
  },
  "steps": [
    {
      "title": {
        "ko": "상태의 의미",
        "en": "State meaning"
      },
      "explanation": {
        "ko": "기준 언어는 지금까지 읽은 1의 홀짝만 기억합니다.",
        "en": "The reference language remembers only parity so far."
      },
      "formula": "q₀=even, q₁=odd",
      "diagram": {
        "ko": "오토마타·형식언어의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Automata and Formal Languages. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "입력 소비",
        "en": "Consume input"
      },
      "explanation": {
        "ko": "101의 세 기호를 순서대로 처리합니다.",
        "en": "Process the three symbols of 101 in order."
      },
      "formula": "0→1→1→0",
      "diagram": {
        "ko": "오토마타·형식언어의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Automata and Formal Languages. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "수용 규칙",
        "en": "Acceptance rule"
      },
      "explanation": {
        "ko": "모든 기호를 읽은 뒤 수용 상태인지 봅니다.",
        "en": "Test the state after all symbols are consumed."
      },
      "formula": "δ*(q₀,101)=q₀∈F",
      "diagram": {
        "ko": "오토마타·형식언어의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Automata and Formal Languages. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "언어와 확률",
        "en": "Language versus probability"
      },
      "explanation": {
        "ko": "길이 3의 8개 단어 중 4개가 수용됩니다.",
        "en": "Four of the eight length-three words are accepted."
      },
      "formula": "|L∩{0,1}³|=4",
      "diagram": {
        "ko": "오토마타·형식언어의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Automata and Formal Languages. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "짝수 개의 1",
        "en": "Even number of ones"
      },
      "predict": {
        "ko": "101을 수용하나요?",
        "en": "Is 101 accepted?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "1이 두 개이므로 수용합니다.",
        "en": "Yes: it contains two ones."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "01로 끝남",
        "en": "Ends with 01"
      },
      "predict": {
        "ko": "길이 2의 01로 끝나는 단어 수는?",
        "en": "How many length-two words end in 01?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "01 하나입니다.",
        "en": "Only 01."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "빈 문자열",
        "en": "Empty word"
      },
      "predict": {
        "ko": "빈 문자열이 01로 끝날 수 있나요?",
        "en": "Can the empty word end in 01?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 짝수 홀짝 언어에서는 수용하지만 이 언어에서는 거부합니다.",
        "en": "No. It is accepted by even parity, but rejected by the ending-01 language."
      },
      "preset": 2
    }
  ]
},
"posets-lattices":{
  "intuition": {
    "ko": "비교할 수 없는 두 집합도 공통 하한과 공통 상한을 가질 수 있습니다.",
    "en": "Two incomparable sets can still have a greatest lower bound and a least upper bound."
  },
  "steps": [
    {
      "title": {
        "ko": "부분순서",
        "en": "Partial order"
      },
      "explanation": {
        "ko": "반사성·반대칭성·추이성을 만족하는 포함 관계입니다.",
        "en": "Inclusion is reflexive, antisymmetric and transitive."
      },
      "formula": "A≤B ⇔ A⊆B",
      "diagram": {
        "ko": "부분순서·격자의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Partial Orders and Lattices. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "비교 불가",
        "en": "Incomparability"
      },
      "explanation": {
        "ko": "A={0,1}, B={0,2} 어느 쪽도 다른 쪽을 포함하지 않습니다.",
        "en": "Neither {0,1} nor {0,2} contains the other."
      },
      "formula": "A∥B",
      "diagram": {
        "ko": "부분순서·격자의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Partial Orders and Lattices. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "만남",
        "en": "Meet"
      },
      "explanation": {
        "ko": "모든 공통 하한 중 가장 큰 것은 {0}입니다.",
        "en": "The greatest common lower bound is {0}."
      },
      "formula": "A∧B={0}",
      "diagram": {
        "ko": "부분순서·격자의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Partial Orders and Lattices. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "이음",
        "en": "Join"
      },
      "explanation": {
        "ko": "모든 공통 상한 중 가장 작은 것은 전체 집합입니다.",
        "en": "The least common upper bound is the universe."
      },
      "formula": "A∨B={0,1,2}",
      "diagram": {
        "ko": "부분순서·격자의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Partial Orders and Lattices. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "비교 불가",
        "en": "Incomparable sets"
      },
      "predict": {
        "ko": "비교 불가이면 meet도 없나요?",
        "en": "Does incomparability mean no meet exists?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 교집합 마스크 1, 합집합 마스크 7입니다.",
        "en": "No. The meet mask is one and join mask seven."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "포함 관계",
        "en": "Nested sets"
      },
      "predict": {
        "ko": "A가 B에 포함되면 meet와 join은?",
        "en": "When A⊆B, what are meet and join?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "meet=A, join=B입니다.",
        "en": "Meet is A and join is B."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "최소와 최대",
        "en": "Bottom and top"
      },
      "predict": {
        "ko": "공집합과 전체 집합의 경계 역할은?",
        "en": "What roles do the empty set and universe play?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "각각 최소와 최대 원소입니다.",
        "en": "They are the bottom and top elements."
      },
      "preset": 2
    }
  ]
},
"finite-state-transitions":{
  "intuition": {
    "ko": "결정적 유한계에서는 같은 상태를 다시 만나면 이후 궤적이 반복됩니다.",
    "en": "In a deterministic finite system, revisiting a state forces the future orbit to repeat."
  },
  "steps": [
    {
      "title": {
        "ko": "전이 함수",
        "en": "Transition map"
      },
      "explanation": {
        "ko": "코드 7은 T=(1,2,0)입니다.",
        "en": "Code seven gives T=(1,2,0)."
      },
      "formula": "7=1+3·2+9·0",
      "diagram": {
        "ko": "유한 상태와 전이의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Finite States and Transitions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "함수 합성",
        "en": "Function composition"
      },
      "explanation": {
        "ko": "0에서 다섯 번 전이하면 2입니다.",
        "en": "Starting at zero, five transitions end at two."
      },
      "formula": "T⁵(0)=2",
      "diagram": {
        "ko": "유한 상태와 전이의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Finite States and Transitions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "반복 보장",
        "en": "Guaranteed repetition"
      },
      "explanation": {
        "ko": "세 상태에서 네 번 관측하면 중복이 있습니다.",
        "en": "Four observations of three possible states force a repeat."
      },
      "formula": "pigeonhole principle",
      "diagram": {
        "ko": "유한 상태와 전이의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Finite States and Transitions. Read the structure and numbers alongside the argument."
      }
    },
    {
      "title": {
        "ko": "과도와 주기",
        "en": "Transient and period"
      },
      "explanation": {
        "ko": "기준 예제는 과도 구간 없이 주기 3입니다.",
        "en": "The reference has no transient and period three."
      },
      "formula": "μ=0, λ=3",
      "diagram": {
        "ko": "유한 상태와 전이의 고정 기준 예제. 본문 논증과 수치·구조를 함께 읽으세요.",
        "en": "Fixed reference example for Finite States and Transitions. Read the structure and numbers alongside the argument."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "3-순환",
        "en": "Three-cycle"
      },
      "predict": {
        "ko": "다섯 번 전이 뒤 상태는?",
        "en": "What is the state after five transitions?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "0,1,2,0,1,2 순서로 최종 2입니다.",
        "en": "The trace 0,1,2,0,1,2 ends at two."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "과도 뒤 2-순환",
        "en": "Transient then two-cycle"
      },
      "predict": {
        "ko": "주기가 2이면 처음부터 반복하나요?",
        "en": "Does period two imply repetition starts immediately?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 코드 17은 과도 길이 1 뒤 2와 1을 반복합니다.",
        "en": "No. Code 17 has transient length one, then alternates two and one."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "흡수 고정점",
        "en": "Absorbing fixed point"
      },
      "predict": {
        "ko": "고정점의 주기는 0인가요?",
        "en": "Is a fixed point’s period zero?"
      },
      "action": {
        "ko": "예제를 불러와 표·구조·수치를 비교하고, 조건 하나를 바꿔 예상과 대조하세요.",
        "en": "Load the example, compare the table, structure and readout, then change one condition and check your prediction."
      },
      "explain": {
        "ko": "아니요. 최소 양의 주기는 1입니다.",
        "en": "No. Its least positive period is one."
      },
      "preset": 2
    }
  ]
},

"numbers-ratios-units":{
  "intuition": {
    "ko": "같은 양의 다른 표현과 서로 다른 차원의 양을 구분합니다.",
    "en": "Distinguish different representations of the same quantity from quantities with different dimensions."
  },
  "steps": [
    {
      "title": {
        "ko": "같은 수",
        "en": "Same number"
      },
      "explanation": {
        "ko": "1/3과 2/6은 같은 유리수입니다.",
        "en": "1/3 and 2/6 represent the same rational number."
      },
      "formula": "1/3 = 2/6",
      "diagram": {
        "ko": "고정 기준 예제의 수·분수·비율과 단위 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Numbers, Fractions, Ratios and Units. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "표현과 근사",
        "en": "Representation and approximation"
      },
      "explanation": {
        "ko": "유한 소수 표시는 정확한 분수와 구분합니다.",
        "en": "Distinguish a finite decimal display from the exact fraction."
      },
      "formula": "1/3 ≠ 0.333",
      "diagram": {
        "ko": "고정 기준 예제의 수·분수·비율과 단위 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Numbers, Fractions, Ratios and Units. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "단위 바꾸기",
        "en": "Convert units"
      },
      "explanation": {
        "ko": "150 cm는 1.5 m와 같은 길이입니다.",
        "en": "150 cm and 1.5 m describe the same length."
      },
      "formula": "150 cm × (1 m / 100 cm)=1.5 m",
      "diagram": {
        "ko": "고정 기준 예제의 수·분수·비율과 단위 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Numbers, Fractions, Ratios and Units. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "차원 확인",
        "en": "Check dimensions"
      },
      "explanation": {
        "ko": "길이를 시간으로 나누면 속도입니다.",
        "en": "Length divided by time has speed dimensions."
      },
      "formula": "1.5 m / 2 s = 0.75 m/s",
      "diagram": {
        "ko": "고정 기준 예제의 수·분수·비율과 단위 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Numbers, Fractions, Ratios and Units. Read its axes and domain alongside the explanation and numeric readout."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "표현",
        "en": "Representation"
      },
      "predict": {
        "ko": "1/3의 표시값 0.333이 정확한 값인가요?",
        "en": "Is displayed 0.333 the exact value of 1/3?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "아닙니다. 길이 1.5 m, 속도 0.75 m/s도 단위와 함께 확인하세요.",
        "en": "No. Also check length 1.5 m and speed 0.75 m/s with units."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "동치 분수",
        "en": "Equivalent fractions"
      },
      "predict": {
        "ko": "분자·분모를 두 배로 하면 값도 두 배인가요?",
        "en": "Does doubling numerator and denominator double the value?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "2/6은 1/3과 같아 같은 위치입니다.",
        "en": "2/6 equals 1/3 and occupies the same location."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "0으로 나누기",
        "en": "Divide by zero"
      },
      "predict": {
        "ko": "분모나 시간이 0이면 어떤 값이 나올까요?",
        "en": "What value should appear for a zero denominator or time?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "정의되지 않으므로 —로 표시합니다. ∞나 0으로 대체하지 않습니다.",
        "en": "The quantity is undefined and shown as —, not replaced by infinity or zero."
      },
      "preset": 2
    }
  ]
},
"real-complex-numbers":{
  "intuition": {
    "ko": "수의 범위를 넓히고 복소수 곱셈을 회전과 크기로 해석합니다.",
    "en": "Expand number systems and interpret complex multiplication through rotation and magnitude."
  },
  "steps": [
    {
      "title": {
        "ko": "수 체계",
        "en": "Number systems"
      },
      "explanation": {
        "ko": "유리수 밖의 실수도 길이로 나타납니다.",
        "en": "Lengths can be real numbers outside the rationals."
      },
      "formula": "|1+i|=√2",
      "diagram": {
        "ko": "고정 기준 예제의 실수와 복소수 평면 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Real Numbers and the Complex Plane. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "복소 좌표",
        "en": "Complex coordinates"
      },
      "explanation": {
        "ko": "a와 b는 서로 다른 좌표 성분입니다.",
        "en": "a and b are distinct coordinate components."
      },
      "formula": "z=1+i",
      "diagram": {
        "ko": "고정 기준 예제의 실수와 복소수 평면 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Real Numbers and the Complex Plane. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "곱셈과 회전",
        "en": "Multiplication and rotation"
      },
      "explanation": {
        "ko": "i를 곱하면 90도 회전합니다.",
        "en": "Multiplication by i rotates by 90 degrees."
      },
      "formula": "i(1+i)=−1+i",
      "diagram": {
        "ko": "고정 기준 예제의 실수와 복소수 평면 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Real Numbers and the Complex Plane. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "크기와 편각",
        "en": "Magnitude and argument"
      },
      "explanation": {
        "ko": "회전은 크기를 보존하지만 영점의 각은 없습니다.",
        "en": "Rotation preserves magnitude, but zero has no angle."
      },
      "formula": "|−1+i|=√2",
      "diagram": {
        "ko": "고정 기준 예제의 실수와 복소수 평면 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Real Numbers and the Complex Plane. Read its axes and domain alongside the explanation and numeric readout."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "회전",
        "en": "Rotation"
      },
      "predict": {
        "ko": "i를 곱하면 어느 좌표로 가나요?",
        "en": "Where does multiplication by i send the point?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "(1,1)은 (−1,1)로 가고 크기 √2를 유지합니다.",
        "en": "(1,1) moves to (−1,1), preserving magnitude √2."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "부호",
        "en": "Sign"
      },
      "predict": {
        "ko": "180도 회전은 어떤 곱셈인가요?",
        "en": "What multiplication gives a 180-degree rotation?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "−1을 곱해 i를 −i로 보냅니다. 작은 잔차는 반올림 오차일 수 있습니다.",
        "en": "Multiply by −1, sending i to −i. Tiny residuals can be roundoff."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "각의 부재",
        "en": "No angle"
      },
      "predict": {
        "ko": "0의 편각을 0도로 정해도 되나요?",
        "en": "Can zero simply be given argument zero?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "수학적으로 고유한 편각이 없어 —로 표시합니다.",
        "en": "There is no defined argument, so it is shown as —."
      },
      "preset": 2
    }
  ]
},
"equations-inequalities":{
  "intuition": {
    "ko": "해를 하나의 숫자라고 가정하지 않고 없는 경우·여러 경우·구간으로 나눕니다.",
    "en": "Treat solutions as sets, including empty, multiple and interval cases."
  },
  "steps": [
    {
      "title": {
        "ko": "동치 변형",
        "en": "Equivalent transformations"
      },
      "explanation": {
        "ko": "0이 아닌 수로 나눌 때만 역변형이 가능합니다.",
        "en": "Division is reversible only for a nonzero divisor."
      },
      "formula": "2x=4 ⇔ x=2",
      "diagram": {
        "ko": "고정 기준 예제의 방정식·부등식과 해의 집합 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Equations, Inequalities and Solution Sets. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "해 없음과 무한",
        "en": "None and infinitely many"
      },
      "explanation": {
        "ko": "a=0일 때 b가 0인지 먼저 확인합니다.",
        "en": "When a=0, check whether b is zero."
      },
      "formula": "0x=0: all real x",
      "diagram": {
        "ko": "고정 기준 예제의 방정식·부등식과 해의 집합 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Equations, Inequalities and Solution Sets. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "근과 부호",
        "en": "Roots and sign"
      },
      "explanation": {
        "ko": "위로 열린 포물선은 두 근 사이에서 0 이하입니다.",
        "en": "An upward parabola is nonpositive between its roots."
      },
      "formula": "x²−3x+2=(x−1)(x−2)",
      "diagram": {
        "ko": "고정 기준 예제의 방정식·부등식과 해의 집합 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Equations, Inequalities and Solution Sets. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "끝점",
        "en": "Endpoints"
      },
      "explanation": {
        "ko": "≤이므로 두 근도 해에 포함합니다.",
        "en": "The ≤ sign includes both roots."
      },
      "formula": "x∈[1,2]",
      "diagram": {
        "ko": "고정 기준 예제의 방정식·부등식과 해의 집합 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Equations, Inequalities and Solution Sets. Read its axes and domain alongside the explanation and numeric readout."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "교점",
        "en": "Intersection"
      },
      "predict": {
        "ko": "2x=4의 해는 어디인가요?",
        "en": "Where is the solution of 2x=4?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "교점의 x좌표 2이며 검사점 1의 잔차는 −2입니다.",
        "en": "The intersection has x=2; residual at probe 1 is −2."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "구간",
        "en": "Interval"
      },
      "predict": {
        "ko": "근만 해인가요, 그 사이도 해인가요?",
        "en": "Are only roots solutions, or also points between them?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "해는 [1,2], x=1.5에서 이차식은 −0.25입니다.",
        "en": "The solution is [1,2]; at x=1.5 the quadratic is −0.25."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "퇴화",
        "en": "Degeneracy"
      },
      "predict": {
        "ko": "0x=0은 해가 0 하나인가요?",
        "en": "Does 0x=0 have only the solution zero?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "모든 실수가 해입니다. 0x=1처럼 바꾸면 해가 없습니다.",
        "en": "Every real number is a solution; changing to 0x=1 gives none."
      },
      "preset": 2
    }
  ]
},
"polynomial-rational-functions":{
  "intuition": {
    "ko": "식을 약분해도 사라지지 않는 정의역의 제한을 그림에서 확인합니다.",
    "en": "See why simplifying an expression does not erase domain restrictions."
  },
  "steps": [
    {
      "title": {
        "ko": "분자 인수분해",
        "en": "Factor the numerator"
      },
      "explanation": {
        "ko": "분자를 인수분해해 약분에 쓰일 인수를 확인합니다.",
        "en": "Factor the numerator to identify the factor used in cancellation."
      },
      "formula": "x²−1=(x−1)(x+1)",
      "diagram": {
        "ko": "고정 기준 예제의 다항함수·유리함수와 정의역 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Polynomial and Rational Functions: Domains Matter. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "나눗셈의 조건",
        "en": "Division condition"
      },
      "explanation": {
        "ko": "원래 분모 0은 허용하지 않습니다.",
        "en": "The original zero denominator is excluded."
      },
      "formula": "x−1≠0",
      "diagram": {
        "ko": "고정 기준 예제의 다항함수·유리함수와 정의역 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Polynomial and Rational Functions: Domains Matter. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "약분과 구멍",
        "en": "Cancellation and a hole"
      },
      "explanation": {
        "ko": "허용 입력에서 x+1과 같지만 x=1은 빠집니다.",
        "en": "It equals x+1 on allowed inputs but omits x=1."
      },
      "formula": "f(1): undefined",
      "diagram": {
        "ko": "고정 기준 예제의 다항함수·유리함수와 정의역 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Polynomial and Rational Functions: Domains Matter. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "극한과 값",
        "en": "Limit and value"
      },
      "explanation": {
        "ko": "주변 값이 2로 가도 원래 함수의 값은 없습니다.",
        "en": "Nearby values approach 2 without defining the original value."
      },
      "formula": "limₓ→₁ f(x)=2",
      "diagram": {
        "ko": "고정 기준 예제의 다항함수·유리함수와 정의역 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Polynomial and Rational Functions: Domains Matter. Read its axes and domain alongside the explanation and numeric readout."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "구멍",
        "en": "Hole"
      },
      "predict": {
        "ko": "약분한 식을 1에 넣어도 원래 값인가요?",
        "en": "Does substituting one after cancellation give the original value?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "원래 f(1)은 정의되지 않지만 극한은 2입니다.",
        "en": "Original f(1) is undefined although the limit is 2."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "극점",
        "en": "Pole"
      },
      "predict": {
        "ko": "x=0.1과 x=0에서 1/x는 어떻게 다른가요?",
        "en": "How do 1/x at x=0.1 and x=0 differ?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "0.1에서는 10, 0에서는 정의되지 않습니다. 극점 양쪽을 잇지 않습니다.",
        "en": "It is 10 at 0.1 and undefined at zero; the branches are not joined."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "제한 없음",
        "en": "No exclusion"
      },
      "predict": {
        "ko": "x²−1도 x=1을 빼야 하나요?",
        "en": "Must x²−1 also exclude x=1?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "아닙니다. 모든 실수에 정의되며 값은 0입니다.",
        "en": "No. It is defined on all reals and gives zero."
      },
      "preset": 2
    }
  ]
},
"exponentials-logarithms":{
  "intuition": {
    "ko": "곱셈을 덧셈으로 바꾸는 로그와 밑·정의역의 조건을 연결합니다.",
    "en": "Connect logarithms, multiplicative scale and the conditions on bases and domains."
  },
  "steps": [
    {
      "title": {
        "ko": "반복 곱",
        "en": "Repeated multiplication"
      },
      "explanation": {
        "ko": "정수 지수에서 시작해 실수 지수로 확장합니다.",
        "en": "Extend integer repeated multiplication to real exponents."
      },
      "formula": "2²=4",
      "diagram": {
        "ko": "고정 기준 예제의 지수·로그와 척도 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Exponentials, Logarithms and Scale. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "역의 질문",
        "en": "Inverse question"
      },
      "explanation": {
        "ko": "4를 만드는 지수를 묻습니다.",
        "en": "Ask which exponent produces four."
      },
      "formula": "log₂4=2",
      "diagram": {
        "ko": "고정 기준 예제의 지수·로그와 척도 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Exponentials, Logarithms and Scale. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "감소하는 지수",
        "en": "Decreasing exponential"
      },
      "explanation": {
        "ko": "0과 1 사이의 밑은 감소함수입니다.",
        "en": "A base between zero and one gives a decreasing exponential."
      },
      "formula": "(1/2)²=1/4",
      "diagram": {
        "ko": "고정 기준 예제의 지수·로그와 척도 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Exponentials, Logarithms and Scale. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "정의역",
        "en": "Domain"
      },
      "explanation": {
        "ko": "밑 1과 0 이하의 로그 입력을 구분합니다.",
        "en": "Exclude base one and nonpositive logarithm inputs."
      },
      "formula": "a>0; a≠1; y>0",
      "diagram": {
        "ko": "고정 기준 예제의 지수·로그와 척도 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Exponentials, Logarithms and Scale. Read its axes and domain alongside the explanation and numeric readout."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "역관계",
        "en": "Inverse relationship"
      },
      "predict": {
        "ko": "2²와 log₂4는 어떤 관계인가요?",
        "en": "How are 2² and log₂4 related?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "4와 2로 서로의 질문을 되돌립니다.",
        "en": "They give 4 and 2, reversing the two questions."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "작은 밑",
        "en": "Small base"
      },
      "predict": {
        "ko": "로그의 밑이 1보다 작아도 가능한가요?",
        "en": "Can a logarithm have a base below one?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "양수이고 1이 아니면 가능합니다. log₀.₅0.25=2입니다.",
        "en": "Yes, if positive and not one: log₀.₅0.25=2."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "역함수 없음",
        "en": "No inverse"
      },
      "predict": {
        "ko": "1ˣ=1에서 역함수를 하나로 정할 수 있나요?",
        "en": "Does 1ˣ=1 determine a unique inverse?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "모든 x가 같은 출력이어서 로그를 정의하지 않습니다.",
        "en": "Every x gives the same output, so no logarithm function is defined."
      },
      "preset": 2
    }
  ]
},
"trigonometry-coordinates":{
  "intuition": {
    "ko": "각도를 좌표로 바꾸고 주기성과 거리 계산을 같은 그림에서 확인합니다.",
    "en": "Convert angles into coordinates and check periodicity and distance in the same diagram."
  },
  "steps": [
    {
      "title": {
        "ko": "라디안",
        "en": "Radians"
      },
      "explanation": {
        "ko": "한 바퀴는 360도이자 2π 라디안입니다.",
        "en": "A full turn is 360 degrees or 2π radians."
      },
      "formula": "60°=π/3",
      "diagram": {
        "ko": "고정 기준 예제의 삼각함수·좌표·거리와 각 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Trigonometry, Coordinates, Distance and Angle. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "좌표",
        "en": "Coordinates"
      },
      "explanation": {
        "ko": "반지름 2에서 cos와 sin을 배율로 사용합니다.",
        "en": "Scale cosine and sine by radius two."
      },
      "formula": "p=(1,√3)",
      "diagram": {
        "ko": "고정 기준 예제의 삼각함수·좌표·거리와 각 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Trigonometry, Coordinates, Distance and Angle. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "거리",
        "en": "Distance"
      },
      "explanation": {
        "ko": "q=(1,0)과의 거리는 좌표 차이의 노름입니다.",
        "en": "Distance to q=(1,0) is the norm of coordinate differences."
      },
      "formula": "‖p−q‖₂=√3",
      "diagram": {
        "ko": "고정 기준 예제의 삼각함수·좌표·거리와 각 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Trigonometry, Coordinates, Distance and Angle. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "주기와 방향",
        "en": "Period and direction"
      },
      "explanation": {
        "ko": "각에 360도를 더해도 같은 좌표입니다. 원점에는 방향이 없습니다.",
        "en": "Adding 360 degrees preserves coordinates; the origin has no direction."
      },
      "formula": "−300° ≡ 60° (mod 360°)",
      "diagram": {
        "ko": "고정 기준 예제의 삼각함수·좌표·거리와 각 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Trigonometry, Coordinates, Distance and Angle. Read its axes and domain alongside the explanation and numeric readout."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "좌표 변환",
        "en": "Coordinate conversion"
      },
      "predict": {
        "ko": "60도, 반지름 2의 좌표는?",
        "en": "What are coordinates at 60 degrees and radius two?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "(1,√3)이며 q까지 거리는 √3입니다.",
        "en": "They are (1,√3), with distance √3 to q."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "주기성",
        "en": "Periodicity"
      },
      "predict": {
        "ko": "−300도는 다른 점인가요?",
        "en": "Does −300 degrees produce a different point?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "60도와 같은 좌표입니다. 주값 각은 원래 입력과 다를 수 있습니다.",
        "en": "It gives the same coordinates as 60 degrees; a principal angle may differ from the original input."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "방향 없음",
        "en": "No direction"
      },
      "predict": {
        "ko": "r=0에서도 θ를 점에서 복원하나요?",
        "en": "Can θ be recovered from the point when r=0?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "모든 각이 같은 원점을 만들어 방향은 복원되지 않습니다.",
        "en": "Every angle gives the origin, so no direction is recovered."
      },
      "preset": 2
    }
  ]
},
"composition-inverse-functions":{
  "intuition": {
    "ko": "계산 순서와 입력 범위가 역변환 가능성을 어떻게 바꾸는지 살펴봅니다.",
    "en": "Inspect how operation order and input restrictions determine invertibility."
  },
  "steps": [
    {
      "title": {
        "ko": "안쪽 먼저",
        "en": "Inner map first"
      },
      "explanation": {
        "ko": "먼저 g(1)=3을 계산합니다.",
        "en": "First compute g(1)=3."
      },
      "formula": "x→g(x)=2x+1",
      "diagram": {
        "ko": "고정 기준 예제의 함수 합성과 역함수의 조건 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Function Composition and Conditions for Inversion. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "바깥 적용",
        "en": "Apply the outer map"
      },
      "explanation": {
        "ko": "다음에 제곱하여 9를 얻습니다.",
        "en": "Then square to obtain nine."
      },
      "formula": "h(1)=f(3)=9",
      "diagram": {
        "ko": "고정 기준 예제의 함수 합성과 역함수의 조건 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Function Composition and Conditions for Inversion. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "전체 역상",
        "en": "Full preimage"
      },
      "explanation": {
        "ko": "9를 만드는 입력은 하나가 아닙니다.",
        "en": "More than one input produces nine."
      },
      "formula": "h⁻¹({9})={−2,1}",
      "diagram": {
        "ko": "고정 기준 예제의 함수 합성과 역함수의 조건 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Function Composition and Conditions for Inversion. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "가지 제한",
        "en": "Restrict a branch"
      },
      "explanation": {
        "ko": "g(x)≥0으로 제한하면 양의 제곱근을 택할 수 있습니다.",
        "en": "Restricting g(x)≥0 selects the positive square root."
      },
      "formula": "h_branch⁻¹(y)=(√y−b)/a",
      "diagram": {
        "ko": "고정 기준 예제의 함수 합성과 역함수의 조건 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Function Composition and Conditions for Inversion. Read its axes and domain alongside the explanation and numeric readout."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "순서",
        "en": "Order"
      },
      "predict": {
        "ko": "f∘g와 g∘f의 값이 같을까요?",
        "en": "Must f∘g and g∘f agree?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "기준 입력에서 9와 3으로 다릅니다.",
        "en": "At the reference input they give 9 and 3."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "가지",
        "en": "Branch"
      },
      "predict": {
        "ko": "음수 안쪽 값을 잊고 양의 제곱근만 쓰면?",
        "en": "What happens if one always chooses the positive root?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "원래 입력 −1 대신 1을 얻습니다. 원래 입력은 선택 가지 밖입니다.",
        "en": "It returns 1 instead of original input −1, which lies outside the selected branch."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "상수",
        "en": "Constant"
      },
      "predict": {
        "ko": "안쪽 배율 0이면 역상을 하나로 정하나요?",
        "en": "Does zero inner scale leave one inverse input?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "모든 실수가 같은 출력 1을 만듭니다. 역함수는 없습니다.",
        "en": "Every real input produces one; there is no inverse function."
      },
      "preset": 2
    }
  ]
},
"sequences-series":{
  "intuition": {
    "ko": "항이 작아지는 것과 누적 합이 수렴하는 것을 구분합니다.",
    "en": "Distinguish shrinking terms from convergence of accumulated sums."
  },
  "steps": [
    {
      "title": {
        "ko": "항과 부분합",
        "en": "Terms and partial sums"
      },
      "explanation": {
        "ko": "수열 항과 누적 합은 다른 수열입니다.",
        "en": "Terms and cumulative sums are different sequences."
      },
      "formula": "uₙ=2⁻ⁿ; Sₖ=Σₙ₌₀ᵏ⁻¹uₙ",
      "diagram": {
        "ko": "고정 기준 예제의 수열·부분합과 급수 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Sequences, Partial Sums and Series. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "유한 합",
        "en": "Finite sum"
      },
      "explanation": {
        "ko": "10항의 합은 아직 2가 아닙니다.",
        "en": "The sum of ten terms is not yet two."
      },
      "formula": "S₁₀=2(1−2⁻¹⁰)=1.998046875",
      "diagram": {
        "ko": "고정 기준 예제의 수열·부분합과 급수 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Sequences, Partial Sums and Series. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "극한",
        "en": "Limit"
      },
      "explanation": {
        "ko": "부분합의 극한이 존재하면 급수의 합입니다.",
        "en": "A series sum is the limit of its partial sums, when it exists."
      },
      "formula": "lim Sₖ=2",
      "diagram": {
        "ko": "고정 기준 예제의 수열·부분합과 급수 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Sequences, Partial Sums and Series. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "조건",
        "en": "Conditions"
      },
      "explanation": {
        "ko": "공비 −1의 부분합은 0과 1 사이에서 진동합니다.",
        "en": "At ratio −1 partial sums oscillate between zero and one."
      },
      "formula": "r=−1: no ordinary sum",
      "diagram": {
        "ko": "고정 기준 예제의 수열·부분합과 급수 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Sequences, Partial Sums and Series. Read its axes and domain alongside the explanation and numeric readout."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "합의 차이",
        "en": "Two kinds of sum"
      },
      "predict": {
        "ko": "10항이면 정확히 2인가요?",
        "en": "Do ten terms sum to exactly two?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "부분합은 1.998046875이며 무한 합 2와 다릅니다.",
        "en": "The partial sum is 1.998046875, distinct from infinite sum two."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "진동",
        "en": "Oscillation"
      },
      "predict": {
        "ko": "항이 유계이면 급수가 수렴하나요?",
        "en": "Do bounded terms guarantee series convergence?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "r=−1이면 부분합은 계속 0과 1을 오갑니다.",
        "en": "At r=−1 partial sums keep alternating between zero and one."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "발산",
        "en": "Divergence"
      },
      "predict": {
        "ko": "30항 같은 유한 계산이 있으면 무한 합도 있나요?",
        "en": "Does a finite computation imply an infinite sum exists?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "공비 1.1은 a=1에서 무한 합이 없습니다.",
        "en": "With a=1 and ratio 1.1 there is no finite ordinary infinite sum."
      },
      "preset": 2
    }
  ]
},
"asymptotic-growth":{
  "intuition": {
    "ko": "유한 관측과 충분히 큰 입력에서의 상한을 구분합니다.",
    "en": "Separate finite observations from an upper bound for sufficiently large inputs."
  },
  "steps": [
    {
      "title": {
        "ko": "양화사",
        "en": "Quantifiers"
      },
      "explanation": {
        "ko": "모든 충분히 큰 n에 통하는 상수를 찾아야 합니다.",
        "en": "Find constants that work for every sufficiently large n."
      },
      "formula": "∃C>0 ∃n₀ ∀n≥n₀",
      "diagram": {
        "ko": "고정 기준 예제의 점근 표기와 성장률 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Asymptotic Notation and Growth Rates. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "비율",
        "en": "Ratio"
      },
      "explanation": {
        "ko": "2n+3을 n으로 나누면 2+3/n입니다.",
        "en": "Dividing 2n+3 by n gives 2+3/n."
      },
      "formula": "f(n)/n=2+3/n",
      "diagram": {
        "ko": "고정 기준 예제의 점근 표기와 성장률 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Asymptotic Notation and Growth Rates. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "시작점",
        "en": "Threshold"
      },
      "explanation": {
        "ko": "C=3이면 n≥3부터 상한이 맞습니다.",
        "en": "For C=3 the bound holds from n=3."
      },
      "formula": "2n+3≤3n ⇔ n≥3",
      "diagram": {
        "ko": "고정 기준 예제의 점근 표기와 성장률 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Asymptotic Notation and Growth Rates. Read its axes and domain alongside the explanation and numeric readout."
      }
    },
    {
      "title": {
        "ko": "증명서와 분류",
        "en": "Certificate versus class"
      },
      "explanation": {
        "ko": "C=2의 실패가 O(n) 실패를 뜻하지는 않습니다.",
        "en": "Failure at C=2 does not mean failure of O(n)."
      },
      "formula": "2n+3=Θ(n)",
      "diagram": {
        "ko": "고정 기준 예제의 점근 표기와 성장률 그림입니다. 축의 의미와 정의역을 설명 및 수치 표와 함께 확인하세요.",
        "en": "Fixed reference plot for Asymptotic Notation and Growth Rates. Read its axes and domain alongside the explanation and numeric readout."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "작은 n",
        "en": "Small n"
      },
      "predict": {
        "ko": "C=3, n₀=1이면 전체 상한인가요?",
        "en": "Does C=3,n₀=1 establish the whole bound?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "n=1에서 5>3이라 이 증명서는 실패합니다.",
        "en": "At n=1, 5>3, so this certificate fails."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "충분한 시작",
        "en": "Sufficient start"
      },
      "predict": {
        "ko": "n₀를 3으로 바꾸면 모든 이후 n도 되나요?",
        "en": "Does n₀=3 make the bound work for every later n?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "2+3/n≤3이므로 모든 n≥3에서 성립합니다.",
        "en": "Since 2+3/n≤3, the bound holds for all n≥3."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "상수 선택",
        "en": "Choose a constant"
      },
      "predict": {
        "ko": "C=2는 시작점을 키우면 될까요?",
        "en": "Can increasing the start point rescue C=2?"
      },
      "action": {
        "ko": "예제를 불러오고 실제 식·그림·수치와 정의역을 비교하세요.",
        "en": "Load the example; compare the actual formula, plot, numbers and domain."
      },
      "explain": {
        "ko": "3/n>0이라 유한 시작점은 없습니다. 그래도 C=3의 O(n) 보장은 남습니다.",
        "en": "Because 3/n>0, no finite start works. The O(n) guarantee with C=3 still holds."
      },
      "preset": 2
    }
  ]
},

"propositions-implications":{
  "intuition": {
    "ko": "조건이 충분하다는 말과 필요하다는 말을 진리표로 구분합니다.",
    "en": "Use truth tables to distinguish sufficient conditions from necessary ones."
  },
  "steps": [
    {
      "title": {
        "ko": "명제",
        "en": "Propositions"
      },
      "explanation": {
        "ko": "참·거짓을 정할 수 있는 문장을 먼저 고정합니다.",
        "en": "First fix statements that have truth values."
      },
      "formula": "P, Q ∈ {⊤,⊥}",
      "diagram": {
        "ko": "고정 기준 예제의 명제·조건과 논리적 함의 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Propositions, Conditions and Implication. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "함의",
        "en": "Implication"
      },
      "explanation": {
        "ko": "P는 참인데 Q가 거짓일 때만 P⇒Q가 거짓입니다.",
        "en": "P⇒Q is false only when P is true and Q is false."
      },
      "formula": "P⇒Q ≡ ¬P∨Q",
      "diagram": {
        "ko": "고정 기준 예제의 명제·조건과 논리적 함의 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Propositions, Conditions and Implication. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "대우",
        "en": "Contrapositive"
      },
      "explanation": {
        "ko": "대우는 원래 함의와 네 행 모두 같습니다.",
        "en": "The contrapositive agrees with the implication on all four rows."
      },
      "formula": "P⇒Q ≡ ¬Q⇒¬P",
      "diagram": {
        "ko": "고정 기준 예제의 명제·조건과 논리적 함의 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Propositions, Conditions and Implication. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "필요와 충분",
        "en": "Necessary and sufficient"
      },
      "explanation": {
        "ko": "P는 Q의 충분조건, Q는 P의 필요조건입니다. 역은 별도 주장입니다.",
        "en": "P is sufficient for Q; Q is necessary for P. The converse is a separate claim."
      },
      "formula": "P⇒Q does not imply Q⇒P",
      "diagram": {
        "ko": "고정 기준 예제의 명제·조건과 논리적 함의 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Propositions, Conditions and Implication. Check truth values or function mappings against the statement and accessible numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "함의 반례",
        "en": "Counterexample to implication"
      },
      "predict": {
        "ko": "어떤 한 행이 함의를 깨뜨리나요?",
        "en": "Which row falsifies implication?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "P=1, Q=0만 원래 함의를 거짓으로 만듭니다.",
        "en": "Only P=1,Q=0 makes the original implication false."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "역은 별도",
        "en": "The converse is separate"
      },
      "predict": {
        "ko": "P⇒Q가 참이면 역도 참일까요?",
        "en": "Does a true P⇒Q force its converse to be true?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "P=0,Q=1이면 원래 함의는 참, 역은 거짓입니다.",
        "en": "At P=0,Q=1, the original implication is true and its converse false."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "한 행과 항진식",
        "en": "One row versus tautology"
      },
      "predict": {
        "ko": "한 행에서 둘 다 참이면 두 식은 동치인가요?",
        "en": "Does agreement on one row establish logical equivalence?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "한 행의 일치는 충분하지 않습니다. 모든 네 행을 비교해야 합니다.",
        "en": "One matching row is insufficient. Compare all four assignments."
      },
      "preset": 2
    }
  ]
},
"quantifiers-counterexamples":{
  "intuition": {
    "ko": "모든 입력마다 다른 답을 고를 수 있는 것과 공통 답 하나가 존재하는 것은 다릅니다.",
    "en": "Choosing a separate answer for every input differs from finding one answer that works for all inputs."
  },
  "steps": [
    {
      "title": {
        "ko": "정의역",
        "en": "Domain"
      },
      "explanation": {
        "ko": "x와 y의 범위 D={0,1,2}를 먼저 정합니다.",
        "en": "First specify D={0,1,2} for both variables."
      },
      "formula": "R(x,y): y=x",
      "diagram": {
        "ko": "고정 기준 예제의 양화사·증인과 반례 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Quantifiers, Witnesses and Counterexamples. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "입력별 선택",
        "en": "Input-dependent choice"
      },
      "explanation": {
        "ko": "각 x마다 y=x를 고를 수 있습니다.",
        "en": "For each x, choose y=x."
      },
      "formula": "∀x∃y R(x,y): ⊤",
      "diagram": {
        "ko": "고정 기준 예제의 양화사·증인과 반례 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Quantifiers, Witnesses and Counterexamples. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "공통 선택",
        "en": "One common choice"
      },
      "explanation": {
        "ko": "서로 다른 x 모두와 같은 y 하나는 없습니다.",
        "en": "No one y equals every different x."
      },
      "formula": "∃y∀x R(x,y): ⊥",
      "diagram": {
        "ko": "고정 기준 예제의 양화사·증인과 반례 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Quantifiers, Witnesses and Counterexamples. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "부정과 범위",
        "en": "Negation and scope"
      },
      "explanation": {
        "ko": "반례는 어떤 x에 대해 가능한 모든 y가 실패해야 합니다.",
        "en": "A counterexample to ∀x∃y requires one x for which every y fails."
      },
      "formula": "¬∀x∃y R ≡ ∃x∀y ¬R",
      "diagram": {
        "ko": "고정 기준 예제의 양화사·증인과 반례 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Quantifiers, Witnesses and Counterexamples. Check truth values or function mappings against the statement and accessible numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "순서",
        "en": "Order"
      },
      "predict": {
        "ko": "같은 = 술어에서 두 양화 문장이 같을까요?",
        "en": "Do both quantifier orders agree for equality?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "N=3에서 ∀x∃y는 참이고 ∃y∀x는 거짓입니다.",
        "en": "At N=3, ∀x∃y is true and ∃y∀x is false."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "정의역",
        "en": "Domain matters"
      },
      "predict": {
        "ko": "유한 집합에서도 항상 더 큰 수를 고를 수 있나요?",
        "en": "Can one always choose a larger number in a finite set?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "x=2에는 D 안의 더 큰 y가 없습니다. 자연수 전체와 다릅니다.",
        "en": "For x=2 no larger y lies in D. This differs from all natural numbers."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "공허한 참",
        "en": "Vacuous truth"
      },
      "predict": {
        "ko": "빈 집합에서 전칭과 존재는 어떻게 되나요?",
        "en": "What happens to universal and existential statements on the empty set?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "외부 전칭은 참, 외부 존재는 거짓입니다. 증인을 만들지는 않습니다.",
        "en": "The outer universal is true and outer existential false. No witness is invented."
      },
      "preset": 2
    }
  ]
},
"sets-functions":{
  "intuition": {
    "ko": "원소의 소속과 함수의 입력·출력 공간을 구분하여 데이터 변환을 읽습니다.",
    "en": "Separate set membership from function domains and codomains when reading data transformations."
  },
  "steps": [
    {
      "title": {
        "ko": "집합 연산",
        "en": "Set operations"
      },
      "explanation": {
        "ko": "A는 짝수 입력, B는 t 이상인 입력입니다.",
        "en": "A contains even inputs; B contains inputs at least t."
      },
      "formula": "A∩B, A∪B, A\\B",
      "diagram": {
        "ko": "고정 기준 예제의 집합·함수와 역상 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Sets, Functions and Preimages. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "함수 정의",
        "en": "Define a function"
      },
      "explanation": {
        "ko": "각 입력을 출력 하나에 대응합니다.",
        "en": "Assign exactly one output to each input."
      },
      "formula": "f(x)=(ax+b) mod n",
      "diagram": {
        "ko": "고정 기준 예제의 집합·함수와 역상 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Sets, Functions and Preimages. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "상과 역상",
        "en": "Image and preimage"
      },
      "explanation": {
        "ko": "역상은 출력 조건을 만족시키는 모든 입력을 모읍니다.",
        "en": "A preimage collects every input satisfying an output condition."
      },
      "formula": "f⁻¹(E)={x∈D:f(x)∈E}",
      "diagram": {
        "ko": "고정 기준 예제의 집합·함수와 역상 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Sets, Functions and Preimages. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "단사와 전사",
        "en": "Injection and surjection"
      },
      "explanation": {
        "ko": "입력 충돌과 사용하지 않은 출력을 각각 검사합니다.",
        "en": "Check input collisions and unused outputs separately."
      },
      "formula": "injective + surjective ⇒ bijective",
      "diagram": {
        "ko": "고정 기준 예제의 집합·함수와 역상 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Sets, Functions and Preimages. Check truth values or function mappings against the statement and accessible numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "항등 대응",
        "en": "Identity mapping"
      },
      "predict": {
        "ko": "각 입력에 다른 출력이 있고 모든 출력이 쓰이나요?",
        "en": "Are outputs distinct and all codomain elements used?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "m=n=3의 항등 대응은 전단사입니다.",
        "en": "The identity mapping with m=n=3 is bijective."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "충돌",
        "en": "Collisions"
      },
      "predict": {
        "ko": "모든 출력이 쓰이면 입력 충돌도 없을까요?",
        "en": "Does using all outputs rule out collisions?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "m=4,n=2이면 전사지만 단사는 아닙니다.",
        "en": "With m=4,n=2 the map is surjective but not injective."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "정의된 출력 공간",
        "en": "Declared codomain"
      },
      "predict": {
        "ko": "출력 공간에 남는 값이 있어도 함수인가요?",
        "en": "Can a valid function leave codomain values unused?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "m=2,n=4이면 단사지만 전사는 아닙니다.",
        "en": "With m=2,n=4 the map is injective but not surjective."
      },
      "preset": 2
    }
  ]
},
"relations-orders":{
  "intuition": {
    "ko": "같다고 묶는 관계와 앞뒤를 비교하는 관계의 조건을 따로 검사합니다.",
    "en": "Test grouping-by-equivalence separately from ordering elements."
  },
  "steps": [
    {
      "title": {
        "ko": "관계의 표",
        "en": "Relation matrix"
      },
      "explanation": {
        "ko": "행 x, 열 y는 xRy의 참·거짓입니다.",
        "en": "Row x, column y records whether xRy holds."
      },
      "formula": "R ⊆ D×D",
      "diagram": {
        "ko": "고정 기준 예제의 동치관계와 부분순서 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Equivalence Relations and Partial Orders. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "동치",
        "en": "Equivalence"
      },
      "explanation": {
        "ko": "자기 자신·역방향·이어진 경로를 검사합니다.",
        "en": "Check self-pairs, reverse pairs and composable paths."
      },
      "formula": "reflexive ∧ symmetric ∧ transitive",
      "diagram": {
        "ko": "고정 기준 예제의 동치관계와 부분순서 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Equivalence Relations and Partial Orders. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "부분순서",
        "en": "Partial order"
      },
      "explanation": {
        "ko": "대칭 대신 반대칭을 요구하며 비교 불가 원소를 허용합니다.",
        "en": "Require antisymmetry instead of symmetry, allowing incomparable elements."
      },
      "formula": "reflexive ∧ antisymmetric ∧ transitive",
      "diagram": {
        "ko": "고정 기준 예제의 동치관계와 부분순서 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Equivalence Relations and Partial Orders. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "실패 증인",
        "en": "Failure witnesses"
      },
      "explanation": {
        "ko": "조건이 실패하면 실제 쌍이나 세 원소를 제시합니다.",
        "en": "When a condition fails, exhibit an actual pair or triple."
      },
      "formula": "xRy ∧ yRz ∧ ¬xRz",
      "diagram": {
        "ko": "고정 기준 예제의 동치관계와 부분순서 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Equivalence Relations and Partial Orders. Check truth values or function mappings against the statement and accessible numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "동치류",
        "en": "Equivalence classes"
      },
      "predict": {
        "ko": "짝홀 합동은 어떤 원소를 묶나요?",
        "en": "Which elements does parity congruence group?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "{1,3}과 {2,4}가 동치류입니다. 반대칭은 실패합니다.",
        "en": "Classes are {1,3} and {2,4}; antisymmetry fails."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "부분 비교",
        "en": "Partial comparison"
      },
      "predict": {
        "ko": "약수 관계에서 모든 쌍을 비교할 수 있나요?",
        "en": "Can divisibility compare every pair?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "2와 3은 비교 불가여서 부분순서지만 전순서는 아닙니다.",
        "en": "2 and 3 are incomparable, so this is partial but not total order."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "엄격한 정의",
        "en": "Strict definitions"
      },
      "predict": {
        "ko": "<도 이 단원의 반사적 부분순서인가요?",
        "en": "Is < a reflexive partial order under this unit’s definition?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "반사성이 없어 아닙니다. 엄격한 순서라는 다른 정의로 다룹니다.",
        "en": "No: it lacks reflexivity. It belongs to the distinct definition of strict order."
      },
      "preset": 2
    }
  ]
},
"proof-strategies":{
  "intuition": {
    "ko": "유한 검사와 일반 증명을 구분하고 직접·대우·모순 논증을 연결합니다.",
    "en": "Distinguish finite checks from general proofs and connect direct, contrapositive and contradiction arguments."
  },
  "steps": [
    {
      "title": {
        "ko": "정의와 가정",
        "en": "Definition and assumptions"
      },
      "explanation": {
        "ko": "짝수는 어떤 정수 k에 대해 n=2k라는 뜻입니다.",
        "en": "Even means n=2k for some integer k."
      },
      "formula": "n∈ℤ; even(n) ⇔ ∃k∈ℤ:n=2k",
      "diagram": {
        "ko": "고정 기준 예제의 정의·가정과 증명 전략 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Definitions, Assumptions and Proof Strategies. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "대우로 전환",
        "en": "Take the contrapositive"
      },
      "explanation": {
        "ko": "제곱이 짝수이면 원래 수가 짝수라는 명제의 대우를 봅니다.",
        "en": "Consider the contrapositive of the square-even theorem."
      },
      "formula": "n odd ⇒ n² odd",
      "diagram": {
        "ko": "고정 기준 예제의 정의·가정과 증명 전략 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Definitions, Assumptions and Proof Strategies. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "정의 대입",
        "en": "Substitute the definition"
      },
      "explanation": {
        "ko": "홀수 n=2k+1을 제곱해 다시 홀수 형태로 씁니다.",
        "en": "Square n=2k+1 and write the result in odd form."
      },
      "formula": "n²=2(2k²+2k)+1",
      "diagram": {
        "ko": "고정 기준 예제의 정의·가정과 증명 전략 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Definitions, Assumptions and Proof Strategies. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "결론과 범위",
        "en": "Conclusion and scope"
      },
      "explanation": {
        "ko": "임의 정수 k에 대한 식이므로 검사 표의 범위 밖도 포함합니다.",
        "en": "The identity holds for arbitrary integer k, beyond the finite table."
      },
      "formula": "∀n∈ℤ: even(n²)⇒even(n)",
      "diagram": {
        "ko": "고정 기준 예제의 정의·가정과 증명 전략 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Definitions, Assumptions and Proof Strategies. Check truth values or function mappings against the statement and accessible numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "유한 검사와 증명",
        "en": "Checking versus proof"
      },
      "predict": {
        "ko": "반례를 못 찾았다는 것만으로 증명인가요?",
        "en": "Does not finding a counterexample constitute a proof?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "아닙니다. 이 정리는 별도의 대우 논증으로 증명합니다.",
        "en": "No. The theorem has a separate contrapositive proof."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "가정 유지",
        "en": "Keep assumptions fixed"
      },
      "predict": {
        "ko": "짝수 결론을 4의 배수로 강화해도 될까요?",
        "en": "Can the conclusion be strengthened to divisible by four?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "n=−2는 제곱이 짝수지만 4의 배수는 아닙니다.",
        "en": "n=−2 has an even square but is not divisible by four."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "결론 검산",
        "en": "Check the conclusion"
      },
      "predict": {
        "ko": "제곱이 홀수인 수가 짝수일까요?",
        "en": "Can an integer with odd square be even?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "n=−3이 즉시 반례입니다.",
        "en": "n=−3 is an immediate counterexample."
      },
      "preset": 2
    }
  ]
},
"mathematical-induction":{
  "intuition": {
    "ko": "시작점과 다음 단계의 연결로 무한한 명제를 증명하고 그럴듯한 패턴을 반례와 비교합니다.",
    "en": "Prove an infinite family through a base and a successor step, and compare plausible patterns with counterexamples."
  },
  "steps": [
    {
      "title": {
        "ko": "명제 고정",
        "en": "Fix the proposition"
      },
      "explanation": {
        "ko": "S(n)=1+…+n의 공식을 n≥0에서 증명합니다.",
        "en": "Prove the sum formula for n≥0."
      },
      "formula": "P(n): S(n)=n(n+1)/2",
      "diagram": {
        "ko": "고정 기준 예제의 수학적 귀납법과 유한 검사의 한계 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Mathematical Induction and the Limits of Finite Tests. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "시작점",
        "en": "Base case"
      },
      "explanation": {
        "ko": "빈 합 S(0)=0으로 시작점을 확인합니다.",
        "en": "Use the empty sum S(0)=0 to establish the base."
      },
      "formula": "P(0): 0=0",
      "diagram": {
        "ko": "고정 기준 예제의 수학적 귀납법과 유한 검사의 한계 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Mathematical Induction and the Limits of Finite Tests. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "귀납 단계",
        "en": "Inductive step"
      },
      "explanation": {
        "ko": "임의 k에서 성립한다고 가정하고 항 k+1을 더합니다.",
        "en": "Assume the claim at arbitrary k, then add k+1."
      },
      "formula": "S(k+1)=k(k+1)/2+(k+1)",
      "diagram": {
        "ko": "고정 기준 예제의 수학적 귀납법과 유한 검사의 한계 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Mathematical Induction and the Limits of Finite Tests. Check truth values or function mappings against the statement and accessible numeric table."
      }
    },
    {
      "title": {
        "ko": "무한 범위",
        "en": "Infinite scope"
      },
      "explanation": {
        "ko": "정리하면 다음 공식이 되어 모든 자연수에 연결됩니다.",
        "en": "Simplifying gives the successor formula, connecting every natural number."
      },
      "formula": "S(k+1)=(k+1)(k+2)/2",
      "diagram": {
        "ko": "고정 기준 예제의 수학적 귀납법과 유한 검사의 한계 시각화입니다. 문장의 참·거짓 또는 함수 대응을 본문과 표에서 함께 확인하세요.",
        "en": "Fixed reference visualization for Mathematical Induction and the Limits of Finite Tests. Check truth values or function mappings against the statement and accessible numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "올바른 귀납",
        "en": "Valid induction"
      },
      "predict": {
        "ko": "여섯 개 사례 검사와 귀납 증명은 어떻게 다른가요?",
        "en": "How do six checked cases differ from an induction proof?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "표는 n=0,…,5, 증명은 시작점과 모든 k의 연결을 다룹니다.",
        "en": "The table checks n=0,…,5; the proof establishes a base and a link for every k."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "시작점만 참",
        "en": "Only the base is true"
      },
      "predict": {
        "ko": "2ⁿ>n²가 n=1에서 참이면 계속 참일까요?",
        "en": "Does 2ⁿ>n² at n=1 make the claim true forever?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "n=2에서 4>4가 거짓입니다. 시작점만으로 부족합니다.",
        "en": "At n=2, 4>4 is false. A base case alone is insufficient."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "그럴듯한 패턴",
        "en": "A plausible pattern"
      },
      "predict": {
        "ko": "n²+n+41이 앞에서 소수이면 항상 소수일까요?",
        "en": "If n²+n+41 is prime initially, is it always prime?"
      },
      "action": {
        "ko": "예제를 불러오고 그림·진리표·검사 범위를 비교하세요.",
        "en": "Load the example and compare its diagram, truth table and checked domain."
      },
      "explain": {
        "ko": "n=40에서 1681=41²입니다. 앞의 40개 사례가 증명은 아닙니다.",
        "en": "At n=40, 1681=41². Forty earlier examples are not a proof."
      },
      "preset": 2
    }
  ]
},

  'vectors-and-forces': {
    intuition: { ko: '힘을 크기 하나로 합치기 전에, 각 방향에 얼마씩 있는지 따로 기록합니다.', en: 'Before combining forces into one magnitude, keep a separate record for each direction.' },
    steps: [
      { title: { ko: '첫 방향 기록', en: 'Record the first direction' }, explanation: { ko: '오른쪽 3 N인 F₁을 x축 방향으로 그립니다. 화살표의 끝은 (3,0), 성분은 (3,0,0)입니다. 이 그림은 힘을 나타내며 입자의 이동 경로가 아닙니다.', en: 'Draw F₁ as 3 N to the right. Its endpoint is (3,0), with components (3,0,0). This is a force diagram, not the path of a particle.' }, formula: 'F₁ = (3, 0, 0) N', diagram: { ko: 'x축 방향으로 원점에서 (3,0)까지 그린 첫 힘 3 N.', en: 'The first force, 3 N, runs from the origin to (3,0) along x.' } },
      { title: { ko: '둘째 힘 이어 붙이기', en: 'Place the second head to tail' }, explanation: { ko: '위쪽 4 N인 F₂를 첫 화살표 끝에 평행 이동해 붙입니다. F₂ 자체의 성분은 (0,4,0)으로 그대로이고, 마지막 끝점은 (3,4)가 됩니다.', en: 'Translate the 4 N upward arrow F₂ to the tip of the first. F₂ still has components (0,4,0), while the final endpoint becomes (3,4).' }, formula: '(3, 0, 0) + (0, 4, 0) = (3, 4, 0)', diagram: { ko: '첫 힘의 끝 (3,0)에서 위쪽으로 4 N인 둘째 힘을 이어 (3,4)에 도착한다. 둘째 힘은 원점에서 그렸을 때와 같은 길이와 방향이다.', en: 'The second force connects (3,0) to (3,4). It has the same direction and length as its dashed copy at the origin.' } },
      { title: { ko: '합력과 크기 읽기', en: 'Read the resultant and its length' }, explanation: { ko: '원점에서 최종 끝점으로 바로 이은 화살표가 합력입니다. 성분은 (3,4,0), 크기는 √(3²+4²)=5 N입니다. 꺾인 두 변의 길이 합 7 N과 구분하세요.', en: 'The arrow directly from the origin to the final tip is the resultant. Its components are (3,4,0), and its magnitude is √(3²+4²)=5 N. Distinguish that from the sum of the two side lengths, 7 N.' }, formula: '|Fnet| = √(3² + 4²) = 5 N', diagram: { ko: '밑변 3 N, 높이 4 N인 직각삼각형과 원점에서 (3,4)로 향하는 합력 5 N.', en: 'A right triangle with sides 3 N and 4 N and resultant 5 N from the origin to (3,4).' } },
    ],
    experiments: [
      { title: { ko: '성분과 길이', en: 'Components and length' }, predict: { ko: '3 N과 4 N을 직각으로 더하면 크기는 7 N일까요?', en: 'Do perpendicular forces of 3 N and 4 N have resultant magnitude 7 N?' }, action: { ko: '기준 예제를 열고 두 힘과 합력의 화살표를 비교한 뒤, 수치 표의 |Fnet|를 읽으세요.', en: 'Load the reference example, compare the two arrows with their resultant, then read |Fnet| in the table.' }, explain: { ko: '합력은 (3,4,0) N이고 크기는 5 N입니다. 같은 축의 성분을 먼저 더한 뒤 길이를 계산합니다.', en: 'The resultant is (3,4,0) N with magnitude 5 N. Add matching components before calculating length.' }, preset: 0 },
      { title: { ko: '힘과 운동 구분', en: 'Force versus motion' }, predict: { ko: '합력이 0인데 이미 움직이고 있다면 2초 뒤에도 원점에 있을까요?', en: 'With zero net force but a nonzero initial velocity, will the particle still be at the origin after two seconds?' }, action: { ko: '이 예제는 초기 속도 (1,0,0) m/s와 2 s를 설정합니다. a, v, p를 각각 비교하세요.', en: 'This example sets initial velocity (1,0,0) m/s and time 2 s. Compare a, v and p separately.' }, explain: { ko: '가속도는 0, 속도는 (1,0,0) m/s, 위치는 (2,0,0) m입니다. 합력이 0이면 속도가 변하지 않습니다.', en: 'Acceleration is zero, velocity is (1,0,0) m/s, and position is (2,0,0) m. Zero net force leaves velocity unchanged.' }, preset: 2, time: 2 },
      { title: { ko: '시간과 에너지', en: 'Time and energy' }, predict: { ko: '같은 힘을 2초 가하면 속도와 위치는 각각 무엇이 될까요?', en: 'What are velocity and position after applying the reference forces for two seconds?' }, action: { ko: '기준 힘, 질량 2 kg, 시간 2 s를 설정합니다. 본문의 시간 표와 운동 에너지를 대조하세요.', en: 'Load the reference forces, mass 2 kg and time 2 s. Compare the time table in the theory with kinetic energy.' }, explain: { ko: '속도 (3,4,0) m/s와 위치 (3,4,0) m는 숫자가 같지만 단위가 다릅니다. 운동 에너지는 25 J이며, 처음 정지했다면 힘이 한 일과 같습니다.', en: 'Velocity (3,4,0) m/s and position (3,4,0) m share numbers but not units. Kinetic energy is 25 J, equal to the work done from rest.' }, preset: 0, time: 2 },
    ],
  },
"inner-products": {
  "intuition": {
    "ko": "벡터를 한 방향에 남는 부분과 그 방향에 수직인 나머지로 나눕니다.",
    "en": "Split a vector into a component along a direction and a perpendicular remainder."
  },
  "steps": [
    {
      "title": {
        "ko": "비교할 방향 정하기",
        "en": "Choose the reference direction"
      },
      "explanation": {
        "ko": "u=(3,2)를 v=(2,0) 방향과 비교합니다. v가 단위 벡터가 아니므로 내적 6을 곧바로 길이나 투영 벡터로 읽지 않습니다.",
        "en": "Compare u=(3,2) with v=(2,0). Since v is not unit length, the dot product 6 is not directly a length or a projection vector."
      },
      "formula": "u·v = 3×2 + 2×0 = 6",
      "diagram": {
        "ko": "두 벡터 u=(3,2), v=(2,0)를 원점에서 그린 그림.",
        "en": "Vectors u=(3,2), v=(2,0), drawn from the origin."
      }
    },
    {
      "title": {
        "ko": "기준 방향에 남는 부분",
        "en": "Keep the component along v"
      },
      "explanation": {
        "ko": "v 방향의 배율은 6/4=1.5입니다. v를 1.5배 한 (3,0)이 투영입니다. v를 길이만 두 배로 바꿔도 같은 직선에 대한 투영은 같습니다.",
        "en": "The coefficient along v is 6/4=1.5. Multiplying v by 1.5 gives projection (3,0). Scaling v alone preserves its line and the projection."
      },
      "formula": "p = (u·v)/(v·v) v = (3, 0)",
      "diagram": {
        "ko": "u에서 x축으로 내려간 발 p=(3,0). v와 p의 길이는 다르다.",
        "en": "The foot p=(3,0) of u on the x-axis. The lengths of v and p differ."
      }
    },
    {
      "title": {
        "ko": "수직인 나머지 확인",
        "en": "Check the perpendicular remainder"
      },
      "explanation": {
        "ko": "잔차 r=u−p=(0,2)는 v와 직교합니다. 같은 직선의 다른 점으로 옮기면 수직 잔차와 추가 가로 이동이 더해져 거리가 커집니다.",
        "en": "Residual r=u−p=(0,2) is orthogonal to v. Any other point on the line adds a horizontal displacement to that perpendicular residual, increasing distance."
      },
      "formula": "u = p + r;  v·r = 0",
      "diagram": {
        "ko": "직각 표시와 함께 u=(3,2), p=(3,0), 수직 잔차 r=(0,2)를 표시한다.",
        "en": "u=(3,2), p=(3,0), perpendicular residual r=(0,2), and a right-angle marker."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "투영과 계수",
        "en": "Projection and coefficient"
      },
      "predict": {
        "ko": "내적 6이면 투영 길이도 6일까요?",
        "en": "Does dot product 6 mean projection length 6?"
      },
      "action": {
        "ko": "기준 투영 예제를 열고 내적·배율·투영을 각각 비교하세요.",
        "en": "Load the reference example and compare dot product, coefficient and projection."
      },
      "explain": {
        "ko": "배율은 1.5, 투영은 (3,0,0)입니다. 내적과 투영 벡터는 다른 객체입니다.",
        "en": "The coefficient is 1.5 and the projection is (3,0,0). A dot product and a projected vector are different objects."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "직교 확인",
        "en": "Orthogonality"
      },
      "predict": {
        "ko": "두 벡터가 직교하면 투영은 어떻게 될까요?",
        "en": "What happens to projection when the vectors are orthogonal?"
      },
      "action": {
        "ko": "직교 예제에서 u·v와 투영을 확인하세요.",
        "en": "Load orthogonal vectors and inspect u·v and the projection."
      },
      "explain": {
        "ko": "내적은 0, 각도는 90°, 투영은 영벡터입니다. 잔차는 u 전체입니다.",
        "en": "The dot product is 0, angle 90°, projection zero. The entire u remains as residual."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "영벡터 경계",
        "en": "The zero-vector boundary"
      },
      "predict": {
        "ko": "방향이 없는 v에도 각도를 정할 수 있을까요?",
        "en": "Can a reference vector with no direction define an angle?"
      },
      "action": {
        "ko": "기준 영벡터 예제를 열고 계산 불가 표시를 확인하세요.",
        "en": "Load zero reference and inspect the undefined readouts."
      },
      "explain": {
        "ko": "v=0이면 분모 v·v=0입니다. 각도나 투영을 임의의 값으로 표시하지 않습니다.",
        "en": "For v=0, denominator v·v is zero. Angle and projection are not assigned arbitrary values."
      },
      "preset": 3
    }
  ]
},
  'linear-transforms': {
    intuition: { ko: '행렬의 열은 변환된 기저입니다. 좌표를 조립하는 계수는 유지하고, 조립 재료를 바꿔 봅니다.', en: 'Matrix columns are the transformed basis. Keep the coordinate coefficients and change the building blocks.' },
    steps: [
      { title: { ko: '기준 방향으로 조립', en: 'Assemble with the original basis' }, explanation: { ko: 'z=0인 평면에서 (1,2)는 e₁ 하나와 e₂ 두 개로 만듭니다. 숫자 1과 2는 각 기준 방향을 얼마나 사용할지 나타냅니다.', en: 'In the plane z=0, assemble (1,2) from one e₁ and two e₂ vectors. The numbers 1 and 2 specify how much of each reference direction to use.' }, formula: 'x = 1e₁ + 2e₂ = (1, 2)', diagram: { ko: '평면의 표준 기저 e₁=(1,0), e₂=(0,1), 입력 점 (1,2).', en: 'Standard planar basis e₁=(1,0), e₂=(0,1), with input point (1,2).' } },
      { title: { ko: '기저 바꾸기', en: 'Change the basis vectors' }, explanation: { ko: '전단 행렬은 e₁을 (1,0), e₂를 (1,1)로 보냅니다. 회색 점선 단위 정사각형이 기울어진 평행사변형으로 바뀝니다. 평행 이동이 아닙니다.', en: 'The shear matrix sends e₁ to (1,0) and e₂ to (1,1). The dashed original unit square becomes a slanted parallelogram. This is not a translation.' }, formula: 'Ae₁ = (1, 0),  Ae₂ = (1, 1)', diagram: { ko: '원래 단위 정사각형과 전단된 평행사변형. 변환된 기저는 (1,0)과 (1,1).', en: 'The original unit square and sheared parallelogram, with transformed basis vectors (1,0) and (1,1).' } },
      { title: { ko: '같은 계수로 다시 조립', en: 'Reassemble with the same coefficients' }, explanation: { ko: '바뀐 e₁ 하나와 바뀐 e₂ 두 개를 더하면 (3,2)입니다. 입력의 계수 1과 2는 그대로이고, 기저만 바뀌었습니다. 이 규칙이 선형성입니다.', en: 'Add one transformed e₁ and two transformed e₂ vectors to get (3,2). The input coefficients 1 and 2 stay the same; the building blocks change. This assembly rule is linearity.' }, formula: 'Ax = 1(1, 0) + 2(1, 1) = (3, 2)', diagram: { ko: '원점에서 (1,0)으로 간 뒤 변환된 둘째 기저 두 개를 더해 (3,2)에 도착한다. 원래 입력 (1,2)은 회색으로 남아 있다.', en: 'Travel to (1,0), then add two transformed second basis vectors to reach (3,2). The original input (1,2) remains gray.' } },
    ],
    experiments: [
      { title: { ko: '행과 열 연결', en: 'Connect rows and columns' }, predict: { ko: '전단에서 입력 (1,2,0)이 어디로 갈까요?', en: 'Where does shear send the input (1,2,0)?' }, action: { ko: '전단 예제를 열고 행렬의 둘째 열과 Ax를 함께 읽으세요. x′=x+y를 손으로 계산해 비교하세요.', en: 'Load the shear example, inspect the second matrix column and Ax, then compare with x′=x+y by hand.' }, explain: { ko: '출력은 (3,2,0)입니다. 첫 기저 하나와 기울어진 둘째 기저 두 개를 더한 결과입니다.', en: 'The output is (3,2,0): one transformed first basis vector plus two transformed second basis vectors.' }, preset: 3 },
      { title: { ko: '부피와 방향', en: 'Volume and orientation' }, predict: { ko: '한 축을 반사하면 부피의 크기도 음수가 될까요?', en: 'Does reflecting one axis make the volume magnitude negative?' }, action: { ko: '반사 예제에서 정육면체가 이동한 방향, det(A), |det(A)|를 비교하세요.', en: 'Load reflection and compare the cube orientation, det(A) and |det(A)|.' }, explain: { ko: 'det(A)는 −1이지만 부피 배율은 1입니다. 부호는 방향 반전을 나타냅니다.', en: 'det(A) is −1, but the volume scale is 1. The sign indicates orientation reversal.' }, preset: 4 },
      { title: { ko: '사라지는 정보', en: 'Information that disappears' }, predict: { ko: 'y 성분을 지운 출력에서 원래 y를 하나로 복원할 수 있을까요?', en: 'Can an output that deletes y uniquely identify the original y?' }, action: { ko: '차원 축소 예제를 연 뒤 입력 y를 2에서 −2로 바꾸세요. Ax가 바뀌는지 확인하세요.', en: 'Load collapse, change input y from 2 to −2, and check whether Ax changes.' }, explain: { ko: '두 입력 모두 (1,0,3)이 됩니다. 서로 다른 입력이 같은 출력으로 가므로 역행렬이 없습니다.', en: 'Both inputs produce (1,0,3). Different inputs share an output, so there is no inverse.' }, preset: 5 },
    ],
  },
"least-squares": {
  "intuition": {
    "ko": "예측할 수 있는 응답 벡터들의 직선에서 관측 벡터와 가장 가까운 점을 찾습니다.",
    "en": "Find the point closest to the observed vector on the line of possible prediction vectors."
  },
  "steps": [
    {
      "title": {
        "ko": "정확히 맞추기 어려운 관측",
        "en": "Observations that do not fit exactly"
      },
      "explanation": {
        "ko": "입력 1,2,3에 응답 1,2,2가 주어졌습니다. 앞 두 점은 기울기 1을 요구하지만 마지막 점은 2/3을 요구합니다. 절편 없는 한 직선으로 모두 맞출 수 없습니다.",
        "en": "Inputs 1,2,3 have responses 1,2,2. The first two points require slope 1, while the third requires 2/3. One intercept-free line cannot satisfy all three."
      },
      "formula": "a=1, a=1, a=2/3: inconsistent",
      "diagram": {
        "ko": "세 관측 점 (1,1), (2,2), (3,2)을 입력과 응답의 평면에 표시한다.",
        "en": "Three observed points (1,1), (2,2), (3,2) in the input-response plane."
      }
    },
    {
      "title": {
        "ko": "잔차를 숫자로 재기",
        "en": "Measure the residual"
      },
      "explanation": {
        "ko": "a=0.5이면 예측은 (0.5,1,1.5), 잔차는 (0.5,1,0.5)입니다. 잔차를 그냥 더하면 부호가 상쇄될 수 있어 제곱합으로 크기를 잽니다.",
        "en": "For a=0.5, predictions are (0.5,1,1.5) and residuals (0.5,1,0.5). Squared residuals avoid cancellation of opposite signs."
      },
      "formula": "L(0.5) = ½(0.5²+1²+0.5²) = 0.75",
      "diagram": {
        "ko": "기울기 0.5인 직선과 각 관측점 사이의 세로 잔차.",
        "en": "A line with slope 0.5 and vertical residuals to the three observations."
      }
    },
    {
      "title": {
        "ko": "최소제곱 해와 직교",
        "en": "Least squares and orthogonality"
      },
      "explanation": {
        "ko": "기울기 11/14에서 제곱합이 가장 작습니다. 잔차가 모두 0인 것은 아니지만, 응답 공간에서 입력 방향 x와 잔차가 직교합니다.",
        "en": "Slope 11/14 minimizes squared error. Residuals are not all zero, but in response space the residual is orthogonal to input direction x."
      },
      "formula": "a* = 11/14; L(a*) = 5/28; x·r = 0",
      "diagram": {
        "ko": "최소제곱 직선과 세 관측의 잔차. 잔차가 남아도 최적임을 표시한다.",
        "en": "The least-squares line and its remaining residuals: nonzero residuals can still be optimal."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "초기 예측 검산",
        "en": "Check the starting prediction"
      },
      "predict": {
        "ko": "기울기 0.5의 손실을 계산할 수 있나요?",
        "en": "Can you calculate the loss for slope 0.5?"
      },
      "action": {
        "ko": "첫 예제를 열고 세 예측과 잔차의 제곱합을 계산하세요.",
        "en": "Load the first example and square the three residuals."
      },
      "explain": {
        "ko": "잔차는 (0.5,1,0.5), 제곱합은 1.5, 손실은 그 절반인 0.75입니다.",
        "en": "Residuals are (0.5,1,0.5), squared error 1.5, and half-squared loss 0.75."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "최적점과 남는 오차",
        "en": "The optimum can retain error"
      },
      "predict": {
        "ko": "최적점이면 모든 관측을 정확하게 맞출까요?",
        "en": "Does an optimum fit every observation exactly?"
      },
      "action": {
        "ko": "최소제곱 해를 열고 x·r, 손실, 기울기를 확인하세요.",
        "en": "Load the least-squares solution and read x·r, loss and slope."
      },
      "explain": {
        "ko": "a*=11/14, x·r=0이지만 손실은 5/28로 0이 아닙니다.",
        "en": "a*=11/14 and x·r=0, but loss is 5/28 rather than zero."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "정확한 해가 있는 경우",
        "en": "A case with an exact solution"
      },
      "predict": {
        "ko": "응답이 (1,2,3)이면 무엇이 달라질까요?",
        "en": "What changes if responses are (1,2,3)?"
      },
      "action": {
        "ko": "정확한 직선 예제를 열고 잔차를 확인하세요.",
        "en": "Load the exact-line example and inspect residuals."
      },
      "explain": {
        "ko": "y=x이므로 a=1에서 잔차와 손실이 모두 0입니다.",
        "en": "Since y=x, slope 1 gives zero residual and zero loss."
      },
      "preset": 2
    }
  ]
},
"eigenvalues": {
  "intuition": {
    "ko": "방향을 유지하는 벡터를 찾으면 복잡한 변환을 방향별 확대·축소로 나누어 읽을 수 있습니다.",
    "en": "Directions preserved by a matrix let us read a transformation as separate directional scalings."
  },
  "steps": [
    {
      "title": {
        "ko": "일반 방향의 변환",
        "en": "Transform an ordinary direction"
      },
      "explanation": {
        "ko": "A=[[2,1],[1,2]]는 (1,0)을 (2,1)로 보냅니다. 보통의 벡터는 길이뿐 아니라 방향도 바뀝니다.",
        "en": "A=[[2,1],[1,2]] sends (1,0) to (2,1). An ordinary vector changes direction as well as length."
      },
      "formula": "A(1,0) = (2,1)",
      "diagram": {
        "ko": "입력 (1,0)과 다른 방향인 출력 (2,1).",
        "en": "Input (1,0) and output (2,1), pointing in different directions."
      }
    },
    {
      "title": {
        "ko": "유지되는 방향 찾기",
        "en": "Find a preserved direction"
      },
      "explanation": {
        "ko": "q₁=(1,1)/√2는 A를 곱해도 같은 직선에 남으며 3배가 됩니다. q₂=(-1,1)/√2는 1배입니다. 두 고유방향은 서로 직교합니다.",
        "en": "q₁=(1,1)/√2 stays on its line and is scaled by 3. q₂=(-1,1)/√2 is scaled by 1. These two eigendirections are orthogonal."
      },
      "formula": "Aq₁ = 3q₁; Aq₂ = q₂",
      "diagram": {
        "ko": "45도 고유방향과 그 3배 출력, 직교하는 둘째 고유방향.",
        "en": "The 45-degree eigenvector, its triple, and the perpendicular second eigendirection."
      }
    },
    {
      "title": {
        "ko": "분산으로 해석하기",
        "en": "Interpret the eigenvalues as variance"
      },
      "explanation": {
        "ko": "A를 공분산으로 보면 고유값 3과 1은 두 방향의 분산입니다. 첫 방향은 총분산 4 중 3, 즉 75%를 담습니다. 타원 반길이는 3이 아니라 √3입니다.",
        "en": "Interpreting A as covariance, eigenvalues 3 and 1 are directional variances. The first retains 3 of total variance 4, or 75%. Its ellipse half-axis is √3, not 3."
      },
      "formula": "λ₁/(λ₁+λ₂) = 3/4 = 75%",
      "diagram": {
        "ko": "공분산 타원과 길이 √3, 1인 직교 반축. 행렬 작용의 3배와 구분한다.",
        "en": "Covariance ellipse with orthogonal half-axes √3 and 1, distinguished from matrix-action scaling by 3."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "고유방향 검산",
        "en": "Check an eigendirection"
      },
      "predict": {
        "ko": "45도 입력은 방향이 바뀔까요?",
        "en": "Will a 45-degree input change its direction?"
      },
      "action": {
        "ko": "고유방향 예제를 열고 v와 Av, 평행 판별식을 비교하세요.",
        "en": "Load eigendirection and compare v, Av and the parallelism diagnostic."
      },
      "explain": {
        "ko": "Av=3v이며 두 벡터가 평행합니다. 입력이 단위 길이라 vᵀAv=3입니다.",
        "en": "Av=3v and the vectors are parallel. For unit v, vᵀAv=3."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "공분산과 PCA",
        "en": "Covariance and PCA"
      },
      "predict": {
        "ko": "첫 주성분이 담는 분산은 몇 퍼센트일까요?",
        "en": "What fraction of variance does the first component retain?"
      },
      "action": {
        "ko": "공분산 예제에서 고유값과 분산 비율을 읽고 반축 길이와 구분하세요.",
        "en": "Load covariance, read eigenvalues and variance fraction, and distinguish them from half-axis lengths."
      },
      "explain": {
        "ko": "고유값은 3,1이고 첫 분산 비율은 75%입니다. 합성점은 평균 0이며 분모 n 공분산이 A입니다.",
        "en": "Eigenvalues are 3 and 1, retaining 75%. The synthetic points are centered and their denominator-n covariance is A."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "같은 고유값",
        "en": "Repeated eigenvalues"
      },
      "predict": {
        "ko": "원형 퍼짐에도 유일한 첫 방향이 있을까요?",
        "en": "Does circular spread have one uniquely preferred first direction?"
      },
      "action": {
        "ko": "같은 고유값 예제를 열고 방향의 유일성과 비율을 확인하세요.",
        "en": "Load repeated eigenvalues and inspect uniqueness and variance fraction."
      },
      "explain": {
        "ko": "고유값이 모두 2라 첫 방향은 유일하지 않습니다. 어느 단위 방향도 분산 2, 즉 50%를 담습니다.",
        "en": "Both eigenvalues are 2, so no first direction is unique. Every unit direction captures variance 2, or 50%."
      },
      "preset": 2
    }
  ]
},
  gradients: {
    intuition: { ko: '각 방향의 작은 변화에 손실이 얼마나 반응하는지 묻고, 그 답으로 다음 매개변수를 정합니다.', en: 'Ask how loss responds to a small change along each direction, then use the answers to update the parameters.' },
    steps: [
      { title: { ko: '현재 위치와 손실', en: 'Current point and loss' }, explanation: { ko: '등고선은 손실이 같은 점들을 잇습니다. a=2, b=1에서 시작점 (2,2)의 손실은 6입니다. 이 평면은 매개변수 x,y를 위에서 본 그림입니다.', en: 'Each contour joins points with equal loss. With a=2 and b=1, the starting point (2,2) has loss 6. This is a top view of parameter coordinates x and y.' }, formula: 'L(2, 2) = (2×2² + 2²)/2 = 6', diagram: { ko: 'x,y 평면의 타원형 등고선과 시작점 (2,2), 손실 6. 원점이 최솟값이다.', en: 'Elliptical contours in the x,y plane and the starting point (2,2), loss 6. The origin is the minimum.' } },
      { title: { ko: '방향과 걸음 크기', en: 'Direction and step size' }, explanation: { ko: '시작 그래디언트는 (4,2)입니다. 학습률 0.25를 곱하면 이동량은 (1,0.5)이고, 이것을 현재 위치에서 뺍니다. 좌표 둘 다 같은 비율로 줄이는 규칙은 아닙니다.', en: 'The initial gradient is (4,2). Multiplying by rate 0.25 gives (1,0.5), which is subtracted from the current point. This does not shrink both coordinates by the same factor.' }, formula: 'p₁ = (2, 2) − 0.25(4, 2) = (1, 1.5)', diagram: { ko: '시작점 (2,2)에서 (1,1.5)로 향하는 갱신 화살표. 화살표는 원점을 직접 가리키지 않는다.', en: 'An update arrow from (2,2) to (1,1.5), which does not point directly toward the origin.' } },
      { title: { ko: '새 위치에서 다시 계산', en: 'Recompute at the new point' }, explanation: { ko: '새 위치의 손실은 2.125입니다. 그곳의 그래디언트 (2,1.5)를 다시 계산하면 다음 위치는 (0.5,1.125)가 됩니다. 이전 그래디언트를 계속 쓰지 않습니다.', en: 'Loss at the new point is 2.125. Recompute gradient (2,1.5) there to obtain the next point (0.5,1.125). Do not keep using the old gradient.' }, formula: 'L(p₁) = 2.125;  p₂ = (0.5, 1.125)', diagram: { ko: '경사하강 경로 (2,2) → (1,1.5) → (0.5,1.125). 각 점에서 다시 계산한 갱신이다.', en: 'The descent path (2,2) → (1,1.5) → (0.5,1.125), recomputing the gradient at each point.' } },
      { title: { ko: '큰 걸음과 초과 이동', en: 'A large step overshoots' }, explanation: { ko: '같은 시작점에서 학습률만 1.25로 바꾸면 첫 위치는 (−3,−0.5), 손실은 9.125입니다. 방향이 하강 방향이어도 너무 멀리 가면 실제 손실이 늘어납니다.', en: 'From the same start, changing only the rate to 1.25 gives first point (−3,−0.5) and loss 9.125. Even a descent direction can increase actual loss when the step is too large.' }, formula: 'η = 1.25:  L(p₁) = 9.125 > 6', diagram: { ko: '같은 시작점에서 학습률 0.25의 끝점 (1,1.5)과 1.25의 끝점 (−3,−0.5)을 비교한다. 후자의 손실은 증가한다.', en: 'Compare endpoints (1,1.5) for rate 0.25 and (−3,−0.5) for rate 1.25 from the same start. The latter increases loss.' } },
    ],
    experiments: [
      { title: { ko: '한 걸음 검산', en: 'Check the first step' }, predict: { ko: '학습률 0.25에서 첫 위치와 손실을 손으로 계산할 수 있나요?', en: 'Can you calculate the first point and loss for rate 0.25 by hand?' }, action: { ko: '기준 예제를 한 단계 진행한 상태로 엽니다. pₖ와 L(pₖ)를 본문의 계산과 대조하세요.', en: 'Load the reference example after one step. Compare pₖ and L(pₖ) with the worked calculation.' }, explain: { ko: 'p₁=(1,1.5), L=2.125입니다. 표의 그래디언트는 새 위치에서 계산한 (2,1.5)입니다.', en: 'p₁=(1,1.5), L=2.125. The gradient in the table is (2,1.5), evaluated at the new point.' }, preset: 0, steps: 1 },
      { title: { ko: '왕복과 수렴', en: 'Alternation and convergence' }, predict: { ko: 'x의 부호가 바뀌면 반드시 발산일까요?', en: 'Does a sign change in x necessarily mean divergence?' }, action: { ko: '학습률 0.75에서 두 단계 진행한 상태를 엽니다. 처음부터 다시 실행해 x의 부호와 손실을 함께 읽으세요.', en: 'Load rate 0.75 after two steps. Restart and compare the sign of x with the loss at each step.' }, explain: { ko: 'x는 2, −1, 0.5로 부호가 바뀌지만 크기가 줄어듭니다. 두 단계 뒤 손실은 0.2578125입니다.', en: 'x alternates 2, −1, 0.5 while its magnitude shrinks. Loss after two steps is 0.2578125.' }, preset: 1, steps: 2 },
      { title: { ko: '큰 학습률', en: 'An excessive rate' }, predict: { ko: '같은 하강 방향에서 걸음만 크게 하면 더 빨리 학습할까요?', en: 'Does taking a bigger step in the same descent direction always learn faster?' }, action: { ko: '학습률 1.25에서 한 단계 진행한 상태를 엽니다. 원래 손실 6과 새 손실을 비교하세요.', en: 'Load rate 1.25 after one step. Compare the new loss with the initial loss of 6.' }, explain: { ko: '첫 위치는 (−3,−0.5), 손실은 9.125입니다. 이 함수에서 모든 시작점의 수렴 범위는 0<η<1입니다.', en: 'The first point is (−3,−0.5), loss 9.125. For this function, convergence from every start requires 0<η<1.' }, preset: 3, steps: 1 },
    ],
  },
"chain-rule": {
  "intuition": {
    "ko": "두 개의 변환기를 직렬로 잇는다고 생각하세요. 각 변환기의 국소 배율을 곱하면 전체 민감도가 됩니다. 유한한 변화에는 곡률로 인한 나머지가 남습니다.",
    "en": "Think of two converters in series. Multiply their local gains to obtain the total sensitivity. A finite change also retains a curvature remainder."
  },
  "steps": [
    {
      "title": {
        "ko": "입력을 변환",
        "en": "Transform the input"
      },
      "explanation": {
        "ko": "x=1을 u=2x+1에 넣으면 3입니다. 이 단계의 기울기는 2입니다.",
        "en": "At x=1, u=2x+1 gives 3. The local slope of this stage is 2."
      },
      "formula": "x=1 → u=3; du/dx=2",
      "diagram": {
        "ko": "입력 1에서 선형 단계 u=3으로 이어지는 계산 그래프.",
        "en": "Computation graph from input 1 to the affine intermediate u=3."
      }
    },
    {
      "title": {
        "ko": "출력을 계산",
        "en": "Evaluate the output"
      },
      "explanation": {
        "ko": "u=3을 제곱하여 y=9를 얻습니다. 제곱의 미분은 현재 u에서 2u=6입니다.",
        "en": "Squaring u=3 gives y=9. The derivative of the square at this u is 2u=6."
      },
      "formula": "u=3 → y=9; dy/du=6",
      "diagram": {
        "ko": "중간값 3을 제곱하여 출력 9를 계산하는 두 단계 그래프.",
        "en": "Two-stage graph squaring the intermediate 3 to produce output 9."
      }
    },
    {
      "title": {
        "ko": "국소 배율을 곱하기",
        "en": "Multiply local gains"
      },
      "explanation": {
        "ko": "x가 아주 조금 바뀌면 u는 약 2배, y는 그 변화의 약 6배 바뀝니다. 전체 배율은 12입니다.",
        "en": "A tiny input change is scaled by 2, then approximately by 6. The total gain is 12."
      },
      "formula": "dy/dx = 6 × 2 = 12",
      "diagram": {
        "ko": "출력에서 입력으로 되짚으며 국소 배율 6과 2를 곱하는 그래프.",
        "en": "Graph tracing from output back to input and multiplying local gains 6 and 2."
      }
    },
    {
      "title": {
        "ko": "유한 변화와 비교",
        "en": "Compare a finite change"
      },
      "explanation": {
        "ko": "Δx=0.1의 일차 예측은 1.2이지만 실제 변화는 1.24입니다. 차이 0.04는 버려진 이차항입니다.",
        "en": "For Δx=0.1 the linear prediction is 1.2 but the actual change is 1.24. The difference 0.04 is the omitted quadratic term."
      },
      "formula": "Δy = 12Δx + 4(Δx)²",
      "diagram": {
        "ko": "기준 계산 그래프와 전체 미분 12를 표시하고 유한 변화의 이차 나머지를 비교합니다.",
        "en": "Reference computation graph showing total derivative 12 for comparison with the finite-change quadratic remainder."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "일차 근사",
        "en": "Linear approximation"
      },
      "predict": {
        "ko": "입력 0.1의 변화가 출력 1.2의 변화와 정확히 같을까요?",
        "en": "Will an input change of 0.1 change the output by exactly 1.2?"
      },
      "action": {
        "ko": "기준 예제를 불러오고 실제 변화·일차 예측·나머지를 비교하세요.",
        "en": "Load the reference and compare actual change, linear prediction and remainder."
      },
      "explain": {
        "ko": "실제 1.24, 일차 예측 1.2, 나머지 0.04입니다. δ를 줄이면 이차 오차가 더 빠르게 줄어듭니다.",
        "en": "Actual change is 1.24, the prediction 1.2 and the remainder 0.04. Shrinking δ makes the quadratic error fall faster."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "기울기가 0인 점",
        "en": "A zero-slope point"
      },
      "predict": {
        "ko": "기울기 0이면 함수가 주변에서도 상수일까요?",
        "en": "Does zero slope make the function locally constant?"
      },
      "action": {
        "ko": "꼭짓점 예제를 불러오고 실제 변화와 예측을 비교하세요.",
        "en": "Load the turning point and compare actual change with prediction."
      },
      "explain": {
        "ko": "일차 예측은 0이지만 실제 변화는 0.04입니다. 미분은 일차 변화만 설명합니다.",
        "en": "The linear prediction is 0 but the actual change is 0.04. A derivative describes only first-order change."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "음의 국소 배율",
        "en": "A negative local gain"
      },
      "predict": {
        "ko": "입력이 증가하면 출력이 감소할 수도 있을까요?",
        "en": "Can increasing the input decrease the output?"
      },
      "action": {
        "ko": "음의 기울기 예제를 불러오고 곡선의 접선과 수치를 읽으세요.",
        "en": "Load the negative-slope case and inspect the tangent and readouts."
      },
      "explain": {
        "ko": "전체 미분은 −4이고 실제 변화는 −0.36입니다. 안쪽 배율이 양수여도 바깥 기울기가 음수일 수 있습니다.",
        "en": "The total derivative is −4 and actual change is −0.36. A positive inner gain can feed a negative outer slope."
      },
      "preset": 2
    }
  ]
},
"jacobians": {
  "intuition": {
    "ko": "Jacobian은 현재 점 근처에서 비선형 함수를 대신하는 행렬입니다. 입력 방향을 앞으로 보내는 Jd와 출력의 중요도를 뒤로 보내는 Jᵀq는 같은 국소 정보를 다르게 사용합니다.",
    "en": "A Jacobian is the matrix replacing a nonlinear function near the current point. Jd pushes an input direction forward; Jᵀq pulls output importance back through the same local information."
  },
  "steps": [
    {
      "title": {
        "ko": "값의 흐름",
        "en": "Flow of values"
      },
      "explanation": {
        "ko": "입력 (2,3)은 곱셈과 제곱·덧셈으로 나뉘어 출력 (6,7)을 만듭니다.",
        "en": "Input (2,3) branches through multiplication and square-plus-add to produce (6,7)."
      },
      "formula": "F(2,3) = (6,7)",
      "diagram": {
        "ko": "두 입력 x와 y가 두 출력 xy와 x²+y로 갈라지는 계산 그래프.",
        "en": "A branching computation graph from inputs x and y to outputs xy and x²+y."
      }
    },
    {
      "title": {
        "ko": "편미분 모으기",
        "en": "Collect partial derivatives"
      },
      "explanation": {
        "ko": "각 출력의 입력별 민감도를 행으로 모읍니다. 행과 열의 순서가 중요합니다.",
        "en": "Collect each output’s input sensitivities in a row. Row and column order matters."
      },
      "formula": "J = [[3,2],[4,1]]",
      "diagram": {
        "ko": "두 출력으로 향하는 네 경로와 Jacobian의 행과 열을 연결한 그래프.",
        "en": "Four input-output paths corresponding to the rows and columns of the Jacobian."
      }
    },
    {
      "title": {
        "ko": "방향을 앞으로 전달",
        "en": "Push a direction forward"
      },
      "explanation": {
        "ko": "d=(0.1,−0.2)는 출력 공간에서 Jd=(−0.1,0.2)로 예측됩니다. 실제 변화에는 이차항이 더해집니다.",
        "en": "d=(0.1,−0.2) predicts output change Jd=(−0.1,0.2). The actual change includes quadratic terms."
      },
      "formula": "Jd = (−0.1,0.2)",
      "diagram": {
        "ko": "입력 방향이 분기된 계산 경로를 따라 출력 방향으로 전달되는 그래프.",
        "en": "Input direction propagated forward through branching paths to predict an output direction."
      }
    },
    {
      "title": {
        "ko": "가중치를 뒤로 전달",
        "en": "Pull weights backward"
      },
      "explanation": {
        "ko": "L=F₁−F₂이면 출력 가중치는 (1,−1)입니다. 입력 기울기는 경로를 합산한 (−1,1)입니다.",
        "en": "For L=F₁−F₂ the output weights are (1,−1). Summing paths gives the input gradient (−1,1)."
      },
      "formula": "Jᵀq = (−1,1); qᵀJd = −0.3",
      "diagram": {
        "ko": "출력 가중치를 입력으로 되돌리고 공유 입력에 도착한 민감도를 더하는 그래프.",
        "en": "Output weights traced backward, adding sensitivities arriving at each shared input."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "두 방향의 같은 정보",
        "en": "Two views of one derivative"
      },
      "predict": {
        "ko": "Jd와 Jᵀq가 서로 다른데도 어떤 스칼라가 같을까요?",
        "en": "Which scalar agrees even though Jd and Jᵀq differ?"
      },
      "action": {
        "ko": "기준 예제를 불러오고 두 내적과 실제 변화를 확인하세요.",
        "en": "Load the reference; compare the two inner products and actual change."
      },
      "explain": {
        "ko": "두 내적은 −0.3입니다. Jd=(−0.1,0.2)와 실제 (−0.12,0.21)는 유한 이동의 이차항만큼 다릅니다.",
        "en": "Both inner products are −0.3. Jd=(−0.1,0.2) differs from actual (−0.12,0.21) by the finite-step quadratic remainder."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "더 작은 이동",
        "en": "A smaller displacement"
      },
      "predict": {
        "ko": "이동을 10분의 1로 줄이면 나머지도 10분의 1일까요?",
        "en": "Does a tenfold smaller displacement give a tenfold smaller remainder?"
      },
      "action": {
        "ko": "작은 이동 예제를 불러와 기준 예제와 비교하세요.",
        "en": "Load the small displacement and compare with the reference."
      },
      "explain": {
        "ko": "Jd=(−0.01,0.02)이고 나머지는 (−0.0002,0.0001)로 100분의 1입니다.",
        "en": "Jd=(−0.01,0.02); the remainder (−0.0002,0.0001) is one hundredth of the original."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "사라지는 일차 방향",
        "en": "A vanishing first-order direction"
      },
      "predict": {
        "ko": "Jd=0이면 유한 이동 후에도 출력이 그대로일까요?",
        "en": "If Jd=0, must the output stay unchanged after a finite move?"
      },
      "action": {
        "ko": "특이 Jacobian 예제를 불러오고 실제 변화와 비교하세요.",
        "en": "Load the singular Jacobian and compare with actual change."
      },
      "explain": {
        "ko": "Jd=0이지만 실제 변화는 (−0.01,0.01)입니다. 한 방향의 일차 정보 손실과 함수의 상수성은 다릅니다.",
        "en": "Jd=0 but actual change is (−0.01,0.01). Losing a first-order direction does not make the function constant."
      },
      "preset": 2
    }
  ]
},
"backpropagation": {
  "intuition": {
    "ko": "역전파는 새 미분 법칙이 아니라 계산 그래프에 연쇄법칙을 적용하는 효율적인 순서입니다. 먼저 값을 저장하고, 손실의 민감도를 거꾸로 전달한 다음 매개변수를 동시에 갱신합니다.",
    "en": "Backpropagation is an efficient ordering of the chain rule on a computation graph. Store forward values, propagate loss sensitivities backward, then update parameters simultaneously."
  },
  "steps": [
    {
      "title": {
        "ko": "순전파",
        "en": "Forward pass"
      },
      "explanation": {
        "ko": "입력 (1,2)에서 z=0.5, ReLU 출력 0.5, 예측 1, 목표 0, 손실 0.5입니다.",
        "en": "Input (1,2) gives z=0.5, ReLU output 0.5, prediction 1, target 0 and loss 0.5."
      },
      "formula": "x → z → h → ŷ → L",
      "diagram": {
        "ko": "입력에서 선형 결합, ReLU, 출력, 제곱 손실로 이어지는 순전파 그래프.",
        "en": "Forward graph from input through an affine combination, ReLU, prediction and squared loss."
      }
    },
    {
      "title": {
        "ko": "손실의 시작 신호",
        "en": "Seed from the loss"
      },
      "explanation": {
        "ko": "손실의 예측에 대한 미분은 오차 1입니다. 출력 가중치 2를 지나면 은닉값의 민감도는 2입니다.",
        "en": "The loss derivative with respect to prediction is error 1. Passing output weight 2 gives hidden sensitivity 2."
      },
      "formula": "∂L/∂ŷ = 1; ∂L/∂h = 2",
      "diagram": {
        "ko": "손실에서 출력과 은닉값으로 민감도 1과 2를 되돌리는 그래프.",
        "en": "Reverse graph passing sensitivities 1 and 2 from loss to prediction and hidden value."
      }
    },
    {
      "title": {
        "ko": "매개변수 기울기",
        "en": "Parameter gradients"
      },
      "explanation": {
        "ko": "ReLU가 열려 있으므로 z의 민감도도 2입니다. 입력을 곱하여 w₁과 w₂의 기울기 2와 4를 얻습니다.",
        "en": "The ReLU gate is open, so z sensitivity is also 2. Multiplying inputs gives weight gradients 2 and 4."
      },
      "formula": "∇θL = (2,4,2,0.5,1)",
      "diagram": {
        "ko": "ReLU의 열린 문을 지나 모든 매개변수로 기울기를 전달하는 역방향 그래프.",
        "en": "Reverse graph through the open ReLU gate, delivering gradients to all parameters."
      }
    },
    {
      "title": {
        "ko": "한 번 갱신하기",
        "en": "Take one update"
      },
      "explanation": {
        "ko": "학습률 0.01에서 모든 기울기를 같은 이전 상태에서 구해 적용하면 손실이 약 0.279827로 줄어듭니다.",
        "en": "At learning rate 0.01, applying gradients from the same old state reduces loss to about 0.279827."
      },
      "formula": "L: 0.5 → 0.279826805",
      "diagram": {
        "ko": "역전파 기울기를 구한 후 매개변수를 동시에 갱신하는 계산 그래프.",
        "en": "Computation graph with backward sensitivities used in a simultaneous parameter update."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "미분과 학습 연결",
        "en": "From derivatives to learning"
      },
      "predict": {
        "ko": "기울기를 확인한 뒤 한 번 갱신하면 손실이 얼마나 줄까요?",
        "en": "How much does loss fall after checking gradients and taking one update?"
      },
      "action": {
        "ko": "기준 예제를 불러오고 한 번 갱신 버튼을 누르세요.",
        "en": "Load the reference and press Apply one update."
      },
      "explain": {
        "ko": "초기 손실 0.5에서 약 0.279827로 줄어듭니다. 수치 미분 표와 역전파 기울기도 비교하세요.",
        "en": "Loss falls from 0.5 to about 0.279827. Compare the finite-difference table with backpropagation gradients too."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "닫힌 ReLU",
        "en": "A closed ReLU gate"
      },
      "predict": {
        "ko": "은닉 기울기가 0이면 어떤 매개변수도 학습하지 못할까요?",
        "en": "If hidden gradients vanish, can any parameter still learn?"
      },
      "action": {
        "ko": "닫힌 ReLU 예제를 불러오고 다섯 기울기를 확인하세요.",
        "en": "Load the inactive ReLU and inspect all five gradients."
      },
      "explain": {
        "ko": "은닉 가중치·편향과 v의 기울기는 0이지만 출력 편향 c의 기울기는 −1입니다.",
        "en": "Hidden weights, hidden bias and v have zero gradients, but output bias c has gradient −1."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "너무 큰 보폭",
        "en": "An oversized step"
      },
      "predict": {
        "ko": "학습률이 크면 한 번에 더 좋아질까요?",
        "en": "Will a larger learning rate necessarily improve loss faster?"
      },
      "action": {
        "ko": "큰 학습률 예제를 불러오고 한 번 갱신하세요.",
        "en": "Load the large-rate case and apply one update."
      },
      "explain": {
        "ko": "학습률 2에서는 손실이 0.5에서 2로 증가합니다. 올바른 기울기가 유한 보폭의 개선을 보장하지 않습니다.",
        "en": "At rate 2, loss rises from 0.5 to 2. A correct gradient does not guarantee improvement for a finite step."
      },
      "preset": 2
    }
  ]
},
"svd":{
  "intuition": {
    "ko": "직사각형 선형변환의 늘어남을 분해하고 버릴 정보의 크기를 계산합니다.",
    "en": "Decompose stretching in a linear map and quantify the information discarded by approximation."
  },
  "steps": [
    {
      "title": {
        "ko": "입력 방향",
        "en": "Input directions"
      },
      "explanation": {
        "ko": "V의 열은 입력 공간의 직교 방향입니다.",
        "en": "Columns of V are orthogonal input directions."
      },
      "formula": "AᵀA vᵢ = σᵢ²vᵢ",
      "diagram": {
        "ko": "고정 기준 예제의 SVD와 저랭크 근사 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for SVD and Low-Rank Approximation. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "늘어남",
        "en": "Stretching"
      },
      "explanation": {
        "ko": "각 방향은 음이 아닌 특잇값만큼 늘어납니다.",
        "en": "Each direction stretches by a nonnegative singular value."
      },
      "formula": "Σ=diag(3,1)",
      "diagram": {
        "ko": "고정 기준 예제의 SVD와 저랭크 근사 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for SVD and Low-Rank Approximation. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "출력 방향",
        "en": "Output directions"
      },
      "explanation": {
        "ko": "U는 늘어난 방향을 출력 공간에 배치합니다.",
        "en": "U places stretched directions in output space."
      },
      "formula": "Avᵢ=σᵢuᵢ",
      "diagram": {
        "ko": "고정 기준 예제의 SVD와 저랭크 근사 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for SVD and Low-Rank Approximation. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "약한 성분 버리기",
        "en": "Discard a weak component"
      },
      "explanation": {
        "ko": "두 번째 항을 버리면 기준 예제의 Frobenius 오차는 1입니다.",
        "en": "Discarding the second term gives Frobenius error 1 in the reference."
      },
      "formula": "A₁=3u₁v₁ᵀ",
      "diagram": {
        "ko": "고정 기준 예제의 SVD와 저랭크 근사 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for SVD and Low-Rank Approximation. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "근사 오차",
        "en": "Approximation error"
      },
      "predict": {
        "ko": "약한 축을 버리면 오차가 얼마인가요?",
        "en": "What error remains after dropping the weak axis?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "특잇값은 (3,1), 랭크 1 오차는 1입니다.",
        "en": "Singular values are (3,1), and rank-one error is 1."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "조건수",
        "en": "Conditioning"
      },
      "predict": {
        "ko": "약한 축이 작아지면 역문제는 어떻게 될까요?",
        "en": "What happens to inversion as the weak axis shrinks?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "특잇값은 (3,0.1), 조건수는 30입니다.",
        "en": "Singular values are (3,0.1), giving condition number 30."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "정확한 저랭크",
        "en": "Exact low rank"
      },
      "predict": {
        "ko": "랭크 1 행렬도 SVD로 표현할 수 있나요?",
        "en": "Can SVD represent a rank-one matrix?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "두 번째 특잇값은 0이고 랭크 1 근사 오차도 0입니다.",
        "en": "The second singular value and rank-one approximation error are zero."
      },
      "preset": 2
    }
  ]
},
"probability-distributions":{
  "intuition": {
    "ko": "가능한 결과에 확률을 배정하고 반복 관측과 모형을 구분합니다.",
    "en": "Assign probabilities to outcomes and distinguish a model from repeated observations."
  },
  "steps": [
    {
      "title": {
        "ko": "결과와 사건",
        "en": "Outcomes and events"
      },
      "explanation": {
        "ko": "K는 n번 시행의 성공 횟수입니다.",
        "en": "K counts successes in n trials."
      },
      "formula": "K∈{0,…,n}",
      "diagram": {
        "ko": "고정 기준 예제의 확률분포와 표본 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Probability Distributions and Samples. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "개수와 확률",
        "en": "Counts and probabilities"
      },
      "explanation": {
        "ko": "같은 성공 횟수의 순서를 조합 수로 셉니다.",
        "en": "Count orders with the same success total using combinations."
      },
      "formula": "P(K=k)=C(n,k)pᵏ(1−p)ⁿ⁻ᵏ",
      "diagram": {
        "ko": "고정 기준 예제의 확률분포와 표본 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Probability Distributions and Samples. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "정규화",
        "en": "Normalization"
      },
      "explanation": {
        "ko": "모든 가능한 결과의 확률 합은 1입니다.",
        "en": "Probabilities of all possible outcomes sum to one."
      },
      "formula": "ΣₖP(K=k)=1",
      "diagram": {
        "ko": "고정 기준 예제의 확률분포와 표본 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Probability Distributions and Samples. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "평균과 변동",
        "en": "Mean and variability"
      },
      "explanation": {
        "ko": "대칭이어도 모든 표본이 평균과 같지는 않습니다.",
        "en": "Symmetry does not make every sample equal the mean."
      },
      "formula": "E[K]=np; Var(K)=np(1−p)",
      "diagram": {
        "ko": "고정 기준 예제의 확률분포와 표본 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Probability Distributions and Samples. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "대칭",
        "en": "Symmetry"
      },
      "predict": {
        "ko": "n=4, p=0.5에서 가장 가능성 높은 횟수는?",
        "en": "Which count is most probable for n=4, p=0.5?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "K=2의 확률은 6/16=0.375입니다.",
        "en": "K=2 has probability 6/16=0.375."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "치우침",
        "en": "Skewness"
      },
      "predict": {
        "ko": "평균은 반드시 정수일까요?",
        "en": "Must the mean be an integer?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "평균은 0.8이고 실제 횟수는 정수입니다.",
        "en": "The mean is 0.8 although every observed count is an integer."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "경계",
        "en": "Boundary"
      },
      "predict": {
        "ko": "p=1이면 분산은 얼마인가요?",
        "en": "What is the variance when p=1?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "K=4의 확률이 1이며 분산은 0입니다.",
        "en": "K=4 has probability one and variance zero."
      },
      "preset": 2
    }
  ]
},
"expectation-variance":{
  "intuition": {
    "ko": "가능한 값의 가중 평균과 평균 주변의 퍼짐을 따로 계산합니다.",
    "en": "Calculate a weighted mean and spread around that mean as distinct quantities."
  },
  "steps": [
    {
      "title": {
        "ko": "가중치 정규화",
        "en": "Normalize weights"
      },
      "explanation": {
        "ko": "가중치 (1,2,1)은 확률 (1/4,1/2,1/4)입니다.",
        "en": "Weights (1,2,1) become probabilities (1/4,1/2,1/4)."
      },
      "formula": "p=(0.25,0.5,0.25)",
      "diagram": {
        "ko": "고정 기준 예제의 기댓값·분산과 불확실성 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Expectation, Variance and Uncertainty. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "가중 평균",
        "en": "Weighted mean"
      },
      "explanation": {
        "ko": "−1과 0과 2를 확률로 가중해 더합니다.",
        "en": "Weight −1, 0 and 2 by their probabilities."
      },
      "formula": "E[X]=0.25",
      "diagram": {
        "ko": "고정 기준 예제의 기댓값·분산과 불확실성 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Expectation, Variance and Uncertainty. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "제곱값 평균",
        "en": "Average squares"
      },
      "explanation": {
        "ko": "먼저 제곱하고 평균하는 것은 평균 후 제곱과 다릅니다.",
        "en": "Squaring before averaging differs from squaring the mean."
      },
      "formula": "E[X²]=1.25",
      "diagram": {
        "ko": "고정 기준 예제의 기댓값·분산과 불확실성 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Expectation, Variance and Uncertainty. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "분산",
        "en": "Variance"
      },
      "explanation": {
        "ko": "평균 주변 제곱 편차의 평균을 구합니다.",
        "en": "Average squared deviations from the mean."
      },
      "formula": "Var(X)=1.1875",
      "diagram": {
        "ko": "고정 기준 예제의 기댓값·분산과 불확실성 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Expectation, Variance and Uncertainty. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "평균과 퍼짐",
        "en": "Mean and spread"
      },
      "predict": {
        "ko": "같은 분포에서 평균과 분산은 어떻게 다른가요?",
        "en": "How do mean and variance differ for the same distribution?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "평균 0.25, 분산 1.1875입니다.",
        "en": "The mean is 0.25 and variance 1.1875."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "변동 없음",
        "en": "No variation"
      },
      "predict": {
        "ko": "한 값만 가능하면 어떻게 되나요?",
        "en": "What if only one value is possible?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "X=2가 확실하므로 평균 2, 분산 0입니다.",
        "en": "X=2 is certain, giving mean 2 and variance 0."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "가중치 변경",
        "en": "Change weights"
      },
      "predict": {
        "ko": "음수 값의 확률을 늘리면 평균은 어떻게 될까요?",
        "en": "How does increasing negative-value mass affect the mean?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "평균은 −0.2, 분산은 1.36입니다.",
        "en": "The mean is −0.2 and variance 1.36."
      },
      "preset": 2
    }
  ]
},
"statistical-estimation":{
  "intuition": {
    "ko": "관측을 고정하고 매개변수를 바꾸며 어떤 모형이 데이터를 잘 설명하는지 비교합니다.",
    "en": "Hold observations fixed and compare how parameter values explain the data."
  },
  "steps": [
    {
      "title": {
        "ko": "관측 고정",
        "en": "Fix observations"
      },
      "explanation": {
        "ko": "성공 7회와 실패 3회를 고정합니다.",
        "en": "Fix seven successes and three failures."
      },
      "formula": "n=10; k=7",
      "diagram": {
        "ko": "고정 기준 예제의 통계적 추정과 가능도 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Statistical Estimation and Likelihood. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "가능도",
        "en": "Likelihood"
      },
      "explanation": {
        "ko": "가능도는 고정 관측의 함수값을 p별로 비교합니다.",
        "en": "Likelihood compares fixed-data function values across p."
      },
      "formula": "L(p)∝p⁷(1−p)³",
      "diagram": {
        "ko": "고정 기준 예제의 통계적 추정과 가능도 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Statistical Estimation and Likelihood. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "로그로 미분",
        "en": "Differentiate the log"
      },
      "explanation": {
        "ko": "양의 가능도에서 로그는 최대점을 보존합니다.",
        "en": "The log preserves maxima of positive likelihood."
      },
      "formula": "ℓ′(p)=7/p−3/(1−p)",
      "diagram": {
        "ko": "고정 기준 예제의 통계적 추정과 가능도 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Statistical Estimation and Likelihood. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "추정과 오차",
        "en": "Estimate and uncertainty"
      },
      "explanation": {
        "ko": "내부 최대점은 0.7이고 표본 변동은 따로 평가합니다.",
        "en": "The interior maximum is 0.7; sampling variability needs separate assessment."
      },
      "formula": "p̂=0.7; SE≈0.1449",
      "diagram": {
        "ko": "고정 기준 예제의 통계적 추정과 가능도 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Statistical Estimation and Likelihood. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "가능도 비교",
        "en": "Compare likelihood"
      },
      "predict": {
        "ko": "후보 0.5가 최대일까요?",
        "en": "Does candidate 0.5 maximize likelihood?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "추정값은 0.7이며 후보 0.5보다 로그 가능도가 큽니다.",
        "en": "The estimate is 0.7 with higher log likelihood than candidate 0.5."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "분모 확인",
        "en": "Check the denominator"
      },
      "predict": {
        "ko": "성공 수는 같고 시행 수만 늘리면 추정값은?",
        "en": "What if successes stay fixed while trials increase?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "7/100=0.07입니다. 비율의 분모를 확인하세요.",
        "en": "It becomes 7/100=0.07. Check the denominator."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "경계 MLE",
        "en": "Boundary MLE"
      },
      "predict": {
        "ko": "모두 성공하면 추정값은 어디 있나요?",
        "en": "Where is the estimate when all trials succeed?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "MLE는 1입니다. plug-in 표준오차 0을 확실성으로 오해하지 마세요.",
        "en": "The MLE is 1. Do not interpret zero plug-in standard error as certainty."
      },
      "preset": 2
    }
  ]
},
"bayesian-inference":{
  "intuition": {
    "ko": "사전 지식과 관측 근거를 결합하고 매개변수 불확실성을 예측에 반영합니다.",
    "en": "Combine prior knowledge with evidence and carry parameter uncertainty into prediction."
  },
  "steps": [
    {
      "title": {
        "ko": "사전",
        "en": "Prior"
      },
      "explanation": {
        "ko": "관측 전 p의 분포를 명시합니다.",
        "en": "Specify a distribution over p before observing data."
      },
      "formula": "p∼Beta(1,1)",
      "diagram": {
        "ko": "고정 기준 예제의 Bayes 추론과 사후예측 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Bayesian Inference and Posterior Prediction. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "근거 곱하기",
        "en": "Multiply evidence"
      },
      "explanation": {
        "ko": "가능도를 사전 밀도에 곱합니다.",
        "en": "Multiply likelihood by prior density."
      },
      "formula": "p⁷(1−p)³ × 1",
      "diagram": {
        "ko": "고정 기준 예제의 Bayes 추론과 사후예측 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Bayesian Inference and Posterior Prediction. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "정규화",
        "en": "Normalize"
      },
      "explanation": {
        "ko": "사후분포를 적분해 1이 되도록 만듭니다.",
        "en": "Normalize the posterior to integrate to one."
      },
      "formula": "p|D∼Beta(8,4)",
      "diagram": {
        "ko": "고정 기준 예제의 Bayes 추론과 사후예측 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Bayesian Inference and Posterior Prediction. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "미래 예측",
        "en": "Predict a new outcome"
      },
      "explanation": {
        "ko": "p를 한 값으로 고정하지 않고 사후분포에 대해 평균합니다.",
        "en": "Average over the posterior rather than fixing p at one value."
      },
      "formula": "P(next=1|D)=8/12",
      "diagram": {
        "ko": "고정 기준 예제의 Bayes 추론과 사후예측 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Bayesian Inference and Posterior Prediction. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "균등 사전",
        "en": "Uniform prior"
      },
      "predict": {
        "ko": "사후 평균이 MLE 0.7과 같을까요?",
        "en": "Does the posterior mean equal the MLE 0.7?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "평균은 2/3이며 사후 분산은 2/117입니다.",
        "en": "The mean is 2/3 and posterior variance is 2/117."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "긍정적 사전",
        "en": "Optimistic prior"
      },
      "predict": {
        "ko": "사전이 강하면 관측 후에도 영향이 남나요?",
        "en": "Can a strong prior retain influence after observing data?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "사후 Beta(16,4)의 평균은 0.8입니다.",
        "en": "Posterior Beta(16,4) has mean 0.8."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "다른 사전",
        "en": "A different prior"
      },
      "predict": {
        "ko": "같은 관측이면 모든 사전이 같은 결론을 줄까요?",
        "en": "Must all priors give the same result for the same data?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "사후 Beta(8,12)의 평균은 0.4입니다. 사전 민감도를 보고해야 합니다.",
        "en": "Posterior Beta(8,12) has mean 0.4. Report prior sensitivity."
      },
      "preset": 2
    }
  ]
},
"information-theory":{
  "intuition": {
    "ko": "불확실성과 잘못된 확률 모형의 추가 비용을 구분합니다.",
    "en": "Distinguish uncertainty from the extra cost of using the wrong probability model."
  },
  "steps": [
    {
      "title": {
        "ko": "놀람",
        "en": "Surprisal"
      },
      "explanation": {
        "ko": "가능성이 낮은 사건일수록 −log 확률이 큽니다.",
        "en": "Less likely events have greater negative log probability."
      },
      "formula": "I(x)=−log P(x)",
      "diagram": {
        "ko": "고정 기준 예제의 엔트로피·교차엔트로피·KL 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Entropy, Cross-Entropy and KL Divergence. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "평균 불확실성",
        "en": "Average uncertainty"
      },
      "explanation": {
        "ko": "참 분포로 놀람을 평균합니다.",
        "en": "Average surprisal under the true distribution."
      },
      "formula": "H(P)=−ΣP log P",
      "diagram": {
        "ko": "고정 기준 예제의 엔트로피·교차엔트로피·KL 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Entropy, Cross-Entropy and KL Divergence. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "다른 모형 사용",
        "en": "Use another model"
      },
      "explanation": {
        "ko": "관측은 P에서 오지만 점수는 Q로 계산합니다.",
        "en": "Outcomes come from P but are scored using Q."
      },
      "formula": "H(P,Q)=−ΣP log Q",
      "diagram": {
        "ko": "고정 기준 예제의 엔트로피·교차엔트로피·KL 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Entropy, Cross-Entropy and KL Divergence. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "추가 비용",
        "en": "Excess cost"
      },
      "explanation": {
        "ko": "평균 추가 비용이 KL입니다.",
        "en": "KL is the extra expected cost."
      },
      "formula": "D_KL=H(P,Q)−H(P)",
      "diagram": {
        "ko": "고정 기준 예제의 엔트로피·교차엔트로피·KL 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Entropy, Cross-Entropy and KL Divergence. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "비교",
        "en": "Compare"
      },
      "predict": {
        "ko": "균등 모형이 치우친 실제 분포에 맞을까요?",
        "en": "Does a uniform model match a skewed true distribution?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "H(P)≈0.562335, 교차엔트로피≈0.693147, KL≈0.130812입니다.",
        "en": "H(P)≈0.562335, cross-entropy≈0.693147 and KL≈0.130812."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "일치",
        "en": "Match"
      },
      "predict": {
        "ko": "Q=P이면 어떤 항이 0인가요?",
        "en": "Which term becomes zero when Q=P?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "KL은 0이지만 실제 분포의 엔트로피는 남습니다.",
        "en": "KL is zero while the true distribution’s entropy remains."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "무한 비용",
        "en": "Infinite cost"
      },
      "predict": {
        "ko": "가능한 사건에 확률 0을 주면?",
        "en": "What if the model gives a possible event probability zero?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "그 사건의 −log Q가 무한대이므로 평균 비용도 무한대입니다.",
        "en": "Its negative log model probability is infinite, making expected cost infinite."
      },
      "preset": 2
    }
  ]
},
"regularization":{
  "intuition": {
    "ko": "약한 관측 방향의 증폭을 줄이는 대신 어떤 편향이 생기는지 살펴봅니다.",
    "en": "Inspect the bias introduced when stabilizing weakly observed directions."
  },
  "steps": [
    {
      "title": {
        "ko": "불안정 방향",
        "en": "Unstable direction"
      },
      "explanation": {
        "ko": "작은 s로 나누면 작은 관측 오차가 증폭됩니다.",
        "en": "Dividing by small s amplifies observation error."
      },
      "formula": "w₂=y₂/s",
      "diagram": {
        "ko": "고정 기준 예제의 정규화와 편향·안정성 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Regularization, Bias and Stability. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "목적함수",
        "en": "Objective"
      },
      "explanation": {
        "ko": "잔차와 계수 크기를 함께 벌점화합니다.",
        "en": "Penalize both residual and coefficient magnitude."
      },
      "formula": "L=½‖y−Xw‖²+λ‖w‖²/2",
      "diagram": {
        "ko": "고정 기준 예제의 정규화와 편향·안정성 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Regularization, Bias and Stability. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "닫힌 해",
        "en": "Closed-form solution"
      },
      "explanation": {
        "ko": "정규방정식의 대각에 λ를 더합니다.",
        "en": "Add λ to the diagonal of the normal matrix."
      },
      "formula": "w₂=sy₂/(s²+λ)",
      "diagram": {
        "ko": "고정 기준 예제의 정규화와 편향·안정성 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Regularization, Bias and Stability. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "절충",
        "en": "Trade-off"
      },
      "explanation": {
        "ko": "작은 계수는 안정적일 수 있지만 참값에서 멀어질 수도 있습니다.",
        "en": "Smaller coefficients can be stable yet farther from the truth."
      },
      "formula": "bias ↔ variance",
      "diagram": {
        "ko": "고정 기준 예제의 정규화와 편향·안정성 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Regularization, Bias and Stability. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "축소",
        "en": "Shrinkage"
      },
      "predict": {
        "ko": "약한 축의 계수는 얼마나 줄어들까요?",
        "en": "How much does the weak-axis coefficient shrink?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "λ=0.1에서 w₂=10/11≈0.909091입니다.",
        "en": "At λ=0.1, w₂=10/11≈0.909091."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "증폭",
        "en": "Amplification"
      },
      "predict": {
        "ko": "λ=0이면 같은 관측의 계수는?",
        "en": "What is the coefficient for the same observation at λ=0?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "w₂=10입니다. 작은 s가 역방향에서 증폭합니다.",
        "en": "w₂=10. The small scale amplifies inversion."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "식별 불가",
        "en": "Nonidentifiability"
      },
      "predict": {
        "ko": "s=0에서 데이터가 w₂를 알려주나요?",
        "en": "Does the data identify w₂ at s=0?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "아닙니다. 표시한 0은 최소 노름 선택이며 유일한 데이터 결론이 아닙니다.",
        "en": "No. Displayed zero is a minimum-norm choice, not a uniquely data-determined conclusion."
      },
      "preset": 2
    }
  ]
},
"stochastic-optimization":{
  "intuition": {
    "ko": "전체 평균 기울기와 무작위 표본 하나의 기울기를 비교합니다.",
    "en": "Compare the full mean gradient with the gradient of one randomly sampled example."
  },
  "steps": [
    {
      "title": {
        "ko": "전체 목적",
        "en": "Full objective"
      },
      "explanation": {
        "ko": "세 표본의 제곱 손실을 평균합니다.",
        "en": "Average squared loss over three examples."
      },
      "formula": "L(w)=mean ½(w−zᵢ)²",
      "diagram": {
        "ko": "고정 기준 예제의 확률적 경사와 학습 잡음 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Stochastic Gradients and Learning Noise. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "평균 경사",
        "en": "Mean gradient"
      },
      "explanation": {
        "ko": "표본 평균은 1이므로 전체 기울기는 w−1입니다.",
        "en": "The sample mean is 1, so the full gradient is w−1."
      },
      "formula": "∇L=w−1",
      "diagram": {
        "ko": "고정 기준 예제의 확률적 경사와 학습 잡음 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Stochastic Gradients and Learning Noise. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "확률적 추정",
        "en": "Stochastic estimate"
      },
      "explanation": {
        "ko": "표본 하나의 경사는 w−zᵢ이며 평균은 전체 경사입니다.",
        "en": "One sample gives w−zᵢ, whose expectation is the full gradient."
      },
      "formula": "E[gᵢ]=w−1; Var(gᵢ)=8/3",
      "diagram": {
        "ko": "고정 기준 예제의 확률적 경사와 학습 잡음 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Stochastic Gradients and Learning Noise. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "반복과 잡음",
        "en": "Iteration and noise"
      },
      "explanation": {
        "ko": "서로 다른 seed는 다른 경로를 만들 수 있습니다.",
        "en": "Different seeds can produce different paths."
      },
      "formula": "w←w−ηgᵢ",
      "diagram": {
        "ko": "고정 기준 예제의 확률적 경사와 학습 잡음 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Stochastic Gradients and Learning Noise. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "평균과 경로",
        "en": "Mean and path"
      },
      "predict": {
        "ko": "불편 추정이면 매번 같은 길을 갈까요?",
        "en": "Does an unbiased estimator follow the same path every time?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "아닙니다. 평균 경사와 표본 경사의 궤적을 구분하세요.",
        "en": "No. Distinguish the mean-gradient path from a sampled path."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "seed 비교",
        "en": "Compare seeds"
      },
      "predict": {
        "ko": "seed만 바꾸면 목적함수도 바뀌나요?",
        "en": "Does changing only the seed change the objective?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "목적함수는 같고 추출 순서와 경로만 바뀝니다.",
        "en": "The objective stays fixed; the sampled sequence and trajectory change."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "불안정",
        "en": "Instability"
      },
      "predict": {
        "ko": "불편한 경사도 큰 보폭에서 안전할까요?",
        "en": "Is an unbiased gradient safe at any step size?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "η=2.1은 이 이차 목적의 전체 경사하강도 불안정하게 합니다.",
        "en": "η=2.1 destabilizes even full gradient descent for this quadratic."
      },
      "preset": 2
    }
  ]
},
"differential-equations":{
  "intuition": {
    "ko": "연속 시간의 감쇠와 유한 보폭 수치해법을 분리해 이해합니다.",
    "en": "Separate continuous-time decay from its finite-step numerical approximation."
  },
  "steps": [
    {
      "title": {
        "ko": "변화 법칙",
        "en": "Evolution law"
      },
      "explanation": {
        "ko": "현재 상태가 다음 순간의 변화율을 결정합니다.",
        "en": "The current state determines its instantaneous rate."
      },
      "formula": "dx/dt=−kx",
      "diagram": {
        "ko": "고정 기준 예제의 미분방정식과 시간 이산화 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Differential Equations and Time Discretization. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "초기 조건",
        "en": "Initial condition"
      },
      "explanation": {
        "ko": "법칙만으로는 궤적이 하나로 정해지지 않습니다.",
        "en": "The law alone does not select one trajectory."
      },
      "formula": "x(0)=x₀",
      "diagram": {
        "ko": "고정 기준 예제의 미분방정식과 시간 이산화 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Differential Equations and Time Discretization. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "수치 보폭",
        "en": "Numerical step"
      },
      "explanation": {
        "ko": "현재 기울기로 다음 상태를 예측합니다.",
        "en": "Use the current rate to predict the next state."
      },
      "formula": "xₙ₊₁=(1−kΔt)xₙ",
      "diagram": {
        "ko": "고정 기준 예제의 미분방정식과 시간 이산화 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Differential Equations and Time Discretization. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "안정성",
        "en": "Stability"
      },
      "explanation": {
        "ko": "참해가 줄어들어도 잘못된 간격은 발산할 수 있습니다.",
        "en": "A poor step can diverge even when the true solution decays."
      },
      "formula": "|1−kΔt|<1",
      "diagram": {
        "ko": "고정 기준 예제의 미분방정식과 시간 이산화 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Differential Equations and Time Discretization. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "해와 근사",
        "en": "Solution and approximation"
      },
      "predict": {
        "ko": "Euler는 정확한 지수해인가요?",
        "en": "Is Euler the exact exponential solution?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "아닙니다. 기준 최종 시간은 5이며 두 값을 비교하세요.",
        "en": "No. The reference final time is 5; compare the two terminal values."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "간격 축소",
        "en": "Step refinement"
      },
      "predict": {
        "ko": "간격을 줄일 때 비교할 시간도 같나요?",
        "en": "Does reducing the step keep the comparison time fixed?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "이 실험은 20회 고정이므로 최종 시간은 1로 바뀝니다.",
        "en": "This lab fixes 20 steps, changing final time to 1."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "수치 발산",
        "en": "Numerical divergence"
      },
      "predict": {
        "ko": "감쇠 방정식이 왜 커지는 수치해를 만들까요?",
        "en": "Why can a decay equation produce a growing numerical solution?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "kΔt=3에서 배율은 −2이므로 진동하며 커집니다.",
        "en": "At kΔt=3 the multiplier is −2, producing growing oscillations."
      },
      "preset": 2
    }
  ]
},
"mechanics-energy":{
  "intuition": {
    "ko": "조화진동자의 힘·운동·에너지를 연결하고 수치 오차를 관찰합니다.",
    "en": "Connect force, motion and energy in an oscillator and inspect numerical error."
  },
  "steps": [
    {
      "title": {
        "ko": "힘",
        "en": "Force"
      },
      "explanation": {
        "ko": "변위 반대 방향의 복원력이 작용합니다.",
        "en": "Restoring force opposes displacement."
      },
      "formula": "F=−kq",
      "diagram": {
        "ko": "고정 기준 예제의 역학·에너지와 구조 보존 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Mechanics, Energy and Structure Preservation. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "운동",
        "en": "Motion"
      },
      "explanation": {
        "ko": "관성 때문에 평형점을 지나 계속 움직입니다.",
        "en": "Inertia carries the mass beyond equilibrium."
      },
      "formula": "ω=√(k/m)",
      "diagram": {
        "ko": "고정 기준 예제의 역학·에너지와 구조 보존 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Mechanics, Energy and Structure Preservation. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "에너지",
        "en": "Energy"
      },
      "explanation": {
        "ko": "운동에너지와 위치에너지가 교환됩니다.",
        "en": "Kinetic and potential energy exchange."
      },
      "formula": "E=½mv²+½kq²",
      "diagram": {
        "ko": "고정 기준 예제의 역학·에너지와 구조 보존 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Mechanics, Energy and Structure Preservation. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "구조와 오차",
        "en": "Structure and error"
      },
      "explanation": {
        "ko": "수치해의 보존량 오차를 실제 에너지와 구분합니다.",
        "en": "Distinguish numerical invariant error from physical energy."
      },
      "formula": "v←v−Δt(k/m)q; q←q+Δt v",
      "diagram": {
        "ko": "고정 기준 예제의 역학·에너지와 구조 보존 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Mechanics, Energy and Structure Preservation. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "오차 관찰",
        "en": "Observe error"
      },
      "predict": {
        "ko": "구조 보존 해법이면 에너지 오차가 0인가요?",
        "en": "Does a structure-preserving method have zero energy error?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "아닙니다. 에너지 편차를 직접 확인하세요.",
        "en": "No. Inspect the measured energy deviation."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "시간 분해능",
        "en": "Time resolution"
      },
      "predict": {
        "ko": "간격이 작아지면 어떤 차이가 보이나요?",
        "en": "What changes when the step is smaller?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "곡선과 에너지 오차가 달라집니다. 같은 최종 시간 비교가 추가로 필요합니다.",
        "en": "The trajectory and energy error change. A fixed-final-time comparison is still needed."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "경계",
        "en": "Boundary"
      },
      "predict": {
        "ko": "ωΔt=2도 안정한가요?",
        "en": "Is ωΔt=2 safely stable?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "엄격한 안정 구간의 경계이며 초기값에 따라 비유계 성장이 생깁니다.",
        "en": "It is the boundary of the strict stability interval; some initial conditions grow without bound."
      },
      "preset": 2
    }
  ]
},
"thermodynamics":{
  "intuition": {
    "ko": "미시 상태의 확률에서 에너지·자유에너지·엔트로피를 계산합니다.",
    "en": "Calculate energy, free energy and entropy from microscopic state probabilities."
  },
  "steps": [
    {
      "title": {
        "ko": "상태와 가중치",
        "en": "States and weights"
      },
      "explanation": {
        "ko": "에너지가 높은 상태는 Boltzmann 가중치가 작습니다.",
        "en": "Higher-energy states have smaller Boltzmann weights."
      },
      "formula": "weight=e⁻ᴱ/ᵀ",
      "diagram": {
        "ko": "고정 기준 예제의 통계역학과 분배함수 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Statistical Mechanics and the Partition Function. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "분배함수",
        "en": "Partition function"
      },
      "explanation": {
        "ko": "모든 미시 가중치를 더해 정규화합니다.",
        "en": "Normalize by summing all microscopic weights."
      },
      "formula": "Z=1+g e⁻Δ/T",
      "diagram": {
        "ko": "고정 기준 예제의 통계역학과 분배함수 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Statistical Mechanics and the Partition Function. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "평균 에너지",
        "en": "Mean energy"
      },
      "explanation": {
        "ko": "들뜬 에너지에 점유 확률을 곱합니다.",
        "en": "Multiply the excited energy by its occupancy probability."
      },
      "formula": "U=ΔP(E=Δ)",
      "diagram": {
        "ko": "고정 기준 예제의 통계역학과 분배함수 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Statistical Mechanics and the Partition Function. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "엔트로피",
        "en": "Entropy"
      },
      "explanation": {
        "ko": "정규화한 미시 확률의 불확실성과 열역학 관계를 확인합니다.",
        "en": "Connect uncertainty of microscopic probabilities with the thermodynamic identity."
      },
      "formula": "S=(U−F)/T",
      "diagram": {
        "ko": "고정 기준 예제의 통계역학과 분배함수 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Statistical Mechanics and the Partition Function. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "정규화",
        "en": "Normalization"
      },
      "predict": {
        "ko": "분배함수 없이 가중치가 확률인가요?",
        "en": "Are unnormalized weights probabilities?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "아닙니다. Z로 나누며 기준 점유 확률은 약 0.268941입니다.",
        "en": "No. Divide by Z; reference excited occupancy is about 0.268941."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "온도",
        "en": "Temperature"
      },
      "predict": {
        "ko": "온도가 높아지면 항상 바닥 상태만 점유하나요?",
        "en": "Does higher temperature keep all mass in the ground state?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "두 상태 점유가 더 비슷해집니다.",
        "en": "The two state probabilities become more similar."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "미시 상태 세기",
        "en": "Count microstates"
      },
      "predict": {
        "ko": "같은 에너지의 상태가 여러 개면 어떤 항이 더해질까요?",
        "en": "What changes when an energy level has multiple microstates?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "전체 엔트로피에는 P(E=Δ) log g가 추가됩니다.",
        "en": "Full entropy adds P(E=Δ) log g to binary occupancy entropy."
      },
      "preset": 2
    }
  ]
},
"stochastic-processes":{
  "intuition": {
    "ko": "한 번의 분포가 아니라 시간에 따라 이동하는 확률을 추적합니다.",
    "en": "Track probability evolving over time rather than only one static distribution."
  },
  "steps": [
    {
      "title": {
        "ko": "상태와 전이",
        "en": "States and transitions"
      },
      "explanation": {
        "ko": "현재 상태만으로 다음 상태의 조건부 분포를 지정합니다.",
        "en": "The current state specifies the conditional next-state distribution."
      },
      "formula": "P=[[1−a,a],[b,1−b]]",
      "diagram": {
        "ko": "고정 기준 예제의 Markov 과정과 평형 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Markov Processes and Equilibrium. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "확률 보존",
        "en": "Conserve probability"
      },
      "explanation": {
        "ko": "상태 0에서 오는 양과 상태 1에 남는 양을 더합니다.",
        "en": "Add incoming mass from state 0 and retained mass in state 1."
      },
      "formula": "p′=a(1−p)+(1−b)p",
      "diagram": {
        "ko": "고정 기준 예제의 Markov 과정과 평형 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Markov Processes and Equilibrium. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "정상분포",
        "en": "Stationary distribution"
      },
      "explanation": {
        "ko": "변화하지 않는 점유를 풉니다.",
        "en": "Solve for unchanged occupancy."
      },
      "formula": "π₁=a/(a+b)",
      "diagram": {
        "ko": "고정 기준 예제의 Markov 과정과 평형 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Markov Processes and Equilibrium. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "수렴 조건",
        "en": "Convergence conditions"
      },
      "explanation": {
        "ko": "두 번째 고유값의 크기로 편차의 감소를 확인합니다.",
        "en": "Inspect decay of deviation through the second eigenvalue."
      },
      "formula": "pₜ−π₁=(1−a−b)ᵗ(p₀−π₁)",
      "diagram": {
        "ko": "고정 기준 예제의 Markov 과정과 평형 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Markov Processes and Equilibrium. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "혼합",
        "en": "Mixing"
      },
      "predict": {
        "ko": "초기값 1이 어디로 가나요?",
        "en": "Where does initial occupancy 1 move?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "정상 점유 0.4로 접근합니다.",
        "en": "It approaches stationary occupancy 0.4."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "정상과 수렴",
        "en": "Stationarity versus convergence"
      },
      "predict": {
        "ko": "정상분포가 있으면 항상 수렴하나요?",
        "en": "Does a stationary distribution guarantee convergence?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "a=b=1이면 정상 점유 0.5가 있지만 초기값 1은 교대로 진동합니다.",
        "en": "At a=b=1, stationary occupancy 0.5 exists, but initial occupancy 1 oscillates."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "비유일성",
        "en": "Nonuniqueness"
      },
      "predict": {
        "ko": "전이가 없으면 정상분포가 하나인가요?",
        "en": "Is there one stationary distribution when no transitions occur?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "모든 초기분포가 정상이므로 유일하지 않습니다.",
        "en": "Every initial distribution is stationary, so it is not unique."
      },
      "preset": 2
    }
  ]
},
"inverse-problems":{
  "intuition": {
    "ko": "관측에서 원인을 복원할 때 정보 부족과 잡음 증폭을 구분합니다.",
    "en": "Separate missing information from noise amplification when reconstructing causes from observations."
  },
  "steps": [
    {
      "title": {
        "ko": "순방향",
        "en": "Forward map"
      },
      "explanation": {
        "ko": "원인은 θ, 관측은 Xθ입니다.",
        "en": "θ is the cause; Xθ is the observation."
      },
      "formula": "θtrue=(1,2)",
      "diagram": {
        "ko": "고정 기준 예제의 역문제·식별성과 관측 설계 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Inverse Problems, Identifiability and Observation Design. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "약한 정보",
        "en": "Weak information"
      },
      "explanation": {
        "ko": "둘째 관측은 두 원인의 차이를 약하게 봅니다.",
        "en": "The second observation weakly measures the parameter contrast."
      },
      "formula": "y₂=ε(θ₁−θ₂)+noise",
      "diagram": {
        "ko": "고정 기준 예제의 역문제·식별성과 관측 설계 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Inverse Problems, Identifiability and Observation Design. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "역방향 증폭",
        "en": "Inverse amplification"
      },
      "explanation": {
        "ko": "차이를 복원하려고 ε로 나누면 잡음도 커집니다.",
        "en": "Dividing by ε to recover contrast also amplifies noise."
      },
      "formula": "δ(θ₁−θ₂)=noise/ε",
      "diagram": {
        "ko": "고정 기준 예제의 역문제·식별성과 관측 설계 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Inverse Problems, Identifiability and Observation Design. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "새 관측",
        "en": "New observation"
      },
      "explanation": {
        "ko": "벌점과 실제 정보 추가는 다른 개입입니다.",
        "en": "Regularization and adding real information are different interventions."
      },
      "formula": "conditioning ≠ identifiability",
      "diagram": {
        "ko": "고정 기준 예제의 역문제·식별성과 관측 설계 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Inverse Problems, Identifiability and Observation Design. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "약한 방향",
        "en": "Weak direction"
      },
      "predict": {
        "ko": "잡음 0.1이 매개변수 오차도 0.1인가요?",
        "en": "Does noise 0.1 imply parameter error 0.1?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "ε=0.1이면 추정은 (1.5,1.5)입니다.",
        "en": "At ε=0.1 the estimate is (1.5,1.5)."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "관측 설계",
        "en": "Observation design"
      },
      "predict": {
        "ko": "같은 잡음에서 대비 관측을 강화하면?",
        "en": "What happens if contrast observation is stronger at the same noise level?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "추정은 (1.05,1.95)로 참값에 더 가깝습니다.",
        "en": "The estimate (1.05,1.95) is closer to truth."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "식별 불가",
        "en": "Nonidentifiability"
      },
      "predict": {
        "ko": "두 원인의 합만 알면 각각을 복원할 수 있나요?",
        "en": "Can one recover both causes from their sum alone?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "합 3을 만족하는 해가 무한히 많습니다. 표시한 (1.5,1.5)는 선택입니다.",
        "en": "Infinitely many solutions sum to 3. Displayed (1.5,1.5) is a chosen solution."
      },
      "preset": 2
    }
  ]
},
"research-methods":{
  "intuition": {
    "ko": "평균 개선값만 보고 결론 내리지 않고 대응 실험과 가정·오차를 함께 검토합니다.",
    "en": "Assess paired experiments, assumptions and uncertainty rather than concluding from mean improvement alone."
  },
  "steps": [
    {
      "title": {
        "ko": "주장 정하기",
        "en": "Define the claim"
      },
      "explanation": {
        "ko": "비교 대상·점수·데이터·예산을 먼저 고정합니다.",
        "en": "Fix the comparator, score, data and budget first."
      },
      "formula": "Δᵢ=score(new,i)−score(base,i)",
      "diagram": {
        "ko": "고정 기준 예제의 재현·불확실성과 연구 주장 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Reproducibility, Uncertainty and Research Claims. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "대응 비교",
        "en": "Paired comparison"
      },
      "explanation": {
        "ko": "같은 조건의 차이를 seed별로 계산합니다.",
        "en": "Compute differences under matched conditions for each seed."
      },
      "formula": "d=[0.1,0.2,0,0.1]",
      "diagram": {
        "ko": "고정 기준 예제의 재현·불확실성과 연구 주장 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Reproducibility, Uncertainty and Research Claims. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "불확실성",
        "en": "Uncertainty"
      },
      "explanation": {
        "ko": "평균뿐 아니라 표준오차와 표본 가정을 확인합니다.",
        "en": "Inspect standard error and sampling assumptions alongside the mean."
      },
      "formula": "SE=s/√n≈0.040825",
      "diagram": {
        "ko": "고정 기준 예제의 재현·불확실성과 연구 주장 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Reproducibility, Uncertainty and Research Claims. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    },
    {
      "title": {
        "ko": "범위 제한",
        "en": "Limit the claim"
      },
      "explanation": {
        "ko": "표본이 작으면 양의 평균만으로 일반적 개선을 결론 내릴 수 없습니다.",
        "en": "A positive mean in a small sample does not establish general improvement."
      },
      "formula": "95% t CI≈[−0.029923,0.229923]",
      "diagram": {
        "ko": "고정 기준 예제의 재현·불확실성과 연구 주장 그림입니다. 파랑과 주황의 의미 및 축은 본문의 설명과 수치 표를 함께 확인하세요.",
        "en": "Fixed reference illustration for Reproducibility, Uncertainty and Research Claims. Read the axes and blue/orange legend alongside the worked example and numeric table."
      }
    }
  ],
  "experiments": [
    {
      "title": {
        "ko": "양의 평균",
        "en": "Positive mean"
      },
      "predict": {
        "ko": "평균 개선 0.1이면 충분한가요?",
        "en": "Is mean improvement 0.1 sufficient?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "구간이 0을 포함합니다. 가정·비용·추가 표본을 검토하세요.",
        "en": "The interval contains zero. Review assumptions, cost and additional samples."
      },
      "preset": 0
    },
    {
      "title": {
        "ko": "크기와 변동",
        "en": "Magnitude and variation"
      },
      "predict": {
        "ko": "평균만 키우면 구간은 어떻게 달라지나요?",
        "en": "How does increasing the mean shift the interval?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "같은 변동에서 구간 전체가 이동합니다. 평가 설계의 타당성은 별도입니다.",
        "en": "At fixed variation the interval shifts. Evaluation validity remains a separate question."
      },
      "preset": 1
    },
    {
      "title": {
        "ko": "부정적 결과",
        "en": "Negative result"
      },
      "predict": {
        "ko": "평균 0이면 기록할 가치가 없나요?",
        "en": "Is a zero-mean result unworthy of recording?"
      },
      "action": {
        "ko": "예제를 불러온 뒤 그림과 수치 표를 비교하세요.",
        "en": "Load the example and compare the plot with the numeric readout."
      },
      "explain": {
        "ko": "실패 조건과 불확실성도 중요한 재현 기록입니다.",
        "en": "Failure conditions and uncertainty are important reproducibility records."
      },
      "preset": 2
    }
  ]
},
};

export interface FoundationControl {label:BookText;min:number;max:number;step:number|'any';initial:number;}
export interface FoundationLabSpec {controls:FoundationControl[];presets:{name:BookText;values:number[]}[];axis:BookText;legend:BookText[];note:BookText;}
export const foundationLabs: Record<string, FoundationLabSpec> = {
  "svd": {
    "controls": [
      {
        "label": {
          "ko": "A11",
          "en": "A11"
        },
        "min": -4,
        "max": 4,
        "step": "any",
        "initial": 3
      },
      {
        "label": {
          "ko": "A12",
          "en": "A12"
        },
        "min": -4,
        "max": 4,
        "step": "any",
        "initial": 0
      },
      {
        "label": {
          "ko": "A21",
          "en": "A21"
        },
        "min": -4,
        "max": 4,
        "step": "any",
        "initial": 0
      },
      {
        "label": {
          "ko": "A22",
          "en": "A22"
        },
        "min": -4,
        "max": 4,
        "step": "any",
        "initial": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "기준 대각 행렬",
          "en": "Reference diagonal"
        },
        "values": [
          3,
          0,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "약한 방향",
          "en": "Weak direction"
        },
        "values": [
          3,
          0,
          0,
          0.1
        ]
      },
      {
        "name": {
          "ko": "랭크 1",
          "en": "Rank one"
        },
        "values": [
          2,
          2,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: 출력 1 · 세로: 출력 2",
      "en": "Horizontal: output 1 · Vertical: output 2"
    },
    "legend": [
      {
        "ko": "A가 보낸 단위원",
        "en": "Unit circle mapped by A"
      },
      {
        "ko": "랭크 1 근사",
        "en": "Rank-one approximation"
      }
    ],
    "note": {
      "ko": "실험은 실수 2×2 행렬입니다. SVD 자체는 직사각형에도 성립합니다. 조건수 미정의 표시는 특이 행렬을 뜻하며 오차 0으로 해석하지 마세요.",
      "en": "The lab uses real 2×2 matrices; SVD also applies to rectangular maps. An undefined condition number denotes singularity, not zero error."
    }
  },
  "probability-distributions": {
    "controls": [
      {
        "label": {
          "ko": "성공 확률 p",
          "en": "Success probability p"
        },
        "min": 0,
        "max": 1,
        "step": "any",
        "initial": 0.5
      },
      {
        "label": {
          "ko": "시행 수 n",
          "en": "Trials n"
        },
        "min": 1,
        "max": 20,
        "step": 1,
        "initial": 4
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "대칭 분포",
          "en": "Symmetric distribution"
        },
        "values": [
          0.5,
          4
        ]
      },
      {
        "name": {
          "ko": "치우친 분포",
          "en": "Skewed distribution"
        },
        "values": [
          0.2,
          4
        ]
      },
      {
        "name": {
          "ko": "확정 결과",
          "en": "Deterministic outcome"
        },
        "values": [
          1,
          4
        ]
      }
    ],
    "axis": {
      "ko": "가로: 성공 횟수 k · 세로: 확률질량",
      "en": "Horizontal: successes k · Vertical: probability mass"
    },
    "legend": [
      {
        "ko": "이항 확률질량",
        "en": "Binomial probability mass"
      }
    ],
    "note": {
      "ko": "독립이고 성공 확률이 같은 시행을 가정합니다. 막대는 정확한 모형 확률이며 무작위 표본의 빈도가 아닙니다.",
      "en": "Trials are independent with the same success probability. Bars show exact model probabilities, not random sample frequencies."
    }
  },
  "expectation-variance": {
    "controls": [
      {
        "label": {
          "ko": "−1의 가중치",
          "en": "Weight at −1"
        },
        "min": 0,
        "max": 5,
        "step": "any",
        "initial": 1
      },
      {
        "label": {
          "ko": "0의 가중치",
          "en": "Weight at 0"
        },
        "min": 0,
        "max": 5,
        "step": "any",
        "initial": 2
      },
      {
        "label": {
          "ko": "2의 가중치",
          "en": "Weight at 2"
        },
        "min": 0.01,
        "max": 5,
        "step": "any",
        "initial": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "기준 분포",
          "en": "Reference distribution"
        },
        "values": [
          1,
          2,
          1
        ]
      },
      {
        "name": {
          "ko": "한 점 집중",
          "en": "Single-point concentration"
        },
        "values": [
          0,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "대칭이 아닌 분포",
          "en": "Asymmetric distribution"
        },
        "values": [
          3,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: X의 값 · 세로: 확률질량",
      "en": "Horizontal: value of X · Vertical: probability mass"
    },
    "legend": [
      {
        "ko": "정규화한 확률",
        "en": "Normalized probabilities"
      },
      {
        "ko": "평균 위치",
        "en": "Mean location"
      }
    ],
    "note": {
      "ko": "가중치는 합으로 나눠 확률로 만듭니다. 주황 선은 확률질량이 아니라 평균의 위치입니다. 분산의 단위는 X 단위의 제곱입니다.",
      "en": "Weights are normalized by their sum. The orange line marks the mean, not probability mass. Variance has squared units of X."
    }
  },
  "statistical-estimation": {
    "controls": [
      {
        "label": {
          "ko": "후보 p",
          "en": "Candidate p"
        },
        "min": 0,
        "max": 1,
        "step": "any",
        "initial": 0.5
      },
      {
        "label": {
          "ko": "표본 수 n",
          "en": "Sample size n"
        },
        "min": 10,
        "max": 100,
        "step": 1,
        "initial": 10
      },
      {
        "label": {
          "ko": "성공 수 k",
          "en": "Successes k"
        },
        "min": 0,
        "max": 10,
        "step": 1,
        "initial": 7
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "7/10 성공",
          "en": "7 successes in 10"
        },
        "values": [
          0.5,
          10,
          7
        ]
      },
      {
        "name": {
          "ko": "더 많은 표본",
          "en": "More observations"
        },
        "values": [
          0.5,
          100,
          7
        ]
      },
      {
        "name": {
          "ko": "경계 관측",
          "en": "Boundary observation"
        },
        "values": [
          1,
          10,
          10
        ]
      }
    ],
    "axis": {
      "ko": "가로: 후보 p · 세로: 최대값 대비 가능도",
      "en": "Horizontal: candidate p · Vertical: likelihood relative to its maximum"
    },
    "legend": [
      {
        "ko": "상대 가능도",
        "en": "Relative likelihood"
      }
    ],
    "note": {
      "ko": "순서가 있는 Bernoulli 관측의 가능도에서 p와 무관한 상수는 생략합니다. 곡선은 p의 확률분포가 아닙니다. 표준오차는 plug-in 근사이며 경계에서 0이더라도 불확실성이 사라지지 않습니다.",
      "en": "Parameter-independent constants are omitted for ordered Bernoulli observations. This curve is not a probability distribution over p. Plug-in standard error can be zero at a boundary without eliminating uncertainty."
    }
  },
  "bayesian-inference": {
    "controls": [
      {
        "label": {
          "ko": "사전 α",
          "en": "Prior α"
        },
        "min": 1,
        "max": 20,
        "step": 1,
        "initial": 1
      },
      {
        "label": {
          "ko": "사전 β",
          "en": "Prior β"
        },
        "min": 1,
        "max": 20,
        "step": 1,
        "initial": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "균등 사전",
          "en": "Uniform prior"
        },
        "values": [
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "성공을 선호하는 사전",
          "en": "Success-favoring prior"
        },
        "values": [
          9,
          1
        ]
      },
      {
        "name": {
          "ko": "실패를 선호하는 사전",
          "en": "Failure-favoring prior"
        },
        "values": [
          1,
          9
        ]
      }
    ],
    "axis": {
      "ko": "가로: 성공 확률 p · 세로: 확률밀도",
      "en": "Horizontal: success probability p · Vertical: probability density"
    },
    "legend": [
      {
        "ko": "사전 밀도",
        "en": "Prior density"
      },
      {
        "ko": "사후 밀도",
        "en": "Posterior density"
      }
    ],
    "note": {
      "ko": "관측은 성공 7회·실패 3회로 고정합니다. 밀도 높이는 한 점의 확률이 아닙니다. 실험의 정수 α·β 제한은 조작 범위이며 Beta 분포는 양의 실수 모양 매개변수도 허용합니다.",
      "en": "Data are fixed at seven successes and three failures. Density height is not point probability. Integer α and β are a lab restriction; beta distributions permit positive real shapes."
    }
  },
  "information-theory": {
    "controls": [
      {
        "label": {
          "ko": "실제 p",
          "en": "True p"
        },
        "min": 0,
        "max": 1,
        "step": "any",
        "initial": 0.75
      },
      {
        "label": {
          "ko": "모형 q",
          "en": "Model q"
        },
        "min": 0,
        "max": 1,
        "step": "any",
        "initial": 0.5
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "불일치",
          "en": "Mismatch"
        },
        "values": [
          0.75,
          0.5
        ]
      },
      {
        "name": {
          "ko": "일치",
          "en": "Matched model"
        },
        "values": [
          0.75,
          0.75
        ]
      },
      {
        "name": {
          "ko": "지원집합 오류",
          "en": "Support mismatch"
        },
        "values": [
          0.75,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: Bernoulli p · 세로: 엔트로피 [nat]",
      "en": "Horizontal: Bernoulli p · Vertical: entropy [nat]"
    },
    "legend": [
      {
        "ko": "가능한 p의 엔트로피",
        "en": "Entropy across p"
      },
      {
        "ko": "현재 P의 위치",
        "en": "Current P location"
      }
    ],
    "note": {
      "ko": "자연로그 단위 nat를 사용합니다. Q가 가능한 사건에 확률 0을 주면 KL과 교차엔트로피는 무한대입니다. 임의의 작은 양수로 숨기지 않습니다.",
      "en": "Natural logarithms give nats. If Q assigns zero to an event possible under P, KL and cross-entropy are infinite; no hidden epsilon is inserted."
    }
  },
  "regularization": {
    "controls": [
      {
        "label": {
          "ko": "벌점 λ",
          "en": "Penalty λ"
        },
        "min": 0,
        "max": 4,
        "step": "any",
        "initial": 0.1
      },
      {
        "label": {
          "ko": "약한 축 s",
          "en": "Weak scale s"
        },
        "min": 0,
        "max": 1,
        "step": "any",
        "initial": 0.1
      },
      {
        "label": {
          "ko": "두 번째 관측 y₂",
          "en": "Second observation y₂"
        },
        "min": -2,
        "max": 2,
        "step": "any",
        "initial": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "약한 축의 축소",
          "en": "Shrink a weak axis"
        },
        "values": [
          0.1,
          0.1,
          1
        ]
      },
      {
        "name": {
          "ko": "벌점 없음",
          "en": "No penalty"
        },
        "values": [
          0,
          0.1,
          1
        ]
      },
      {
        "name": {
          "ko": "정보 없는 축",
          "en": "Unobserved axis"
        },
        "values": [
          0,
          0,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: λ · 세로: 계수 값",
      "en": "Horizontal: λ · Vertical: coefficient value"
    },
    "legend": [
      {
        "ko": "첫 계수 w₁",
        "en": "First coefficient w₁"
      },
      {
        "ko": "둘째 계수 w₂",
        "en": "Second coefficient w₂"
      }
    ],
    "note": {
      "ko": "X=diag(3,s), y=(3,y₂)의 ridge입니다. λ=0이고 s=0이면 최소 노름 해 하나를 표시합니다. 벌점은 선택한 좌표의 크기를 선호하며 무편향성을 보장하지 않습니다.",
      "en": "Ridge uses X=diag(3,s), y=(3,y₂). At λ=s=0 the displayed solution is the minimum-norm choice. The penalty favors scale in the chosen coordinates and does not guarantee unbiasedness."
    }
  },
  "stochastic-optimization": {
    "controls": [
      {
        "label": {
          "ko": "학습률 η",
          "en": "Learning rate η"
        },
        "min": 0.01,
        "max": 2.1,
        "step": "any",
        "initial": 0.1
      },
      {
        "label": {
          "ko": "시작 w₀",
          "en": "Initial w₀"
        },
        "min": -4,
        "max": 4,
        "step": "any",
        "initial": 4
      },
      {
        "label": {
          "ko": "난수 seed",
          "en": "Random seed"
        },
        "min": 1,
        "max": 9999,
        "step": 1,
        "initial": 123
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "기준 seed",
          "en": "Reference seed"
        },
        "values": [
          0.1,
          4,
          123
        ]
      },
      {
        "name": {
          "ko": "다른 seed",
          "en": "Different seed"
        },
        "values": [
          0.1,
          4,
          321
        ]
      },
      {
        "name": {
          "ko": "큰 보폭",
          "en": "Large step"
        },
        "values": [
          2.1,
          4,
          123
        ]
      }
    ],
    "axis": {
      "ko": "가로: 갱신 횟수 · 세로: 매개변수 w",
      "en": "Horizontal: update count · Vertical: parameter w"
    },
    "legend": [
      {
        "ko": "전체 경사하강",
        "en": "Full gradient descent"
      },
      {
        "ko": "표본 경사하강",
        "en": "Sample gradient descent"
      }
    ],
    "note": {
      "ko": "표본 z∈{−1,1,3}을 복원 추출합니다. seed는 재현을 위한 결정적 난수열을 정합니다. 상수 학습률 SGD가 매번 손실을 줄이거나 정확히 최적점에 멈춘다고 가정하지 않습니다.",
      "en": "Sample z∈{−1,1,3} uniformly with replacement. A seed selects a reproducible pseudorandom sequence. Constant-rate SGD need not decrease loss every step or stop exactly at the optimum."
    }
  },
  "differential-equations": {
    "controls": [
      {
        "label": {
          "ko": "감쇠율 k",
          "en": "Decay rate k"
        },
        "min": 0.1,
        "max": 5,
        "step": "any",
        "initial": 1
      },
      {
        "label": {
          "ko": "시간 간격 Δt",
          "en": "Time step Δt"
        },
        "min": 0.01,
        "max": 1,
        "step": "any",
        "initial": 0.25
      },
      {
        "label": {
          "ko": "초기값 x₀",
          "en": "Initial x₀"
        },
        "min": -2,
        "max": 2,
        "step": "any",
        "initial": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "기준 간격",
          "en": "Reference step"
        },
        "values": [
          1,
          0.25,
          1
        ]
      },
      {
        "name": {
          "ko": "작은 간격",
          "en": "Small step"
        },
        "values": [
          1,
          0.05,
          1
        ]
      },
      {
        "name": {
          "ko": "불안정 간격",
          "en": "Unstable step"
        },
        "values": [
          3,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: 시간 t · 세로: 상태 x",
      "en": "Horizontal: time t · Vertical: state x"
    },
    "legend": [
      {
        "ko": "정확한 지수해",
        "en": "Exact exponential solution"
      },
      {
        "ko": "전진 Euler",
        "en": "Forward Euler"
      }
    ],
    "note": {
      "ko": "20개 시간 간격을 계산하므로 Δt를 바꾸면 최종 시간도 바뀝니다. 오차 비교에는 같은 최종 시간을 맞춰야 합니다. 실제 감쇠와 수치 진동·발산은 다릅니다.",
      "en": "There are 20 steps, so changing Δt also changes final time. Compare errors at matching final times. Physical decay differs from numerical oscillation or divergence."
    }
  },
  "mechanics-energy": {
    "controls": [
      {
        "label": {
          "ko": "시간 간격 Δt [s]",
          "en": "Time step Δt [s]"
        },
        "min": 0.01,
        "max": 1,
        "step": "any",
        "initial": 0.1
      },
      {
        "label": {
          "ko": "강성 k [N/m]",
          "en": "Stiffness k [N/m]"
        },
        "min": 0.25,
        "max": 4,
        "step": "any",
        "initial": 1
      },
      {
        "label": {
          "ko": "초기 위치 q₀ [m]",
          "en": "Initial position q₀ [m]"
        },
        "min": 0.1,
        "max": 2,
        "step": "any",
        "initial": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "기준 진동",
          "en": "Reference oscillator"
        },
        "values": [
          0.1,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "작은 간격",
          "en": "Smaller step"
        },
        "values": [
          0.05,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "안정 경계",
          "en": "Stability boundary"
        },
        "values": [
          1,
          4,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: 시간 [s] · 세로: 위치 [m]",
      "en": "Horizontal: time [s] · Vertical: position [m]"
    },
    "legend": [
      {
        "ko": "정확한 진동",
        "en": "Exact oscillator"
      },
      {
        "ko": "symplectic Euler",
        "en": "Symplectic Euler"
      }
    ],
    "note": {
      "ko": "질량 1 kg, 초기 속도 0, 마찰 없는 용수철입니다. 속도 먼저·위치 나중의 symplectic Euler는 에너지를 정확히 보존하지 않습니다. 100회 고정이므로 간격을 바꾸면 최종 시간도 달라집니다.",
      "en": "Mass is 1 kg, initial velocity is zero and friction is absent. Velocity-first symplectic Euler does not preserve exact energy. Fixed 100 steps mean changing step size also changes final time."
    }
  },
  "thermodynamics": {
    "controls": [
      {
        "label": {
          "ko": "에너지 간격 Δ",
          "en": "Energy gap Δ"
        },
        "min": 0.1,
        "max": 4,
        "step": "any",
        "initial": 1
      },
      {
        "label": {
          "ko": "온도 T",
          "en": "Temperature T"
        },
        "min": 0.05,
        "max": 5,
        "step": "any",
        "initial": 1
      },
      {
        "label": {
          "ko": "들뜬 상태 축퇴도 g",
          "en": "Excited degeneracy g"
        },
        "min": 1,
        "max": 5,
        "step": 1,
        "initial": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "두 상태",
          "en": "Two states"
        },
        "values": [
          1,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "높은 온도",
          "en": "Higher temperature"
        },
        "values": [
          1,
          5,
          1
        ]
      },
      {
        "name": {
          "ko": "축퇴 상태",
          "en": "Degenerate states"
        },
        "values": [
          1,
          1,
          3
        ]
      }
    ],
    "axis": {
      "ko": "가로: 온도 T · 세로: 점유 확률 또는 S/kB",
      "en": "Horizontal: temperature T · Vertical: occupation probability or S/kB"
    },
    "legend": [
      {
        "ko": "들뜬 에너지 점유",
        "en": "Excited-level occupancy"
      },
      {
        "ko": "무차원 엔트로피 S/kB",
        "en": "Dimensionless entropy S/kB"
      }
    ],
    "note": {
      "ko": "kB=1인 단위를 사용한 평형 canonical 모형입니다. 들뜬 에너지 Δ에 미시 상태 g개가 있습니다. g>1이면 이진 점유 엔트로피와 전체 미시 엔트로피가 다릅니다.",
      "en": "Canonical equilibrium uses units with kB=1. Energy Δ has g microscopic states. For g>1, binary occupancy entropy differs from full microscopic entropy."
    }
  },
  "stochastic-processes": {
    "controls": [
      {
        "label": {
          "ko": "0→1 확률 a",
          "en": "0→1 probability a"
        },
        "min": 0,
        "max": 1,
        "step": "any",
        "initial": 0.2
      },
      {
        "label": {
          "ko": "1→0 확률 b",
          "en": "1→0 probability b"
        },
        "min": 0,
        "max": 1,
        "step": "any",
        "initial": 0.3
      },
      {
        "label": {
          "ko": "초기 점유 p₀",
          "en": "Initial occupancy p₀"
        },
        "min": 0,
        "max": 1,
        "step": "any",
        "initial": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "혼합",
          "en": "Mixing"
        },
        "values": [
          0.2,
          0.3,
          1
        ]
      },
      {
        "name": {
          "ko": "주기 2",
          "en": "Period two"
        },
        "values": [
          1,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "변화 없음",
          "en": "No transitions"
        },
        "values": [
          0,
          0,
          0.7
        ]
      }
    ],
    "axis": {
      "ko": "가로: 이산 시간 t · 세로: 상태 1의 확률",
      "en": "Horizontal: discrete time t · Vertical: probability of state 1"
    },
    "legend": [
      {
        "ko": "시간별 점유",
        "en": "Occupancy over time"
      },
      {
        "ko": "정상 점유",
        "en": "Stationary occupancy"
      }
    ],
    "note": {
      "ko": "정확한 분포 갱신이며 하나의 무작위 입자 궤적이 아닙니다. 정상분포의 존재와 그 분포로의 수렴은 다릅니다. 전이 확률만으로 물리적 온도나 엔트로피 생산을 정하지 않습니다.",
      "en": "This evolves the exact distribution, not one random particle trajectory. Existence of a stationary distribution differs from convergence to it. Transition probabilities alone do not specify physical temperature or entropy production."
    }
  },
  "inverse-problems": {
    "controls": [
      {
        "label": {
          "ko": "관측 대비 ε",
          "en": "Observation contrast ε"
        },
        "min": 0,
        "max": 1,
        "step": "any",
        "initial": 0.1
      },
      {
        "label": {
          "ko": "두 번째 관측 잡음",
          "en": "Second-observation noise"
        },
        "min": -0.2,
        "max": 0.2,
        "step": "any",
        "initial": 0.1
      },
      {
        "label": {
          "ko": "ridge λ",
          "en": "Ridge λ"
        },
        "min": 0,
        "max": 1,
        "step": "any",
        "initial": 0
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "약한 관측",
          "en": "Weak observation"
        },
        "values": [
          0.1,
          0.1,
          0
        ]
      },
      {
        "name": {
          "ko": "더 강한 관측",
          "en": "Stronger observation"
        },
        "values": [
          1,
          0.1,
          0
        ]
      },
      {
        "name": {
          "ko": "관측 부족",
          "en": "Missing contrast"
        },
        "values": [
          0,
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 두 번째 관측 잡음 · 세로: 추정 θ₁",
      "en": "Horizontal: second-observation noise · Vertical: estimated θ₁"
    },
    "legend": [
      {
        "ko": "잡음에 따른 추정",
        "en": "Estimate across noise"
      },
      {
        "ko": "참 θ₁ 위치",
        "en": "True θ₁ location"
      }
    ],
    "note": {
      "ko": "참 θ=(1,2), X=[[1,1],[ε,−ε]]입니다. ε=λ=0의 해는 최소 노름 선택이며 복원 성공을 뜻하지 않습니다. 실험은 선형 역문제이며 실제 물리 모형의 불일치를 포함하지 않습니다.",
      "en": "Truth is θ=(1,2), X=[[1,1],[ε,−ε]]. At ε=λ=0, minimum norm is a choice, not successful recovery. This linear inverse example excludes real physical model discrepancy."
    }
  },
  "research-methods": {
    "controls": [
      {
        "label": {
          "ko": "평균 효과 Δ",
          "en": "Mean effect Δ"
        },
        "min": -0.5,
        "max": 0.5,
        "step": "any",
        "initial": 0.1
      },
      {
        "label": {
          "ko": "효과 변동 b",
          "en": "Effect variation b"
        },
        "min": 0,
        "max": 0.5,
        "step": "any",
        "initial": 0.1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "작은 표본",
          "en": "Small sample"
        },
        "values": [
          0.1,
          0.1
        ]
      },
      {
        "name": {
          "ko": "큰 개선",
          "en": "Larger improvement"
        },
        "values": [
          0.3,
          0.1
        ]
      },
      {
        "name": {
          "ko": "효과 없음",
          "en": "No mean effect"
        },
        "values": [
          0,
          0.1
        ]
      }
    ],
    "axis": {
      "ko": "가로: 대응 seed 번호 · 세로: 점수 차이",
      "en": "Horizontal: paired seed index · Vertical: score difference"
    },
    "legend": [
      {
        "ko": "대응 차이 4개",
        "en": "Four paired differences"
      },
      {
        "ko": "평균 차이",
        "en": "Mean difference"
      }
    ],
    "note": {
      "ko": "차이는 [Δ,Δ+b,Δ−b,Δ]로 만든 교육용 합성값이며 실제 연구 성과가 아닙니다. df=3의 t 구간은 독립·정규 대응 차이를 가정합니다. 여러 가설·선택적 보고·데이터 누출을 보정하지 않습니다.",
      "en": "Differences [Δ,Δ+b,Δ−b,Δ] are synthetic teaching values, not research results. The df=3 t interval assumes independent normal paired differences. It does not adjust for multiple hypotheses, selective reporting or leakage."
    }
  }
};

export interface LogicLabSpec {controls:FoundationControl[];presets:{name:BookText;values:number[]}[];note:BookText;}
export const logicLabs:Record<string,LogicLabSpec> = {
  "propositions-implications": {
    "controls": [
      {
        "label": {
          "ko": "P (0/1)",
          "en": "P (0/1)"
        },
        "min": 0,
        "max": 1,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "Q (0/1)",
          "en": "Q (0/1)"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "거짓인 함의",
          "en": "False implication"
        },
        "values": [
          1,
          0
        ]
      },
      {
        "name": {
          "ko": "역의 함정",
          "en": "Converse trap"
        },
        "values": [
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "둘 다 참",
          "en": "Both true"
        },
        "values": [
          1,
          1
        ]
      }
    ],
    "note": {
      "ko": "고전 이값 논리를 사용합니다. 함의의 참은 원인의 입증이 아닙니다. 조건을 만족하는 사례가 없는 경우의 참도 구분하세요.",
      "en": "This is classical two-valued logic. A true implication is not causal evidence. Distinguish vacuous truth when no premise-satisfying case exists."
    }
  },
  "quantifiers-counterexamples": {
    "controls": [
      {
        "label": {
          "ko": "정의역 크기 N",
          "en": "Domain size N"
        },
        "min": 0,
        "max": 6,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "술어: 0은 =, 1은 >, 2는 ≥",
          "en": "Predicate: 0 =, 1 >, 2 ≥"
        },
        "min": 0,
        "max": 2,
        "initial": 0,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "입력별 증인",
          "en": "Separate witnesses"
        },
        "values": [
          3,
          0
        ]
      },
      {
        "name": {
          "ko": "유한 최대값",
          "en": "Finite maximum"
        },
        "values": [
          3,
          1
        ]
      },
      {
        "name": {
          "ko": "빈 정의역",
          "en": "Empty domain"
        },
        "values": [
          0,
          0
        ]
      }
    ],
    "note": {
      "ko": "D={0,…,N−1}의 유한 모형을 전수 검사합니다. 이 결과를 정수·실수 전체로 확대하지 마세요. 술어 코드는 R(x,y)의 y와 x를 비교합니다.",
      "en": "This exhaustively checks the finite model D={0,…,N−1}. It does not establish the result over all integers or reals. Predicate codes compare y with x in R(x,y)."
    }
  },
  "sets-functions": {
    "controls": [
      {
        "label": {
          "ko": "입력 개수 m",
          "en": "Domain size m"
        },
        "min": 1,
        "max": 5,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "출력 공간 크기 n",
          "en": "Codomain size n"
        },
        "min": 1,
        "max": 5,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "배율 a",
          "en": "Multiplier a"
        },
        "min": 0,
        "max": 4,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "이동 b",
          "en": "Offset b"
        },
        "min": 0,
        "max": 4,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "B의 하한 t",
          "en": "Threshold t for B"
        },
        "min": 0,
        "max": 5,
        "initial": 1,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "전단사",
          "en": "Bijection"
        },
        "values": [
          3,
          3,
          1,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "여러 입력·한 출력",
          "en": "Many-to-one"
        },
        "values": [
          4,
          2,
          1,
          0,
          2
        ]
      },
      {
        "name": {
          "ko": "남는 출력",
          "en": "Unused outputs"
        },
        "values": [
          2,
          4,
          1,
          0,
          1
        ]
      }
    ],
    "note": {
      "ko": "D={0,…,m−1}, C={0,…,n−1}, A는 짝수 입력, B={x∈D:x≥t}입니다. 역상은 역함수와 다릅니다.",
      "en": "D={0,…,m−1}, C={0,…,n−1}, A is the even-input set and B={x∈D:x≥t}. A preimage is not an inverse function."
    }
  },
  "relations-orders": {
    "controls": [
      {
        "label": {
          "ko": "원소 수 N",
          "en": "Number of elements N"
        },
        "min": 1,
        "max": 5,
        "initial": 4,
        "step": 1
      },
      {
        "label": {
          "ko": "관계: 0 합동, 1 약수, 2 <, 3 ≤",
          "en": "Relation: 0 congruence, 1 divides, 2 <, 3 ≤"
        },
        "min": 0,
        "max": 3,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "합동의 법 k",
          "en": "Congruence modulus k"
        },
        "min": 2,
        "max": 3,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "합동 관계",
          "en": "Congruence relation"
        },
        "values": [
          4,
          0,
          2
        ]
      },
      {
        "name": {
          "ko": "약수 순서",
          "en": "Divisibility order"
        },
        "values": [
          4,
          1,
          2
        ]
      },
      {
        "name": {
          "ko": "엄격한 순서",
          "en": "Strict order"
        },
        "values": [
          4,
          2,
          2
        ]
      }
    ],
    "note": {
      "ko": "D={1,…,N}에서 조건을 전수 검사합니다. 반대칭은 비대칭과 다릅니다. <와 ≤의 정의를 섞지 마세요.",
      "en": "Properties are exhaustively checked on D={1,…,N}. Antisymmetry differs from asymmetry. Keep the definitions of < and ≤ separate."
    }
  },
  "proof-strategies": {
    "controls": [
      {
        "label": {
          "ko": "주장: 0 참정리, 1 강한 결론, 2 틀린 결론",
          "en": "Claim: 0 theorem, 1 stronger conclusion, 2 wrong conclusion"
        },
        "min": 0,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "정수 검사 한계 B",
          "en": "Integer check bound B"
        },
        "min": 1,
        "max": 8,
        "initial": 3,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "참 정리",
          "en": "Valid theorem"
        },
        "values": [
          0,
          3
        ]
      },
      {
        "name": {
          "ko": "더 강한 주장",
          "en": "Stronger claim"
        },
        "values": [
          1,
          3
        ]
      },
      {
        "name": {
          "ko": "틀린 결론",
          "en": "Wrong conclusion"
        },
        "values": [
          2,
          3
        ]
      }
    ],
    "note": {
      "ko": "표는 −B부터 B까지 정수만 확인합니다. 일반 증명은 참 정리 예제에 별도로 제공합니다. 반례 없음과 증명 완료를 혼동하지 마세요.",
      "en": "The table checks only integers from −B to B. A general proof is supplied separately for the valid theorem. Absence of a counterexample is not proof completion."
    }
  },
  "mathematical-induction": {
    "controls": [
      {
        "label": {
          "ko": "주장: 0 합, 1 거듭제곱, 2 소수 패턴",
          "en": "Claim: 0 sum, 1 powers, 2 prime pattern"
        },
        "min": 0,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "최대 검사 n",
          "en": "Largest checked n"
        },
        "min": 0,
        "max": 45,
        "initial": 5,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "합 공식",
          "en": "Sum formula"
        },
        "values": [
          0,
          5
        ]
      },
      {
        "name": {
          "ko": "끊어진 단계",
          "en": "Broken step"
        },
        "values": [
          1,
          4
        ]
      },
      {
        "name": {
          "ko": "늦게 나오는 반례",
          "en": "Late counterexample"
        },
        "values": [
          2,
          40
        ]
      }
    ],
    "note": {
      "ko": "합과 소수 패턴은 n=0부터, 거듭제곱 주장은 n=1부터 검사합니다. 유한 표의 참 결과를 무한 명제의 증명으로 표시하지 않습니다.",
      "en": "Sum and prime-pattern checks start at n=0; the power claim starts at n=1. Passing finite checks is not displayed as proof of an infinite statement."
    }
  }
};

export const elementaryLabs:Record<string,FoundationLabSpec> = {
  "numbers-ratios-units": {
    "controls": [
      {
        "label": {
          "ko": "분자 a",
          "en": "Numerator a"
        },
        "min": -6,
        "max": 6,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "분모 b",
          "en": "Denominator b"
        },
        "min": 0,
        "max": 6,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "길이 L [cm]",
          "en": "Length L [cm]"
        },
        "min": 0,
        "max": 200,
        "initial": 150,
        "step": "any"
      },
      {
        "label": {
          "ko": "시간 t [s]",
          "en": "Time t [s]"
        },
        "min": 0,
        "max": 10,
        "initial": 2,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "기준 비율",
          "en": "Reference ratio"
        },
        "values": [
          1,
          3,
          150,
          2
        ]
      },
      {
        "name": {
          "ko": "같은 분수",
          "en": "Equivalent fraction"
        },
        "values": [
          2,
          6,
          150,
          2
        ]
      },
      {
        "name": {
          "ko": "정의되지 않는 나눗셈",
          "en": "Undefined division"
        },
        "values": [
          1,
          0,
          150,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 무차원 비율 · 세로: 위치 표시용",
      "en": "Horizontal: dimensionless ratio · Vertical: marker only"
    },
    "legend": [
      {
        "ko": "0에서 비율까지",
        "en": "From zero to the ratio"
      },
      {
        "ko": "비율의 위치",
        "en": "Ratio location"
      }
    ],
    "note": {
      "ko": "그림은 무차원 분수만 표시합니다. 길이와 속도는 표에서 단위를 붙여 계산합니다. 0인 분모·시간에 값을 임의로 넣지 않습니다.",
      "en": "The plot shows only the dimensionless fraction. Length and speed are computed with units in the readout. Zero denominator or time is not assigned an invented value."
    }
  },
  "real-complex-numbers": {
    "controls": [
      {
        "label": {
          "ko": "실수부 a",
          "en": "Real part a"
        },
        "min": -3,
        "max": 3,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "허수부 b",
          "en": "Imaginary part b"
        },
        "min": -3,
        "max": 3,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "회전 θ [°]",
          "en": "Rotation θ [°]"
        },
        "min": -180,
        "max": 180,
        "initial": 90,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "90도 회전",
          "en": "90-degree rotation"
        },
        "values": [
          1,
          1,
          90
        ]
      },
      {
        "name": {
          "ko": "180도 회전",
          "en": "180-degree rotation"
        },
        "values": [
          0,
          1,
          180
        ]
      },
      {
        "name": {
          "ko": "영복소수",
          "en": "Zero complex number"
        },
        "values": [
          0,
          0,
          90
        ]
      }
    ],
    "axis": {
      "ko": "가로: 실수부 · 세로: 허수부 · 동일 단위 길이",
      "en": "Horizontal: real part · Vertical: imaginary part · Equal unit length"
    },
    "legend": [
      {
        "ko": "z",
        "en": "z"
      },
      {
        "ko": "회전한 w",
        "en": "Rotated w"
      }
    ],
    "note": {
      "ko": "점은 복소수를 실수 좌표 두 개로 표현합니다. 회전은 exp(iθ)를 곱합니다. 영복소수의 편각은 정의하지 않습니다. 표시 편각은 atan2의 주값으로 분기 경계에서 점프할 수 있습니다.",
      "en": "Points represent complex numbers using two real coordinates. Rotation multiplies by exp(iθ). Zero has no defined argument. Displayed principal arguments use atan2 and may jump at the branch cut."
    }
  },
  "equations-inequalities": {
    "controls": [
      {
        "label": {
          "ko": "모드: 0 일차식, 1 이차부등식",
          "en": "Mode: 0 linear, 1 quadratic inequality"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "계수 a",
          "en": "Coefficient a"
        },
        "min": -4,
        "max": 4,
        "initial": 2,
        "step": "any"
      },
      {
        "label": {
          "ko": "상수 b",
          "en": "Constant b"
        },
        "min": -4,
        "max": 4,
        "initial": 4,
        "step": "any"
      },
      {
        "label": {
          "ko": "검사점 x₀",
          "en": "Probe x₀"
        },
        "min": -4,
        "max": 4,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "일차 방정식",
          "en": "Linear equation"
        },
        "values": [
          0,
          2,
          4,
          1
        ]
      },
      {
        "name": {
          "ko": "이차 부등식",
          "en": "Quadratic inequality"
        },
        "values": [
          1,
          -3,
          2,
          1.5
        ]
      },
      {
        "name": {
          "ko": "항등식",
          "en": "Identity"
        },
        "values": [
          0,
          0,
          0,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 세로: 각 변 또는 이차식 값",
      "en": "Horizontal: x · Vertical: equation sides or quadratic value"
    },
    "legend": [
      {
        "ko": "ax 또는 이차식",
        "en": "ax or the quadratic"
      },
      {
        "ko": "b 또는 0",
        "en": "b or zero"
      }
    ],
    "note": {
      "ko": "일차식은 ax=b, 이차 모드는 x²+ax+b≤0입니다. 이차 모드의 해 구간은 닫힌 구간입니다. 실수 해가 없으면 ∅, 모든 실수가 해이면 해 개수 ∞로 구분합니다.",
      "en": "Linear mode is ax=b; quadratic mode is x²+ax+b≤0. The quadratic solution interval is closed. No real solutions are shown as ∅; all-real solutions have count ∞."
    }
  },
  "polynomial-rational-functions": {
    "controls": [
      {
        "label": {
          "ko": "모드: 0 구멍, 1 극점, 2 다항식",
          "en": "Mode: 0 hole, 1 pole, 2 polynomial"
        },
        "min": 0,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "매개변수 c",
          "en": "Parameter c"
        },
        "min": -2,
        "max": 2,
        "initial": 0,
        "step": "any"
      },
      {
        "label": {
          "ko": "검사점 x₀",
          "en": "Probe x₀"
        },
        "min": -3,
        "max": 3,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "제거된 한 점",
          "en": "A removed point"
        },
        "values": [
          0,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "수직 점근선",
          "en": "Vertical asymptote"
        },
        "values": [
          1,
          0,
          0.1
        ]
      },
      {
        "name": {
          "ko": "다항함수",
          "en": "Polynomial"
        },
        "values": [
          2,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 세로: f(x) · 제외점에서 선을 잇지 않음",
      "en": "Horizontal: x · Vertical: f(x) · No joining across excluded points"
    },
    "legend": [
      {
        "ko": "함수의 가지",
        "en": "Function branch"
      },
      {
        "ko": "다른 가지",
        "en": "Other branch"
      }
    ],
    "note": {
      "ko": "빈 원은 정의되지 않는 점 (1,2)입니다. 극점 양쪽은 간격 0.08, 구멍 양쪽은 0.02를 두고 표본화합니다. 유한 그림은 극한값이나 무한대를 대신하지 않습니다.",
      "en": "An open circle marks undefined point (1,2). Sampling stops 0.08 from a pole and 0.02 from a hole. A finite plot does not replace a limit or infinity."
    }
  },
  "exponentials-logarithms": {
    "controls": [
      {
        "label": {
          "ko": "밑 a",
          "en": "Base a"
        },
        "min": 0.25,
        "max": 4,
        "initial": 2,
        "step": "any"
      },
      {
        "label": {
          "ko": "지수 검사점 x₀",
          "en": "Exponent probe x₀"
        },
        "min": -2,
        "max": 3,
        "initial": 2,
        "step": "any"
      },
      {
        "label": {
          "ko": "로그 입력 y₀",
          "en": "Log input y₀"
        },
        "min": 0,
        "max": 4,
        "initial": 4,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "밑 2",
          "en": "Base two"
        },
        "values": [
          2,
          2,
          4
        ]
      },
      {
        "name": {
          "ko": "1보다 작은 밑",
          "en": "Base below one"
        },
        "values": [
          0.5,
          2,
          0.25
        ]
      },
      {
        "name": {
          "ko": "밑 1",
          "en": "Base one"
        },
        "values": [
          1,
          2,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 세로: aˣ · 로그는 수치 표에서 계산",
      "en": "Horizontal: x · Vertical: aˣ · Logarithms are evaluated in the readout"
    },
    "legend": [
      {
        "ko": "지수함수",
        "en": "Exponential function"
      }
    ],
    "note": {
      "ko": "그림은 지수함수만 표시합니다. 실수 로그는 a>0, a≠1, y>0에서 정의합니다. 밑 1의 상수 함수에는 역함수가 없습니다.",
      "en": "The plot shows only the exponential. Real logarithms require a>0, a≠1 and y>0. The constant exponential at base one has no inverse."
    }
  },
  "trigonometry-coordinates": {
    "controls": [
      {
        "label": {
          "ko": "각 θ [°]",
          "en": "Angle θ [°]"
        },
        "min": -360,
        "max": 360,
        "initial": 60,
        "step": "any"
      },
      {
        "label": {
          "ko": "반지름 r",
          "en": "Radius r"
        },
        "min": 0,
        "max": 3,
        "initial": 2,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "기준 좌표",
          "en": "Reference coordinates"
        },
        "values": [
          60,
          2
        ]
      },
      {
        "name": {
          "ko": "같은 방향의 다른 각",
          "en": "Another angle for the same direction"
        },
        "values": [
          -300,
          2
        ]
      },
      {
        "name": {
          "ko": "원점",
          "en": "Origin"
        },
        "values": [
          60,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 세로: y · 동일 단위 길이",
      "en": "Horizontal: x · Vertical: y · Equal unit length"
    },
    "legend": [
      {
        "ko": "단위원",
        "en": "Unit circle"
      },
      {
        "ko": "위치 벡터 p",
        "en": "Position vector p"
      }
    ],
    "note": {
      "ko": "좌표는 무차원입니다. 단위원은 기준 길이 1이고 q=(1,0)입니다. 입력은 도 단위, 계산은 라디안입니다. r=0이면 방향각은 정의하지 않습니다.",
      "en": "Coordinates are dimensionless. The reference circle has radius one and q=(1,0). Inputs use degrees; calculations use radians. At r=0 the direction angle is undefined."
    }
  },
  "composition-inverse-functions": {
    "controls": [
      {
        "label": {
          "ko": "배율 a",
          "en": "Scale a"
        },
        "min": -3,
        "max": 3,
        "initial": 2,
        "step": "any"
      },
      {
        "label": {
          "ko": "이동 b",
          "en": "Offset b"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "입력 x₀",
          "en": "Input x₀"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "기준 합성",
          "en": "Reference composition"
        },
        "values": [
          2,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "분기 밖 입력",
          "en": "Input outside the selected branch"
        },
        "values": [
          2,
          0,
          -1
        ]
      },
      {
        "name": {
          "ko": "상수 안쪽 함수",
          "en": "Constant inner map"
        },
        "values": [
          0,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 세로: h(x)=(ax+b)²",
      "en": "Horizontal: x · Vertical: h(x)=(ax+b)²"
    },
    "legend": [
      {
        "ko": "합성함수 h",
        "en": "Composite h"
      }
    ],
    "note": {
      "ko": "그림은 합성함수만 그립니다. 제한 역함수는 g(x)≥0인 입력 가지를 사용합니다. 전체 역상과 역함수를 구분하고, a=0이면 역함수를 표시하지 않습니다.",
      "en": "The plot shows only the composition. The restricted inverse uses the branch g(x)≥0. Distinguish the full preimage from an inverse function; a=0 has no inverse."
    }
  },
  "sequences-series": {
    "controls": [
      {
        "label": {
          "ko": "공비 r",
          "en": "Ratio r"
        },
        "min": -1.5,
        "max": 1.5,
        "initial": 0.5,
        "step": "any"
      },
      {
        "label": {
          "ko": "첫 항 a",
          "en": "First term a"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "항 개수 N",
          "en": "Number of terms N"
        },
        "min": 0,
        "max": 30,
        "initial": 10,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "수렴하는 기하급수",
          "en": "Convergent geometric series"
        },
        "values": [
          0.5,
          1,
          10
        ]
      },
      {
        "name": {
          "ko": "진동하는 부분합",
          "en": "Oscillating partial sums"
        },
        "values": [
          -1,
          1,
          10
        ]
      },
      {
        "name": {
          "ko": "증가하는 부분합",
          "en": "Growing partial sums"
        },
        "values": [
          1.1,
          1,
          10
        ]
      }
    ],
    "axis": {
      "ko": "가로: 더한 항 수 k · 세로: 부분합 Sₖ",
      "en": "Horizontal: number of added terms k · Vertical: partial sum Sₖ"
    },
    "legend": [
      {
        "ko": "유한 부분합",
        "en": "Finite partial sums"
      },
      {
        "ko": "존재하는 무한 합",
        "en": "Infinite sum when it exists"
      }
    ],
    "note": {
      "ko": "기하급수만 계산합니다. a≠0일 때 |r|<1에서 수렴하며, a=0이면 모든 r에서 영급수입니다. N=0은 빈 합 0입니다. 유한 부분합을 무한 합으로 표시하지 않습니다.",
      "en": "Only geometric series are computed. For a≠0 convergence requires |r|<1; a=0 gives the zero series for every r. N=0 is the empty sum zero. Finite partial sums are not reported as infinite sums."
    }
  },
  "asymptotic-growth": {
    "controls": [
      {
        "label": {
          "ko": "기울기 a",
          "en": "Slope a"
        },
        "min": 1,
        "max": 5,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "상수 b",
          "en": "Constant b"
        },
        "min": 0,
        "max": 20,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "후보 상한 C",
          "en": "Candidate bound C"
        },
        "min": 1,
        "max": 5,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "후보 시작 n₀",
          "en": "Candidate start n₀"
        },
        "min": 1,
        "max": 64,
        "initial": 1,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "이른 시작",
          "en": "Too early a start"
        },
        "values": [
          2,
          3,
          3,
          1
        ]
      },
      {
        "name": {
          "ko": "유효한 증명서",
          "en": "Valid certificate"
        },
        "values": [
          2,
          3,
          3,
          3
        ]
      },
      {
        "name": {
          "ko": "작은 상한 상수",
          "en": "Insufficient bound constant"
        },
        "values": [
          2,
          3,
          2,
          3
        ]
      }
    ],
    "axis": {
      "ko": "가로: 입력 크기 n · 세로: f(n)/n 또는 C",
      "en": "Horizontal: input size n · Vertical: f(n)/n or C"
    },
    "legend": [
      {
        "ko": "비율 f(n)/n",
        "en": "Ratio f(n)/n"
      },
      {
        "ko": "후보 상수 C",
        "en": "Candidate constant C"
      }
    ],
    "note": {
      "ko": "합성 비용 함수 an+b를 사용합니다. 실제 실행 시간 측정이 아닙니다. 후보 C,n₀가 실패해도 다른 상수로 Big-O가 성립할 수 있습니다.",
      "en": "This uses the synthetic cost function an+b, not measured runtime. Failure of one candidate C,n₀ does not disprove Big-O with different constants."
    }
  }
};

export const discreteLabs:Record<string,FoundationLabSpec> = {
  "counting-permutations": {
    "controls": [
      {
        "label": {
          "ko": "종류 수 n",
          "en": "Number of types n"
        },
        "min": 0,
        "max": 8,
        "initial": 5,
        "step": 1
      },
      {
        "label": {
          "ko": "선택 수 k",
          "en": "Selection size k"
        },
        "min": 0,
        "max": 8,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "방식 0~3",
          "en": "Mode 0–3"
        },
        "min": 0,
        "max": 3,
        "initial": 0,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "순서 있는 선택",
          "en": "Ordered selection"
        },
        "values": [
          5,
          3,
          0
        ]
      },
      {
        "name": {
          "ko": "순서 없는 선택",
          "en": "Unordered selection"
        },
        "values": [
          5,
          3,
          1
        ]
      },
      {
        "name": {
          "ko": "빈 선택",
          "en": "Empty selection"
        },
        "values": [
          0,
          0,
          3
        ]
      }
    ],
    "axis": {
      "ko": "가로: 선택 수 k · 세로: 정확한 경우의 수",
      "en": "Horizontal: selection size k · Vertical: exact count"
    },
    "legend": [],
    "note": {
      "ko": "0 순서·무중복, 1 무순서·무중복, 2 순서·중복, 3 무순서·중복입니다. 공집합에서 0개 선택은 한 가지입니다.",
      "en": "0 ordered/no replacement; 1 unordered/no replacement; 2 ordered/replacement; 3 unordered/replacement. Selecting nothing has one outcome."
    }
  },
  "binomial-inclusion-exclusion": {
    "controls": [
      {
        "label": {
          "ko": "집합 A 비트마스크",
          "en": "Set A bitmask"
        },
        "min": 0,
        "max": 63,
        "initial": 7,
        "step": 1
      },
      {
        "label": {
          "ko": "집합 B 비트마스크",
          "en": "Set B bitmask"
        },
        "min": 0,
        "max": 63,
        "initial": 28,
        "step": 1
      },
      {
        "label": {
          "ko": "부분집합 크기 k",
          "en": "Subset size k"
        },
        "min": 0,
        "max": 6,
        "initial": 3,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "한 원소 중첩",
          "en": "One shared element"
        },
        "values": [
          7,
          28,
          3
        ]
      },
      {
        "name": {
          "ko": "같은 집합",
          "en": "Identical sets"
        },
        "values": [
          7,
          7,
          0
        ]
      },
      {
        "name": {
          "ko": "전체와 공집합",
          "en": "Full and empty sets"
        },
        "values": [
          63,
          0,
          6
        ]
      }
    ],
    "axis": {
      "ko": "가로: k=0,…,6 · 세로: C(6,k) · 아래 표: 원소별 소속",
      "en": "Horizontal: k=0,…,6 · Vertical: C(6,k) · Table: element membership"
    },
    "legend": [],
    "note": {
      "ko": "전체 집합은 {0,…,5}입니다. 비트 i가 1이면 원소 i를 포함합니다.",
      "en": "The universe is {0,…,5}; bit i includes element i."
    }
  },
  "recurrences-generating-functions": {
    "controls": [
      {
        "label": {
          "ko": "계수 p",
          "en": "Coefficient p"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "계수 q",
          "en": "Coefficient q"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "마지막 지수 N",
          "en": "Last exponent N"
        },
        "min": 0,
        "max": 12,
        "initial": 8,
        "step": 1
      },
      {
        "label": {
          "ko": "평가점 x",
          "en": "Evaluation point x"
        },
        "min": -0.8,
        "max": 0.8,
        "initial": 0.5,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "수렴 범위 안",
          "en": "Inside convergence radius"
        },
        "values": [
          1,
          1,
          8,
          0.5
        ]
      },
      {
        "name": {
          "ko": "수렴 범위 밖",
          "en": "Outside convergence radius"
        },
        "values": [
          1,
          1,
          8,
          0.8
        ]
      },
      {
        "name": {
          "ko": "유한 다항식",
          "en": "Finite polynomial"
        },
        "values": [
          0,
          0,
          8,
          0.5
        ]
      }
    ],
    "axis": {
      "ko": "가로: 정수 n · 세로: 계수 aₙ · 표: aₙxⁿ",
      "en": "Horizontal: integer n · Vertical: coefficient aₙ · Table: aₙxⁿ"
    },
    "legend": [],
    "note": {
      "ko": "a₀=0,a₁=1입니다. 유리식의 값이 존재해도 무한급수가 그 점에서 수렴하는 것은 아닙니다. p=q=0이면 생성함수는 다항식 x이며 반지름은 제한이 없습니다.",
      "en": "a₀=0,a₁=1. A rational expression can exist where its power series diverges. For p=q=0 the generating function is the polynomial x, with unrestricted radius."
    }
  },
  "graphs-trees-dags": {
    "controls": [
      {
        "label": {
          "ko": "방향 간선 마스크",
          "en": "Directed edge mask"
        },
        "min": 0,
        "max": 63,
        "initial": 9,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "사슬 DAG",
          "en": "Chain DAG"
        },
        "values": [
          9
        ]
      },
      {
        "name": {
          "ko": "방향 순환",
          "en": "Directed cycle"
        },
        "values": [
          25
        ]
      },
      {
        "name": {
          "ko": "왕복 간선 트리",
          "en": "Tree with reciprocal edges"
        },
        "values": [
          13
        ]
      }
    ],
    "axis": {
      "ko": "화살표: 방향 간선 · 표: 비트별 간선 · 위상 순서는 수치 표에 표시",
      "en": "Arrows: directed edges · Table: edge bits · Topological order in readout"
    },
    "legend": [],
    "note": {
      "ko": "세 정점, 자기 간선 없는 단순 방향 그래프입니다. bit 0~5는 01,02,10,12,20,21입니다. 트리는 방향과 중복을 지운 기저 그래프 기준입니다.",
      "en": "Three vertices; a simple directed graph without self-loops. Bits 0–5 encode 01,02,10,12,20,21. Tree refers to the underlying simple undirected graph."
    }
  },
  "paths-cycles-connectivity": {
    "controls": [
      {
        "label": {
          "ko": "방향 간선 마스크",
          "en": "Directed edge mask"
        },
        "min": 0,
        "max": 63,
        "initial": 9,
        "step": 1
      },
      {
        "label": {
          "ko": "출발 정점 s",
          "en": "Source s"
        },
        "min": 0,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "도착 정점 t",
          "en": "Target t"
        },
        "min": 0,
        "max": 2,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "두 단계 경로",
          "en": "Two-edge path"
        },
        "values": [
          9,
          0,
          2
        ]
      },
      {
        "name": {
          "ko": "강연결 순환",
          "en": "Strongly connected cycle"
        },
        "values": [
          25,
          0,
          0
        ]
      },
      {
        "name": {
          "ko": "도달 불가",
          "en": "Unreachable target"
        },
        "values": [
          0,
          0,
          2
        ]
      }
    ],
    "axis": {
      "ko": "주황: 선택한 최단 경로 · 표: 길이 0 포함 도달성",
      "en": "Orange: selected shortest path · Table: reachability including length zero"
    },
    "legend": [],
    "note": {
      "ko": "거리는 가중치 없는 방향 간선 개수입니다. A²는 길이 2 보행을 세며 단순 경로만 세는 것이 아닙니다.",
      "en": "Distance counts unweighted directed edges. A² counts length-two walks, which may repeat vertices."
    }
  },
  "boolean-algebra": {
    "controls": [
      {
        "label": {
          "ko": "진리표 마스크",
          "en": "Truth-table mask"
        },
        "min": 0,
        "max": 15,
        "initial": 6,
        "step": 1
      },
      {
        "label": {
          "ko": "입력 x",
          "en": "Input x"
        },
        "min": 0,
        "max": 1,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "입력 y",
          "en": "Input y"
        },
        "min": 0,
        "max": 1,
        "initial": 1,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "XOR",
          "en": "XOR"
        },
        "values": [
          6,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "NAND",
          "en": "NAND"
        },
        "values": [
          7,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "상수 거짓",
          "en": "Constant false"
        },
        "values": [
          0,
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 행 번호 2x+y · 세로: 출력 F · 표: 입력과 출력",
      "en": "Horizontal: row index 2x+y · Vertical: output F · Table: inputs and output"
    },
    "legend": [],
    "note": {
      "ko": "ANF에서 덧셈은 XOR, 곱셈은 AND입니다. 논리 OR나 실수 덧셈과 혼동하지 마세요.",
      "en": "In ANF, addition is XOR and multiplication is AND; neither ordinary real addition nor logical OR is the same operation."
    }
  },
  "automata-formal-languages": {
    "controls": [
      {
        "label": {
          "ko": "입력 정수 w",
          "en": "Input integer w"
        },
        "min": 0,
        "max": 255,
        "initial": 5,
        "step": 1
      },
      {
        "label": {
          "ko": "문자열 길이 N",
          "en": "Word length N"
        },
        "min": 0,
        "max": 8,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "언어 방식 0/1",
          "en": "Language mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "짝수 개의 1",
          "en": "Even number of ones"
        },
        "values": [
          5,
          3,
          0
        ]
      },
      {
        "name": {
          "ko": "01로 끝남",
          "en": "Ends with 01"
        },
        "values": [
          1,
          2,
          1
        ]
      },
      {
        "name": {
          "ko": "빈 문자열",
          "en": "Empty word"
        },
        "values": [
          0,
          0,
          1
        ]
      }
    ],
    "axis": {
      "ko": "화살표 숫자: 소비한 기호 · 이중 원: 수용 상태 · 주황 원: 최종 상태",
      "en": "Arrow label: consumed symbol · Double circle: accepting state · Orange circle: final state"
    },
    "legend": [],
    "note": {
      "ko": "방식 0은 1의 개수가 짝수인 언어, 1은 01로 끝나는 언어입니다. 입력은 정수의 하위 N비트를 앞에 0을 채워 왼쪽부터 읽습니다.",
      "en": "Mode zero accepts even parity of ones; mode one accepts words ending in 01. Read the lowest N bits left to right, padding with leading zeros."
    }
  },
  "posets-lattices": {
    "controls": [
      {
        "label": {
          "ko": "집합 A 마스크",
          "en": "Set A mask"
        },
        "min": 0,
        "max": 7,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "집합 B 마스크",
          "en": "Set B mask"
        },
        "min": 0,
        "max": 7,
        "initial": 5,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "비교 불가",
          "en": "Incomparable sets"
        },
        "values": [
          3,
          5
        ]
      },
      {
        "name": {
          "ko": "포함 관계",
          "en": "Nested sets"
        },
        "values": [
          1,
          3
        ]
      },
      {
        "name": {
          "ko": "최소와 최대",
          "en": "Bottom and top"
        },
        "values": [
          0,
          7
        ]
      }
    ],
    "axis": {
      "ko": "Hasse 그림: 아래 집합에서 위 집합으로 원소 하나 추가 · 반사·추이 간선 생략",
      "en": "Hasse diagram: add one element upward · Reflexive and transitive edges omitted"
    },
    "legend": [],
    "note": {
      "ko": "전체 격자는 {0,1,2}의 8개 부분집합입니다. 마스크의 비트 i는 원소 i입니다. 위쪽이 포함 관계의 큰 쪽입니다.",
      "en": "The lattice consists of all eight subsets of {0,1,2}; bit i encodes element i. Higher means larger under inclusion."
    }
  },
  "finite-state-transitions": {
    "controls": [
      {
        "label": {
          "ko": "전이 코드 c",
          "en": "Transition code c"
        },
        "min": 0,
        "max": 26,
        "initial": 7,
        "step": 1
      },
      {
        "label": {
          "ko": "초기 상태 s₀",
          "en": "Initial state s₀"
        },
        "min": 0,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "전이 횟수 N",
          "en": "Number of transitions N"
        },
        "min": 0,
        "max": 12,
        "initial": 5,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "3-순환",
          "en": "Three-cycle"
        },
        "values": [
          7,
          0,
          5
        ]
      },
      {
        "name": {
          "ko": "과도 뒤 2-순환",
          "en": "Transient then two-cycle"
        },
        "values": [
          17,
          0,
          5
        ]
      },
      {
        "name": {
          "ko": "흡수 고정점",
          "en": "Absorbing fixed point"
        },
        "values": [
          26,
          0,
          5
        ]
      }
    ],
    "axis": {
      "ko": "각 정점의 화살표 하나: 결정적 다음 상태 · 주황: N회 전이 뒤 상태",
      "en": "One outgoing arrow per state: deterministic next state · Orange: state after N transitions"
    },
    "legend": [],
    "note": {
      "ko": "c=T(0)+3T(1)+9T(2)입니다. 상태 번호 사이의 거리는 정의하지 않습니다. 확률 전이나 연속 물리 시간이 아닌 결정적 이산계입니다.",
      "en": "c=T(0)+3T(1)+9T(2). State labels have no metric. This is a deterministic discrete system, not a probabilistic transition or continuous physical time."
    }
  }
};

export const matrixStructureLabs:Record<string,FoundationLabSpec> = {
  "vector-spaces-rank": {
    "controls": [
      {
        "label": {
          "ko": "원소 a",
          "en": "Entry a"
        },
        "min": -4,
        "max": 4,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "원소 b",
          "en": "Entry b"
        },
        "min": -4,
        "max": 4,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "원소 c",
          "en": "Entry c"
        },
        "min": -4,
        "max": 4,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "원소 d",
          "en": "Entry d"
        },
        "min": -4,
        "max": 4,
        "initial": 4,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "랭크 1",
          "en": "Rank one"
        },
        "values": [
          1,
          2,
          2,
          4
        ]
      },
      {
        "name": {
          "ko": "랭크 2",
          "en": "Rank two"
        },
        "values": [
          1,
          0,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "영 사상",
          "en": "Zero map"
        },
        "values": [
          0,
          0,
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로·세로: 출력 공간 좌표 · 파랑: 첫 열 · 주황: 둘째 열",
      "en": "Axes: output coordinates · Blue: first column · Orange: second column"
    },
    "legend": [
      {
        "ko": "첫 열",
        "en": "First column"
      },
      {
        "ko": "둘째 열",
        "en": "Second column"
      }
    ],
    "note": {
      "ko": "작은 정수 행렬의 랭크는 정확한 행렬식과 영 여부로 판정합니다. 랭크 0의 영공간은 R² 전체, 랭크 2의 영공간은 {0}입니다.",
      "en": "Rank uses exact determinant and zero predicates for small integer matrices. Rank zero has nullspace R²; rank two has nullspace {0}."
    }
  },
  "determinants-traces": {
    "controls": [
      {
        "label": {
          "ko": "원소 a",
          "en": "Entry a"
        },
        "min": -4,
        "max": 4,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "원소 b",
          "en": "Entry b"
        },
        "min": -4,
        "max": 4,
        "initial": -1,
        "step": 1
      },
      {
        "label": {
          "ko": "원소 c",
          "en": "Entry c"
        },
        "min": -4,
        "max": 4,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "원소 d",
          "en": "Entry d"
        },
        "min": -4,
        "max": 4,
        "initial": 0,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "90도 회전",
          "en": "Quarter turn"
        },
        "values": [
          0,
          -1,
          1,
          0
        ]
      },
      {
        "name": {
          "ko": "반사",
          "en": "Reflection"
        },
        "values": [
          1,
          0,
          0,
          -1
        ]
      },
      {
        "name": {
          "ko": "면적 붕괴",
          "en": "Area collapse"
        },
        "values": [
          1,
          2,
          2,
          4
        ]
      }
    ],
    "axis": {
      "ko": "같은 단위 좌표 · 파랑: 열벡터의 평행사변형 · 주황: 단위 정사각형",
      "en": "Equal-unit coordinates · Blue: column parallelogram · Orange: unit square"
    },
    "legend": [
      {
        "ko": "변환된 정사각형",
        "en": "Transformed square"
      },
      {
        "ko": "단위 정사각형",
        "en": "Unit square"
      }
    ],
    "note": {
      "ko": "실수 고유값이 없을 수도 있습니다. 복소 켤레쌍은 (공통 실수부, 허수부 절댓값)으로 표시합니다.",
      "en": "Real eigenvalues need not exist. A complex conjugate pair is represented by its common real part and absolute imaginary part."
    }
  },
  "positive-semidefinite-matrices": {
    "controls": [
      {
        "label": {
          "ko": "대각 a",
          "en": "Diagonal a"
        },
        "min": -3,
        "max": 3,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "대칭 원소 b",
          "en": "Off-diagonal b"
        },
        "min": -3,
        "max": 3,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "대각 c",
          "en": "Diagonal c"
        },
        "min": -3,
        "max": 3,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "양의 정부호",
          "en": "Positive definite"
        },
        "values": [
          2,
          1,
          2
        ]
      },
      {
        "name": {
          "ko": "준정부호·특이",
          "en": "Semidefinite and singular"
        },
        "values": [
          1,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "양의 대각·부정부호",
          "en": "Positive diagonal, indefinite"
        },
        "values": [
          1,
          2,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: 단위 벡터 각도 θ [도] · 파랑: vᵀAv · 주황: 0",
      "en": "Horizontal: unit-vector angle θ [degrees] · Blue: vᵀAv · Orange: zero"
    },
    "legend": [
      {
        "ko": "이차형식",
        "en": "Quadratic form"
      },
      {
        "ko": "0 기준",
        "en": "Zero reference"
      }
    ],
    "note": {
      "ko": "2×2 실대칭 행렬만 조작합니다. PSD 판정은 a≥0,c≥0,ac−b²≥0의 정확한 정수 조건입니다. 표본 각도만 보고 판정하지 않습니다.",
      "en": "Only real symmetric 2×2 matrices. PSD is certified by exact integer conditions a≥0,c≥0,ac−b²≥0, not by sampled angles."
    }
  },
  "quadratic-forms-orders": {
    "controls": [
      {
        "label": {
          "ko": "대각 a",
          "en": "Diagonal a"
        },
        "min": -3,
        "max": 3,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "대칭 원소 b",
          "en": "Off-diagonal b"
        },
        "min": -3,
        "max": 3,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "대각 c",
          "en": "Diagonal c"
        },
        "min": -3,
        "max": 3,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "검사 각도 θ [도]",
          "en": "Probe angle θ [degrees]"
        },
        "min": -180,
        "max": 180,
        "initial": 45,
        "step": 1
      },
      {
        "label": {
          "ko": "후보 상한 C",
          "en": "Candidate bound C"
        },
        "min": -3,
        "max": 4,
        "initial": 3,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "성립하는 상한",
          "en": "Valid bound"
        },
        "values": [
          2,
          1,
          2,
          45,
          3
        ]
      },
      {
        "name": {
          "ko": "한 방향만 통과",
          "en": "One direction passes"
        },
        "values": [
          2,
          1,
          2,
          0,
          2
        ]
      },
      {
        "name": {
          "ko": "영 행렬",
          "en": "Zero matrix"
        },
        "values": [
          0,
          0,
          0,
          180,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 단위 방향 각도 θ [도] · 파랑: vᵀAv · 주황: 후보 C",
      "en": "Horizontal: unit-direction angle θ [degrees] · Blue: vᵀAv · Orange: candidate C"
    },
    "legend": [
      {
        "ko": "이차형식",
        "en": "Quadratic form"
      },
      {
        "ko": "후보 상한 C",
        "en": "Candidate C"
      }
    ],
    "note": {
      "ko": "Loewner 순서는 CI−A가 PSD라는 뜻이며 원소별 부등식과 다릅니다. 프로브 비교는 반올림 오차 허용 10⁻¹², 행렬 판정은 정확한 정수 조건을 사용합니다.",
      "en": "Loewner order means CI−A is PSD, not entrywise inequality. The probe comparison allows 10⁻¹² roundoff; the matrix certificate uses exact integer predicates."
    }
  },
  "kronecker-products": {
    "controls": [
      {
        "label": {
          "ko": "전단 s",
          "en": "Shear s"
        },
        "min": -3,
        "max": 3,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "배율 t",
          "en": "Scale t"
        },
        "min": -3,
        "max": 3,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "열 우선 벡터화",
          "en": "Column-major vectorization"
        },
        "values": [
          1,
          2
        ]
      },
      {
        "name": {
          "ko": "항등 인자",
          "en": "Identity factors"
        },
        "values": [
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "특이 인자",
          "en": "Singular factor"
        },
        "values": [
          -1,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 열 우선 성분 번호 · 파랑: vec(AXBᵀ) · 주황: (A⊗B)vec(X)",
      "en": "Horizontal: column-major component index · Blue: vec(AXBᵀ) · Orange: (A⊗B)vec(X)"
    },
    "legend": [
      {
        "ko": "직접 행렬 곱",
        "en": "Direct matrix product"
      },
      {
        "ko": "순서가 다른 인자",
        "en": "Reversed factor order"
      }
    ],
    "note": {
      "ko": "X=[[1,2],[3,4]]이므로 vec(X)=(1,3,2,4)입니다. 아래 표는 A⊗B이고 항등식의 인자 순서는 B⊗A입니다. 성분 번호는 연속 좌표가 아닙니다.",
      "en": "X=[[1,2],[3,4]] gives vec(X)=(1,3,2,4). The table is A⊗B; the vectorization identity uses B⊗A. Component indices are not continuous coordinates."
    }
  }
};

export const analysisLabs:Record<string,FoundationLabSpec> = {
  "limits-continuity-completeness": {
    "controls": [
      {
        "label": {
          "ko": "방식 0/1",
          "en": "Mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "점의 값 f(1)",
          "en": "Assigned value f(1)"
        },
        "min": 0,
        "max": 3,
        "initial": 0,
        "step": "any"
      },
      {
        "label": {
          "ko": "주변 반경 δ",
          "en": "Neighborhood radius δ"
        },
        "min": 0.01,
        "max": 1,
        "initial": 0.5,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "제거 가능한 불연속",
          "en": "Removable discontinuity"
        },
        "values": [
          0,
          0,
          0.5
        ]
      },
      {
        "name": {
          "ko": "연속으로 채우기",
          "en": "Continuous extension"
        },
        "values": [
          0,
          2,
          0.1
        ]
      },
      {
        "name": {
          "ko": "점프",
          "en": "Jump"
        },
        "values": [
          1,
          0.5,
          0.1
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 세로: f(x) · 빈 원: 제외된 값 · 점: 실제 f(1)",
      "en": "Horizontal: x · Vertical: f(x) · Open circle: excluded value · Dot: actual f(1)"
    },
    "legend": [
      {
        "ko": "함수·원래 모형",
        "en": "Function or original model"
      },
      {
        "ko": "비교 모형",
        "en": "Comparison model"
      }
    ],
    "note": {
      "ko": "δ는 그리는 근방 크기이며 유한 그림은 ε–δ 논증의 증명이 아닙니다.",
      "en": "δ sets the plotted neighborhood; a finite picture is not an epsilon–delta proof."
    }
  },
  "differentiability-mean-value": {
    "controls": [
      {
        "label": {
          "ko": "함수 방식 0/1",
          "en": "Function mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "기준점 x₀",
          "en": "Base point x₀"
        },
        "min": -1,
        "max": 1,
        "initial": 0.5,
        "step": "any"
      },
      {
        "label": {
          "ko": "전진 폭 h",
          "en": "Forward step h"
        },
        "min": 0,
        "max": 1,
        "initial": 0.1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "매끄러운 함수",
          "en": "Smooth function"
        },
        "values": [
          0,
          0.5,
          0.1
        ]
      },
      {
        "name": {
          "ko": "모서리",
          "en": "Corner"
        },
        "values": [
          1,
          0,
          0.1
        ]
      },
      {
        "name": {
          "ko": "폭 0",
          "en": "Zero step"
        },
        "values": [
          0,
          0.5,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 파랑: 함수 · 주황: 존재할 때 기준 접선",
      "en": "Horizontal: x · Blue: function · Orange: base tangent when defined"
    },
    "legend": [
      {
        "ko": "함수·원래 모형",
        "en": "Function or original model"
      },
      {
        "ko": "비교 모형",
        "en": "Comparison model"
      }
    ],
    "note": {
      "ko": "평균값정리는 [−1,1]에서 검사합니다. |x|는 0에서 미분 불가라 가정을 충족하지 않습니다. h=0의 차분은 정의하지 않습니다.",
      "en": "The MVT interval is [−1,1]. |x| fails differentiability at zero. A finite difference with h=0 is undefined."
    }
  },
  "integration-fundamental-theorem": {
    "controls": [
      {
        "label": {
          "ko": "누적 끝점 x",
          "en": "Accumulation endpoint x"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "분할 수 N",
          "en": "Partition count N"
        },
        "min": 1,
        "max": 64,
        "initial": 4,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "네 직사각형",
          "en": "Four rectangles"
        },
        "values": [
          1,
          4
        ]
      },
      {
        "name": {
          "ko": "분할 세분",
          "en": "Refined partition"
        },
        "values": [
          1,
          32
        ]
      },
      {
        "name": {
          "ko": "빈 구간",
          "en": "Zero-width interval"
        },
        "values": [
          0,
          4
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 파랑: f(x)=x² · 주황: F(x)=x³/3 · 무차원 예제",
      "en": "Horizontal: x · Blue: f(x)=x² · Orange: F(x)=x³/3 · Dimensionless example"
    },
    "legend": [
      {
        "ko": "함수 f",
        "en": "Integrand f"
      },
      {
        "ko": "누적 F",
        "en": "Accumulation F"
      }
    ],
    "note": {
      "ko": "왼쪽 합은 실제 N개 직사각형으로 계산합니다. 두 곡선은 함수와 누적량으로 의미가 다릅니다.",
      "en": "The left sum uses N actual rectangles. The two curves represent a function and its accumulation, different quantities."
    }
  },
  "taylor-remainders": {
    "controls": [
      {
        "label": {
          "ko": "평가점 x",
          "en": "Evaluation point x"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "다항 차수 n",
          "en": "Polynomial degree n"
        },
        "min": 0,
        "max": 8,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "이차 근사",
          "en": "Quadratic approximation"
        },
        "values": [
          1,
          2
        ]
      },
      {
        "name": {
          "ko": "고차 근사",
          "en": "Higher-degree approximation"
        },
        "values": [
          1,
          8
        ]
      },
      {
        "name": {
          "ko": "원점",
          "en": "Expansion center"
        },
        "values": [
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 파랑: exp(x) · 주황: 원점 Taylor 다항식",
      "en": "Horizontal: x · Blue: exp(x) · Orange: Taylor polynomial at zero"
    },
    "legend": [
      {
        "ko": "exp(x)",
        "en": "exp(x)"
      },
      {
        "ko": "Taylor 다항식",
        "en": "Taylor polynomial"
      }
    ],
    "note": {
      "ko": "Lagrange 상한은 exp(|x|)|x|ⁿ⁺¹/(n+1)!입니다. 실제 오차와 같은 값일 필요는 없습니다.",
      "en": "The Lagrange bound is exp(|x|)|x|ⁿ⁺¹/(n+1)!. It need not equal actual error."
    }
  },
  "series-convergence-tests": {
    "controls": [
      {
        "label": {
          "ko": "급수 방식 0~2",
          "en": "Series mode 0–2"
        },
        "min": 0,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "항 수 N",
          "en": "Number of terms N"
        },
        "min": 1,
        "max": 128,
        "initial": 8,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "조화급수",
          "en": "Harmonic series"
        },
        "values": [
          0,
          8
        ]
      },
      {
        "name": {
          "ko": "교대 조화급수",
          "en": "Alternating harmonic series"
        },
        "values": [
          1,
          8
        ]
      },
      {
        "name": {
          "ko": "제곱 역수 급수",
          "en": "Reciprocal-square series"
        },
        "values": [
          2,
          8
        ]
      }
    ],
    "axis": {
      "ko": "가로: 항 수 · 파랑: 부분합 · 주황: 존재할 때 무한합",
      "en": "Horizontal: term count · Blue: partial sum · Orange: infinite sum when defined"
    },
    "legend": [
      {
        "ko": "부분합",
        "en": "Partial sum"
      },
      {
        "ko": "무한합",
        "en": "Infinite sum"
      }
    ],
    "note": {
      "ko": "교대급수 오차 ≤1/(N+1), 제곱 역수 꼬리 ≤1/N입니다. π²/6의 평가는 여기서 증명하지 않고 적분 비교로 수렴을 증명합니다.",
      "en": "Alternating error ≤1/(N+1); reciprocal-square tail ≤1/N. The value π²/6 is stated, while convergence is proved by integral comparison."
    }
  },
  "pointwise-uniform-convergence": {
    "controls": [
      {
        "label": {
          "ko": "함수열 방식 0/1",
          "en": "Sequence mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "인덱스 n",
          "en": "Index n"
        },
        "min": 1,
        "max": 128,
        "initial": 8,
        "step": 1
      },
      {
        "label": {
          "ko": "검사점 x₀",
          "en": "Probe x₀"
        },
        "min": 0,
        "max": 1,
        "initial": 0.5,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "점별 수렴",
          "en": "Pointwise convergence"
        },
        "values": [
          0,
          8,
          0.5
        ]
      },
      {
        "name": {
          "ko": "끝점 근처",
          "en": "Near the endpoint"
        },
        "values": [
          0,
          64,
          0.99
        ]
      },
      {
        "name": {
          "ko": "균등 수렴",
          "en": "Uniform convergence"
        },
        "values": [
          1,
          8,
          0.5
        ]
      }
    ],
    "axis": {
      "ko": "가로: x∈[0,1] · 파랑: fₙ · 주황: 점별 극한",
      "en": "Horizontal: x∈[0,1] · Blue: fₙ · Orange: pointwise limit"
    },
    "legend": [
      {
        "ko": "함수·원래 모형",
        "en": "Function or original model"
      },
      {
        "ko": "비교 모형",
        "en": "Comparison model"
      }
    ],
    "note": {
      "ko": "xⁿ의 오차 supremum은 1이나 최대값으로 달성되지 않습니다. 유한 표본 최대와 구분하세요.",
      "en": "For xⁿ the error supremum is one but is not attained. Distinguish it from a finite sample maximum."
    }
  },
  "compactness-extreme-values": {
    "controls": [
      {
        "label": {
          "ko": "구간 방식 0/1",
          "en": "Interval mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "검사점 x₀",
          "en": "Probe x₀"
        },
        "min": 0,
        "max": 1,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "닫힌 구간",
          "en": "Closed interval"
        },
        "values": [
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "열린 구간",
          "en": "Open interval"
        },
        "values": [
          1,
          0.5
        ]
      },
      {
        "name": {
          "ko": "제외된 끝점",
          "en": "Excluded endpoint"
        },
        "values": [
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 세로: f(x)=x · 빈 원: 정의역 제외 끝점",
      "en": "Horizontal: x · Vertical: f(x)=x · Open circles: excluded endpoints"
    },
    "legend": [
      {
        "ko": "함수·원래 모형",
        "en": "Function or original model"
      },
      {
        "ko": "비교 모형",
        "en": "Comparison model"
      }
    ],
    "note": {
      "ko": "Rⁿ의 Heine–Borel 정리를 쓰며 무한차원 공간으로 일반화하지 않습니다.",
      "en": "Uses the Heine–Borel theorem in Rⁿ; do not generalize it to arbitrary infinite-dimensional spaces."
    }
  },
  "convexity-first-order": {
    "controls": [
      {
        "label": {
          "ko": "곡률 계수 a",
          "en": "Curvature coefficient a"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "접점 x",
          "en": "Tangent base x"
        },
        "min": -1,
        "max": 1,
        "initial": 0,
        "step": "any"
      },
      {
        "label": {
          "ko": "비교점 y",
          "en": "Comparison point y"
        },
        "min": -1,
        "max": 1,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "볼록 이차식",
          "en": "Convex quadratic"
        },
        "values": [
          1,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "오목 이차식",
          "en": "Concave quadratic"
        },
        "values": [
          -1,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "같은 점",
          "en": "Same point"
        },
        "values": [
          -1,
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 입력 · 파랑: ax² · 주황: 선택한 접선",
      "en": "Horizontal: input · Blue: ax² · Orange: selected tangent"
    },
    "legend": [
      {
        "ko": "함수",
        "en": "Function"
      },
      {
        "ko": "접선",
        "en": "Tangent"
      }
    ],
    "note": {
      "ko": "전역 판정은 a≥0이라는 해석 조건입니다. 같은 점에서 간격 0이라는 사실만으로 볼록성을 증명하지 않습니다.",
      "en": "Global convexity is certified by a≥0. Zero gap at the same point does not establish convexity."
    }
  },
  "limits-integration-interchange": {
    "controls": [
      {
        "label": {
          "ko": "모형 방식 0/1",
          "en": "Model mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "인덱스 n",
          "en": "Index n"
        },
        "min": 2,
        "max": 64,
        "initial": 4,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "움직이는 봉우리",
          "en": "Shrinking spike"
        },
        "values": [
          0,
          4
        ]
      },
      {
        "name": {
          "ko": "더 좁은 봉우리",
          "en": "Narrower spike"
        },
        "values": [
          0,
          32
        ]
      },
      {
        "name": {
          "ko": "지배되는 함수열",
          "en": "Dominated sequence"
        },
        "values": [
          1,
          8
        ]
      }
    ],
    "axis": {
      "ko": "가로: x∈[0,1] · 파랑: fₙ · 주황: 점별 극한 0",
      "en": "Horizontal: x∈[0,1] · Blue: fₙ · Orange: pointwise limit zero"
    },
    "legend": [
      {
        "ko": "함수·원래 모형",
        "en": "Function or original model"
      },
      {
        "ko": "비교 모형",
        "en": "Comparison model"
      }
    ],
    "note": {
      "ko": "봉우리 꼭짓점을 해석적으로 넣어 얇은 지지를 표본 격자에서 놓치지 않습니다.",
      "en": "Analytic spike vertices prevent a sampling grid from missing the narrow support."
    }
  },
  "implicit-inverse-functions": {
    "controls": [
      {
        "label": {
          "ko": "정리 방식 0/1",
          "en": "Theorem mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "기준점 x₀",
          "en": "Base point x₀"
        },
        "min": -1,
        "max": 1,
        "initial": 0.6,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "정칙 원의 가지",
          "en": "Regular circle branch"
        },
        "values": [
          0,
          0.6
        ]
      },
      {
        "name": {
          "ko": "수직 접선",
          "en": "Vertical tangent"
        },
        "values": [
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "역함수는 있지만",
          "en": "Inverse without regular derivative"
        },
        "values": [
          1,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 세로: 양의 원 가지 또는 세제곱과 그 역함수",
      "en": "Horizontal: x · Vertical: upper circle branch or cube and its inverse"
    },
    "legend": [
      {
        "ko": "함수·원래 모형",
        "en": "Function or original model"
      },
      {
        "ko": "비교 모형",
        "en": "Comparison model"
      }
    ],
    "note": {
      "ko": "원 모형은 양의 가지입니다. x³은 전역 가역이지만 원점에서 역함수가 유한 도함수를 갖지 않습니다.",
      "en": "The circle uses the upper branch. x³ is globally invertible, but its inverse lacks a finite derivative at zero."
    }
  },
  "matrix-trace-logdet-derivatives": {
    "controls": [
      {
        "label": {
          "ko": "대각 a",
          "en": "Diagonal a"
        },
        "min": 1,
        "max": 3,
        "initial": 2,
        "step": "any"
      },
      {
        "label": {
          "ko": "대칭 원소 b",
          "en": "Shared off-diagonal b"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "대각 c",
          "en": "Diagonal c"
        },
        "min": 1,
        "max": 3,
        "initial": 2,
        "step": "any"
      },
      {
        "label": {
          "ko": "차분 폭 h",
          "en": "Difference step h"
        },
        "min": 0,
        "max": 0.1,
        "initial": 0.01,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "SPD 내부",
          "en": "Inside SPD domain"
        },
        "values": [
          2,
          1,
          2,
          0.01
        ]
      },
      {
        "name": {
          "ko": "특이 경계",
          "en": "Singular boundary"
        },
        "values": [
          1,
          1,
          1,
          0.01
        ]
      },
      {
        "name": {
          "ko": "분리된 성분",
          "en": "Diagonal example"
        },
        "values": [
          2,
          0,
          2,
          0.01
        ]
      }
    ],
    "axis": {
      "ko": "가로: A₀₀의 변화 t · 세로: 정의역 안의 logdet(A+tE₀₀)",
      "en": "Horizontal: perturbation t of A₀₀ · Vertical: logdet on its domain"
    },
    "legend": [
      {
        "ko": "함수·원래 모형",
        "en": "Function or original model"
      },
      {
        "ko": "비교 모형",
        "en": "Comparison model"
      }
    ],
    "note": {
      "ko": "실수 SPD 정의역만 다룹니다. 차분의 양쪽 행렬이 SPD가 아니면 중앙차분을 정의하지 않습니다.",
      "en": "Only the real SPD domain is used. Central differences require both perturbed matrices to stay SPD."
    }
  },
  "vector-fields-divergence-curl": {
    "controls": [
      {
        "label": {
          "ko": "계수 a",
          "en": "Coefficient a"
        },
        "min": -2,
        "max": 2,
        "initial": 0,
        "step": "any"
      },
      {
        "label": {
          "ko": "계수 b",
          "en": "Coefficient b"
        },
        "min": -2,
        "max": 2,
        "initial": -1,
        "step": "any"
      },
      {
        "label": {
          "ko": "계수 c",
          "en": "Coefficient c"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "계수 d",
          "en": "Coefficient d"
        },
        "min": -2,
        "max": 2,
        "initial": 0,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "회전장",
          "en": "Rotational field"
        },
        "values": [
          0,
          -1,
          1,
          0
        ]
      },
      {
        "name": {
          "ko": "방사장",
          "en": "Radial field"
        },
        "values": [
          1,
          0,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "전단장",
          "en": "Shear field"
        },
        "values": [
          0,
          1,
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "같은 단위 xy 좌표 · 화살표: F의 0.3배 · 입자 궤적이 아님",
      "en": "Equal-unit xy coordinates · Arrows: 0.3F · Not particle trajectories"
    },
    "legend": [
      {
        "ko": "위치별 장",
        "en": "Field at each position"
      }
    ],
    "note": {
      "ko": "발산과 curl은 Jacobian에서 계산합니다. 그림의 유한 화살표 길이는 미분 연산 자체가 아닙니다.",
      "en": "Divergence and curl come from the Jacobian. Finite drawn arrows are not themselves derivative operations."
    }
  },
  "line-surface-volume-integrals": {
    "controls": [
      {
        "label": {
          "ko": "장 방식 0/1",
          "en": "Field mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "반지름 R",
          "en": "Radius R"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "원 위의 회전",
          "en": "Rotation on a circle"
        },
        "values": [
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "원 밖으로 유출",
          "en": "Outward radial flux"
        },
        "values": [
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "퇴화 원",
          "en": "Zero-radius limit"
        },
        "values": [
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "같은 단위 xy 좌표 · 원은 반시계 방향 · 화살표: 0.25F",
      "en": "Equal-unit xy coordinates · Circle oriented counterclockwise · Arrows: 0.25F"
    },
    "legend": [
      {
        "ko": "원 경계",
        "en": "Circle boundary"
      },
      {
        "ko": "위치별 장",
        "en": "Field arrows"
      }
    ],
    "note": {
      "ko": "여기서는 원 경계의 선적분과 원판 면적을 계산합니다. 일반 곡면·체적 적분은 본문에서 매개화·Jacobian과 함께 설명합니다.",
      "en": "The lab computes circle-boundary integrals and disk area; the theory explains surface and volume integrals with parametrizations and Jacobians."
    }
  },
  "gauss-stokes-theorems": {
    "controls": [
      {
        "label": {
          "ko": "반지름 R",
          "en": "Radius R"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "방사 계수 a",
          "en": "Radial coefficient a"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "회전 계수 b",
          "en": "Rotation coefficient b"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "유출과 순환",
          "en": "Flux and circulation"
        },
        "values": [
          1,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "방향 반전 장",
          "en": "Reversed field"
        },
        "values": [
          1,
          -1,
          -1
        ]
      },
      {
        "name": {
          "ko": "퇴화 극한",
          "en": "Degenerate limit"
        },
        "values": [
          0,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "xy 적도 단면 · 구면 전체 그림이 아님 · 적도는 +z에서 반시계",
      "en": "Equatorial xy cross-section, not the full sphere · Equator is CCW viewed from +z"
    },
    "legend": [
      {
        "ko": "함수·원래 모형",
        "en": "Function or original model"
      },
      {
        "ko": "비교 모형",
        "en": "Comparison model"
      }
    ],
    "note": {
      "ko": "구면 유출은 sinφ를 Simpson 64분할로, 순환은 벡터 내적의 중점 합으로 계산해 내부 적분의 해석값과 대조합니다. R=0은 정리의 정칙 표면이 아닌 퇴화 극한입니다.",
      "en": "Sphere flux uses 64-panel Simpson quadrature of sinφ; circulation uses midpoint vector dot products. Compare with analytic interior integrals. R=0 is a degenerate limit, not a regular surface."
    }
  },
  "higher-order-autodiff": {
    "controls": [
      {
        "label": {
          "ko": "기준 x",
          "en": "Base x"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "기준 y",
          "en": "Base y"
        },
        "min": -2,
        "max": 2,
        "initial": 0,
        "step": "any"
      },
      {
        "label": {
          "ko": "방향 vₓ",
          "en": "Direction vₓ"
        },
        "min": -1,
        "max": 1,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "방향 vᵧ",
          "en": "Direction vᵧ"
        },
        "min": -1,
        "max": 1,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "혼합 방향",
          "en": "Mixed direction"
        },
        "values": [
          1,
          0,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "x 방향",
          "en": "Along x"
        },
        "values": [
          1,
          0,
          1,
          0
        ]
      },
      {
        "name": {
          "ko": "영 방향",
          "en": "Zero direction"
        },
        "values": [
          1,
          0,
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 경로 매개변수 t · 파랑: f(x+tv) · 주황: 이차 Taylor 모형",
      "en": "Horizontal: path parameter t · Blue: f(x+tv) · Orange: quadratic Taylor model"
    },
    "legend": [
      {
        "ko": "실제 경로 함수",
        "en": "Actual path function"
      },
      {
        "ko": "이차 모형",
        "en": "Quadratic model"
      }
    ],
    "note": {
      "ko": "f=½x²y+sin y의 정확한 해석 도함수입니다. 자동 미분 엔진 실행과 동일시하지 않으며 중앙차분으로 독립 검산합니다.",
      "en": "These are analytic derivatives of f=½x²y+sin y, not a claim of running an autodiff engine. Independent central differences verify them."
    }
  }
};

export const complexLabs:Record<string,FoundationLabSpec> = {
  "complex-polar-maps": {
    "controls": [
      {
        "label": {
          "ko": "크기 r",
          "en": "Magnitude r"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "각도 θ [도]",
          "en": "Angle θ [degrees]"
        },
        "min": -180,
        "max": 180,
        "initial": 45,
        "step": "any"
      },
      {
        "label": {
          "ko": "거듭제곱 k",
          "en": "Power k"
        },
        "min": 1,
        "max": 4,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "두 배 각도",
          "en": "Double angle"
        },
        "values": [
          1,
          45,
          2
        ]
      },
      {
        "name": {
          "ko": "크기 확대",
          "en": "Magnitude expansion"
        },
        "values": [
          2,
          45,
          2
        ]
      },
      {
        "name": {
          "ko": "영점",
          "en": "Zero"
        },
        "values": [
          0,
          45,
          2
        ]
      }
    ],
    "axis": {
      "ko": "복소평면 · 파랑: z · 주황: zᵏ · 같은 축 단위",
      "en": "Complex plane · Blue: z · Orange: zᵏ · Equal axis units"
    },
    "legend": [
      {
        "ko": "입력 z",
        "en": "Input z"
      },
      {
        "ko": "출력 zᵏ",
        "en": "Output zᵏ"
      }
    ],
    "note": {
      "ko": "삼각함수 내부는 라디안입니다. 영점의 각도는 정의하지 않습니다. 표시 주값 각도와 연속적으로 누적한 각도를 구분하세요.",
      "en": "Trigonometric arguments are radians. Zero has no argument. Distinguish principal and accumulated angles."
    }
  },
  "cauchy-riemann-holomorphic": {
    "controls": [
      {
        "label": {
          "ko": "함수 방식 0~2",
          "en": "Function mode 0–2"
        },
        "min": 0,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "실수부 x",
          "en": "Real part x"
        },
        "min": -1,
        "max": 1,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "허수부 y",
          "en": "Imaginary part y"
        },
        "min": -1,
        "max": 1,
        "initial": 0,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "정칙 제곱",
          "en": "Holomorphic square"
        },
        "values": [
          0,
          1,
          0
        ]
      },
      {
        "name": {
          "ko": "켤레 함수",
          "en": "Conjugation"
        },
        "values": [
          1,
          0,
          0
        ]
      },
      {
        "name": {
          "ko": "한 점에서만 미분 가능",
          "en": "Differentiable only at one point"
        },
        "values": [
          2,
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "출력 복소평면 · 파랑: 수평 입력선의 상 · 주황: 수직 입력선의 상",
      "en": "Output complex plane · Blue: image of horizontal input line · Orange: image of vertical input line"
    },
    "legend": [
      {
        "ko": "수평선의 상",
        "en": "Horizontal-line image"
      },
      {
        "ko": "수직선의 상",
        "en": "Vertical-line image"
      }
    ],
    "note": {
      "ko": "세 모형은 실수 C¹ 함수입니다. 원점의 |z|²는 미분 가능하지만 어느 열린 근방에서도 정칙이 아닙니다.",
      "en": "These models are real C¹. |z|² is differentiable at zero but not holomorphic on any neighborhood."
    }
  },
  "complex-contour-integrals": {
    "controls": [
      {
        "label": {
          "ko": "적분 함수 0/1",
          "en": "Integrand mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "반지름 R",
          "en": "Radius R"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "방향 0/1",
          "en": "Direction 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "정칙 함수",
          "en": "Holomorphic integrand"
        },
        "values": [
          0,
          1,
          0
        ]
      },
      {
        "name": {
          "ko": "원점 특이점",
          "en": "Pole at zero"
        },
        "values": [
          1,
          1,
          0
        ]
      },
      {
        "name": {
          "ko": "방향 반전",
          "en": "Reversed orientation"
        },
        "values": [
          1,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "입력 복소평면 · 파랑: 경로 · 주황 화살표: 진행 방향",
      "en": "Input complex plane · Blue: contour · Orange arrows: traversal direction"
    },
    "legend": [
      {
        "ko": "적분 경로",
        "en": "Contour"
      },
      {
        "ko": "진행 방향",
        "en": "Traversal direction"
      }
    ],
    "note": {
      "ko": "중점 구적과 해석값을 별도로 계산합니다. R=0에서 1/z 적분은 정의하지 않습니다.",
      "en": "Midpoint quadrature and analytic values are separate. The 1/z integral is undefined at radius zero."
    }
  },
  "residues-winding": {
    "controls": [
      {
        "label": {
          "ko": "극 위치 a",
          "en": "Pole position a"
        },
        "min": -2,
        "max": 2,
        "initial": 0,
        "step": "any"
      },
      {
        "label": {
          "ko": "원 반지름 R",
          "en": "Circle radius R"
        },
        "min": 0.25,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "극 차수 m",
          "en": "Pole order m"
        },
        "min": 1,
        "max": 3,
        "initial": 1,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "내부 단순극",
          "en": "Enclosed simple pole"
        },
        "values": [
          0,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "외부 극",
          "en": "Pole outside"
        },
        "values": [
          2,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "경로 위 극",
          "en": "Pole on contour"
        },
        "values": [
          1,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "입력 복소평면 · 원: 반시계 경로 · 빈 원: 제외된 극",
      "en": "Input complex plane · Circle: CCW contour · Open marker: excluded pole"
    },
    "legend": [
      {
        "ko": "경로",
        "en": "Contour"
      },
      {
        "ko": "극 위치",
        "en": "Pole location"
      }
    ],
    "note": {
      "ko": "경로에 가까운 극은 구적 오차를 키울 수 있습니다. 해석 감김 수와 수치 적분을 혼동하지 마세요.",
      "en": "Near-contour poles can increase quadrature error. Do not confuse analytic winding with numerical quadrature."
    }
  },
  "analytic-functions-singularities": {
    "controls": [
      {
        "label": {
          "ko": "특이점 방식 0~2",
          "en": "Singularity mode 0–2"
        },
        "min": 0,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "검사 반지름 r",
          "en": "Probe radius r"
        },
        "min": 0,
        "max": 1,
        "initial": 0.5,
        "step": "any"
      },
      {
        "label": {
          "ko": "접근 각도 θ [도]",
          "en": "Approach angle θ [degrees]"
        },
        "min": -180,
        "max": 180,
        "initial": 0,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "제거 가능한 점",
          "en": "Removable point"
        },
        "values": [
          0,
          0,
          0
        ]
      },
      {
        "name": {
          "ko": "단순극",
          "en": "Simple pole"
        },
        "values": [
          1,
          0.5,
          0
        ]
      },
      {
        "name": {
          "ko": "본질적 특이점",
          "en": "Essential singularity"
        },
        "values": [
          2,
          0.5,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 반지름 r · 파랑: Re f · 주황: Im f · 그림은 r=0.1,…,1",
      "en": "Horizontal: radius r · Blue: Re f · Orange: Im f · Plot range r=0.1,…,1"
    },
    "legend": [
      {
        "ko": "실수부",
        "en": "Real part"
      },
      {
        "ko": "허수부",
        "en": "Imaginary part"
      }
    ],
    "note": {
      "ko": "검사점은 그림 범위 밖 r=0도 허용합니다. 숫자 표현 범위 초과와 함수의 미정의를 별도로 표시합니다.",
      "en": "The probe may include r=0 outside the sampled plot. Numeric overflow and an undefined function are reported separately."
    }
  },
  "analytic-continuation-branches": {
    "controls": [
      {
        "label": {
          "ko": "종점 각도 θ [도]",
          "en": "Endpoint angle θ [degrees]"
        },
        "min": -170,
        "max": 170,
        "initial": 90,
        "step": "any"
      },
      {
        "label": {
          "ko": "감김 횟수 k",
          "en": "Winding count k"
        },
        "min": -2,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "반지름 r",
          "en": "Radius r"
        },
        "min": 0.25,
        "max": 2,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "감김 없음",
          "en": "No extra winding"
        },
        "values": [
          90,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "한 바퀴 추가",
          "en": "One extra turn"
        },
        "values": [
          90,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "두 바퀴 추가",
          "en": "Two extra turns"
        },
        "values": [
          90,
          2,
          1
        ]
      }
    ],
    "axis": {
      "ko": "복소평면 · 파랑: 입력 경로 z · 주황: 연속시킨 제곱근 경로",
      "en": "Complex plane · Blue: input path z · Orange: continued square-root path"
    },
    "legend": [
      {
        "ko": "입력 경로",
        "en": "Input path"
      },
      {
        "ko": "연속 제곱근",
        "en": "Continued square root"
      }
    ],
    "note": {
      "ko": "경로를 따른 각도 lift이며 원점을 뺀 평면 전체의 단일값 로그를 주장하지 않습니다. 종점은 주가지 절단 위를 피합니다.",
      "en": "This is an angle lift along a path, not a global single-valued logarithm on the punctured plane. Endpoints avoid the principal cut."
    }
  },
  "complex-linear-algebra": {
    "controls": [
      {
        "label": {
          "ko": "위상 θ [도]",
          "en": "Phase θ [degrees]"
        },
        "min": -180,
        "max": 180,
        "initial": 90,
        "step": "any"
      },
      {
        "label": {
          "ko": "벡터 w 배율",
          "en": "Scale of w"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "같은 벡터",
          "en": "Matching vectors"
        },
        "values": [
          90,
          1
        ]
      },
      {
        "name": {
          "ko": "직교 벡터",
          "en": "Orthogonal vectors"
        },
        "values": [
          -90,
          1
        ]
      },
      {
        "name": {
          "ko": "영벡터",
          "en": "Zero vector"
        },
        "values": [
          90,
          0
        ]
      }
    ],
    "axis": {
      "ko": "복소평면 · 파랑: w의 첫 성분 · 주황: 둘째 성분 · C² 전체의 2D 그림이 아님",
      "en": "Complex plane · Blue: first component of w · Orange: second component · Not a 2D embedding of all C²"
    },
    "legend": [
      {
        "ko": "성분 w₀",
        "en": "Component w₀"
      },
      {
        "ko": "성분 w₁",
        "en": "Component w₁"
      }
    ],
    "note": {
      "ko": "v=(1,i)는 고정입니다. 내적은 첫 인자에 켤레를 취합니다. 영벡터와의 정규화 유사도는 정의하지 않습니다.",
      "en": "v=(1,i) is fixed. The first argument is conjugated. Normalized similarity with a zero vector is undefined."
    }
  },
  "wirtinger-gradients": {
    "controls": [
      {
        "label": {
          "ko": "실수부 x",
          "en": "Real part x"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "허수부 y",
          "en": "Imaginary part y"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "보폭 η",
          "en": "Step η"
        },
        "min": 0,
        "max": 1.5,
        "initial": 0.25,
        "step": "any"
      },
      {
        "label": {
          "ko": "목표 방식 0/1",
          "en": "Target mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "수축하는 갱신",
          "en": "Contracting update"
        },
        "values": [
          1,
          1,
          0.25,
          0
        ]
      },
      {
        "name": {
          "ko": "진동하는 갱신",
          "en": "Oscillating update"
        },
        "values": [
          1,
          1,
          1,
          0
        ]
      },
      {
        "name": {
          "ko": "이동된 목표",
          "en": "Shifted target"
        },
        "values": [
          0,
          0,
          0.25,
          1
        ]
      }
    ],
    "axis": {
      "ko": "복소평면 · 파랑: 실제 gradient 갱신 궤적 · 주황 점: 최소점",
      "en": "Complex plane · Blue: actual gradient-update trajectory · Orange point: minimizer"
    },
    "legend": [
      {
        "ko": "갱신 궤적",
        "en": "Update trajectory"
      },
      {
        "ko": "최소점",
        "en": "Minimizer"
      }
    ],
    "note": {
      "ko": "유클리드 실수 gradient의 복소 표현은 2∂L/∂z̄입니다. 라이브러리의 gradient 규약은 따로 확인해야 합니다.",
      "en": "The Euclidean real gradient encoded as complex is 2∂L/∂z̄. Check each library’s convention separately."
    }
  },
  "complex-signals-phase": {
    "controls": [
      {
        "label": {
          "ko": "첫 위상 φ₁ [도]",
          "en": "First phase φ₁ [degrees]"
        },
        "min": -180,
        "max": 180,
        "initial": 0,
        "step": "any"
      },
      {
        "label": {
          "ko": "둘째 위상 φ₂ [도]",
          "en": "Second phase φ₂ [degrees]"
        },
        "min": -180,
        "max": 180,
        "initial": 0,
        "step": "any"
      },
      {
        "label": {
          "ko": "둘째 진폭 A₂",
          "en": "Second amplitude A₂"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "보강 간섭",
          "en": "Constructive interference"
        },
        "values": [
          0,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "상쇄 간섭",
          "en": "Destructive interference"
        },
        "values": [
          0,
          180,
          1
        ]
      },
      {
        "name": {
          "ko": "직교 위상",
          "en": "Quadrature phases"
        },
        "values": [
          0,
          90,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: 무차원 위상 시간 t · 파랑: 첫 신호 · 주황: 두 신호 합",
      "en": "Horizontal: dimensionless phase time t · Blue: first signal · Orange: combined signal"
    },
    "legend": [
      {
        "ko": "첫 신호",
        "en": "First signal"
      },
      {
        "ko": "합 신호",
        "en": "Combined signal"
      }
    ],
    "note": {
      "ko": "같은 주파수의 실수 신호를 phasor로 표현합니다. 진폭 제곱은 정규화한 양이며 실제 전력 단위는 별도 물리 모형이 필요합니다.",
      "en": "Phasors represent equal-frequency real signals. Squared amplitude is normalized; physical power needs an additional model and units."
    }
  }
};

export const tensorLabs:Record<string,FoundationLabSpec> = {
  "tensor-order-shapes": {
    "controls": [
      {
        "label": {
          "ko": "배치 B",
          "en": "Batch B"
        },
        "min": 1,
        "max": 3,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "길이 T",
          "en": "Length T"
        },
        "min": 1,
        "max": 4,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "채널 C",
          "en": "Channels C"
        },
        "min": 1,
        "max": 4,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "축 순열 0/1",
          "en": "Axis permutation 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "원래 축",
          "en": "Original axes"
        },
        "values": [
          1,
          2,
          3,
          0
        ]
      },
      {
        "name": {
          "ko": "축 교환",
          "en": "Swap axes"
        },
        "values": [
          1,
          2,
          3,
          1
        ]
      },
      {
        "name": {
          "ko": "배치 추가",
          "en": "Add a batch"
        },
        "values": [
          2,
          2,
          3,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 행 우선 평탄 인덱스 · 세로: 실제 원소 · 표: 출력 축 좌표",
      "en": "Horizontal: row-major flat index · Vertical: actual entry · Table: output coordinates"
    },
    "legend": [
      {
        "ko": "출력 원소",
        "en": "Output entries"
      }
    ],
    "note": {
      "ko": "차수는 축 개수입니다. 마지막 축이 가장 빨리 변하는 저장 규약을 사용합니다. 순열은 reshape와 다릅니다.",
      "en": "Order counts axes. Last-axis-fastest storage is used. Axis permutation differs from reshape."
    }
  },
  "tensor-indices-contractions": {
    "controls": [
      {
        "label": {
          "ko": "수축 축",
          "en": "Contracted axis"
        },
        "min": 0,
        "max": 2,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "벡터 v₀",
          "en": "Vector v₀"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "벡터 v₁",
          "en": "Vector v₁"
        },
        "min": -2,
        "max": 2,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "마지막 축 수축",
          "en": "Contract the last axis"
        },
        "values": [
          2,
          1,
          2
        ]
      },
      {
        "name": {
          "ko": "첫 축 수축",
          "en": "Contract the first axis"
        },
        "values": [
          0,
          1,
          2
        ]
      },
      {
        "name": {
          "ko": "영 공변벡터",
          "en": "Zero covector"
        },
        "values": [
          2,
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 출력 행 우선 인덱스 · 세로: 수축 값 · 표: 남은 두 축",
      "en": "Horizontal: output row-major index · Vertical: contracted value · Table: two free axes"
    },
    "legend": [
      {
        "ko": "수축 결과",
        "en": "Contracted values"
      }
    ],
    "note": {
      "ko": "고정된 실수 Euclidean 기저의 작은 정수 예제입니다. 합산 축만 사라지고 자유축 순서는 보존됩니다.",
      "en": "A small integer example in fixed real Euclidean bases. Only the summed axis disappears; free-axis order is preserved."
    }
  },
  "multilinear-maps": {
    "controls": [
      {
        "label": {
          "ko": "u₀",
          "en": "u₀"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "u₁",
          "en": "u₁"
        },
        "min": -2,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "v₀",
          "en": "v₀"
        },
        "min": -2,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "v₁",
          "en": "v₁"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "기저 입력",
          "en": "Basis inputs"
        },
        "values": [
          1,
          0,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "모든 성분 1",
          "en": "All-one inputs"
        },
        "values": [
          1,
          1,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "한 슬롯 0",
          "en": "One zero slot"
        },
        "values": [
          0,
          0,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: 출력 성분 k · 파랑: F(u,v) · 주황: F(2u,v)",
      "en": "Horizontal: output component k · Blue: F(u,v) · Orange: F(2u,v)"
    },
    "legend": [
      {
        "ko": "F(u,v)",
        "en": "F(u,v)"
      },
      {
        "ko": "F(2u,v)",
        "en": "F(2u,v)"
      }
    ],
    "note": {
      "ko": "한 슬롯 두 배는 출력 두 배, 두 슬롯 두 배는 출력 네 배입니다. 일반적인 선형층과 같은 의미의 joint linearity가 아닙니다.",
      "en": "Doubling one slot doubles the output; doubling both quadruples it. This is not joint linearity of a usual linear layer."
    }
  },
  "broadcasting-semantics": {
    "controls": [
      {
        "label": {
          "ko": "shape 방식 0~3",
          "en": "Shape mode 0–3"
        },
        "min": 0,
        "max": 3,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "배율 b",
          "en": "Scale b"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "연산 0 더하기 / 1 곱하기",
          "en": "Operation 0 add / 1 multiply"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "채널 편향",
          "en": "Channel bias"
        },
        "values": [
          0,
          1,
          0
        ]
      },
      {
        "name": {
          "ko": "바깥 곱 모양",
          "en": "Outer-product layout"
        },
        "values": [
          2,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "호환 불가",
          "en": "Incompatible shapes"
        },
        "values": [
          3,
          1,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 출력 평탄 인덱스 · 세로: 원소별 연산 결과",
      "en": "Horizontal: flattened output index · Vertical: elementwise result"
    },
    "legend": [
      {
        "ko": "연산 결과",
        "en": "Output entries"
      }
    ],
    "note": {
      "ko": "오른쪽 축부터 비교합니다. shape가 맞지 않으면 출력 자체가 없으며 0 텐서로 대신하지 않습니다.",
      "en": "Compare trailing axes. Incompatible shapes have no output, not an invented zero tensor."
    }
  },
  "tensor-cp-tucker": {
    "controls": [
      {
        "label": {
          "ko": "첫 가중치 λ₀",
          "en": "First weight λ₀"
        },
        "min": -2,
        "max": 2,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "둘째 가중치 λ₁",
          "en": "Second weight λ₁"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "유지할 항 수",
          "en": "Term budget"
        },
        "min": 0,
        "max": 2,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "정확한 두 항",
          "en": "Exact two-term model"
        },
        "values": [
          2,
          1,
          2
        ]
      },
      {
        "name": {
          "ko": "한 항 근사",
          "en": "One-term approximation"
        },
        "values": [
          2,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "항 없음",
          "en": "No retained terms"
        },
        "values": [
          2,
          1,
          0
        ]
      }
    ],
    "axis": {
      "ko": "그림: 선택한 Tucker core·인자와 자유축 · 표: 원본과 복원값",
      "en": "Diagram: selected Tucker core, factors and free axes · Table: original and reconstruction"
    },
    "legend": [
      {
        "ko": "원본",
        "en": "Original"
      },
      {
        "ko": "근사",
        "en": "Approximation"
      }
    ],
    "note": {
      "ko": "명시적 직교 대각 예제입니다. 일반 CP 최적화나 HOSVD를 실행하지 않습니다. 작은 예제에서는 분해 저장량이 원본보다 클 수 있습니다.",
      "en": "An explicit orthogonal diagonal example, not a general CP solver or HOSVD run. Factor storage may exceed dense storage here."
    }
  },
  "tensor-trains": {
    "controls": [
      {
        "label": {
          "ko": "내부 bond 예산 r",
          "en": "Internal bond budget r"
        },
        "min": 1,
        "max": 2,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "둘째 성분 t",
          "en": "Second component t"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "Gauge 배율 g",
          "en": "Gauge scale g"
        },
        "min": 0.25,
        "max": 2,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "정확한 TT",
          "en": "Exact TT"
        },
        "values": [
          2,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "Gauge 변경",
          "en": "Gauge change"
        },
        "values": [
          2,
          1,
          2
        ]
      },
      {
        "name": {
          "ko": "Bond 제한",
          "en": "Restricted bond"
        },
        "values": [
          1,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "파랑: 합산 bond · 주황: 물리 자유축 · 경계 bond 차원은 1",
      "en": "Blue: summed bonds · Orange: physical free axes · Boundary bond dimensions are one"
    },
    "legend": [
      {
        "ko": "복원값",
        "en": "Reconstruction"
      }
    ],
    "note": {
      "ko": "정해진 core의 직접 수축입니다. 일반 TT-SVD 구현이 아니며 선택 bond가 항상 최소 랭크라는 뜻도 아닙니다.",
      "en": "Direct contraction of specified cores, not a general TT-SVD implementation. Configured bonds need not be minimal ranks."
    }
  },
  "tensor-networks-contraction": {
    "controls": [
      {
        "label": {
          "ko": "차원 n",
          "en": "Dimension n"
        },
        "min": 1,
        "max": 4,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "차원 m",
          "en": "Dimension m"
        },
        "min": 1,
        "max": 4,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "차원 p",
          "en": "Dimension p"
        },
        "min": 1,
        "max": 4,
        "initial": 4,
        "step": 1
      },
      {
        "label": {
          "ko": "차원 q",
          "en": "Dimension q"
        },
        "min": 1,
        "max": 4,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "오른쪽 먼저가 유리",
          "en": "Right-first is cheaper"
        },
        "values": [
          2,
          3,
          4,
          2
        ]
      },
      {
        "name": {
          "ko": "왼쪽 먼저가 유리",
          "en": "Left-first is cheaper"
        },
        "values": [
          4,
          2,
          1,
          4
        ]
      },
      {
        "name": {
          "ko": "비용 같음",
          "en": "Equal costs"
        },
        "values": [
          1,
          1,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "상자: 실제 행렬 인자 · 파랑: 수축 지표 · 주황: 출력 자유 지표",
      "en": "Boxes: matrix factors · Blue: contracted indices · Orange: output free indices"
    },
    "legend": [
      {
        "ko": "출력",
        "en": "Output"
      }
    ],
    "note": {
      "ko": "직접 조밀 행렬곱의 스칼라 곱셈만 셉니다. 실행 시간·통신·캐시·병렬 비용은 포함하지 않습니다.",
      "en": "Counts scalar multiplications in direct dense products; runtime, communication, cache and parallel costs are excluded."
    }
  },
  "tensor-low-rank-approximation": {
    "controls": [
      {
        "label": {
          "ko": "가중치 a",
          "en": "Weight a"
        },
        "min": -4,
        "max": 4,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "가중치 b",
          "en": "Weight b"
        },
        "min": -4,
        "max": 4,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "유지 예산 R",
          "en": "Retained budget R"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "한 항 유지",
          "en": "Keep one term"
        },
        "values": [
          3,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "모두 유지",
          "en": "Keep both terms"
        },
        "values": [
          3,
          1,
          2
        ]
      },
      {
        "name": {
          "ko": "영 텐서",
          "en": "Zero tensor"
        },
        "values": [
          0,
          0,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: 행 우선 원소 번호 · 파랑: X · 주황: 절단 근사",
      "en": "Horizontal: row-major index · Blue: X · Orange: truncated approximation"
    },
    "legend": [
      {
        "ko": "원본",
        "en": "Original"
      },
      {
        "ko": "근사",
        "en": "Approximation"
      }
    ],
    "note": {
      "ko": "이 특별한 직교 대각 텐서의 폐형식 절단이며 임의 텐서의 CP 최적화 알고리즘이 아닙니다.",
      "en": "Closed-form truncation of this special orthogonal diagonal tensor, not a general CP optimizer."
    }
  },
  "tensor-symmetry-antisymmetry": {
    "controls": [
      {
        "label": {
          "ko": "원소 a",
          "en": "Entry a"
        },
        "min": -3,
        "max": 3,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "원소 b",
          "en": "Entry b"
        },
        "min": -3,
        "max": 3,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "원소 c",
          "en": "Entry c"
        },
        "min": -3,
        "max": 3,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "원소 d",
          "en": "Entry d"
        },
        "min": -3,
        "max": 3,
        "initial": 3,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "혼합 성분",
          "en": "Mixed components"
        },
        "values": [
          1,
          2,
          0,
          3
        ]
      },
      {
        "name": {
          "ko": "대칭",
          "en": "Symmetric"
        },
        "values": [
          1,
          2,
          2,
          3
        ]
      },
      {
        "name": {
          "ko": "반대칭",
          "en": "Antisymmetric"
        },
        "values": [
          0,
          2,
          -2,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 2×2 행 우선 인덱스 · 파랑: S · 주황: K · 표: A=S+K",
      "en": "Horizontal: 2×2 row-major index · Blue: S · Orange: K · Table: A=S+K"
    },
    "legend": [
      {
        "ko": "대칭 부분",
        "en": "Symmetric part"
      },
      {
        "ko": "반대칭 부분",
        "en": "Antisymmetric part"
      }
    ],
    "note": {
      "ko": "실수의 두 지표 교환 예제입니다. 복소 Hermitian 분해에는 켤레전치가 필요합니다.",
      "en": "A real two-index example. Complex Hermitian decomposition requires conjugate transpose."
    }
  },
  "tensor-storage-cost": {
    "controls": [
      {
        "label": {
          "ko": "차수 d",
          "en": "Order d"
        },
        "min": 2,
        "max": 8,
        "initial": 6,
        "step": 1
      },
      {
        "label": {
          "ko": "축 크기 n",
          "en": "Axis size n"
        },
        "min": 2,
        "max": 8,
        "initial": 4,
        "step": 1
      },
      {
        "label": {
          "ko": "설정 랭크 r",
          "en": "Configured rank r"
        },
        "min": 1,
        "max": 4,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "원소 바이트 방식 0/1",
          "en": "Scalar-byte mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 1,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "고차 예제",
          "en": "Higher-order example"
        },
        "values": [
          6,
          4,
          2,
          1
        ]
      },
      {
        "name": {
          "ko": "작은 원본",
          "en": "Small dense tensor"
        },
        "values": [
          2,
          2,
          4,
          1
        ]
      },
      {
        "name": {
          "ko": "2바이트 저장",
          "en": "Two-byte storage"
        },
        "values": [
          6,
          4,
          2,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 차수 d · 세로: 저장 원소 수의 log₁₀ · 파랑: dense · 주황: TT",
      "en": "Horizontal: order d · Vertical: log₁₀ stored scalars · Blue: dense · Orange: TT"
    },
    "legend": [
      {
        "ko": "Dense",
        "en": "Dense"
      },
      {
        "ko": "TT",
        "en": "TT"
      }
    ],
    "note": {
      "ko": "훈련 상태·활성값·복원 버퍼는 제외합니다. 같은 r은 같은 표현력이나 같은 정확도를 뜻하지 않습니다.",
      "en": "Excludes training states, activations and reconstruction buffers. Equal r does not mean equal expressiveness or accuracy."
    }
  }
};

export const abstractAlgebraLabs:Record<string,FoundationLabSpec> = {
  "groups-rings-fields": {
    "controls": [
      {
        "label": {
          "ko": "법 n",
          "en": "Modulus n"
        },
        "min": 2,
        "max": 9,
        "initial": 5,
        "step": 1
      },
      {
        "label": {
          "ko": "원소 a",
          "en": "Element a"
        },
        "min": 0,
        "max": 8,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "소수 계수",
          "en": "Prime modulus"
        },
        "values": [
          5,
          2
        ]
      },
      {
        "name": {
          "ko": "영인자",
          "en": "Zero divisor"
        },
        "values": [
          6,
          2
        ]
      },
      {
        "name": {
          "ko": "영 원소",
          "en": "Zero element"
        },
        "values": [
          5,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 정수 입력 또는 평탄 지표 · 세로: 실제 계산값 · 표: 원소별 계산",
      "en": "Horizontal: integer input or flat index · Vertical: evaluated value · Table: entrywise calculations"
    },
    "legend": [],
    "note": {
      "ko": "선택한 유한 구조의 정확한 계산입니다. 각 절의 계수·작용·나눗셈 조건을 확인하세요.",
      "en": "Exact calculations in the selected finite structure. Check the stated scalar, action and division assumptions."
    }
  },
  "homomorphisms-isomorphisms": {
    "controls": [
      {
        "label": {
          "ko": "출발 법 m",
          "en": "Source modulus m"
        },
        "min": 2,
        "max": 8,
        "initial": 6,
        "step": 1
      },
      {
        "label": {
          "ko": "도착 법 n",
          "en": "Target modulus n"
        },
        "min": 2,
        "max": 8,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "배율 k",
          "en": "Multiplier k"
        },
        "min": 0,
        "max": 7,
        "initial": 1,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "자연스러운 사상",
          "en": "Natural map"
        },
        "values": [
          6,
          3,
          1
        ]
      },
      {
        "name": {
          "ko": "대표원 불일치",
          "en": "Inconsistent representatives"
        },
        "values": [
          3,
          6,
          1
        ]
      },
      {
        "name": {
          "ko": "동형",
          "en": "Isomorphism"
        },
        "values": [
          5,
          5,
          2
        ]
      }
    ],
    "axis": {
      "ko": "가로: 정수 입력 또는 평탄 지표 · 세로: 실제 계산값 · 표: 원소별 계산",
      "en": "Horizontal: integer input or flat index · Vertical: evaluated value · Table: entrywise calculations"
    },
    "legend": [],
    "note": {
      "ko": "선택한 유한 구조의 정확한 계산입니다. 각 절의 계수·작용·나눗셈 조건을 확인하세요.",
      "en": "Exact calculations in the selected finite structure. Check the stated scalar, action and division assumptions."
    }
  },
  "subgroups-quotients": {
    "controls": [
      {
        "label": {
          "ko": "법 n",
          "en": "Modulus n"
        },
        "min": 2,
        "max": 9,
        "initial": 6,
        "step": 1
      },
      {
        "label": {
          "ko": "생성원 h",
          "en": "Generator h"
        },
        "min": 0,
        "max": 8,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "짝수 부분군",
          "en": "Even subgroup"
        },
        "values": [
          6,
          2
        ]
      },
      {
        "name": {
          "ko": "영 부분군",
          "en": "Zero subgroup"
        },
        "values": [
          6,
          0
        ]
      },
      {
        "name": {
          "ko": "전체 부분군",
          "en": "Whole subgroup"
        },
        "values": [
          6,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: 정수 입력 또는 평탄 지표 · 세로: 실제 계산값 · 표: 원소별 계산",
      "en": "Horizontal: integer input or flat index · Vertical: evaluated value · Table: entrywise calculations"
    },
    "legend": [],
    "note": {
      "ko": "선택한 유한 구조의 정확한 계산입니다. 각 절의 계수·작용·나눗셈 조건을 확인하세요.",
      "en": "Exact calculations in the selected finite structure. Check the stated scalar, action and division assumptions."
    }
  },
  "polynomials-finite-fields": {
    "controls": [
      {
        "label": {
          "ko": "법 다항식 방식 0/1",
          "en": "Polynomial modulus mode 0/1"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "다항식 a 비트",
          "en": "Polynomial a bits"
        },
        "min": 0,
        "max": 7,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "다항식 b 비트",
          "en": "Polynomial b bits"
        },
        "min": 0,
        "max": 7,
        "initial": 4,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "기약 다항식",
          "en": "Irreducible modulus"
        },
        "values": [
          0,
          2,
          4
        ]
      },
      {
        "name": {
          "ko": "가약 다항식",
          "en": "Reducible modulus"
        },
        "values": [
          1,
          2,
          5
        ]
      },
      {
        "name": {
          "ko": "영 다항식",
          "en": "Zero polynomial"
        },
        "values": [
          0,
          0,
          4
        ]
      }
    ],
    "axis": {
      "ko": "가로: 정수 입력 또는 평탄 지표 · 세로: 실제 계산값 · 표: 원소별 계산",
      "en": "Horizontal: integer input or flat index · Vertical: evaluated value · Table: entrywise calculations"
    },
    "legend": [],
    "note": {
      "ko": "선택한 유한 구조의 정확한 계산입니다. 각 절의 계수·작용·나눗셈 조건을 확인하세요.",
      "en": "Exact calculations in the selected finite structure. Check the stated scalar, action and division assumptions."
    }
  },
  "modules-linear-actions": {
    "controls": [
      {
        "label": {
          "ko": "계수 법 n",
          "en": "Scalar modulus n"
        },
        "min": 2,
        "max": 7,
        "initial": 6,
        "step": 1
      },
      {
        "label": {
          "ko": "대각 계수 a",
          "en": "Diagonal coefficient a"
        },
        "min": 0,
        "max": 6,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "대각 계수 b",
          "en": "Diagonal coefficient b"
        },
        "min": 0,
        "max": 6,
        "initial": 3,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "환의 영인자",
          "en": "Ring zero divisors"
        },
        "values": [
          6,
          2,
          3
        ]
      },
      {
        "name": {
          "ko": "체 위 선형사상",
          "en": "Linear map over a field"
        },
        "values": [
          5,
          2,
          3
        ]
      },
      {
        "name": {
          "ko": "영 사상",
          "en": "Zero map"
        },
        "values": [
          5,
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: 정수 입력 또는 평탄 지표 · 세로: 실제 계산값 · 표: 원소별 계산",
      "en": "Horizontal: integer input or flat index · Vertical: evaluated value · Table: entrywise calculations"
    },
    "legend": [],
    "note": {
      "ko": "선택한 유한 구조의 정확한 계산입니다. 각 절의 계수·작용·나눗셈 조건을 확인하세요.",
      "en": "Exact calculations in the selected finite structure. Check the stated scalar, action and division assumptions."
    }
  },
  "algebraic-symmetries": {
    "controls": [
      {
        "label": {
          "ko": "회전 횟수 r",
          "en": "Quarter-turn count r"
        },
        "min": 0,
        "max": 3,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "반사 여부 f",
          "en": "Reflection flag f"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "점 방식 0~2",
          "en": "Point mode 0–2"
        },
        "min": 0,
        "max": 2,
        "initial": 0,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "꼭짓점",
          "en": "Corner"
        },
        "values": [
          1,
          0,
          0
        ]
      },
      {
        "name": {
          "ko": "축 위 점",
          "en": "Axis point"
        },
        "values": [
          1,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "중심",
          "en": "Center"
        },
        "values": [
          3,
          1,
          2
        ]
      }
    ],
    "axis": {
      "ko": "화살표: 선택한 군 원소의 궤도 순열 · 표: 여덟 군 원소의 작용",
      "en": "Arrows: orbit permutation under the selected element · Table: actions of all eight elements"
    },
    "legend": [],
    "note": {
      "ko": "선택한 유한 구조의 정확한 계산입니다. 각 절의 계수·작용·나눗셈 조건을 확인하세요.",
      "en": "Exact calculations in the selected finite structure. Check the stated scalar, action and division assumptions."
    }
  },
  "polynomial-systems": {
    "controls": [
      {
        "label": {
          "ko": "체 방식 0 F5 / 1 F7",
          "en": "Field mode 0 F5 / 1 F7"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "합 s",
          "en": "Sum s"
        },
        "min": 0,
        "max": 6,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "곱 q",
          "en": "Product q"
        },
        "min": 0,
        "max": 6,
        "initial": 2,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "두 순서 해",
          "en": "Two ordered solutions"
        },
        "values": [
          0,
          3,
          2
        ]
      },
      {
        "name": {
          "ko": "중근",
          "en": "Repeated root"
        },
        "values": [
          0,
          2,
          1
        ]
      },
      {
        "name": {
          "ko": "해 없음",
          "en": "No solution"
        },
        "values": [
          0,
          0,
          2
        ]
      }
    ],
    "axis": {
      "ko": "가로: 정수 입력 또는 평탄 지표 · 세로: 실제 계산값 · 표: 원소별 계산",
      "en": "Horizontal: integer input or flat index · Vertical: evaluated value · Table: entrywise calculations"
    },
    "legend": [],
    "note": {
      "ko": "선택한 유한 구조의 정확한 계산입니다. 각 절의 계수·작용·나눗셈 조건을 확인하세요.",
      "en": "Exact calculations in the selected finite structure. Check the stated scalar, action and division assumptions."
    }
  },
  "invariant-constructions": {
    "controls": [
      {
        "label": {
          "ko": "좌표 x",
          "en": "Coordinate x"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "좌표 y",
          "en": "Coordinate y"
        },
        "min": -2,
        "max": 2,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "특징 0 x / 1 x² / 2 xy",
          "en": "Feature 0 x / 1 x² / 2 xy"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "제곱 특징",
          "en": "Squared feature"
        },
        "values": [
          1,
          2,
          1
        ]
      },
      {
        "name": {
          "ko": "선형 특징",
          "en": "Linear feature"
        },
        "values": [
          1,
          2,
          0
        ]
      },
      {
        "name": {
          "ko": "교차 특징",
          "en": "Cross feature"
        },
        "values": [
          1,
          2,
          2
        ]
      }
    ],
    "axis": {
      "ko": "가로: 회전 번호 k · 파랑: 특징 값 · 주황: 군 평균",
      "en": "Horizontal: rotation index k · Blue: feature value · Orange: group average"
    },
    "legend": [],
    "note": {
      "ko": "선택한 유한 구조의 정확한 계산입니다. 각 절의 계수·작용·나눗셈 조건을 확인하세요.",
      "en": "Exact calculations in the selected finite structure. Check the stated scalar, action and division assumptions."
    }
  },
  "algebraic-representations-scope": {
    "controls": [
      {
        "label": {
          "ko": "행렬 a",
          "en": "Matrix a"
        },
        "min": -2,
        "max": 2,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "행렬 b",
          "en": "Matrix b"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "행렬 c",
          "en": "Matrix c"
        },
        "min": -2,
        "max": 2,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "행렬 d",
          "en": "Matrix d"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "일반 행렬",
          "en": "General matrix"
        },
        "values": [
          2,
          1,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "회전 등변",
          "en": "Rotation equivariant"
        },
        "values": [
          1,
          -1,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "모든 대칭 등변",
          "en": "All square symmetries"
        },
        "values": [
          1,
          0,
          0,
          1
        ]
      }
    ],
    "axis": {
      "ko": "가로: 행 우선 지표 · 파랑: 원본 M · 주황: C4 등변 사영",
      "en": "Horizontal: row-major index · Blue: original M · Orange: C4 equivariant projection"
    },
    "legend": [],
    "note": {
      "ko": "선택한 유한 구조의 정확한 계산입니다. 각 절의 계수·작용·나눗셈 조건을 확인하세요.",
      "en": "Exact calculations in the selected finite structure. Check the stated scalar, action and division assumptions."
    }
  }
};

export const probabilityLabs:Record<string,FoundationLabSpec> = {
  "probability-spaces-events": {
    "controls": [
      {
        "label": {
          "ko": "가중치 방식 0~2",
          "en": "Weight mode 0–2"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "사건 A 마스크",
          "en": "Event A mask"
        },
        "min": 0,
        "max": 15,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "사건 B 마스크",
          "en": "Event B mask"
        },
        "min": 0,
        "max": 15,
        "initial": 6,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "가중 사건",
          "en": "Weighted events"
        },
        "values": [
          1,
          3,
          6
        ]
      },
      {
        "name": {
          "ko": "서로소 사건",
          "en": "Disjoint events"
        },
        "values": [
          0,
          3,
          12
        ]
      },
      {
        "name": {
          "ko": "확률 0인 비공집합",
          "en": "Nonempty null events"
        },
        "values": [
          2,
          4,
          8
        ]
      }
    ],
    "axis": {
      "ko": "가로: 결과 ω=0,…,3 · 세로: 한 점의 확률 · 표: A,B 소속",
      "en": "Horizontal: outcome ω=0,…,3 · Vertical: point probability · Table: event membership"
    },
    "legend": [],
    "note": {
      "ko": "표시값은 반올림합니다. 본문의 가정과 정의되지 않는 조건을 확인하세요.",
      "en": "Readouts are rounded. Check the assumptions and undefined cases in the text."
    }
  },
  "conditional-probability-independence": {
    "controls": [
      {
        "label": {
          "ko": "A와 B 개수",
          "en": "Count A and B"
        },
        "min": 0,
        "max": 10,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "A 아님과 B 개수",
          "en": "Count not A and B"
        },
        "min": 0,
        "max": 10,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "A와 B 아님 개수",
          "en": "Count A and not B"
        },
        "min": 0,
        "max": 10,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "둘 다 아님 개수",
          "en": "Count neither"
        },
        "min": 0,
        "max": 10,
        "initial": 6,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "다른 조건부",
          "en": "Different conditionals"
        },
        "values": [
          1,
          2,
          1,
          6
        ]
      },
      {
        "name": {
          "ko": "독립 사건",
          "en": "Independent events"
        },
        "values": [
          1,
          2,
          2,
          4
        ]
      },
      {
        "name": {
          "ko": "조건 사건 확률 0",
          "en": "Zero-probability condition"
        },
        "values": [
          0,
          0,
          2,
          6
        ]
      }
    ],
    "axis": {
      "ko": "가로: 1_A · 세로: 1_B · 칸 숫자: 결합확률",
      "en": "Horizontal: 1_A · Vertical: 1_B · Cell numbers: joint probabilities"
    },
    "legend": [],
    "note": {
      "ko": "표시값은 반올림합니다. 본문의 가정과 정의되지 않는 조건을 확인하세요.",
      "en": "Readouts are rounded. Check the assumptions and undefined cases in the text."
    }
  },
  "mass-density-cdf": {
    "controls": [
      {
        "label": {
          "ko": "분포 방식 0 Bernoulli / 1 Uniform",
          "en": "Distribution 0 Bernoulli / 1 Uniform"
        },
        "min": 0,
        "max": 1,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "Uniform 폭 w",
          "en": "Uniform width w"
        },
        "min": 0.5,
        "max": 3,
        "initial": 0.5,
        "step": "any"
      },
      {
        "label": {
          "ko": "질의 x",
          "en": "Query x"
        },
        "min": -1,
        "max": 4,
        "initial": 0.25,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "높은 밀도",
          "en": "Density above one"
        },
        "values": [
          1,
          0.5,
          0.25
        ]
      },
      {
        "name": {
          "ko": "점프가 있는 분포",
          "en": "Distribution with jumps"
        },
        "values": [
          0,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "지지집합 밖",
          "en": "Outside support"
        },
        "values": [
          1,
          0.5,
          0.75
        ]
      }
    ],
    "axis": {
      "ko": "가로: x · 세로: F(x) · 빈 원: 제외 값 · 채운 원: 점프 뒤 값",
      "en": "Horizontal: x · Vertical: F(x) · Open circles: excluded values · Filled circles: values after jumps"
    },
    "legend": [],
    "note": {
      "ko": "표시값은 반올림합니다. 본문의 가정과 정의되지 않는 조건을 확인하세요.",
      "en": "Readouts are rounded. Check the assumptions and undefined cases in the text."
    }
  },
  "joint-marginal-conditional": {
    "controls": [
      {
        "label": {
          "ko": "개수 00",
          "en": "Count 00"
        },
        "min": 0,
        "max": 8,
        "initial": 3,
        "step": 1
      },
      {
        "label": {
          "ko": "개수 01",
          "en": "Count 01"
        },
        "min": 0,
        "max": 8,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "개수 10",
          "en": "Count 10"
        },
        "min": 0,
        "max": 8,
        "initial": 1,
        "step": 1
      },
      {
        "label": {
          "ko": "개수 11",
          "en": "Count 11"
        },
        "min": 0,
        "max": 8,
        "initial": 3,
        "step": 1
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "양의 공분산",
          "en": "Positive covariance"
        },
        "values": [
          3,
          1,
          1,
          3
        ]
      },
      {
        "name": {
          "ko": "독립 결합",
          "en": "Independent joint law"
        },
        "values": [
          1,
          2,
          2,
          4
        ]
      },
      {
        "name": {
          "ko": "비어 있는 조건 행",
          "en": "Empty conditioning row"
        },
        "values": [
          3,
          1,
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: X · 세로: Y · 칸 숫자: 결합확률 · 표: 주변확률",
      "en": "Horizontal: X · Vertical: Y · Cell numbers: joint mass · Table: marginals"
    },
    "legend": [],
    "note": {
      "ko": "표시값은 반올림합니다. 본문의 가정과 정의되지 않는 조건을 확인하세요.",
      "en": "Readouts are rounded. Check the assumptions and undefined cases in the text."
    }
  },
  "multivariate-gaussians": {
    "controls": [
      {
        "label": {
          "ko": "표준편차 σx",
          "en": "Standard deviation σx"
        },
        "min": 0.5,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "표준편차 σy",
          "en": "Standard deviation σy"
        },
        "min": 0.5,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "상관 ρ",
          "en": "Correlation ρ"
        },
        "min": -1,
        "max": 1,
        "initial": 0.5,
        "step": "any"
      },
      {
        "label": {
          "ko": "조건값 x",
          "en": "Condition value x"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "상관 Gaussian",
          "en": "Correlated Gaussian"
        },
        "values": [
          1,
          1,
          0.5,
          1
        ]
      },
      {
        "name": {
          "ko": "독립 Gaussian",
          "en": "Independent Gaussian"
        },
        "values": [
          1,
          1,
          0,
          1
        ]
      },
      {
        "name": {
          "ko": "특이 Gaussian",
          "en": "Singular Gaussian"
        },
        "values": [
          1,
          2,
          1,
          1
        ]
      }
    ],
    "axis": {
      "ko": "두 축은 같은 단위 비율입니다. 곡선: 단위원을 공분산 인자로 변환한 경계",
      "en": "Axes use equal unit scales. Curve: a unit circle transformed by a covariance factor"
    },
    "legend": [],
    "note": {
      "ko": "표시값은 반올림합니다. 본문의 가정과 정의되지 않는 조건을 확인하세요.",
      "en": "Readouts are rounded. Check the assumptions and undefined cases in the text."
    }
  },
  "random-variable-transformations": {
    "controls": [
      {
        "label": {
          "ko": "변환 방식 0 선형 / 1 제곱",
          "en": "Transform 0 linear / 1 square"
        },
        "min": 0,
        "max": 1,
        "initial": 0,
        "step": 1
      },
      {
        "label": {
          "ko": "정수 배율 a",
          "en": "Integer scale a"
        },
        "min": -2,
        "max": 2,
        "initial": 2,
        "step": 1
      },
      {
        "label": {
          "ko": "질의 y",
          "en": "Query y"
        },
        "min": -2,
        "max": 2,
        "initial": 1,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "두 배 확대",
          "en": "Double the scale"
        },
        "values": [
          0,
          2,
          1
        ]
      },
      {
        "name": {
          "ko": "두 역상 제곱",
          "en": "Two-branch square"
        },
        "values": [
          1,
          1,
          0.25
        ]
      },
      {
        "name": {
          "ko": "한 점으로 붕괴",
          "en": "Collapse to one point"
        },
        "values": [
          0,
          0,
          0
        ]
      }
    ],
    "axis": {
      "ko": "가로: y · 세로: 변환 뒤 누적분포 F_Y(y) · 표: 밀도와 점 질량",
      "en": "Horizontal: y · Vertical: transformed CDF F_Y(y) · Table: density and point mass"
    },
    "legend": [],
    "note": {
      "ko": "표시값은 반올림합니다. 본문의 가정과 정의되지 않는 조건을 확인하세요.",
      "en": "Readouts are rounded. Check the assumptions and undefined cases in the text."
    }
  },
  "laws-large-numbers-clt": {
    "controls": [
      {
        "label": {
          "ko": "표본 수 n",
          "en": "Sample count n"
        },
        "min": 1,
        "max": 30,
        "initial": 10,
        "step": 1
      },
      {
        "label": {
          "ko": "Bernoulli 확률 p",
          "en": "Bernoulli probability p"
        },
        "min": 0,
        "max": 1,
        "initial": 0.5,
        "step": "any"
      },
      {
        "label": {
          "ko": "편차 기준 ε",
          "en": "Deviation threshold ε"
        },
        "min": 0.1,
        "max": 1,
        "initial": 0.2,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "열 표본",
          "en": "Ten observations"
        },
        "values": [
          10,
          0.5,
          0.2
        ]
      },
      {
        "name": {
          "ko": "서른 표본",
          "en": "Thirty observations"
        },
        "values": [
          30,
          0.5,
          0.2
        ]
      },
      {
        "name": {
          "ko": "분산 0",
          "en": "Zero variance"
        },
        "values": [
          10,
          0,
          0.2
        ]
      }
    ],
    "axis": {
      "ko": "가로: 표본평균 k/n · 파랑: 이항 질량 · 주황: 연속성 보정 정규 구간 질량",
      "en": "Horizontal: sample mean k/n · Blue: binomial mass · Orange: continuity-corrected normal bin mass"
    },
    "legend": [],
    "note": {
      "ko": "표시값은 반올림합니다. 본문의 가정과 정의되지 않는 조건을 확인하세요.",
      "en": "Readouts are rounded. Check the assumptions and undefined cases in the text."
    }
  },
  "reparameterization-gradients": {
    "controls": [
      {
        "label": {
          "ko": "위치 μ",
          "en": "Location μ"
        },
        "min": -2,
        "max": 2,
        "initial": 0,
        "step": "any"
      },
      {
        "label": {
          "ko": "척도 σ",
          "en": "Scale σ"
        },
        "min": 0,
        "max": 2,
        "initial": 1,
        "step": "any"
      },
      {
        "label": {
          "ko": "고정 잡음 ε",
          "en": "Fixed noise ε"
        },
        "min": -2,
        "max": 2,
        "initial": 0,
        "step": "any"
      }
    ],
    "presets": [
      {
        "name": {
          "ko": "중앙 잡음",
          "en": "Central noise"
        },
        "values": [
          0,
          1,
          0
        ]
      },
      {
        "name": {
          "ko": "한 표본의 최적값",
          "en": "One sample at target"
        },
        "values": [
          0,
          1,
          1
        ]
      },
      {
        "name": {
          "ko": "척도 0",
          "en": "Zero scale"
        },
        "values": [
          1,
          0,
          2
        ]
      }
    ],
    "axis": {
      "ko": "가로: 고정 잡음 ε · 세로: 이차 표본 손실 · 표: 고정된 세 잡음 값",
      "en": "Horizontal: fixed noise ε · Vertical: quadratic sample loss · Table: three fixed noise values"
    },
    "legend": [],
    "note": {
      "ko": "표시값은 반올림합니다. 본문의 가정과 정의되지 않는 조건을 확인하세요.",
      "en": "Readouts are rounded. Check the assumptions and undefined cases in the text."
    }
  }
};
