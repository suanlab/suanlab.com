/** Bilingual preparations only; numerical evaluation stays inside workers. */
import {masterKinds,isMasterInput,snapshotMasterInput,masterInputKey,masterParameterKeys,type MasterKind,type MasterInput}from './learning-master-equation-protocol';
export {masterKinds,masterInputKey};export type {MasterKind,MasterInput};
export type MasterText={ko:string;en:string};
export const masterText=(ko:string,en:string):MasterText=>({ko,en});
export const isMasterKind=(v:string):v is MasterKind=>(masterKinds as readonly string[]).includes(v);
export const masterTitles:Record<MasterKind,MasterText>={
 'two-state-continuous-time-generator-euler':masterText('연속시간 생성자와 Euler 근사','Continuous-time generators and Euler approximation'),
 'poisson-uniformization-potential-events':masterText('Poisson 시계와 잠재 사건','Poisson clocks and potential events'),
 'state-dependent-holding-time-jump-occupation':masterText('체류시간과 방문·시간 점유','Holding times, visits and time occupation'),
};
export const masterPresets={
  "two-state-continuous-time-generator-euler": [
    {
      "id": "master-0-0",
      "label": {
        "ko": "기준:두 상태 완화",
        "en": "Reference: two-state relaxation"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 2,
          "backwardNumerator": 1,
          "rateDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialOneNumerator": 0,
          "initialDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "master-0-1",
      "label": {
        "ko": "음수가 되는 Euler",
        "en": "Negative Euler values"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 2,
          "backwardNumerator": 1,
          "rateDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialOneNumerator": 0,
          "initialDenominator": 1,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "master-0-2",
      "label": {
        "ko": "수축하지만 확률은 아님",
        "en": "Contracting but not stochastic"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 4,
          "backwardNumerator": 1,
          "rateDenominator": 1,
          "timeNumerator": 7,
          "timeDenominator": 20,
          "initialOneNumerator": 0,
          "initialDenominator": 1,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "master-0-3",
      "label": {
        "ko": "Euler 주기2",
        "en": "Period-two Euler"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 1,
          "backwardNumerator": 1,
          "rateDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "initialOneNumerator": 0,
          "initialDenominator": 1,
          "eulerSteps": 4
        }
      }
    },
    {
      "id": "master-0-4",
      "label": {
        "ko": "한 단계의 거짓 평형화",
        "en": "One-step artificial equilibration"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 2,
          "backwardNumerator": 1,
          "rateDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 3,
          "initialOneNumerator": 0,
          "initialDenominator": 1,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "master-0-5",
      "label": {
        "ko": "정상 초기값의 함정",
        "en": "Stationary initialization hides failure"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 2,
          "backwardNumerator": 1,
          "rateDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialOneNumerator": 2,
          "initialDenominator": 3,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "master-0-6",
      "label": {
        "ko": "상태0으로 흡수",
        "en": "Absorption at zero"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 0,
          "backwardNumerator": 2,
          "rateDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialOneNumerator": 1,
          "initialDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "master-0-7",
      "label": {
        "ko": "상태1로 흡수",
        "en": "Absorption at one"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 2,
          "backwardNumerator": 0,
          "rateDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialOneNumerator": 0,
          "initialDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "master-0-8",
      "label": {
        "ko": "영 전이율·정상 가족",
        "en": "Zero rates and stationary family"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 0,
          "backwardNumerator": 0,
          "rateDenominator": 1,
          "timeNumerator": 3,
          "timeDenominator": 1,
          "initialOneNumerator": 1,
          "initialDenominator": 3,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "master-0-9",
      "label": {
        "ko": "물리적 시간0",
        "en": "Zero physical time"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 2,
          "backwardNumerator": 1,
          "rateDenominator": 1,
          "timeNumerator": 0,
          "timeDenominator": 1,
          "initialOneNumerator": 3,
          "initialDenominator": 4,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "master-0-10",
      "label": {
        "ko": "반올림 뒤의 비영 전류",
        "en": "Nonzero current behind rounding"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 5,
          "backwardNumerator": 5,
          "rateDenominator": 1,
          "timeNumerator": 10,
          "timeDenominator": 1,
          "initialOneNumerator": 0,
          "initialDenominator": 1,
          "eulerSteps": 128
        }
      }
    },
    {
      "id": "master-0-11",
      "label": {
        "ko": "아주 작은 양의 변화",
        "en": "Tiny positive change"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 1,
          "backwardNumerator": 2,
          "rateDenominator": 1000000,
          "timeNumerator": 1,
          "timeDenominator": 1000000,
          "initialOneNumerator": 0,
          "initialDenominator": 1,
          "eulerSteps": 128
        }
      }
    },
    {
      "id": "master-0-12",
      "label": {
        "ko": "전이율·시간 재척도",
        "en": "Rate/time rescaling"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 4,
          "backwardNumerator": 2,
          "rateDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 2,
          "initialOneNumerator": 0,
          "initialDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "master-0-13",
      "label": {
        "ko": "음의 순전류",
        "en": "Negative net current"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 1,
          "backwardNumerator": 4,
          "rateDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialOneNumerator": 1,
          "initialDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "master-0-14",
      "label": {
        "ko": "복귀와 무점프",
        "en": "Return versus no jump"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 1,
          "backwardNumerator": 1,
          "rateDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialOneNumerator": 0,
          "initialDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "master-0-15",
      "label": {
        "ko": "유리수 정상 초기값",
        "en": "Rational stationary initialization"
      },
      "input": {
        "kind": "two-state-continuous-time-generator-euler",
        "parameters": {
          "forwardNumerator": 1,
          "backwardNumerator": 2,
          "rateDenominator": 2,
          "timeNumerator": 3,
          "timeDenominator": 2,
          "initialOneNumerator": 1,
          "initialDenominator": 3,
          "eulerSteps": 16
        }
      }
    }
  ],
  "poisson-uniformization-potential-events": [
    {
      "id": "master-1-0",
      "label": {
        "ko": "단방향 고리",
        "en": "One-way cycle"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 1,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 12
        }
      }
    },
    {
      "id": "master-1-1",
      "label": {
        "ko": "주기적 내장 사슬·긴 시간",
        "en": "Periodic embedded chain, later time"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 1,
          "clockDenominator": 1,
          "timeNumerator": 10,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 80
        }
      }
    },
    {
      "id": "master-1-2",
      "label": {
        "ko": "같은 생성자·두 배 시계",
        "en": "Same generator, doubled clock"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 2,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 24
        }
      }
    },
    {
      "id": "master-1-3",
      "label": {
        "ko": "사건0번까지만 합산",
        "en": "Retain only zero events"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 1,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 0
        }
      }
    },
    {
      "id": "master-1-4",
      "label": {
        "ko": "대칭 양방향 전이율",
        "en": "Symmetric directional rates"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 1,
          "rateDenominator": 1,
          "clockNumerator": 2,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 12
        }
      }
    },
    {
      "id": "master-1-5",
      "label": {
        "ko": "반대 정상 순환",
        "en": "Opposite stationary circulation"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 4,
          "rateDenominator": 1,
          "clockNumerator": 5,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 40
        }
      }
    },
    {
      "id": "master-1-6",
      "label": {
        "ko": "비균일 초기 무게",
        "en": "Nonuniform initial weights"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 1,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 2,
          "initialOneWeight": 3,
          "initialTwoWeight": 5,
          "poissonCutoff": 24
        }
      }
    },
    {
      "id": "master-1-7",
      "label": {
        "ko": "정상 초기값·불완전 합",
        "en": "Stationary initialization, partial sum"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 1,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 1,
          "initialTwoWeight": 1,
          "poissonCutoff": 0
        }
      }
    },
    {
      "id": "master-1-8",
      "label": {
        "ko": "물리적 시간0",
        "en": "Zero physical time"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 1,
          "clockDenominator": 1,
          "timeNumerator": 0,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 20
        }
      }
    },
    {
      "id": "master-1-9",
      "label": {
        "ko": "생성자와 시계가 모두0",
        "en": "Zero generator and zero clock"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 0,
          "counterclockwiseNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 0,
          "clockDenominator": 1,
          "timeNumerator": 10,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 0
        }
      }
    },
    {
      "id": "master-1-10",
      "label": {
        "ko": "허구 사건만 발생",
        "en": "Fictitious events only"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 0,
          "counterclockwiseNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 2,
          "clockDenominator": 1,
          "timeNumerator": 3,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 24
        }
      }
    },
    {
      "id": "master-1-11",
      "label": {
        "ko": "큰 평균·부족한 절단",
        "en": "Large mean, insufficient cutoff"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 5,
          "counterclockwiseNumerator": 5,
          "rateDenominator": 1,
          "clockNumerator": 10,
          "clockDenominator": 1,
          "timeNumerator": 10,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 0
        }
      }
    },
    {
      "id": "master-1-12",
      "label": {
        "ko": "큰 평균·충분한 절단",
        "en": "Large mean, adequate cutoff"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 5,
          "counterclockwiseNumerator": 5,
          "rateDenominator": 1,
          "clockNumerator": 10,
          "clockDenominator": 1,
          "timeNumerator": 10,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 256
        }
      }
    },
    {
      "id": "master-1-13",
      "label": {
        "ko": "뺄셈 없이 작은 꼬리",
        "en": "Tiny tail without subtraction"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 0,
          "rateDenominator": 1000000,
          "clockNumerator": 1,
          "clockDenominator": 1000000,
          "timeNumerator": 1,
          "timeDenominator": 1000000,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-1-14",
      "label": {
        "ko": "표시 범위 밖 양의 꼬리",
        "en": "Positive tail outside display range"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 0,
          "rateDenominator": 1000000,
          "clockNumerator": 1,
          "clockDenominator": 1000000,
          "timeNumerator": 1,
          "timeDenominator": 1000000,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 256
        }
      }
    },
    {
      "id": "master-1-15",
      "label": {
        "ko": "유리수 시계·시간",
        "en": "Rational clock and time"
      },
      "input": {
        "kind": "poisson-uniformization-potential-events",
        "parameters": {
          "clockwiseNumerator": 1,
          "counterclockwiseNumerator": 1,
          "rateDenominator": 3,
          "clockNumerator": 5,
          "clockDenominator": 4,
          "timeNumerator": 7,
          "timeDenominator": 3,
          "initialZeroWeight": 1,
          "initialOneWeight": 2,
          "initialTwoWeight": 1,
          "poissonCutoff": 32
        }
      }
    }
  ],
  "state-dependent-holding-time-jump-occupation": [
    {
      "id": "master-2-0",
      "label": {
        "ko": "체류율1·2·4",
        "en": "Holding rates one, two, four"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 1,
          "rateOneNumerator": 2,
          "rateTwoNumerator": 4,
          "rateDenominator": 1,
          "clockNumerator": 4,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-1",
      "label": {
        "ko": "균일 방문형 초기값",
        "en": "Uniform visit-style initialization"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 1,
          "rateOneNumerator": 2,
          "rateTwoNumerator": 4,
          "rateDenominator": 1,
          "clockNumerator": 4,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 1,
          "initialTwoWeight": 1,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-2",
      "label": {
        "ko": "시간 정상 초기값4:2:1",
        "en": "Time-stationary initialization4:2:1"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 1,
          "rateOneNumerator": 2,
          "rateTwoNumerator": 4,
          "rateDenominator": 1,
          "clockNumerator": 4,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 4,
          "initialOneWeight": 2,
          "initialTwoWeight": 1,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-3",
      "label": {
        "ko": "같은 이탈률",
        "en": "Equal exit rates"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 1,
          "rateOneNumerator": 1,
          "rateTwoNumerator": 1,
          "rateDenominator": 1,
          "clockNumerator": 1,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-4",
      "label": {
        "ko": "같은 생성자·더 빠른 시계",
        "en": "Same generator, faster clock"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 1,
          "rateOneNumerator": 2,
          "rateTwoNumerator": 4,
          "rateDenominator": 1,
          "clockNumerator": 8,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 64
        }
      }
    },
    {
      "id": "master-2-5",
      "label": {
        "ko": "절반 전이율·두 배 시간",
        "en": "Half rates, doubled time"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 1,
          "rateOneNumerator": 2,
          "rateTwoNumerator": 4,
          "rateDenominator": 2,
          "clockNumerator": 2,
          "clockDenominator": 1,
          "timeNumerator": 2,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-6",
      "label": {
        "ko": "과도 상태에서 흡수",
        "en": "Absorption from a transient state"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 0,
          "rateOneNumerator": 2,
          "rateTwoNumerator": 4,
          "rateDenominator": 1,
          "clockNumerator": 4,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 0,
          "initialOneWeight": 1,
          "initialTwoWeight": 0,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-7",
      "label": {
        "ko": "이미 흡수된 초기값",
        "en": "Already absorbed initialization"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 0,
          "rateOneNumerator": 2,
          "rateTwoNumerator": 4,
          "rateDenominator": 1,
          "clockNumerator": 4,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-8",
      "label": {
        "ko": "두 흡수 상태",
        "en": "Two absorbing states"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 0,
          "rateOneNumerator": 2,
          "rateTwoNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 4,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 2,
          "initialOneWeight": 3,
          "initialTwoWeight": 5,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-9",
      "label": {
        "ko": "흡수 정상 가족의 한 법칙",
        "en": "One law in the absorbing family"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 0,
          "rateOneNumerator": 2,
          "rateTwoNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 4,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 2,
          "initialOneWeight": 0,
          "initialTwoWeight": 5,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-10",
      "label": {
        "ko": "모든 전이율0",
        "en": "All exit rates zero"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 0,
          "rateOneNumerator": 0,
          "rateTwoNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 0,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-11",
      "label": {
        "ko": "허구 사건만 있는 체류",
        "en": "Holding with fictitious events only"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 0,
          "rateOneNumerator": 0,
          "rateTwoNumerator": 0,
          "rateDenominator": 1,
          "clockNumerator": 2,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 3,
          "initialTwoWeight": 1,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-12",
      "label": {
        "ko": "물리적 시간0",
        "en": "Zero physical time"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 1,
          "rateOneNumerator": 2,
          "rateTwoNumerator": 4,
          "rateDenominator": 1,
          "clockNumerator": 4,
          "clockDenominator": 1,
          "timeNumerator": 0,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 32
        }
      }
    },
    {
      "id": "master-2-13",
      "label": {
        "ko": "매우 다른 체류시간",
        "en": "Very different holding times"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 1,
          "rateOneNumerator": 1000000,
          "rateTwoNumerator": 500000,
          "rateDenominator": 200000,
          "clockNumerator": 5,
          "clockDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 64
        }
      }
    },
    {
      "id": "master-2-14",
      "label": {
        "ko": "큰 평균·부족한 절단",
        "en": "Large mean, insufficient cutoff"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 5,
          "rateOneNumerator": 5,
          "rateTwoNumerator": 5,
          "rateDenominator": 1,
          "clockNumerator": 10,
          "clockDenominator": 1,
          "timeNumerator": 10,
          "timeDenominator": 1,
          "initialZeroWeight": 1,
          "initialOneWeight": 0,
          "initialTwoWeight": 0,
          "poissonCutoff": 0
        }
      }
    },
    {
      "id": "master-2-15",
      "label": {
        "ko": "유리수 시간·초기 무게",
        "en": "Rational time and initial weights"
      },
      "input": {
        "kind": "state-dependent-holding-time-jump-occupation",
        "parameters": {
          "rateZeroNumerator": 1,
          "rateOneNumerator": 2,
          "rateTwoNumerator": 4,
          "rateDenominator": 3,
          "clockNumerator": 7,
          "clockDenominator": 3,
          "timeNumerator": 3,
          "timeDenominator": 2,
          "initialZeroWeight": 1,
          "initialOneWeight": 2,
          "initialTwoWeight": 4,
          "poissonCutoff": 40
        }
      }
    }
  ]
} as unknown as Record<MasterKind,{id:string;label:MasterText;input:MasterInput}[]>;

export function masterPresetInput(kind:MasterKind,index:number):MasterInput{const p=masterPresets[kind][index];if(!Number.isInteger(index)||!p)throw new RangeError('Unknown master-equation preset');return snapshotMasterInput(p.input);}
const names:Record<string,MasterText>={
 forwardNumerator:masterText('0→1 전이율 분자','Rate0→1 numerator'),backwardNumerator:masterText('1→0 전이율 분자','Rate1→0 numerator'),rateDenominator:masterText('전이율 공통 분모','Common rate denominator'),
 timeNumerator:masterText('물리적 시간 분자','Physical-time numerator'),timeDenominator:masterText('물리적 시간 분모','Physical-time denominator'),initialOneNumerator:masterText('초기 상태1 확률 분자','Initial state-one probability numerator'),initialDenominator:masterText('초기 확률 분모','Initial probability denominator'),eulerSteps:masterText('같은 시간 구간의 Euler 단계 수','Euler steps over the same duration'),
 clockwiseNumerator:masterText('시계 방향 전이율 분자','Clockwise-rate numerator'),counterclockwiseNumerator:masterText('반시계 방향 전이율 분자','Counterclockwise-rate numerator'),clockNumerator:masterText('잠재 시계 전이율 분자','Potential-clock rate numerator'),clockDenominator:masterText('잠재 시계 전이율 분모','Potential-clock rate denominator'),
 initialZeroWeight:masterText('초기 상태0 무게','Initial state-zero weight'),initialOneWeight:masterText('초기 상태1 무게','Initial state-one weight'),initialTwoWeight:masterText('초기 상태2 무게','Initial state-two weight'),poissonCutoff:masterText('보존할 최대 Poisson 횟수 M','Maximum retained Poisson count M'),
 rateZeroNumerator:masterText('상태0 이탈률 분자','State-zero exit-rate numerator'),rateOneNumerator:masterText('상태1 이탈률 분자','State-one exit-rate numerator'),rateTwoNumerator:masterText('상태2 이탈률 분자','State-two exit-rate numerator'),
};
export function masterFields(input:MasterInput){return masterParameterKeys[masterKinds.indexOf(input.kind)].map(id=>({id,label:names[id],value:(input.parameters as unknown as Record<string,number>)[id],minimum:id.endsWith('Denominator')||id==='eulerSteps'?1:0,maximum:id==='eulerSteps'?128:id==='poissonCutoff'?256:1000000}));}
export function editMasterDraft(input:MasterInput,id:string,raw:string):MasterInput|null{const d=masterFields(input).find(x=>x.id===id),text=raw.trim();if(!d||!/^\d{1,7}$/.test(text))return null;const value=Number(text);if(value<d.minimum||value>d.maximum)return null;const next={kind:input.kind,parameters:{...input.parameters,[id]:value}};return isMasterInput(next)?snapshotMasterInput(next):null;}
export function advanceMasterTime(input:MasterInput):MasterInput|null{const p=input.parameters,maximum=Math.min(1000000,10*p.timeDenominator);if(p.timeNumerator>=maximum)return null;return editMasterDraft(input,'timeNumerator',String(Math.min(maximum,p.timeNumerator+p.timeDenominator)));}
export function rewindMasterTime(input:MasterInput):MasterInput{return editMasterDraft(input,'timeNumerator','0')!;}
