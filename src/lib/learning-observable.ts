/** Authored fixed preparations; all calculations stay in Workers. */
import {observableKinds,snapshotObservableInput,observableInputKey,type ObservableInput} from './learning-observable-protocol';
import {parseNumberDraft} from './learning-number-input';
export {observableKinds,observableInputKey};export type {ObservableInput};export type ObservableKind=typeof observableKinds[number];
export const isObservableKind=(x:string):x is ObservableKind=>(observableKinds as readonly string[]).includes(x);
export const observableText=(ko:string,en:string)=>({ko,en});const t=observableText;
export const observableTitles={
 [observableKinds[0]]:t('Hermitian 스펙트럼·Born 확률·기대값','Hermitian spectra, Born probabilities and expectation'),
 [observableKinds[1]]:t('축퇴 사영과 순차 관측','Degenerate projectors and sequential observation'),
};
const preparations:Record<ObservableKind,{id:string;label:{ko:string;en:string};input:ObservableInput}[]>={
  "hermitian-spectrum-born-expectation": [
    {
      "id": "Z with imaginary relative phase",
      "label": {
        "ko": "Z와 허수 상대 위상",
        "en": "Z and imaginary relative phase"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 1,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1
        }
      }
    },
    {
      "id": "Z eigenstate",
      "label": {
        "ko": "Z 고유상태",
        "en": "Z eigenstate"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1
        }
      }
    },
    {
      "id": "X eigenstate",
      "label": {
        "ko": "X 고유상태",
        "en": "X eigenstate"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1
        }
      }
    },
    {
      "id": "Y eigenstate",
      "label": {
        "ko": "Y 고유상태",
        "en": "Y eigenstate"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 1,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 1,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1
        }
      }
    },
    {
      "id": "fully degenerate zero",
      "label": {
        "ko": "같은 평균·영 연산자",
        "en": "Same mean, zero operator"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 1,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 0,
          "gapDenominator": 1
        }
      }
    },
    {
      "id": "fully degenerate nonzero",
      "label": {
        "ko": "완전 축퇴7/3",
        "en": "Fully degenerate7/3"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 1,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 0,
          "centerNumerator": 7,
          "centerDenominator": 3,
          "gapNumerator": 0,
          "gapDenominator": 1
        }
      }
    },
    {
      "id": "negative gap",
      "label": {
        "ko": "음수 간격과 정렬",
        "en": "Negative gap and sorting"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 1,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": -3,
          "gapDenominator": 7
        }
      }
    },
    {
      "id": "shift and scale",
      "label": {
        "ko": "유리수 중심·간격",
        "en": "Rational center and gap"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 1,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 0,
          "centerNumerator": -2,
          "centerDenominator": 3,
          "gapNumerator": 5,
          "gapDenominator": 11
        }
      }
    },
    {
      "id": "passive complex basis",
      "label": {
        "ko": "수동 복소 기저 변경",
        "en": "Passive complex basis change"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 1,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "basisReal0": 2,
          "basisImag0": 3,
          "basisReal1": -4,
          "basisImag1": 5,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1
        }
      }
    },
    {
      "id": "global imaginary phase",
      "label": {
        "ko": "전역 위상만 변경",
        "en": "Global phase only"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 0,
          "stateImag0": 1,
          "stateReal1": -1,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1
        }
      }
    },
    {
      "id": "small first state coordinate",
      "label": {
        "ko": "작지만 영 아닌 확률",
        "en": "Small nonzero probability"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 1000000,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1
        }
      }
    },
    {
      "id": "complex raw state",
      "label": {
        "ko": "복소 원시 준비",
        "en": "Complex raw preparation"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": -3,
          "stateImag0": 2,
          "stateReal1": 5,
          "stateImag1": -7,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1
        }
      }
    },
    {
      "id": "imaginary passive basis",
      "label": {
        "ko": "허수 성분의 수동 기저",
        "en": "Imaginary passive basis"
      },
      "input": {
        "kind": "hermitian-spectrum-born-expectation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 1,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 0,
          "basisImag1": 1,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1
        }
      }
    }
  ],
  "degenerate-projectors-sequential-observation": [
    {
      "id": "degenerate coherence preserved",
      "label": {
        "ko": "축퇴 공간의 중첩 보존",
        "en": "Coherence in a degenerate space"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 0,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "stateReal2": 1,
          "stateImag2": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisAReal2": 0,
          "axisAImag2": 0,
          "axisBReal0": 0,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "axisBReal2": 1,
          "axisBImag2": 0,
          "aPositiveNumerator": 1,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": 2,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": 1,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 0,
          "bNegativeDenominator": 1
        }
      }
    },
    {
      "id": "rank-one zero complementary branch",
      "label": {
        "ko": "확률0인 여공간 분기",
        "en": "Zero-probability complementary branch"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "stateReal2": 0,
          "stateImag2": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisAReal2": 0,
          "axisAImag2": 0,
          "axisBReal0": 0,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "axisBReal2": 1,
          "axisBImag2": 0,
          "aPositiveNumerator": 1,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": 2,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": 1,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 0,
          "bNegativeDenominator": 1
        }
      }
    },
    {
      "id": "imaginary coherence",
      "label": {
        "ko": "허수 coherence",
        "en": "Imaginary coherence"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 0,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "stateReal2": 0,
          "stateImag2": 1,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisAReal2": 0,
          "axisAImag2": 0,
          "axisBReal0": 0,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "axisBReal2": 0,
          "axisBImag2": 1,
          "aPositiveNumerator": 1,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": 2,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": 1,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 0,
          "bNegativeDenominator": 1
        }
      }
    },
    {
      "id": "order-dependent first plane",
      "label": {
        "ko": "관측 순서의 차이",
        "en": "Measurement order matters"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "stateReal2": 0,
          "stateImag2": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisAReal2": 0,
          "axisAImag2": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "axisBReal2": 0,
          "axisBImag2": 0,
          "aPositiveNumerator": 1,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": 2,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": 1,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 0,
          "bNegativeDenominator": 1
        }
      }
    },
    {
      "id": "same state statistics do not imply commuting",
      "label": {
        "ko": "한 준비가 가리는 비가환성",
        "en": "Noncommutativity hidden by one preparation"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 0,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "stateReal2": 1,
          "stateImag2": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisAReal2": 0,
          "axisAImag2": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "axisBReal2": 0,
          "axisBImag2": 0,
          "aPositiveNumerator": 1,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": 2,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": 1,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 0,
          "bNegativeDenominator": 1
        }
      }
    },
    {
      "id": "fully degenerate A",
      "label": {
        "ko": "A 고유값 병합",
        "en": "Merge A eigenvalues"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 0,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "stateReal2": 1,
          "stateImag2": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisAReal2": 0,
          "axisAImag2": 0,
          "axisBReal0": 0,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "axisBReal2": 1,
          "axisBImag2": 0,
          "aPositiveNumerator": 1,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": 1,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": 1,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 0,
          "bNegativeDenominator": 1
        }
      }
    },
    {
      "id": "both fully degenerate",
      "label": {
        "ko": "두 관측량 모두 스칼라",
        "en": "Both observables are scalar"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 0,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "stateReal2": 1,
          "stateImag2": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisAReal2": 0,
          "axisAImag2": 0,
          "axisBReal0": 0,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "axisBReal2": 1,
          "axisBImag2": 0,
          "aPositiveNumerator": 1,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": 1,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": 1,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 1,
          "bNegativeDenominator": 1
        }
      }
    },
    {
      "id": "negative eigenvalue sorting",
      "label": {
        "ko": "음수 고유값 정렬",
        "en": "Sort negative eigenvalues"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 0,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "stateReal2": 1,
          "stateImag2": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisAReal2": 0,
          "axisAImag2": 0,
          "axisBReal0": 0,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "axisBReal2": 1,
          "axisBImag2": 0,
          "aPositiveNumerator": -3,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": -7,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": -2,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 0,
          "bNegativeDenominator": 1
        }
      }
    },
    {
      "id": "complex rank-one projector",
      "label": {
        "ko": "복소 rank1 고유공간",
        "en": "Complex rank-one eigenspace"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 0,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "stateReal2": 1,
          "stateImag2": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 1,
          "axisAReal2": 1,
          "axisAImag2": -1,
          "axisBReal0": 0,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "axisBReal2": 1,
          "axisBImag2": 0,
          "aPositiveNumerator": 1,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": 2,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": 1,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 0,
          "bNegativeDenominator": 1
        }
      }
    },
    {
      "id": "small rank-one probability",
      "label": {
        "ko": "작지만 영 아닌 분기",
        "en": "Small nonzero branch"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 1000000,
          "stateImag1": 0,
          "stateReal2": 1000000,
          "stateImag2": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisAReal2": 0,
          "axisAImag2": 0,
          "axisBReal0": 0,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "axisBReal2": 1,
          "axisBImag2": 0,
          "aPositiveNumerator": 1,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": 2,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": 1,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 0,
          "bNegativeDenominator": 1
        }
      }
    },
    {
      "id": "commuting identical axes",
      "label": {
        "ko": "같은 고유공간",
        "en": "Identical eigenspaces"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 0,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "stateReal2": 1,
          "stateImag2": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisAReal2": 0,
          "axisAImag2": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 0,
          "axisBImag1": 0,
          "axisBReal2": 0,
          "axisBImag2": 0,
          "aPositiveNumerator": 1,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": 2,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": 1,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 0,
          "bNegativeDenominator": 1
        }
      }
    },
    {
      "id": "complex second projector",
      "label": {
        "ko": "복소 후속 관측",
        "en": "Complex subsequent observation"
      },
      "input": {
        "kind": "degenerate-projectors-sequential-observation",
        "parameters": {
          "stateReal0": 0,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "stateReal2": 1,
          "stateImag2": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisAReal2": 0,
          "axisAImag2": 0,
          "axisBReal0": -2,
          "axisBImag0": 1,
          "axisBReal1": 3,
          "axisBImag1": -2,
          "axisBReal2": 1,
          "axisBImag2": 1,
          "aPositiveNumerator": 1,
          "aPositiveDenominator": 1,
          "aNegativeNumerator": 2,
          "aNegativeDenominator": 1,
          "bPositiveNumerator": 1,
          "bPositiveDenominator": 1,
          "bNegativeNumerator": 0,
          "bNegativeDenominator": 1
        }
      }
    }
  ]
};
export const observablePresets=Object.fromEntries(observableKinds.map(kind=>[kind,preparations[kind].map(p=>({...p,input:snapshotObservableInput(p.input)}))])) as typeof preparations;
export function observablePresetInput(kind:ObservableKind,index:number){const p=observablePresets[kind][index];if(!Number.isInteger(index)||!p)throw new RangeError('Unknown observable preparation');return snapshotObservableInput(p.input);}
export function observableFields(input:ObservableInput){return Object.entries(input.parameters).map(([id,value])=>{
 let label=t(id,id);const v=/^(state|axisA|axisB|axis|basis)(Real|Imag)([012])$/.exec(id),r=/^(center|gap|aPositive|aNegative|bPositive|bNegative)(Numerator|Denominator)$/.exec(id);
 if(v){const names={state:t('원시 상태','Raw state'),axis:t('관측 고유공간 방향','Observable eigenspace direction'),axisA:t('A의 rank1 방향','A rank-one direction'),axisB:t('B의 rank1 방향','B rank-one direction'),basis:t('수동 기저 열','Passive basis column')},name=names[v[1] as keyof typeof names];label=t(`${name.ko} ${v[3]} · ${v[2]==='Real'?'실수':'허수'}`,`${name.en} ${v[3]} · ${v[2]==='Real'?'real':'imaginary'}`);}
 if(r){const names={center:t('중심 a','Center a'),gap:t('부호 있는 간격 b','Signed gap b'),aPositive:t('A · P 고유값 λ','A · P eigenvalue λ'),aNegative:t('A · Q 고유값 μ','A · Q eigenvalue μ'),bPositive:t('B · rank1 고유값','B · rank-one eigenvalue'),bNegative:t('B · rank2 고유값','B · rank-two eigenvalue')},name=names[r[1] as keyof typeof names];label=t(`${name.ko} · ${r[2]==='Numerator'?'분자':'분모'}`,`${name.en} · ${r[2]==='Numerator'?'numerator':'denominator'}`);}
 return{id,value,label};
});}
export function editObservableField(input:ObservableInput,id:string,raw:string):ObservableInput|null{if(!Object.hasOwn(input.parameters,id))return null;const value=parseNumberDraft(raw,-1000000,1000000,true);if(value===null)return null;try{return snapshotObservableInput({...input,parameters:{...input.parameters,[id]:value}} as ObservableInput);}catch{return null;}}
