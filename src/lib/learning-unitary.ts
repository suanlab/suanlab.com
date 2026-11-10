/** Authored preparations and bilingual controls; calculation engines stay in Workers. */
import {unitaryKinds,snapshotUnitaryInput,unitaryInputKey,type UnitaryInput} from './learning-unitary-protocol';
import {parseNumberDraft} from './learning-number-input';
export {unitaryKinds,unitaryInputKey};export type {UnitaryInput};export type UnitaryKind=typeof unitaryKinds[number];
export const isUnitaryKind=(value:string):value is UnitaryKind=>(unitaryKinds as readonly string[]).includes(value);
export const unitaryText=(ko:string,en:string)=>({ko,en});const t=unitaryText;
export const unitaryTitles={
 [unitaryKinds[0]]:t('스펙트럼과 Schrödinger 시간 진화','Spectral Schrödinger time evolution'),
 [unitaryKinds[1]]:t('Hamiltonian pulse와 시간 순서','Hamiltonian pulses and time ordering'),
};
const preparations:Record<UnitaryKind,{id:string;label:{ko:string;en:string};input:UnitaryInput}[]>={
  "two-level-spectral-schrodinger-evolution": [
    {
      "id": "X evolution",
      "label": {
        "ko": "X 진화와 전이",
        "en": "X evolution"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 16
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
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 0,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 16
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
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 16
        }
      }
    },
    {
      "id": "imaginary relative phase",
      "label": {
        "ko": "허수 상대 위상",
        "en": "Imaginary relative phase"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 1,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 16
        }
      }
    },
    {
      "id": "energy shift only",
      "label": {
        "ko": "공통 에너지 이동",
        "en": "Energy shift only"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 1,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 16
        }
      }
    },
    {
      "id": "hbar twice as large",
      "label": {
        "ko": "ℏ 수치 두 배",
        "en": "Hbar twice as large"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 2,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 16
        }
      }
    },
    {
      "id": "hbar half as large",
      "label": {
        "ko": "ℏ 수치 절반",
        "en": "Hbar half as large"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 2,
          "sampleSteps": 4,
          "eulerSteps": 16
        }
      }
    },
    {
      "id": "fine Euler grid",
      "label": {
        "ko": "촘촘한 Euler 격자",
        "en": "Fine Euler grid"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 64
        }
      }
    },
    {
      "id": "single Euler step",
      "label": {
        "ko": "Euler 한 걸음",
        "en": "Single Euler step"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 1,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "scalar Hamiltonian",
      "label": {
        "ko": "Scalar Hamiltonian",
        "en": "Scalar Hamiltonian"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 2,
          "centerDenominator": 1,
          "gapNumerator": 0,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 16
        }
      }
    },
    {
      "id": "zero duration",
      "label": {
        "ko": "영 지속 시간",
        "en": "Zero duration"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 0,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 16
        }
      }
    },
    {
      "id": "complex Hamiltonian axis",
      "label": {
        "ko": "복소 Hamiltonian 방향",
        "en": "Complex Hamiltonian axis"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 2,
          "axisImag0": 3,
          "axisReal1": -4,
          "axisImag1": 5,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 16
        }
      }
    },
    {
      "id": "negative gap",
      "label": {
        "ko": "음수 에너지 간격",
        "en": "Negative gap"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": -1,
          "gapDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 16
        }
      }
    },
    {
      "id": "near pi revival",
      "label": {
        "ko": "가까운 ray 재귀",
        "en": "Near pi revival"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1,
          "gapDenominator": 1,
          "timeNumerator": 355,
          "timeDenominator": 113,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4,
          "eulerSteps": 16
        }
      }
    },
    {
      "id": "Euler overflow retained exactly",
      "label": {
        "ko": "큰 Euler 성장",
        "en": "Euler overflow retained exactly"
      },
      "input": {
        "kind": "two-level-spectral-schrodinger-evolution",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisReal0": 1,
          "axisImag0": 0,
          "axisReal1": 1,
          "axisImag1": 0,
          "centerNumerator": 0,
          "centerDenominator": 1,
          "gapNumerator": 1000000,
          "gapDenominator": 1,
          "timeNumerator": 1000000,
          "timeDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1000000,
          "sampleSteps": 4,
          "eulerSteps": 64
        }
      }
    }
  ],
  "piecewise-hamiltonian-pulse-time-ordering": [
    {
      "id": "X then Z",
      "label": {
        "ko": "X 다음 Z",
        "en": "X then Z"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 0,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 1,
          "durationADenominator": 1,
          "durationBNumerator": 1,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "unequal durations",
      "label": {
        "ko": "다른 지속 시간",
        "en": "Unequal durations"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 0,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 2,
          "durationADenominator": 1,
          "durationBNumerator": 1,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "off-grid switch times",
      "label": {
        "ko": "격자 사이의 전환 시각",
        "en": "Off-grid switch times"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 0,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 2,
          "durationADenominator": 3,
          "durationBNumerator": 4,
          "durationBDenominator": 5,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "nonorthogonal axes",
      "label": {
        "ko": "비직교 방향",
        "en": "Nonorthogonal axes"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 2,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 1,
          "durationADenominator": 1,
          "durationBNumerator": 1,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "commuting unequal schedule",
      "label": {
        "ko": "가환하지만 다른 중간 경로",
        "en": "Commuting unequal schedule"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 0,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 0,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 2,
          "bGapDenominator": 1,
          "durationANumerator": 1,
          "durationADenominator": 1,
          "durationBNumerator": 1,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "identical Hamiltonians unequal durations",
      "label": {
        "ko": "같은 Hamiltonian과 다른 지속 시간",
        "en": "Identical Hamiltonians unequal durations"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 1,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 2,
          "durationADenominator": 1,
          "durationBNumerator": 1,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "opposite axes cancellation",
      "label": {
        "ko": "서로 상쇄되는 작용",
        "en": "Opposite axes cancellation"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": -1,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 1,
          "durationADenominator": 1,
          "durationBNumerator": 1,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "scalar mean with cancelled traceless action",
      "label": {
        "ko": "Traceless 상쇄와 scalar 평균",
        "en": "Scalar mean with cancelled traceless action"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": -1,
          "axisBImag1": 0,
          "aCenterNumerator": 3,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": -1,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 1,
          "durationADenominator": 1,
          "durationBNumerator": 1,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "small pulses",
      "label": {
        "ko": "작은 두 pulse",
        "en": "Small pulses"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 0,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 1,
          "durationADenominator": 100,
          "durationBNumerator": 1,
          "durationBDenominator": 100,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "half small pulses",
      "label": {
        "ko": "절반 크기의 작은 pulse",
        "en": "Half small pulses"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 0,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 1,
          "durationADenominator": 200,
          "durationBNumerator": 1,
          "durationBDenominator": 200,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "zero first duration",
      "label": {
        "ko": "첫 지속 시간0",
        "en": "Zero first duration"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 0,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 0,
          "durationADenominator": 1,
          "durationBNumerator": 1,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "zero total duration",
      "label": {
        "ko": "전체 지속 시간0",
        "en": "Zero total duration"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 0,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 0,
          "durationADenominator": 1,
          "durationBNumerator": 0,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "complex axes",
      "label": {
        "ko": "복소 pulse 방향",
        "en": "Complex axes"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 2,
          "axisAImag0": 3,
          "axisAReal1": -4,
          "axisAImag1": 5,
          "axisBReal0": -2,
          "axisBImag0": 1,
          "axisBReal1": 3,
          "axisBImag1": -4,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 1,
          "durationADenominator": 1,
          "durationBNumerator": 1,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "common energy shift",
      "label": {
        "ko": "공통 에너지 이동",
        "en": "Common energy shift"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 0,
          "axisBImag1": 0,
          "aCenterNumerator": 1,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 1,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 1,
          "durationADenominator": 1,
          "durationBNumerator": 1,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "near opposite tiny actions",
      "label": {
        "ko": "거의 상쇄되는 작은 작용",
        "en": "Near opposite tiny actions"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1000000,
          "axisAImag0": 0,
          "axisAReal1": 999999,
          "axisAImag1": 0,
          "axisBReal0": -999998,
          "axisBImag0": 0,
          "axisBReal1": 999999,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1000000,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1000000,
          "durationANumerator": 1,
          "durationADenominator": 1000000,
          "durationBNumerator": 1,
          "durationBDenominator": 1000000,
          "hbarNumerator": 1000000,
          "hbarDenominator": 1,
          "sampleSteps": 4
        }
      }
    },
    {
      "id": "sixteen intervals",
      "label": {
        "ko": "더 많은 물리 시간 표본",
        "en": "Sixteen intervals"
      },
      "input": {
        "kind": "piecewise-hamiltonian-pulse-time-ordering",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "axisAReal0": 1,
          "axisAImag0": 0,
          "axisAReal1": 1,
          "axisAImag1": 0,
          "axisBReal0": 1,
          "axisBImag0": 0,
          "axisBReal1": 0,
          "axisBImag1": 0,
          "aCenterNumerator": 0,
          "aCenterDenominator": 1,
          "aGapNumerator": 1,
          "aGapDenominator": 1,
          "bCenterNumerator": 0,
          "bCenterDenominator": 1,
          "bGapNumerator": 1,
          "bGapDenominator": 1,
          "durationANumerator": 1,
          "durationADenominator": 1,
          "durationBNumerator": 1,
          "durationBDenominator": 1,
          "hbarNumerator": 1,
          "hbarDenominator": 1,
          "sampleSteps": 16
        }
      }
    }
  ]
};
export const unitaryPresets=Object.fromEntries(unitaryKinds.map(kind=>[kind,preparations[kind].map(p=>({...p,input:snapshotUnitaryInput(p.input)}))])) as typeof preparations;
export function unitaryPresetInput(kind:UnitaryKind,index:number){const p=unitaryPresets[kind][index];if(!Number.isInteger(index)||!p)throw new RangeError('Unknown unitary preparation');return snapshotUnitaryInput(p.input);}
export function unitaryFields(input:UnitaryInput){return Object.entries(input.parameters).map(([id,value])=>{
 let label=t(id,id);const v=/^(state|axisA|axisB|axis)(Real|Imag)([01])$/.exec(id),r=/^(center|gap|aCenter|aGap|bCenter|bGap|time|durationA|durationB|hbar)(Numerator|Denominator)$/.exec(id);
 if(v){const names={state:t('원시 상태','Raw state'),axis:t('Hamiltonian 방향','Hamiltonian direction'),axisA:t('A 방향','A direction'),axisB:t('B 방향','B direction')},name=names[v[1] as keyof typeof names];label=t(`${name.ko} ${v[3]} · ${v[2]==='Real'?'실수':'허수'}`,`${name.en} ${v[3]} · ${v[2]==='Real'?'real':'imaginary'}`);}
 if(r){const names={center:t('중심 a · E0','Center a · E0'),gap:t('간격 b · E0','Gap b · E0'),aCenter:t('A 중심 · E0','A center · E0'),aGap:t('A 간격 · E0','A gap · E0'),bCenter:t('B 중심 · E0','B center · E0'),bGap:t('B 간격 · E0','B gap · E0'),time:t('최종 시간 · T0','Final time · T0'),durationA:t('A 지속 시간 · T0','A duration · T0'),durationB:t('B 지속 시간 · T0','B duration · T0'),hbar:t('ℏ · E0T0','ℏ · E0T0')},name=names[r[1] as keyof typeof names];label=t(`${name.ko} · ${r[2]==='Numerator'?'분자':'분모'}`,`${name.en} · ${r[2]==='Numerator'?'numerator':'denominator'}`);}
 if(id==='sampleSteps')label=t('물리 시간 관측 구간 수','Physical time sample intervals');if(id==='eulerSteps')label=t('고정 Euler 단계 수','Fixed Euler steps');return{id,value,label};
});}
export function editUnitaryField(input:UnitaryInput,id:string,raw:string):UnitaryInput|null{if(!Object.hasOwn(input.parameters,id))return null;const value=parseNumberDraft(raw,-1000000,1000000,true);if(value===null)return null;try{return snapshotUnitaryInput({...input,parameters:{...input.parameters,[id]:value}} as UnitaryInput);}catch{return null;}}
