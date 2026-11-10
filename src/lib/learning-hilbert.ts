/** Fixed teaching preparations; computation stays in Workers. */
import{hilbertKinds,snapshotHilbertInput,hilbertInputKey,type HilbertInput}from'./learning-hilbert-protocol';
import{parseNumberDraft}from'./learning-number-input';
export{hilbertKinds,hilbertInputKey};export type{HilbertInput};export type HilbertKind=typeof hilbertKinds[number];
export const isHilbertKind=(x:string):x is HilbertKind=>(hilbertKinds as readonly string[]).includes(x);
export const hilbertText=(ko:string,en:string)=>({ko,en});const t=hilbertText;
export const hilbertTitles={
 [hilbertKinds[0]]:t('복소 상태·ray·기저 진폭','Complex states, rays and basis amplitudes'),
 [hilbertKinds[1]]:t('파동함수·L² 사영·완비성','Wavefunctions, L² projections and completeness'),
};
const preparations:Record<HilbertKind,{id:string;label:{ko:string;en:string};input:HilbertInput}[]>={
  "complex-state-rays-basis-amplitudes": [
    {
      "id": "equal Z statistics orthogonal rays",
      "label": {
        "ko": "같은 Z 확률·직교 상태",
        "en": "Equal Z probabilities, orthogonal states"
      },
      "input": {
        "kind": "complex-state-rays-basis-amplitudes",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "referenceReal0": 1,
          "referenceImag0": 0,
          "referenceReal1": -1,
          "referenceImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 1,
          "basisImag1": 0,
          "phaseA": 1,
          "phaseB": 1,
          "scaleNumerator": 1,
          "scaleDenominator": 1
        }
      }
    },
    {
      "id": "positive imaginary relative phase",
      "label": {
        "ko": "양의 허수 상대 위상",
        "en": "Positive imaginary relative phase"
      },
      "input": {
        "kind": "complex-state-rays-basis-amplitudes",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 1,
          "referenceReal0": 1,
          "referenceImag0": 0,
          "referenceReal1": -1,
          "referenceImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 1,
          "basisImag1": 0,
          "phaseA": 1,
          "phaseB": 1,
          "scaleNumerator": 1,
          "scaleDenominator": 1
        }
      }
    },
    {
      "id": "negative imaginary relative phase",
      "label": {
        "ko": "음의 허수 상대 위상",
        "en": "Negative imaginary relative phase"
      },
      "input": {
        "kind": "complex-state-rays-basis-amplitudes",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": -1,
          "referenceReal0": 1,
          "referenceImag0": 0,
          "referenceReal1": -1,
          "referenceImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 1,
          "basisImag1": 0,
          "phaseA": 1,
          "phaseB": 1,
          "scaleNumerator": 1,
          "scaleDenominator": 1
        }
      }
    },
    {
      "id": "same ray imaginary scaling",
      "label": {
        "ko": "같은 ray의 다른 대표",
        "en": "Different representatives of one ray"
      },
      "input": {
        "kind": "complex-state-rays-basis-amplitudes",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "referenceReal0": 0,
          "referenceImag0": 1,
          "referenceReal1": 0,
          "referenceImag1": 1,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 1,
          "basisImag1": 0,
          "phaseA": 1,
          "phaseB": 1,
          "scaleNumerator": 1,
          "scaleDenominator": 1
        }
      }
    },
    {
      "id": "identity global phase",
      "label": {
        "ko": "전역 위상 1",
        "en": "Global phase 1"
      },
      "input": {
        "kind": "complex-state-rays-basis-amplitudes",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "referenceReal0": 1,
          "referenceImag0": 0,
          "referenceReal1": -1,
          "referenceImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 1,
          "basisImag1": 0,
          "phaseA": 1,
          "phaseB": 0,
          "scaleNumerator": 1,
          "scaleDenominator": 1
        }
      }
    },
    {
      "id": "negative global phase",
      "label": {
        "ko": "전역 위상 −1",
        "en": "Global phase −1"
      },
      "input": {
        "kind": "complex-state-rays-basis-amplitudes",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "referenceReal0": 1,
          "referenceImag0": 0,
          "referenceReal1": -1,
          "referenceImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 1,
          "basisImag1": 0,
          "phaseA": 0,
          "phaseB": 1,
          "scaleNumerator": 1,
          "scaleDenominator": 1
        }
      }
    },
    {
      "id": "skew basis coefficient mass exceeds one",
      "label": {
        "ko": "기울어진 기저의 제곱 합2",
        "en": "Skew-basis square sum 2"
      },
      "input": {
        "kind": "complex-state-rays-basis-amplitudes",
        "parameters": {
          "stateReal0": 0,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "referenceReal0": 1,
          "referenceImag0": 0,
          "referenceReal1": 0,
          "referenceImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 1,
          "basisImag1": 0,
          "phaseA": 1,
          "phaseB": 1,
          "scaleNumerator": 1,
          "scaleDenominator": 1
        }
      }
    },
    {
      "id": "skew basis coefficient mass below one",
      "label": {
        "ko": "기울어진 기저의 제곱 합1/2",
        "en": "Skew-basis square sum 1/2"
      },
      "input": {
        "kind": "complex-state-rays-basis-amplitudes",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 1,
          "stateImag1": 0,
          "referenceReal0": 1,
          "referenceImag0": 0,
          "referenceReal1": 1,
          "referenceImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 1,
          "basisImag1": 0,
          "phaseA": 1,
          "phaseB": 1,
          "scaleNumerator": 1,
          "scaleDenominator": 1
        }
      }
    },
    {
      "id": "general complex states",
      "label": {
        "ko": "일반 복소 상태·기저",
        "en": "General complex states and basis"
      },
      "input": {
        "kind": "complex-state-rays-basis-amplitudes",
        "parameters": {
          "stateReal0": -3,
          "stateImag0": 2,
          "stateReal1": 5,
          "stateImag1": -7,
          "referenceReal0": 4,
          "referenceImag0": -2,
          "referenceReal1": 1,
          "referenceImag1": 3,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 1,
          "basisImag1": 0,
          "phaseA": 1,
          "phaseB": 1,
          "scaleNumerator": 1,
          "scaleDenominator": 1
        }
      }
    },
    {
      "id": "zero second component",
      "label": {
        "ko": "영 진폭의 위상 미정의",
        "en": "Undefined phase at zero amplitude"
      },
      "input": {
        "kind": "complex-state-rays-basis-amplitudes",
        "parameters": {
          "stateReal0": 1,
          "stateImag0": 0,
          "stateReal1": 0,
          "stateImag1": 0,
          "referenceReal0": 1,
          "referenceImag0": 0,
          "referenceReal1": -1,
          "referenceImag1": 0,
          "basisReal0": 1,
          "basisImag0": 0,
          "basisReal1": 1,
          "basisImag1": 0,
          "phaseA": 1,
          "phaseB": 1,
          "scaleNumerator": 1,
          "scaleDenominator": 1
        }
      }
    }
  ],
  "wavefunction-l2-projection-completeness": [
    {
      "id": "half interval step",
      "label": {
        "ko": "반구간 계단·3차 사영",
        "en": "Half-interval step, degree 3"
      },
      "input": {
        "kind": "wavefunction-l2-projection-completeness",
        "parameters": {
          "lengthNumerator": 1,
          "lengthDenominator": 1,
          "breakNumerator": 1,
          "breakDenominator": 2,
          "leftRealConstant": 1,
          "leftRealSlope": 0,
          "leftImagConstant": 0,
          "leftImagSlope": 0,
          "rightRealConstant": 0,
          "rightRealSlope": 0,
          "rightImagConstant": 0,
          "rightImagSlope": 0,
          "cutoffDegree": 3,
          "bins": 8,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "constant projection of step",
      "label": {
        "ko": "계단의 상수 사영",
        "en": "Constant projection of the step"
      },
      "input": {
        "kind": "wavefunction-l2-projection-completeness",
        "parameters": {
          "lengthNumerator": 1,
          "lengthDenominator": 1,
          "breakNumerator": 1,
          "breakDenominator": 2,
          "leftRealConstant": 1,
          "leftRealSlope": 0,
          "leftImagConstant": 0,
          "leftImagSlope": 0,
          "rightRealConstant": 0,
          "rightRealSlope": 0,
          "rightImagConstant": 0,
          "rightImagSlope": 0,
          "cutoffDegree": 0,
          "bins": 8,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "linear projection of step",
      "label": {
        "ko": "계단의 1차 사영",
        "en": "Linear projection of the step"
      },
      "input": {
        "kind": "wavefunction-l2-projection-completeness",
        "parameters": {
          "lengthNumerator": 1,
          "lengthDenominator": 1,
          "breakNumerator": 1,
          "breakDenominator": 2,
          "leftRealConstant": 1,
          "leftRealSlope": 0,
          "leftImagConstant": 0,
          "leftImagSlope": 0,
          "rightRealConstant": 0,
          "rightRealSlope": 0,
          "rightImagConstant": 0,
          "rightImagSlope": 0,
          "cutoffDegree": 1,
          "bins": 8,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "zero projection of sign-changing state",
      "label": {
        "ko": "부호가 바뀌는 상태의 영 사영",
        "en": "Zero projection of a sign-changing state"
      },
      "input": {
        "kind": "wavefunction-l2-projection-completeness",
        "parameters": {
          "lengthNumerator": 1,
          "lengthDenominator": 1,
          "breakNumerator": 1,
          "breakDenominator": 2,
          "leftRealConstant": 1,
          "leftRealSlope": 0,
          "leftImagConstant": 0,
          "leftImagSlope": 0,
          "rightRealConstant": -1,
          "rightRealSlope": 0,
          "rightImagConstant": 0,
          "rightImagSlope": 0,
          "cutoffDegree": 0,
          "bins": 8,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "global affine complex state",
      "label": {
        "ko": "정확히 재구성되는 복소 affine",
        "en": "Exactly reconstructed complex affine"
      },
      "input": {
        "kind": "wavefunction-l2-projection-completeness",
        "parameters": {
          "lengthNumerator": 1,
          "lengthDenominator": 1,
          "breakNumerator": 1,
          "breakDenominator": 2,
          "leftRealConstant": 1,
          "leftRealSlope": 2,
          "leftImagConstant": -1,
          "leftImagSlope": 3,
          "rightRealConstant": 1,
          "rightRealSlope": 2,
          "rightImagConstant": -1,
          "rightImagSlope": 3,
          "cutoffDegree": 1,
          "bins": 8,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "small support missed by coarse samples",
      "label": {
        "ko": "성긴 샘플이 놓치는 질량",
        "en": "Mass missed by coarse samples"
      },
      "input": {
        "kind": "wavefunction-l2-projection-completeness",
        "parameters": {
          "lengthNumerator": 1,
          "lengthDenominator": 1,
          "breakNumerator": 1,
          "breakDenominator": 1000000,
          "leftRealConstant": 1,
          "leftRealSlope": 0,
          "leftImagConstant": 0,
          "leftImagSlope": 0,
          "rightRealConstant": 0,
          "rightRealSlope": 0,
          "rightImagConstant": 0,
          "rightImagSlope": 0,
          "cutoffDegree": 12,
          "bins": 8,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "long physical interval",
      "label": {
        "ko": "긴 물리 구간의 밀도",
        "en": "Density on a longer interval"
      },
      "input": {
        "kind": "wavefunction-l2-projection-completeness",
        "parameters": {
          "lengthNumerator": 10,
          "lengthDenominator": 1,
          "breakNumerator": 1,
          "breakDenominator": 2,
          "leftRealConstant": 1,
          "leftRealSlope": 0,
          "leftImagConstant": 0,
          "leftImagSlope": 0,
          "rightRealConstant": 0,
          "rightRealSlope": 0,
          "rightImagConstant": 0,
          "rightImagSlope": 0,
          "cutoffDegree": 3,
          "bins": 8,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "split at one",
      "label": {
        "ko": "끝점만 바뀐 대표",
        "en": "Representative changed at one endpoint"
      },
      "input": {
        "kind": "wavefunction-l2-projection-completeness",
        "parameters": {
          "lengthNumerator": 1,
          "lengthDenominator": 1,
          "breakNumerator": 1,
          "breakDenominator": 1,
          "leftRealConstant": 1,
          "leftRealSlope": 0,
          "leftImagConstant": 0,
          "leftImagSlope": 0,
          "rightRealConstant": 999,
          "rightRealSlope": 0,
          "rightImagConstant": 0,
          "rightImagSlope": 0,
          "cutoffDegree": 0,
          "bins": 8,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "split at zero",
      "label": {
        "ko": "빈 왼쪽 조각",
        "en": "Empty left segment"
      },
      "input": {
        "kind": "wavefunction-l2-projection-completeness",
        "parameters": {
          "lengthNumerator": 1,
          "lengthDenominator": 1,
          "breakNumerator": 0,
          "breakDenominator": 2,
          "leftRealConstant": 999,
          "leftRealSlope": 0,
          "leftImagConstant": 0,
          "leftImagSlope": 0,
          "rightRealConstant": 1,
          "rightRealSlope": 0,
          "rightImagConstant": 1,
          "rightImagSlope": 0,
          "cutoffDegree": 0,
          "bins": 8,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "interior complex discontinuity",
      "label": {
        "ko": "복소 점프와 유한 사영",
        "en": "Complex jump and finite projection"
      },
      "input": {
        "kind": "wavefunction-l2-projection-completeness",
        "parameters": {
          "lengthNumerator": 1,
          "lengthDenominator": 1,
          "breakNumerator": 2,
          "breakDenominator": 3,
          "leftRealConstant": -2,
          "leftRealSlope": 3,
          "leftImagConstant": 1,
          "leftImagSlope": -2,
          "rightRealConstant": 3,
          "rightRealSlope": -1,
          "rightImagConstant": -4,
          "rightImagSlope": 5,
          "cutoffDegree": 7,
          "bins": 8,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "degree twelve step",
      "label": {
        "ko": "계단의 12차 사영",
        "en": "Degree-12 projection of the step"
      },
      "input": {
        "kind": "wavefunction-l2-projection-completeness",
        "parameters": {
          "lengthNumerator": 1,
          "lengthDenominator": 1,
          "breakNumerator": 1,
          "breakDenominator": 2,
          "leftRealConstant": 1,
          "leftRealSlope": 0,
          "leftImagConstant": 0,
          "leftImagSlope": 0,
          "rightRealConstant": 0,
          "rightRealSlope": 0,
          "rightImagConstant": 0,
          "rightImagSlope": 0,
          "cutoffDegree": 12,
          "bins": 8,
          "sampleSteps": 8
        }
      }
    }
  ]
};
export const hilbertPresets=Object.fromEntries(hilbertKinds.map(kind=>[kind,preparations[kind].map(p=>({...p,input:snapshotHilbertInput(p.input)}))]))as typeof preparations;
export function hilbertPresetInput(kind:HilbertKind,index:number){const p=hilbertPresets[kind][index];if(!Number.isInteger(index)||!p)throw new RangeError('Unknown Hilbert-space preparation');return snapshotHilbertInput(p.input);}
const special:Record<string,ReturnType<typeof t>>={phaseA:t('단위 전역 phasor의 a','Unit global phasor a'),phaseB:t('단위 전역 phasor의 b','Unit global phasor b'),scaleNumerator:t('전역 실수 배율 · 분자','Global real scale · numerator'),scaleDenominator:t('전역 실수 배율 · 분모','Global real scale · denominator'),lengthNumerator:t('물리 길이 L · 분자','Physical length L · numerator'),lengthDenominator:t('물리 길이 L · 분모','Physical length L · denominator'),breakNumerator:t('분할 위치 u · 분자','Split position u · numerator'),breakDenominator:t('분할 위치 u · 분모','Split position u · denominator'),cutoffDegree:t('남길 최고 다항식 차수 N','Maximum retained polynomial degree N'),bins:t('정확 적분 bin 수','Exact-integration bin count'),sampleSteps:t('그림 샘플 간격 수','Drawing sample intervals')};
export function hilbertFields(input:HilbertInput){return Object.entries(input.parameters).map(([id,value])=>{
 let label=special[id];const z=/^(state|reference|basis)(Real|Imag)([01])$/.exec(id),piece=/^(left|right)(Real|Imag)(Constant|Slope)$/.exec(id);
 if(z){const names={state:t('원시 상태','Raw state'),reference:t('원시 비교 상태','Raw reference'),basis:t('원시 기저 열','Raw basis column')},a=names[z[1]as keyof typeof names];label=t(`${a.ko} ${z[3]} · ${z[2]==='Real'?'실수':'허수'}`,`${a.en} ${z[3]} · ${z[2]==='Real'?'real':'imaginary'}`);}
 if(piece)label=t(`${piece[1]==='left'?'왼쪽':'오른쪽'} g · ${piece[2]==='Real'?'실수':'허수'} ${piece[3]==='Constant'?'상수':'기울기'}`,`${piece[1]==='left'?'Left':'Right'} g · ${piece[2]==='Real'?'real':'imaginary'} ${piece[3]==='Constant'?'constant':'slope'}`);
 return{id,value,label:label??t(id,id)};
});}
export function editHilbertField(input:HilbertInput,id:string,raw:string):HilbertInput|null{if(!Object.hasOwn(input.parameters,id))return null;const value=parseNumberDraft(raw,-1000000,1000000,true);if(value===null)return null;try{return snapshotHilbertInput({...input,parameters:{...input.parameters,[id]:value}}as HilbertInput);}catch{return null;}}
