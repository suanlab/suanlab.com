/** Bilingual input preparations; numerical work stays in native workers. */
import {currentKinds,currentParameterKeys,isCurrentInput,snapshotCurrentInput,currentInputKey,type CurrentInput,type CurrentKind}from './learning-probability-current-protocol';
export {currentKinds,currentInputKey};export type{CurrentInput,CurrentKind};
export type CurrentText={ko:string;en:string};export const currentText=(ko:string,en:string):CurrentText=>({ko,en});
export const isCurrentKind=(v:string):v is CurrentKind=>(currentKinds as readonly string[]).includes(v);
export function currentPresetInput(kind:CurrentKind,index:number){const p=currentPresets[kind][index];if(!Number.isInteger(index)||!p)throw new RangeError('Unknown current preparation');return snapshotCurrentInput(p.input);}
const names:Record<string,CurrentText>={relaxationNumerator:currentText('감쇠 γ 분자 [1/s]','Damping γ numerator [1/s]'),diffusionNumerator:currentText('확산 D 분자 [m²/s]','Diffusion D numerator [m²/s]'),coefficientDenominator:currentText('γ,D 공통 분모','Common γ,D denominator'),angularNumerator:currentText('각속도 ω 분자 [rad/s]','Angular rate ω numerator [rad/s]'),angularDenominator:currentText('각속도 분모','Angular-rate denominator'),probeXNumerator:currentText('탐침 x₁ 분자 [m]','Probe x₁ numerator [m]'),probeYNumerator:currentText('탐침 x₂ 분자 [m]','Probe x₂ numerator [m]'),spaceDenominator:currentText('탐침 공통 분모','Common probe denominator'),timeNumerator:currentText('물리 시간 T 분자 [s]','Physical time T numerator [s]'),timeDenominator:currentText('시간 분모','Time denominator'),eulerSteps:currentText('같은 구간의 Euler 단계 N','Euler steps N over the same duration')};
export function currentFields(input:CurrentInput){return currentParameterKeys.map(id=>({id,label:names[id],value:input.parameters[id],minimum:['angularNumerator','probeXNumerator','probeYNumerator'].includes(id)?-1000000:id==='timeNumerator'?0:1,maximum:id==='eulerSteps'?64:1000000}));}
export function editCurrentDraft(input:CurrentInput,id:string,raw:string):CurrentInput|null{const field=currentFields(input).find(f=>f.id===id),text=raw.trim();if(!field||!/^[-]?\d{1,7}$/.test(text))return null;const value=Number(text);if(value<field.minimum||value>field.maximum)return null;const next={kind:input.kind,parameters:{...input.parameters,[id]:value}};return isCurrentInput(next)?snapshotCurrentInput(next):null;}
export function advanceCurrentTime(input:CurrentInput){const p=input.parameters,upper=Math.min(1000000,4*p.timeDenominator);return p.timeNumerator>=upper?null:editCurrentDraft(input,'timeNumerator',String(Math.min(upper,p.timeNumerator+p.timeDenominator)));}
export const rewindCurrentTime=(input:CurrentInput)=>editCurrentDraft(input,'timeNumerator','0')!;
export const currentTitles:Record<CurrentKind,CurrentText>={'stationary-gaussian-circulating-current':currentText('정상 Gaussian과 순환 확률 흐름','Stationary Gaussians and circulating probability currents')};
export const currentPresets={
  "stationary-gaussian-circulating-current": [
    {
      "id": "current-0",
      "label": {
        "ko": "기준 반시계 흐름",
        "en": "Reference counterclockwise flow"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "current-1",
      "label": {
        "ko": "회전0·전체 흐름0",
        "en": "Zero rotation, zero full current"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 0,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "current-2",
      "label": {
        "ko": "시계 방향 흐름",
        "en": "Clockwise flow"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": -1,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "current-3",
      "label": {
        "ko": "원점의 영 탐침",
        "en": "Zero probe at the origin"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 2,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "probeXNumerator": 0,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "current-4",
      "label": {
        "ko": "드리프트 성분 상쇄",
        "en": "Drift-component cancellation"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 1,
          "spaceDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "current-5",
      "label": {
        "ko": "수축하지만 편향된 Euler",
        "en": "Contracting but biased Euler"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 2
        }
      }
    },
    {
      "id": "current-6",
      "label": {
        "ko": "90도 Euler 경계",
        "en": "Quarter-turn Euler boundary"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "current-7",
      "label": {
        "ko": "유리수 경계 승수",
        "en": "Rational boundary multiplier"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 2,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 2,
          "timeDenominator": 5,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "current-8",
      "label": {
        "ko": "안정성 경계 바로 아래",
        "en": "Just below stability boundary"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 2,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 3999,
          "timeDenominator": 10000,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "current-9",
      "label": {
        "ko": "안정성 경계 바로 위",
        "en": "Just above stability boundary"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 2,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 4001,
          "timeDenominator": 10000,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "current-10",
      "label": {
        "ko": "빠른 회전과 팽창",
        "en": "Fast rotation and expansion"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 2,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 4,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "current-11",
      "label": {
        "ko": "같은 폭·느린 시계",
        "en": "Same width, slower clock"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "angularNumerator": 1,
          "angularDenominator": 2,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 2,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "current-12",
      "label": {
        "ko": "영 물리 시간",
        "en": "Zero physical time"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 0,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "current-13",
      "label": {
        "ko": "작은 감쇠·최대 각도",
        "en": "Tiny damping, maximum angle"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1000000,
          "angularNumerator": 4,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "eulerSteps": 64
        }
      }
    },
    {
      "id": "current-14",
      "label": {
        "ko": "아주 작은 양의 장",
        "en": "Tiny positive field"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1000000,
          "probeXNumerator": 1,
          "probeYNumerator": 1,
          "spaceDenominator": 1000000,
          "timeNumerator": 1,
          "timeDenominator": 1000000,
          "eulerSteps": 64
        }
      }
    },
    {
      "id": "current-15",
      "label": {
        "ko": "양의 밀도·선형 표시 미해결",
        "en": "Positive density, unresolved linear display"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1000000,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1000000,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "probeXNumerator": 4,
          "probeYNumerator": 4,
          "spaceDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "current-16",
      "label": {
        "ko": "Euler 평균 승수0",
        "en": "Zero Euler mean multiplier"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 0,
          "angularDenominator": 1,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "current-17",
      "label": {
        "ko": "355/113은 정확한 pi가 아님",
        "en": "355/113 is not exactly pi"
      },
      "input": {
        "kind": "stationary-gaussian-circulating-current",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 355,
          "angularDenominator": 113,
          "probeXNumerator": 1,
          "probeYNumerator": 0,
          "spaceDenominator": 1,
          "timeNumerator": 2,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    }
  ]
} as unknown as Record<CurrentKind,{id:string;label:CurrentText;input:CurrentInput}[]>;
