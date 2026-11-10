/** Bilingual preparations and bounded input edits. Numerical evaluation stays in workers. */
import {fokkerKinds,isFokkerInput,snapshotFokkerInput,fokkerInputKey,fokkerParameterKeys,type FokkerInput,type FokkerKind}from './learning-fokker-planck-protocol';
export {fokkerKinds,fokkerInputKey};export type{FokkerInput,FokkerKind};
export type FokkerText={ko:string;en:string};export const fokkerText=(ko:string,en:string):FokkerText=>({ko,en});
export const isFokkerKind=(v:string):v is FokkerKind=>(fokkerKinds as readonly string[]).includes(v);
export function fokkerPresetInput(kind:FokkerKind,index:number){const v=fokkerPresets[kind][index];if(!Number.isInteger(index)||!v)throw new RangeError('Unknown Fokker–Planck preset');return snapshotFokkerInput(v.input);}
const names:Record<string,FokkerText>={
 relaxationNumerator:fokkerText('완화율 γ 분자 [1/s]','Relaxation γ numerator [1/s]'),diffusionNumerator:fokkerText('확산 D 분자 [m²/s]','Diffusion D numerator [m²/s]'),coefficientDenominator:fokkerText('계수 공통 분모','Common coefficient denominator'),
 initialMeanNumerator:fokkerText('초기 평균 분자 [m]','Initial mean numerator [m]'),initialVarianceNumerator:fokkerText('초기 분산 분자 [m²]','Initial variance numerator [m²]'),initialDenominator:fokkerText('초기 모멘트 공통 분모','Common initial-moment denominator'),
 timeNumerator:fokkerText('물리 시간 T 분자 [s]','Physical time T numerator [s]'),timeDenominator:fokkerText('물리 시간 분모','Physical-time denominator'),eulerSteps:fokkerText('같은 구간의 Euler 단계 N','Euler steps N over the same duration'),
 cells:fokkerText('셀 수 K·초기 밀도 보존 재분배','Cells K; conservative initial-density remap'),halfWidthNumerator:fokkerText('반폭 L 분자 [m]','Half-width L numerator [m]'),spaceDenominator:fokkerText('반폭 분모','Half-width denominator'),windNumerator:fokkerText('바람 c 분자 [m/s]','Wind c numerator [m/s]'),poissonCutoff:fokkerText('Poisson 절단 M','Poisson cutoff M'),
};
export function fokkerFields(input:FokkerInput){const p=input.parameters as unknown as Record<string,unknown>;const keys=fokkerParameterKeys[fokkerKinds.indexOf(input.kind)].filter(k=>typeof p[k]==='number');return[...keys.map(id=>({id,label:names[id],value:p[id]as number,minimum:['initialMeanNumerator','windNumerator'].includes(id)?-1000000:id==='cells'?2:id.endsWith('Denominator')||['eulerSteps','halfWidthNumerator'].includes(id)?1:0,maximum:id==='cells'?8:id==='eulerSteps'?128:id==='poissonCutoff'?256:1000000})),...('initialWeights'in input.parameters?input.parameters.initialWeights.map((value,i)=>({id:`weight-${i}`,label:fokkerText(`초기 셀 ${i+1} 무게`,`Initial cell ${i+1} weight`),value,minimum:0,maximum:1000000})):[])];}
/** Integrate the old piecewise constant density over new uniform cells. */
export function remapFokkerWeights(weights:readonly number[],cells:number):number[]|null{
 if(!Number.isInteger(cells)||cells<2||cells>8||weights.length<2||weights.length>8||!Array.from(weights).every(x=>Number.isInteger(x)&&x>=0&&x<=1000000))return null;
 const n=weights.length,next=Array.from({length:cells},(_,i)=>weights.reduce((s,w,j)=>s+w*Math.max(0,Math.min((i+1)*n,(j+1)*cells)-Math.max(i*n,j*cells)),0));
 const gcd=(a:number,b:number):number=>b===0?a:gcd(b,a%b),d=next.reduce(gcd,0);if(d===0)return null;const result=next.map(x=>x/d);return result.every(x=>x<=1000000)?result:null;
}
export function editFokkerDraft(input:FokkerInput,id:string,raw:string):FokkerInput|null{
 const field=fokkerFields(input).find(f=>f.id===id),text=raw.trim();if(!field||!/^[-]?\d{1,7}$/.test(text))return null;
 const value=Number(text);if(value<field.minimum||value>field.maximum)return null;
 let parameters:unknown={...input.parameters,[id]:value};
 if(input.kind===fokkerKinds[1]){const p={...input.parameters,initialWeights:[...input.parameters.initialWeights]};if(id.startsWith('weight-'))p.initialWeights[Number(id.slice(7))]=value;else if(id==='cells'){const weights=remapFokkerWeights(p.initialWeights,value);if(!weights)return null;p.cells=value;p.initialWeights=weights;}else Object.assign(p,{[id]:value});parameters=p;}
 const next={kind:input.kind,parameters};return isFokkerInput(next)?snapshotFokkerInput(next):null;
}
export function advanceFokkerTime(input:FokkerInput):FokkerInput|null{
 const p=input.parameters,low=p.timeNumerator,high=Math.min(1000000,10*p.timeDenominator,low+p.timeDenominator);let a=low,b=high;
 while(a<b){const mid=Math.ceil((a+b)/2);if(editFokkerDraft(input,'timeNumerator',String(mid)))a=mid;else b=mid-1;}
 return a===low?null:editFokkerDraft(input,'timeNumerator',String(a));
}
export const rewindFokkerTime=(input:FokkerInput)=>editFokkerDraft(input,'timeNumerator','0')!;
export const fokkerTitles:Record<FokkerKind,FokkerText>={
 'ornstein-uhlenbeck-density-euler-moments':fokkerText('OU 밀도와 Euler 모멘트','OU densities and Euler moments'),
 'fokker-planck-cell-mass-boundaries':fokkerText('셀 질량·경계와 확률 수지','Cell masses, boundaries and probability budgets'),
};
export const fokkerPresets={
  "ornstein-uhlenbeck-density-euler-moments": [
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-0",
      "label": {
        "ko": "기준 OU 완화",
        "en": "Reference OU relaxation"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 2,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "initialMeanNumerator": 8,
          "initialVarianceNumerator": 1,
          "initialDenominator": 4,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-1",
      "label": {
        "ko": "정상 Gaussian",
        "en": "Stationary Gaussian"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 2,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "initialMeanNumerator": 0,
          "initialVarianceNumerator": 1,
          "initialDenominator": 2,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-2",
      "label": {
        "ko": "Brownian 확산",
        "en": "Brownian spreading"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "initialMeanNumerator": 8,
          "initialVarianceNumerator": 1,
          "initialDenominator": 4,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-3",
      "label": {
        "ko": "항등과 정상 가족",
        "en": "Identity and stationary family"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 0,
          "diffusionNumerator": 0,
          "coefficientDenominator": 1,
          "initialMeanNumerator": 8,
          "initialVarianceNumerator": 1,
          "initialDenominator": 4,
          "timeNumerator": 3,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-4",
      "label": {
        "ko": "결정론적 점 질량",
        "en": "Deterministic point mass"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 0,
          "coefficientDenominator": 1,
          "initialMeanNumerator": 2,
          "initialVarianceNumerator": 0,
          "initialDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-5",
      "label": {
        "ko": "잡음 없는 무작위 초기값",
        "en": "Random initial law without forcing noise"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 0,
          "coefficientDenominator": 1,
          "initialMeanNumerator": 2,
          "initialVarianceNumerator": 1,
          "initialDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-6",
      "label": {
        "ko": "영 시간의 원자",
        "en": "Atomic zero time"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 2,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "initialMeanNumerator": 2,
          "initialVarianceNumerator": 0,
          "initialDenominator": 1,
          "timeNumerator": 0,
          "timeDenominator": 1,
          "eulerSteps": 8
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-7",
      "label": {
        "ko": "Euler 평균 초기화",
        "en": "Euler mean reset"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 2,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "initialMeanNumerator": 2,
          "initialVarianceNumerator": 0,
          "initialDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-8",
      "label": {
        "ko": "Euler 안정성 경계",
        "en": "Euler stability boundary"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 2,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "initialMeanNumerator": 2,
          "initialVarianceNumerator": 0,
          "initialDenominator": 1,
          "timeNumerator": 2,
          "timeDenominator": 1,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-9",
      "label": {
        "ko": "팽창하는 Euler 평균",
        "en": "Expanding Euler mean"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 2,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "initialMeanNumerator": 2,
          "initialVarianceNumerator": 0,
          "initialDenominator": 1,
          "timeNumerator": 3,
          "timeDenominator": 1,
          "eulerSteps": 1
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-10",
      "label": {
        "ko": "빠른 완화와 거친 Euler",
        "en": "Fast relaxation, coarse Euler"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 5,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "initialMeanNumerator": 10,
          "initialVarianceNumerator": 1,
          "initialDenominator": 5,
          "timeNumerator": 10,
          "timeDenominator": 1,
          "eulerSteps": 4
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-11",
      "label": {
        "ko": "아주 작은 양의 확산",
        "en": "Tiny positive diffusion"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1000000,
          "initialMeanNumerator": 0,
          "initialVarianceNumerator": 0,
          "initialDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1000000,
          "eulerSteps": 128
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-12",
      "label": {
        "ko": "정상 폭과 이동 평균",
        "en": "Stationary width, moving mean"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 5,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "initialMeanNumerator": 10,
          "initialVarianceNumerator": 1,
          "initialDenominator": 5,
          "timeNumerator": 10,
          "timeDenominator": 1,
          "eulerSteps": 128
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-13",
      "label": {
        "ko": "정상 원자",
        "en": "Stationary atom"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 0,
          "coefficientDenominator": 1,
          "initialMeanNumerator": 0,
          "initialVarianceNumerator": 0,
          "initialDenominator": 1,
          "timeNumerator": 10,
          "timeDenominator": 1,
          "eulerSteps": 128
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-14",
      "label": {
        "ko": "넓은 초기 법칙",
        "en": "Wide initial law"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 1,
          "diffusionNumerator": 2,
          "coefficientDenominator": 3,
          "initialMeanNumerator": -3,
          "initialVarianceNumerator": 5,
          "initialDenominator": 1,
          "timeNumerator": 7,
          "timeDenominator": 2,
          "eulerSteps": 32
        }
      }
    },
    {
      "id": "fokker-ornstein-uhlenbeck-density-euler-moments-15",
      "label": {
        "ko": "세밀한 Euler",
        "en": "Fine Euler"
      },
      "input": {
        "kind": "ornstein-uhlenbeck-density-euler-moments",
        "parameters": {
          "relaxationNumerator": 2,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "initialMeanNumerator": 8,
          "initialVarianceNumerator": 1,
          "initialDenominator": 4,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 128
        }
      }
    }
  ],
  "fokker-planck-cell-mass-boundaries": [
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-0",
      "label": {
        "ko": "반사 OU 격자",
        "en": "Reflecting OU grid"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 4,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 2,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 16,
          "poissonCutoff": 64,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            0,
            1,
            0,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-1",
      "label": {
        "ko": "양쪽 흡수",
        "en": "Two absorbing walls"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 4,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 2,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 16,
          "poissonCutoff": 64,
          "leftBoundary": "absorbing",
          "rightBoundary": "absorbing",
          "initialWeights": [
            0,
            1,
            0,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-2",
      "label": {
        "ko": "왼쪽 반사·오른쪽 흡수",
        "en": "Reflect left, absorb right"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 4,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 2,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 16,
          "poissonCutoff": 64,
          "leftBoundary": "reflecting",
          "rightBoundary": "absorbing",
          "initialWeights": [
            0,
            1,
            0,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-3",
      "label": {
        "ko": "바람과 반사",
        "en": "Wind and reflection"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 4,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 0,
          "windNumerator": 4,
          "diffusionNumerator": 1,
          "coefficientDenominator": 4,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 16,
          "poissonCutoff": 64,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            0,
            1,
            0,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-4",
      "label": {
        "ko": "순수 이류와 탈출",
        "en": "Pure advection and escape"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 4,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 0,
          "windNumerator": 1,
          "diffusionNumerator": 0,
          "coefficientDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "eulerSteps": 16,
          "poissonCutoff": 64,
          "leftBoundary": "absorbing",
          "rightBoundary": "absorbing",
          "initialWeights": [
            1,
            0,
            0,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-5",
      "label": {
        "ko": "항등 격자",
        "en": "Identity grid"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 4,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 0,
          "windNumerator": 0,
          "diffusionNumerator": 0,
          "coefficientDenominator": 2,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 16,
          "poissonCutoff": 64,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            0,
            1,
            0,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-6",
      "label": {
        "ko": "영 물리 시간",
        "en": "Zero physical time"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 4,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 2,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "timeNumerator": 0,
          "timeDenominator": 1,
          "eulerSteps": 16,
          "poissonCutoff": 64,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            0,
            1,
            0,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-7",
      "label": {
        "ko": "두 셀 양수성 경계",
        "en": "Two-cell positivity boundary"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 2,
          "halfWidthNumerator": 1,
          "spaceDenominator": 1,
          "relaxationNumerator": 0,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 1,
          "poissonCutoff": 64,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            1,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-8",
      "label": {
        "ko": "보존되는 음수 Euler",
        "en": "Conservative signed Euler failure"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 2,
          "halfWidthNumerator": 1,
          "spaceDenominator": 1,
          "relaxationNumerator": 0,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "timeNumerator": 3,
          "timeDenominator": 2,
          "eulerSteps": 1,
          "poissonCutoff": 64,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            1,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-9",
      "label": {
        "ko": "두 셀 시간 세분화",
        "en": "Two-cell time refinement"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 2,
          "halfWidthNumerator": 1,
          "spaceDenominator": 1,
          "relaxationNumerator": 0,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 16,
          "poissonCutoff": 64,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            1,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-10",
      "label": {
        "ko": "반 셀 거리의 흡수",
        "en": "Half-cell absorbing distance"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 2,
          "halfWidthNumerator": 1,
          "spaceDenominator": 1,
          "relaxationNumerator": 0,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 2,
          "eulerSteps": 8,
          "poissonCutoff": 64,
          "leftBoundary": "absorbing",
          "rightBoundary": "absorbing",
          "initialWeights": [
            1,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-11",
      "label": {
        "ko": "격자 정상 초기화",
        "en": "Grid-stationary initialization"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 4,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 2,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 16,
          "poissonCutoff": 64,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            1,
            3,
            3,
            1
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-12",
      "label": {
        "ko": "영 확산과 분리된 클래스",
        "en": "Zero diffusion, split classes"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 4,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 2,
          "windNumerator": 0,
          "diffusionNumerator": 0,
          "coefficientDenominator": 2,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 16,
          "poissonCutoff": 64,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            1,
            0,
            0,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-13",
      "label": {
        "ko": "같은 밀도의 여덟 셀",
        "en": "Eight cells, same initial density"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 8,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 2,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 32,
          "poissonCutoff": 96,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-14",
      "label": {
        "ko": "Poisson 절단0",
        "en": "Poisson cutoff zero"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 4,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 2,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 2,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "eulerSteps": 16,
          "poissonCutoff": 0,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            0,
            1,
            0,
            0
          ]
        }
      }
    },
    {
      "id": "fokker-fokker-planck-cell-mass-boundaries-15",
      "label": {
        "ko": "극소 양의 누락 꼬리",
        "en": "Tiny positive omitted tail"
      },
      "input": {
        "kind": "fokker-planck-cell-mass-boundaries",
        "parameters": {
          "cells": 4,
          "halfWidthNumerator": 2,
          "spaceDenominator": 1,
          "relaxationNumerator": 0,
          "windNumerator": 0,
          "diffusionNumerator": 1,
          "coefficientDenominator": 1000000,
          "timeNumerator": 1,
          "timeDenominator": 1000000,
          "eulerSteps": 4,
          "poissonCutoff": 16,
          "leftBoundary": "reflecting",
          "rightBoundary": "reflecting",
          "initialWeights": [
            1,
            0,
            0,
            0
          ]
        }
      }
    }
  ]
} as unknown as Record<FokkerKind,{id:string;label:FokkerText;input:FokkerInput}[]>;
