/** Teaching preparations only. Numerical engines stay inside Workers. */
import {coarseKinds,snapshotCoarseInput,coarseInputKey,type CoarseInput} from './learning-coarse-protocol';
import {parseNumberDraft} from './learning-number-input';
export {coarseKinds,coarseInputKey};export type {CoarseInput};
export type CoarseKind=typeof coarseKinds[number];
export const isCoarseKind=(s:string):s is CoarseKind=>(coarseKinds as readonly string[]).includes(s);
export const coarseText=(ko:string,en:string)=>({ko,en});const t=coarseText;
export const coarseTitles={
 [coarseKinds[0]]:t('집계한 상태의 닫힘과 기억','Closure and memory of aggregated states'),
 [coarseKinds[1]]:t('숨은 변수의 기억과 Markov 근사','Hidden-variable memory and the Markov approximation'),
};
const preparations:Record<CoarseKind,{id:string;label:{ko:string;en:string};input:CoarseInput}[]>={
  "markov-partition-lumpability-memory": [
    {
      "id": "stationary-matched nonlumpable",
      "label": {
        "ko": "같은 거시값·초기 상태 0",
        "en": "Same macro, initial state 0"
      },
      "input": {
        "kind": "markov-partition-lumpability-memory",
        "parameters": {
          "weight01": 1,
          "weight02": 1,
          "weight10": 1,
          "weight12": 3,
          "weight20": 1,
          "weight21": 3,
          "transitionDenominator": 4,
          "initialWeights": [
            1,
            0,
            0
          ],
          "liftNumerator": 1,
          "liftDenominator": 2,
          "steps": 8,
          "memoryDepth": 1
        }
      }
    },
    {
      "id": "same macro different hidden",
      "label": {
        "ko": "같은 거시값·초기 상태 1",
        "en": "Same macro, initial state 1"
      },
      "input": {
        "kind": "markov-partition-lumpability-memory",
        "parameters": {
          "weight01": 1,
          "weight02": 1,
          "weight10": 1,
          "weight12": 3,
          "weight20": 1,
          "weight21": 3,
          "transitionDenominator": 4,
          "initialWeights": [
            0,
            1,
            0
          ],
          "liftNumerator": 1,
          "liftDenominator": 2,
          "steps": 8,
          "memoryDepth": 1
        }
      }
    },
    {
      "id": "stationary law",
      "label": {
        "ko": "보존되는 정상 법칙",
        "en": "Preserved stationary law"
      },
      "input": {
        "kind": "markov-partition-lumpability-memory",
        "parameters": {
          "weight01": 1,
          "weight02": 1,
          "weight10": 1,
          "weight12": 3,
          "weight20": 1,
          "weight21": 3,
          "transitionDenominator": 4,
          "initialWeights": [
            1,
            1,
            1
          ],
          "liftNumerator": 1,
          "liftDenominator": 2,
          "steps": 8,
          "memoryDepth": 1
        }
      }
    },
    {
      "id": "lifted initial law",
      "label": {
        "ko": "초기 숨은 성분 0",
        "en": "Zero initial hidden component"
      },
      "input": {
        "kind": "markov-partition-lumpability-memory",
        "parameters": {
          "weight01": 1,
          "weight02": 1,
          "weight10": 1,
          "weight12": 3,
          "weight20": 1,
          "weight21": 3,
          "transitionDenominator": 4,
          "initialWeights": [
            1,
            1,
            0
          ],
          "liftNumerator": 1,
          "liftDenominator": 2,
          "steps": 8,
          "memoryDepth": 1
        }
      }
    },
    {
      "id": "strong lumpability",
      "label": {
        "ko": "모든 초기 법칙에서 닫힘",
        "en": "Closure for all initial laws"
      },
      "input": {
        "kind": "markov-partition-lumpability-memory",
        "parameters": {
          "weight01": 1,
          "weight02": 1,
          "weight10": 1,
          "weight12": 1,
          "weight20": 1,
          "weight21": 3,
          "transitionDenominator": 4,
          "initialWeights": [
            1,
            0,
            0
          ],
          "liftNumerator": 1,
          "liftDenominator": 2,
          "steps": 8,
          "memoryDepth": 1
        }
      }
    },
    {
      "id": "memoryless approximation",
      "label": {
        "ko": "기억 지연을 남기지 않기",
        "en": "Retain no memory lags"
      },
      "input": {
        "kind": "markov-partition-lumpability-memory",
        "parameters": {
          "weight01": 1,
          "weight02": 1,
          "weight10": 1,
          "weight12": 3,
          "weight20": 1,
          "weight21": 3,
          "transitionDenominator": 4,
          "initialWeights": [
            1,
            0,
            0
          ],
          "liftNumerator": 1,
          "liftDenominator": 2,
          "steps": 8,
          "memoryDepth": 0
        }
      }
    },
    {
      "id": "full finite memory",
      "label": {
        "ko": "표시 궤적의 모든 기억",
        "en": "All memory for the displayed rollout"
      },
      "input": {
        "kind": "markov-partition-lumpability-memory",
        "parameters": {
          "weight01": 1,
          "weight02": 1,
          "weight10": 1,
          "weight12": 3,
          "weight20": 1,
          "weight21": 3,
          "transitionDenominator": 4,
          "initialWeights": [
            1,
            0,
            0
          ],
          "liftNumerator": 1,
          "liftDenominator": 2,
          "steps": 8,
          "memoryDepth": 32
        }
      }
    },
    {
      "id": "negative truncated pseudo-law",
      "label": {
        "ko": "확률 공간을 벗어난 근사",
        "en": "Approximation outside probability space"
      },
      "input": {
        "kind": "markov-partition-lumpability-memory",
        "parameters": {
          "weight01": 0,
          "weight02": 4,
          "weight10": 1,
          "weight12": 0,
          "weight20": 2,
          "weight21": 2,
          "transitionDenominator": 4,
          "initialWeights": [
            1,
            0,
            0
          ],
          "liftNumerator": 0,
          "liftDenominator": 2,
          "steps": 8,
          "memoryDepth": 0
        }
      }
    },
    {
      "id": "nonzero underflow memory",
      "label": {
        "ko": "표시 범위 아래의 비영 기억",
        "en": "Nonzero memory below display range"
      },
      "input": {
        "kind": "markov-partition-lumpability-memory",
        "parameters": {
          "weight01": 0,
          "weight02": 1000000,
          "weight10": 0,
          "weight12": 999999,
          "weight20": 500000,
          "weight21": 500000,
          "transitionDenominator": 1000000,
          "initialWeights": [
            1,
            0,
            0
          ],
          "liftNumerator": 1,
          "liftDenominator": 1000000,
          "steps": 32,
          "memoryDepth": 32
        }
      }
    }
  ],
  "linear-hidden-memory-markov-limit": [
    {
      "id": "two real modes",
      "label": {
        "ko": "서로 다른 두 감쇠 모드",
        "en": "Two distinct decaying modes"
      },
      "input": {
        "kind": "linear-hidden-memory-markov-limit",
        "parameters": {
          "aNumerator": 2,
          "aDenominator": 1,
          "bNumerator": 1,
          "bDenominator": 1,
          "cNumerator": 2,
          "cDenominator": 1,
          "dNumerator": 3,
          "dDenominator": 1,
          "epsilonNumerator": 1,
          "epsilonDenominator": 1,
          "xNumerator": 1,
          "xDenominator": 1,
          "yNumerator": 0,
          "yDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "prepared hidden offset",
      "label": {
        "ko": "다른 숨은 초기값",
        "en": "Different hidden initial value"
      },
      "input": {
        "kind": "linear-hidden-memory-markov-limit",
        "parameters": {
          "aNumerator": 2,
          "aDenominator": 1,
          "bNumerator": 1,
          "bDenominator": 1,
          "cNumerator": 2,
          "cDenominator": 1,
          "dNumerator": 3,
          "dDenominator": 1,
          "epsilonNumerator": 1,
          "epsilonDenominator": 1,
          "xNumerator": 1,
          "xDenominator": 1,
          "yNumerator": 1,
          "yDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "initially quasisteady",
      "label": {
        "ko": "처음에만 준정상 준비",
        "en": "Initially quasisteady preparation"
      },
      "input": {
        "kind": "linear-hidden-memory-markov-limit",
        "parameters": {
          "aNumerator": 2,
          "aDenominator": 1,
          "bNumerator": 1,
          "bDenominator": 1,
          "cNumerator": 2,
          "cDenominator": 1,
          "dNumerator": 3,
          "dDenominator": 1,
          "epsilonNumerator": 1,
          "epsilonDenominator": 1,
          "xNumerator": 1,
          "xDenominator": 1,
          "yNumerator": 2,
          "yDenominator": 3,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "epsilon half",
      "label": {
        "ko": "숨은 시간척도 ε=1/2",
        "en": "Hidden timescale epsilon=1/2"
      },
      "input": {
        "kind": "linear-hidden-memory-markov-limit",
        "parameters": {
          "aNumerator": 2,
          "aDenominator": 1,
          "bNumerator": 1,
          "bDenominator": 1,
          "cNumerator": 2,
          "cDenominator": 1,
          "dNumerator": 3,
          "dDenominator": 1,
          "epsilonNumerator": 1,
          "epsilonDenominator": 2,
          "xNumerator": 1,
          "xDenominator": 1,
          "yNumerator": 0,
          "yDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "epsilon sixteenth",
      "label": {
        "ko": "숨은 시간척도 ε=1/16",
        "en": "Hidden timescale epsilon=1/16"
      },
      "input": {
        "kind": "linear-hidden-memory-markov-limit",
        "parameters": {
          "aNumerator": 2,
          "aDenominator": 1,
          "bNumerator": 1,
          "bDenominator": 1,
          "cNumerator": 2,
          "cDenominator": 1,
          "dNumerator": 3,
          "dDenominator": 1,
          "epsilonNumerator": 1,
          "epsilonDenominator": 16,
          "xNumerator": 1,
          "xDenominator": 1,
          "yNumerator": 0,
          "yDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "no feedback to observation",
      "label": {
        "ko": "관측으로 돌아오는 결합 0",
        "en": "Zero feedback to observation"
      },
      "input": {
        "kind": "linear-hidden-memory-markov-limit",
        "parameters": {
          "aNumerator": 2,
          "aDenominator": 1,
          "bNumerator": 0,
          "bDenominator": 1,
          "cNumerator": 2,
          "cDenominator": 1,
          "dNumerator": 3,
          "dDenominator": 1,
          "epsilonNumerator": 1,
          "epsilonDenominator": 1,
          "xNumerator": 1,
          "xDenominator": 1,
          "yNumerator": 2,
          "yDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "no return to hidden",
      "label": {
        "ko": "기억 없이 남은 초기 강제",
        "en": "Initial forcing without memory"
      },
      "input": {
        "kind": "linear-hidden-memory-markov-limit",
        "parameters": {
          "aNumerator": 2,
          "aDenominator": 1,
          "bNumerator": 1,
          "bDenominator": 1,
          "cNumerator": 0,
          "cDenominator": 1,
          "dNumerator": 3,
          "dDenominator": 1,
          "epsilonNumerator": 1,
          "epsilonDenominator": 1,
          "xNumerator": 1,
          "xDenominator": 1,
          "yNumerator": 2,
          "yDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "damped oscillation",
      "label": {
        "ko": "감쇠 진동과 근사",
        "en": "Damped oscillation and approximation"
      },
      "input": {
        "kind": "linear-hidden-memory-markov-limit",
        "parameters": {
          "aNumerator": 1,
          "aDenominator": 1,
          "bNumerator": 2,
          "bDenominator": 1,
          "cNumerator": -2,
          "cDenominator": 1,
          "dNumerator": 1,
          "dDenominator": 1,
          "epsilonNumerator": 1,
          "epsilonDenominator": 1,
          "xNumerator": 1,
          "xDenominator": 1,
          "yNumerator": 0,
          "yDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "stiff hidden layer",
      "label": {
        "ko": "매우 빠른 초기 층",
        "en": "Extremely fast initial layer"
      },
      "input": {
        "kind": "linear-hidden-memory-markov-limit",
        "parameters": {
          "aNumerator": 2,
          "aDenominator": 1,
          "bNumerator": 1,
          "bDenominator": 1,
          "cNumerator": 2,
          "cDenominator": 1,
          "dNumerator": 8,
          "dDenominator": 1,
          "epsilonNumerator": 1,
          "epsilonDenominator": 1000000000,
          "xNumerator": 1,
          "xDenominator": 1,
          "yNumerator": 1,
          "yDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "marginal stationary",
      "label": {
        "ko": "경계의 정확 정상 준비",
        "en": "Exact marginal stationary preparation"
      },
      "input": {
        "kind": "linear-hidden-memory-markov-limit",
        "parameters": {
          "aNumerator": 1,
          "aDenominator": 1,
          "bNumerator": 1,
          "bDenominator": 1,
          "cNumerator": 1,
          "cDenominator": 1,
          "dNumerator": 1,
          "dDenominator": 1,
          "epsilonNumerator": 1,
          "epsilonDenominator": 1,
          "xNumerator": 1,
          "xDenominator": 1,
          "yNumerator": 1,
          "yDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "Jordan with nonzero memory",
      "label": {
        "ko": "중복 고유값과 기억",
        "en": "Repeated eigenvalue with memory"
      },
      "input": {
        "kind": "linear-hidden-memory-markov-limit",
        "parameters": {
          "aNumerator": 1,
          "aDenominator": 1,
          "bNumerator": 1,
          "bDenominator": 1,
          "cNumerator": -1,
          "cDenominator": 1,
          "dNumerator": 3,
          "dDenominator": 1,
          "epsilonNumerator": 1,
          "epsilonDenominator": 1,
          "xNumerator": 1,
          "xDenominator": 1,
          "yNumerator": 0,
          "yDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    }
  ]
};
export const coarsePresets=Object.fromEntries(coarseKinds.map(kind=>[kind,preparations[kind].map(p=>({...p,input:snapshotCoarseInput(p.input)}))])) as typeof preparations;
export function coarsePresetInput(kind:CoarseKind,index:number){const p=coarsePresets[kind][index];if(!Number.isInteger(index)||!p)throw new RangeError('Unknown coarse-graining preparation');return snapshotCoarseInput(p.input);}
const labels:Record<string,ReturnType<typeof t>>={a:t('관측 감쇠 a [s⁻¹]','Observed decay a [s⁻¹]'),b:t('숨은 값의 관측 결합 b [s⁻¹]','Hidden-to-observed coupling b [s⁻¹]'),c:t('관측의 숨은 결합 c [s⁻¹]','Observed-to-hidden coupling c [s⁻¹]'),d:t('숨은 감쇠 d [s⁻¹]','Hidden decay d [s⁻¹]'),epsilon:t('시간척도 ε','Timescale epsilon'),x:t('초기 관측 x₀','Initial observed x₀'),y:t('초기 숨은 값 y₀','Initial hidden y₀'),time:t('최종 물리 시간 [s]','Final physical time [s]'),lift:t('블록 A의 lifting 비율 α','Lifting fraction alpha in block A')};
const specials:Record<string,ReturnType<typeof t>>={transitionDenominator:t('전이 확률의 공통 분모','Common transition denominator'),steps:t('최종 이산 갱신 횟수','Final discrete update count'),memoryDepth:t('유지할 기억 지연 수 L','Retained memory lag count L'),sampleSteps:t('관측 시간 간격 수','Observation time intervals')};
export function coarseFields(input:CoarseInput){return Object.entries(input.parameters).flatMap(([id,value])=>{
 if(Array.isArray(value))return value.map((v,i)=>({id:`${id}.${i}`,value:v,label:t(`초기 미시 상태 ${i}의 무게`,`Initial micro-state ${i} weight`)}));
 const denominator=id.endsWith('Denominator'),base=id.replace(/Numerator$|Denominator$/,''),name=labels[base],edge=/^weight([0-2])([0-2])$/.exec(id);
 const label=specials[id]??(edge?t(`전이 ${edge[1]}→${edge[2]} · 분자`,`Transition ${edge[1]}→${edge[2]} · numerator`):name?t(name.ko+(denominator?' · 분모':' · 분자'),name.en+(denominator?' · denominator':' · numerator')):t(id,id));
 return[{id,value:value as number,label}];
});}
export function editCoarseField(input:CoarseInput,id:string,raw:string):CoarseInput|null{
 const value=parseNumberDraft(raw,-1000000000,1000000000,true);if(value===null)return null;
 const initial=/^initialWeights\.([0-2])$/.exec(id);
 if(initial){if(input.kind!==coarseKinds[0])return null;const weights=[...input.parameters.initialWeights];weights[Number(initial[1])]=value;try{return snapshotCoarseInput({...input,parameters:{...input.parameters,initialWeights:weights}});}catch{return null;}}
 if(!Object.hasOwn(input.parameters,id)||typeof(input.parameters as unknown as Record<string,unknown>)[id]!=='number')return null;
 try{return snapshotCoarseInput({...input,parameters:{...input.parameters,[id]:value}} as CoarseInput);}catch{return null;}
}
