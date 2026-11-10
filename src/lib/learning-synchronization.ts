/** Fixed teaching preparations. All numerical calculation remains in Workers. */
import {synchronizationKinds,snapshotSynchronizationInput,synchronizationInputKey,type SynchronizationInput} from './learning-synchronization-protocol';
import {parseNumberDraft} from './learning-number-input';
export {synchronizationKinds,synchronizationInputKey};export type {SynchronizationInput};
export type SynchronizationKind=typeof synchronizationKinds[number];
export const isSynchronizationKind=(s:string):s is SynchronizationKind=>(synchronizationKinds as readonly string[]).includes(s);
export const synchronizationText=(ko:string,en:string)=>({ko,en});
const t=synchronizationText;
export const synchronizationTitles={
 [synchronizationKinds[0]]:t('진폭의 탄생과 임계 이완','Amplitude onset and critical relaxation'),
 [synchronizationKinds[1]]:t('두 진동자의 잠김과 미끄러짐','Two-oscillator locking and phase slips'),
 [synchronizationKinds[2]]:t('유한 집단의 질서와 동조','Order and locking in finite populations'),
};
const preparations:Record<SynchronizationKind,{id:string;label:{ko:string;en:string};input:SynchronizationInput}[]>={
  "hopf-amplitude-critical-relaxation": [
    {
      "id": "below onset",
      "label": {
        "ko": "임계 아래",
        "en": "Below onset"
      },
      "input": {
        "kind": "hopf-amplitude-critical-relaxation",
        "parameters": {
          "muNumerator": -1,
          "muDenominator": 1,
          "gNumerator": 1,
          "gDenominator": 1,
          "shearNumerator": 0,
          "shearDenominator": 1,
          "omegaNumerator": 2,
          "omegaDenominator": 1,
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
      "id": "critical",
      "label": {
        "ko": "임계의 대수 감쇠",
        "en": "Critical algebraic decay"
      },
      "input": {
        "kind": "hopf-amplitude-critical-relaxation",
        "parameters": {
          "muNumerator": 0,
          "muDenominator": 1,
          "gNumerator": 1,
          "gDenominator": 1,
          "shearNumerator": 0,
          "shearDenominator": 1,
          "omegaNumerator": 2,
          "omegaDenominator": 1,
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
      "id": "on periodic circle",
      "label": {
        "ko": "주기 궤도",
        "en": "Periodic orbit"
      },
      "input": {
        "kind": "hopf-amplitude-critical-relaxation",
        "parameters": {
          "muNumerator": 1,
          "muDenominator": 1,
          "gNumerator": 1,
          "gDenominator": 1,
          "shearNumerator": 0,
          "shearDenominator": 1,
          "omegaNumerator": 2,
          "omegaDenominator": 1,
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
      "id": "circle of equilibria",
      "label": {
        "ko": "평형점들의 원",
        "en": "Circle of equilibria"
      },
      "input": {
        "kind": "hopf-amplitude-critical-relaxation",
        "parameters": {
          "muNumerator": 1,
          "muDenominator": 1,
          "gNumerator": 1,
          "gDenominator": 1,
          "shearNumerator": 2,
          "shearDenominator": 1,
          "omegaNumerator": 2,
          "omegaDenominator": 1,
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
      "id": "origin above onset",
      "label": {
        "ko": "불안정한 정확 원점",
        "en": "Exact unstable origin"
      },
      "input": {
        "kind": "hopf-amplitude-critical-relaxation",
        "parameters": {
          "muNumerator": 1,
          "muDenominator": 1,
          "gNumerator": 1,
          "gDenominator": 1,
          "shearNumerator": 0,
          "shearDenominator": 1,
          "omegaNumerator": 2,
          "omegaDenominator": 1,
          "xNumerator": 0,
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
      "id": "worked critical shear",
      "label": {
        "ko": "임계의 위상 전단",
        "en": "Critical phase shear"
      },
      "input": {
        "kind": "hopf-amplitude-critical-relaxation",
        "parameters": {
          "muNumerator": 0,
          "muDenominator": 1,
          "gNumerator": 1,
          "gDenominator": 1,
          "shearNumerator": 1,
          "shearDenominator": 1,
          "omegaNumerator": 2,
          "omegaDenominator": 1,
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
  ],
  "two-oscillator-locking-phase-slips": [
    {
      "id": "stable exact fixed phase",
      "label": {
        "ko": "안정 잠김",
        "en": "Stable locked phase"
      },
      "input": {
        "kind": "two-oscillator-locking-phase-slips",
        "parameters": {
          "omega1Numerator": 3,
          "omega1Denominator": 1,
          "omega2Numerator": 0,
          "omega2Denominator": 1,
          "couplingNumerator": 5,
          "couplingDenominator": 1,
          "initialHalfA": 1,
          "initialHalfB": 3,
          "meanPhaseNumerator": 0,
          "meanPhaseDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "unstable exact fixed phase",
      "label": {
        "ko": "불안정 잠김",
        "en": "Unstable locked phase"
      },
      "input": {
        "kind": "two-oscillator-locking-phase-slips",
        "parameters": {
          "omega1Numerator": 3,
          "omega1Denominator": 1,
          "omega2Numerator": 0,
          "omega2Denominator": 1,
          "couplingNumerator": 5,
          "couplingDenominator": 1,
          "initialHalfA": 3,
          "initialHalfB": 1,
          "meanPhaseNumerator": 0,
          "meanPhaseDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "critical from beyond the bottleneck",
      "label": {
        "ko": "임계 병목 통과",
        "en": "Critical bottleneck passage"
      },
      "input": {
        "kind": "two-oscillator-locking-phase-slips",
        "parameters": {
          "omega1Numerator": 1,
          "omega1Denominator": 1,
          "omega2Numerator": 0,
          "omega2Denominator": 1,
          "couplingNumerator": 1,
          "couplingDenominator": 1,
          "initialHalfA": 2,
          "initialHalfB": 1,
          "meanPhaseNumerator": 0,
          "meanPhaseDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "drift positive",
      "label": {
        "ko": "양의 미끄러짐",
        "en": "Positive phase slips"
      },
      "input": {
        "kind": "two-oscillator-locking-phase-slips",
        "parameters": {
          "omega1Numerator": 6,
          "omega1Denominator": 1,
          "omega2Numerator": 0,
          "omega2Denominator": 1,
          "couplingNumerator": 3,
          "couplingDenominator": 1,
          "initialHalfA": 0,
          "initialHalfB": 1,
          "meanPhaseNumerator": 0,
          "meanPhaseDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "neutral antiphase",
      "label": {
        "ko": "중립 반대 위상",
        "en": "Neutral antiphase"
      },
      "input": {
        "kind": "two-oscillator-locking-phase-slips",
        "parameters": {
          "omega1Numerator": 2,
          "omega1Denominator": 1,
          "omega2Numerator": 2,
          "omega2Denominator": 1,
          "couplingNumerator": 0,
          "couplingDenominator": 1,
          "initialHalfA": 1,
          "initialHalfB": 0,
          "meanPhaseNumerator": 0,
          "meanPhaseDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    }
  ],
  "finite-kuramoto-collective-order": [
    {
      "id": "heterogeneous four",
      "label": {
        "ko": "서로 다른 네 주파수",
        "en": "Four different frequencies"
      },
      "input": {
        "kind": "finite-kuramoto-collective-order",
        "parameters": {
          "omegaNumerators": [
            -1,
            0,
            1,
            2
          ],
          "phasePiNumerators": [
            0,
            1,
            2,
            3
          ],
          "omegaDenominator": 1,
          "phasePiDenominator": 2,
          "couplingNumerator": 3,
          "couplingDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "integrationSteps": 128,
          "sampleSteps": 8,
          "referenceWidthNumerator": 1,
          "referenceWidthDenominator": 1
        }
      }
    },
    {
      "id": "identical aligned",
      "label": {
        "ko": "완전 정렬",
        "en": "Aligned phases"
      },
      "input": {
        "kind": "finite-kuramoto-collective-order",
        "parameters": {
          "omegaNumerators": [
            1,
            1,
            1,
            1
          ],
          "phasePiNumerators": [
            0,
            0,
            0,
            0
          ],
          "omegaDenominator": 1,
          "phasePiDenominator": 2,
          "couplingNumerator": 3,
          "couplingDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "integrationSteps": 128,
          "sampleSteps": 8,
          "referenceWidthNumerator": 1,
          "referenceWidthDenominator": 1
        }
      }
    },
    {
      "id": "identical antiphase clusters",
      "label": {
        "ko": "두 반대 집단",
        "en": "Antiphase clusters"
      },
      "input": {
        "kind": "finite-kuramoto-collective-order",
        "parameters": {
          "omegaNumerators": [
            1,
            1,
            1,
            1
          ],
          "phasePiNumerators": [
            0,
            0,
            2,
            2
          ],
          "omegaDenominator": 1,
          "phasePiDenominator": 2,
          "couplingNumerator": 3,
          "couplingDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "integrationSteps": 128,
          "sampleSteps": 8,
          "referenceWidthNumerator": 1,
          "referenceWidthDenominator": 1
        }
      }
    },
    {
      "id": "identical quarter roots",
      "label": {
        "ko": "사분점 배열",
        "en": "Quarter roots"
      },
      "input": {
        "kind": "finite-kuramoto-collective-order",
        "parameters": {
          "omegaNumerators": [
            1,
            1,
            1,
            1
          ],
          "phasePiNumerators": [
            0,
            1,
            2,
            3
          ],
          "omegaDenominator": 1,
          "phasePiDenominator": 2,
          "couplingNumerator": 3,
          "couplingDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "integrationSteps": 128,
          "sampleSteps": 8,
          "referenceWidthNumerator": 1,
          "referenceWidthDenominator": 1
        }
      }
    },
    {
      "id": "antiphase roundoff-sensitive",
      "label": {
        "ko": "대칭과 계산 오차",
        "en": "Symmetry and roundoff"
      },
      "input": {
        "kind": "finite-kuramoto-collective-order",
        "parameters": {
          "omegaNumerators": [
            0,
            0
          ],
          "phasePiNumerators": [
            0,
            1
          ],
          "omegaDenominator": 1,
          "phasePiDenominator": 1,
          "couplingNumerator": 16,
          "couplingDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "integrationSteps": 128,
          "sampleSteps": 8,
          "referenceWidthNumerator": 1,
          "referenceWidthDenominator": 1
        }
      }
    },
    {
      "id": "coarse insufficient grid",
      "label": {
        "ko": "의도적으로 거친 격자",
        "en": "Deliberately coarse grid"
      },
      "input": {
        "kind": "finite-kuramoto-collective-order",
        "parameters": {
          "omegaNumerators": [
            -1,
            0,
            1,
            2
          ],
          "phasePiNumerators": [
            0,
            1,
            2,
            3
          ],
          "omegaDenominator": 1,
          "phasePiDenominator": 2,
          "couplingNumerator": 16,
          "couplingDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "integrationSteps": 1,
          "sampleSteps": 1,
          "referenceWidthNumerator": 1,
          "referenceWidthDenominator": 1
        }
      }
    },
    {
      "id": "regular32roots",
      "label": {
        "ko": "32개 균형 위상",
        "en": "32 balanced phases"
      },
      "input": {
        "kind": "finite-kuramoto-collective-order",
        "parameters": {
          "omegaNumerators": [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1
          ],
          "phasePiNumerators": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31
          ],
          "omegaDenominator": 1,
          "phasePiDenominator": 16,
          "couplingNumerator": 3,
          "couplingDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "integrationSteps": 64,
          "sampleSteps": 8,
          "referenceWidthNumerator": 1,
          "referenceWidthDenominator": 1
        }
      }
    },
    {
      "id": "zero time",
      "label": {
        "ko": "시간 0",
        "en": "Zero time"
      },
      "input": {
        "kind": "finite-kuramoto-collective-order",
        "parameters": {
          "omegaNumerators": [
            -1,
            0,
            1,
            2
          ],
          "phasePiNumerators": [
            0,
            1,
            2,
            3
          ],
          "omegaDenominator": 1,
          "phasePiDenominator": 2,
          "couplingNumerator": 3,
          "couplingDenominator": 1,
          "timeNumerator": 0,
          "timeDenominator": 1,
          "integrationSteps": 128,
          "sampleSteps": 8,
          "referenceWidthNumerator": 1,
          "referenceWidthDenominator": 1
        }
      }
    }
  ]
};
export const synchronizationPresets=Object.fromEntries(synchronizationKinds.map(kind=>[kind,preparations[kind].map(p=>({...p,input:snapshotSynchronizationInput(p.input)}))])) as typeof preparations;
export function synchronizationPresetInput(kind:SynchronizationKind,index:number){const p=synchronizationPresets[kind][index];if(!Number.isInteger(index)||!p)throw new RangeError('Unknown synchronization preparation');return snapshotSynchronizationInput(p.input);}
const labels:Record<string,ReturnType<typeof t>>={mu:t('선형 성장률 μ','Linear growth μ'),g:t('양의 포화 g','Positive saturation g'),shear:t('위상 전단 c','Phase shear c'),omega:t('각주파수 ω','Angular frequency ω'),x:t('초기 실수 좌표 x₀','Initial real coordinate x₀'),y:t('초기 허수 좌표 y₀','Initial imaginary coordinate y₀'),omega1:t('고유 주파수 ω₁','Natural frequency ω₁'),omega2:t('고유 주파수 ω₂','Natural frequency ω₂'),coupling:t('전체 결합 K','Total coupling K'),meanPhase:t('초기 평균 위상 Φ₀','Initial mean phase Φ₀'),time:t('최종 물리 시간 [s]','Final physical time [s]'),referenceWidth:t('별도 Lorentzian 폭 γ','Separate Lorentzian width γ'),phasePi:t('초기 위상 / π','Initial phase / π')};
const specials:Record<string,ReturnType<typeof t>>={initialHalfA:t('초기 상대 반각의 a','Initial relative half-angle a'),initialHalfB:t('초기 상대 반각의 b','Initial relative half-angle b'),sampleSteps:t('표시 기록 간격 수','Display record intervals'),integrationSteps:t('기본 RK4 간격 수 n','Base RK4 steps n')};
export function synchronizationFields(input:SynchronizationInput){return Object.entries(input.parameters).flatMap(([id,value])=>{
 if(typeof value!=='number')return[];const denominator=id.endsWith('Denominator'),base=id.replace(/Numerator$|Denominator$/,''),name=labels[base];
 const label=specials[id]??(name?t(name.ko+(denominator?' · 분모':' · 분자'),name.en+(denominator?' · denominator':' · numerator')):t(id,id));
 return[{id,value,label}];
});}
export function editSynchronizationField(input:SynchronizationInput,id:string,raw:string):SynchronizationInput|null{
 if(!Object.hasOwn(input.parameters,id)||typeof(input.parameters as unknown as Record<string,unknown>)[id]!=='number')return null;
 const v=parseNumberDraft(raw,-1000000000,1000000000,true);if(v===null)return null;
 try{return snapshotSynchronizationInput({...input,parameters:{...input.parameters,[id]:v}} as SynchronizationInput);}catch{return null;}
}
