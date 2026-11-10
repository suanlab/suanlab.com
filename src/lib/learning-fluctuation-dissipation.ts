/** UI-safe bilingual preparations; numerical work stays in the native worker. */
import {fdtKinds,fdtParameterKeys,isFdtInput,snapshotFdtInput,fdtInputKey,type FdtInput,type FdtKind} from './learning-fluctuation-dissipation-protocol';
export {fdtKinds,fdtInputKey};export type{FdtInput,FdtKind};
export type FdtText={ko:string;en:string};export const fdtText=(ko:string,en:string):FdtText=>({ko,en});
export const isFdtKind=(v:string):v is FdtKind=>(fdtKinds as readonly string[]).includes(v);
export const fdtTitles:Record<FdtKind,FdtText>={'thermal-ou-fluctuation-response':fdtText('열적 요동과 힘에 대한 응답','Thermal fluctuations and force response')};
const names:Record<typeof fdtParameterKeys[number],FdtText>={
 mobilityNumerator:fdtText('이동도 μ 분자 [m/(N·s)]','Mobility μ numerator [m/(N·s)]'),stiffnessNumerator:fdtText('강성 κ 분자 [N/m]','Stiffness κ numerator [N/m]'),thermalEnergyNumerator:fdtText('열에너지 θ 분자 [J]','Thermal energy θ numerator [J]'),coefficientDenominator:fdtText('μ·κ·θ 공통 분모','Common μ,κ,θ denominator'),
 angularNumerator:fdtText('회전율 ω 분자 [rad/s]','Rotation ω numerator [rad/s]'),angularDenominator:fdtText('회전율 분모','Rotation denominator'),
 forceXNumerator:fdtText('계단 힘 F₀ 분자 [N]','Step force F₀ numerator [N]'),forceYNumerator:fdtText('계단 힘 F₁ 분자 [N]','Step force F₁ numerator [N]'),forceDenominator:fdtText('계단 힘 공통 분모','Common step-force denominator'),
 impulseXNumerator:fdtText('충격 적분 I₀ 분자 [N·s]','Impulse integral I₀ numerator [N·s]'),impulseYNumerator:fdtText('충격 적분 I₁ 분자 [N·s]','Impulse integral I₁ numerator [N·s]'),impulseDenominator:fdtText('충격 적분 공통 분모','Common impulse denominator'),
 timeNumerator:fdtText('최종 시차 T 분자 [s]','Final lag T numerator [s]'),timeDenominator:fdtText('시차 분모','Lag denominator'),sampleSteps:fdtText('같은 시간 구간의 표본 간격 수','Sample intervals over the same duration'),
};
const signed=['angularNumerator','forceXNumerator','forceYNumerator','impulseXNumerator','impulseYNumerator'];
export const fdtFields=(input:FdtInput)=>fdtParameterKeys.map(id=>({id,label:names[id],value:input.parameters[id],minimum:signed.includes(id)?-1000000:id==='timeNumerator'?0:1,maximum:id==='sampleSteps'?64:1000000}));
export function editFdtDraft(input:FdtInput,id:string,raw:string):FdtInput|null {
 const field=fdtFields(input).find(f=>f.id===id),text=raw.trim();if(!field||!/^[-]?\d{1,7}$/.test(text))return null;
 const next={kind:input.kind,parameters:{...input.parameters,[id]:Number(text)}};return isFdtInput(next)?snapshotFdtInput(next):null;
}
export function fdtPresetInput(kind:FdtKind,index:number){const p=fdtPresets[kind][index];if(!Number.isInteger(index)||!p)throw new RangeError('Unknown thermal preparation');return snapshotFdtInput(p.input);}
export const fdtPresets:Record<FdtKind,{id:string;label:FdtText;input:FdtInput}[]>={
  "thermal-ou-fluctuation-response": [
    {
      "id": "fdt-0",
      "label": {
        "ko": "평형 기준",
        "en": "Equilibrium reference"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 0,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-1",
      "label": {
        "ko": "두 배 이동도",
        "en": "Twice the mobility"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 2,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 0,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-2",
      "label": {
        "ko": "두 배 강성",
        "en": "Twice the stiffness"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 2,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 0,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-3",
      "label": {
        "ko": "세 배 열에너지",
        "en": "Threefold thermal energy"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 3,
          "coefficientDenominator": 1,
          "angularNumerator": 0,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-4",
      "label": {
        "ko": "반시계 방향 구동",
        "en": "Counterclockwise driven background"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-5",
      "label": {
        "ko": "시계 방향 구동",
        "en": "Clockwise driven background"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": -1,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-6",
      "label": {
        "ko": "가로 방향 힘과 충격",
        "en": "Transverse force and impulse"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "forceXNumerator": 0,
          "forceYNumerator": 1,
          "forceDenominator": 1,
          "impulseXNumerator": 0,
          "impulseYNumerator": 1,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-7",
      "label": {
        "ko": "서로 다른 힘·충격 방향",
        "en": "Distinct force and impulse directions"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 0,
          "impulseYNumerator": 1,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-8",
      "label": {
        "ko": "영 시간·평형 오른쪽 극한",
        "en": "Zero time equilibrium right limit"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 0,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 0,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-9",
      "label": {
        "ko": "영 시간·구동 오른쪽 극한",
        "en": "Zero time driven right limit"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 0,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-10",
      "label": {
        "ko": "짧은 시차·촘촘한 기록",
        "en": "Short lag and dense records"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1000000,
          "sampleSteps": 64
        }
      }
    },
    {
      "id": "fdt-11",
      "label": {
        "ko": "최대 각도·양방향 성분",
        "en": "Largest angle and both force components"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 4,
          "angularDenominator": 1,
          "forceXNumerator": 4,
          "forceYNumerator": -4,
          "forceDenominator": 1,
          "impulseXNumerator": -4,
          "impulseYNumerator": 4,
          "impulseDenominator": 1,
          "timeNumerator": 4,
          "timeDenominator": 1,
          "sampleSteps": 64
        }
      }
    },
    {
      "id": "fdt-12",
      "label": {
        "ko": "아주 작은 감쇠와 순환",
        "en": "Tiny damping and driven circulation"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1000000,
          "angularNumerator": 4,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 64
        }
      }
    },
    {
      "id": "fdt-13",
      "label": {
        "ko": "양의 응답·선형 표시 미해결",
        "en": "Positive response beyond linear display precision"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1000000,
          "stiffnessNumerator": 1000000,
          "thermalEnergyNumerator": 1000000,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-14",
      "label": {
        "ko": "같은 Gaussian 폭·다른 강성",
        "en": "Same Gaussian width different restoring force"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 2,
          "thermalEnergyNumerator": 2,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-15",
      "label": {
        "ko": "대각 응답0 근처",
        "en": "Near a zero diagonal response"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 355,
          "angularDenominator": 226,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 16
        }
      }
    },
    {
      "id": "fdt-16",
      "label": {
        "ko": "유리수 각도는 정확한 pi가 아님",
        "en": "Rational angle is not exactly pi"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 355,
          "angularDenominator": 113,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 16
        }
      }
    },
    {
      "id": "fdt-17",
      "label": {
        "ko": "힘과 충격 모두0",
        "en": "Zero applied force and impulse"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "forceXNumerator": 0,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 0,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-18",
      "label": {
        "ko": "빠른 시계·같은 시차 위상",
        "en": "Same lag phase on a faster clock"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 2,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 2,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 2,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-19",
      "label": {
        "ko": "서로 다른 계수와 부호 힘",
        "en": "Unequal coefficients and signed forcing"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 2,
          "stiffnessNumerator": 3,
          "thermalEnergyNumerator": 5,
          "coefficientDenominator": 1,
          "angularNumerator": -4,
          "angularDenominator": 1,
          "forceXNumerator": 3,
          "forceYNumerator": -2,
          "forceDenominator": 2,
          "impulseXNumerator": -1,
          "impulseYNumerator": 2,
          "impulseDenominator": 3,
          "timeNumerator": 1,
          "timeDenominator": 1,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-20",
      "label": {
        "ko": "음수 성분 진단량",
        "en": "Negative component diagnostic"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 1,
          "stiffnessNumerator": 1,
          "thermalEnergyNumerator": 1,
          "coefficientDenominator": 1,
          "angularNumerator": 1,
          "angularDenominator": 1,
          "forceXNumerator": 1,
          "forceYNumerator": 0,
          "forceDenominator": 1,
          "impulseXNumerator": 1,
          "impulseYNumerator": 0,
          "impulseDenominator": 1,
          "timeNumerator": 1,
          "timeDenominator": 4,
          "sampleSteps": 8
        }
      }
    },
    {
      "id": "fdt-21",
      "label": {
        "ko": "같은 준비·다른 정수 표현",
        "en": "Same exact preparation different integer encoding"
      },
      "input": {
        "kind": "thermal-ou-fluctuation-response",
        "parameters": {
          "mobilityNumerator": 2,
          "stiffnessNumerator": 2,
          "thermalEnergyNumerator": 2,
          "coefficientDenominator": 2,
          "angularNumerator": 2,
          "angularDenominator": 2,
          "forceXNumerator": 2,
          "forceYNumerator": 0,
          "forceDenominator": 2,
          "impulseXNumerator": 2,
          "impulseYNumerator": 0,
          "impulseDenominator": 2,
          "timeNumerator": 2,
          "timeDenominator": 2,
          "sampleSteps": 8
        }
      }
    }
  ]
};
