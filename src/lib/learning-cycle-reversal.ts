/** Bilingual preparations and edits. This module imports no numerical evaluator. */
import { reversalKinds, reversalScalarKeys, isReversalInput, snapshotReversalInput, reversalInputKey, type ReversalKind, type ReversalInput } from './learning-cycle-reversal-protocol';
import { parseNumberJsonDraft } from './learning-json-number-input';
export { reversalKinds, reversalInputKey };
export type { ReversalKind, ReversalInput };
export type ReversalText = { ko: string; en: string };
export const reversalText = (ko: string, en: string): ReversalText => ({ ko, en });
export const isReversalKind = (v: string): v is ReversalKind => (reversalKinds as readonly string[]).includes(v);
export const reversalTitles: Record<ReversalKind, ReversalText> = { 'continuous-cycle-path-reversal': reversalText('연속시간 경로와 상세 균형 위반', 'Continuous-time paths and broken detailed balance') };
const names: Record<typeof reversalScalarKeys[number], ReversalText> = {
  clockwise01: reversalText('0→1 전이율 분자 [1/s]', '0→1 rate numerator [1/s]'),
  clockwise12: reversalText('1→2 전이율 분자 [1/s]', '1→2 rate numerator [1/s]'),
  clockwise20: reversalText('2→0 전이율 분자 [1/s]', '2→0 rate numerator [1/s]'),
  counter10: reversalText('1→0 전이율 분자 [1/s]', '1→0 rate numerator [1/s]'),
  counter21: reversalText('2→1 전이율 분자 [1/s]', '2→1 rate numerator [1/s]'),
  counter02: reversalText('0→2 전이율 분자 [1/s]', '0→2 rate numerator [1/s]'),
  rateDenominator: reversalText('전이율 공통 분모', 'Common rate denominator'),
  timeDenominator: reversalText('대기시간 공통 분모', 'Common dwell-time denominator'),
};
export const reversalFields = (input: ReversalInput) => reversalScalarKeys.map(id => ({ id, label: names[id], value: input.parameters[id], minimum: 1, maximum: 1000000 }));
export function editReversalDraft(input: ReversalInput, id: string, raw: string): ReversalInput | null {
  const field = reversalFields(input).find(f => f.id === id), text = raw.trim();
  if (!field || !/^\d{1,7}$/.test(text)) return null;
  const next = { kind: input.kind, parameters: { ...input.parameters, [id]: Number(text) } };
  return isReversalInput(next) ? snapshotReversalInput(next) : null;
}
export function editReversalJson(input: ReversalInput, raw: string): ReversalInput | null {
  try {
    if (raw.length > 32768) return null;
    const parsed = parseNumberJsonDraft(`{"parameters":${raw}}`, ['parameters']) as { parameters: unknown };
    const next = { kind: input.kind, parameters: parsed.parameters };
    return isReversalInput(next) ? snapshotReversalInput(next) : null;
  } catch { return null; }
}
export function reversalPresetInput(kind: ReversalKind, index: number) {
  const preset = reversalPresets[kind][index];
  if (!Number.isInteger(index) || !preset) throw new RangeError('Unknown reversal preparation');
  return snapshotReversalInput(preset.input);
}
export const reversalPresets: Record<ReversalKind, { id: string; label: ReversalText; input: ReversalInput }[]> = {
  "continuous-cycle-path-reversal": [
    {
      "id": "reversal-0",
      "label": {
        "ko": "대칭 순환",
        "en": "Symmetric cycle"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 1,
          "clockwise12": 1,
          "clockwise20": 1,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2,
            0
          ],
          "dwellNumerators": [
            1,
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-1",
      "label": {
        "ko": "비균일 가역·열린 경로",
        "en": "Nonuniform reversible open path"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 2,
          "clockwise12": 3,
          "clockwise20": 1,
          "counter10": 1,
          "counter21": 2,
          "counter02": 3,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2
          ],
          "dwellNumerators": [
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-2",
      "label": {
        "ko": "비균일 시계 방향 순환",
        "en": "Nonuniform clockwise cycle"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 2,
          "clockwise12": 3,
          "clockwise20": 4,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2,
            0
          ],
          "dwellNumerators": [
            1,
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-3",
      "label": {
        "ko": "같은 생성자의 역순환",
        "en": "Same generator reverse cycle"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 2,
          "clockwise12": 3,
          "clockwise20": 4,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            0,
            2,
            1,
            0
          ],
          "dwellNumerators": [
            1,
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-4",
      "label": {
        "ko": "비가역이어도 되짚기",
        "en": "Nonreversible backtrack"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 2,
          "clockwise12": 3,
          "clockwise20": 4,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            0
          ],
          "dwellNumerators": [
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-5",
      "label": {
        "ko": "서로 다른 대기율",
        "en": "Unequal waiting rates"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 1,
          "clockwise12": 2,
          "clockwise20": 7,
          "counter10": 3,
          "counter21": 5,
          "counter02": 11,
          "rateDenominator": 1,
          "states": [
            2,
            0,
            1,
            2
          ],
          "dwellNumerators": [
            1,
            3,
            2,
            4
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-6",
      "label": {
        "ko": "같은 상태·다른 대기 배분",
        "en": "Same state path different waits"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 1,
          "clockwise12": 2,
          "clockwise20": 7,
          "counter10": 3,
          "counter21": 5,
          "counter02": 11,
          "rateDenominator": 1,
          "states": [
            2,
            0,
            1,
            2
          ],
          "dwellNumerators": [
            4,
            2,
            3,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-7",
      "label": {
        "ko": "같은 히스토그램·가역",
        "en": "Same histogram reversible"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 1,
          "clockwise12": 1,
          "clockwise20": 1,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2
          ],
          "dwellNumerators": [
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-8",
      "label": {
        "ko": "같은 히스토그램·순환",
        "en": "Same histogram circulating"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 3,
          "clockwise12": 3,
          "clockwise20": 3,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2
          ],
          "dwellNumerators": [
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-9",
      "label": {
        "ko": "열 배 빠른 시계",
        "en": "Tenfold faster clock"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 20,
          "clockwise12": 30,
          "clockwise20": 40,
          "counter10": 10,
          "counter21": 10,
          "counter02": 10,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2,
            0
          ],
          "dwellNumerators": [
            1,
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-10",
      "label": {
        "ko": "상세 균형에 가까운 준비",
        "en": "Near detailed balance"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 1000000,
          "clockwise12": 1,
          "clockwise20": 1,
          "counter10": 999999,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2,
            0
          ],
          "dwellNumerators": [
            1,
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-11",
      "label": {
        "ko": "강한 반시계 방향 편향",
        "en": "Strong counterclockwise bias"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 1,
          "clockwise12": 1,
          "clockwise20": 1,
          "counter10": 1000000,
          "counter21": 1000000,
          "counter02": 1000000,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2,
            0
          ],
          "dwellNumerators": [
            1,
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-12",
      "label": {
        "ko": "영 시간·무점프",
        "en": "Zero duration no jump"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 2,
          "clockwise12": 3,
          "clockwise20": 4,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            1
          ],
          "dwellNumerators": [
            0
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-13",
      "label": {
        "ko": "양의 시간·무점프",
        "en": "No jump positive duration"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 2,
          "clockwise12": 3,
          "clockwise20": 4,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            1
          ],
          "dwellNumerators": [
            3
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-14",
      "label": {
        "ko": "관측 끝점의 점프",
        "en": "Final jump at observation endpoint"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 2,
          "clockwise12": 3,
          "clockwise20": 4,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            0,
            1
          ],
          "dwellNumerators": [
            1,
            0
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-15",
      "label": {
        "ko": "같은 전이율·다른 분수 입력",
        "en": "Same rates different exact encoding"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 4,
          "clockwise12": 6,
          "clockwise20": 8,
          "counter10": 2,
          "counter21": 2,
          "counter02": 2,
          "rateDenominator": 2,
          "states": [
            0,
            1,
            2,
            0
          ],
          "dwellNumerators": [
            1,
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-16",
      "label": {
        "ko": "비균일 가역 순환",
        "en": "Nonuniform reversible loop"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 2,
          "clockwise12": 3,
          "clockwise20": 1,
          "counter10": 1,
          "counter21": 2,
          "counter02": 3,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2,
            0
          ],
          "dwellNumerators": [
            1,
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-17",
      "label": {
        "ko": "아주 작은 양의 경로비",
        "en": "Positive underflow path ratio"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 1,
          "clockwise12": 1,
          "clockwise20": 1,
          "counter10": 1000000,
          "counter21": 1000000,
          "counter02": 1000000,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1
          ],
          "dwellNumerators": [
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
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-18",
      "label": {
        "ko": "매우 큰 경로비",
        "en": "Huge path ratio"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 1000000,
          "clockwise12": 1000000,
          "clockwise20": 1000000,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1,
            2,
            0,
            1
          ],
          "dwellNumerators": [
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
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    },
    {
      "id": "reversal-19",
      "label": {
        "ko": "아주 긴 대기",
        "en": "Very long waits"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 1000000,
          "clockwise12": 1000000,
          "clockwise20": 1000000,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            2,
            0
          ],
          "dwellNumerators": [
            1000000,
            1000000,
            1000000,
            1000000
          ],
          "timeDenominator": 1
        }
      }
    },
    {
      "id": "reversal-20",
      "label": {
        "ko": "작은 전이율과 짧은 대기",
        "en": "Tiny rate and short dwell"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 1,
          "clockwise12": 1,
          "clockwise20": 1,
          "counter10": 1,
          "counter21": 1,
          "counter02": 1,
          "rateDenominator": 1000000,
          "states": [
            0,
            2,
            1
          ],
          "dwellNumerators": [
            1,
            1,
            1
          ],
          "timeDenominator": 1000000
        }
      }
    },
    {
      "id": "reversal-21",
      "label": {
        "ko": "드문 상태의 비균일 점유",
        "en": "Rare nonuniform occupation"
      },
      "input": {
        "kind": "continuous-cycle-path-reversal",
        "parameters": {
          "clockwise01": 1,
          "clockwise12": 1,
          "clockwise20": 1,
          "counter10": 1000000,
          "counter21": 1,
          "counter02": 1000000,
          "rateDenominator": 1,
          "states": [
            0,
            1,
            0,
            2,
            0
          ],
          "dwellNumerators": [
            1,
            1,
            1,
            1,
            1
          ],
          "timeDenominator": 4
        }
      }
    }
  ]
};
