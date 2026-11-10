/** Persist applied inputs, unfinished edits and learning choices, never results or playback. */
import { blockOrderData as data, blockOrderDefault, readBlockOrderInput, type BlockOrderInput } from './learning-block-order-input';
import { blockOrderDraft, type BlockOrderDraft, type BlockOrderMode } from './learning-block-order-view';
export const blockOrderStorageKey = 'suanlab-block-order-v1';
export interface BlockOrderLimits { steps: number; guides: number; questions: number; options: number }
export interface BlockOrderDisplay { mode: BlockOrderMode; example: number; flat: boolean; page: number }
export interface BlockOrderSavedState {
  input: BlockOrderInput; draft: BlockOrderDraft; step: number; motion: boolean; guide: number; explain: boolean;
  answers: (number | null)[]; display: BlockOrderDisplay;
}
const integer = (x: unknown, lo: number, hi: number): x is number => typeof x === 'number' && Number.isInteger(x) && !Object.is(x,-0) && x >= lo && x <= hi;
export function readBlockOrderDraft(value: unknown): BlockOrderDraft | null {
  const d = data(value,['shape','blockShape','boundary','disagreementActivity','blockUpActivities','selectedUpCounts','witnessPair']);
  if (!d || !['open','periodic'].includes(d.boundary as string)) return null;
  const fields = { shape: 12, blockShape: 12, disagreementActivity: 13, blockUpActivities: 64, selectedUpCounts: 20, witnessPair: 12 };
  for (const [k,max] of Object.entries(fields)) if (typeof d[k] !== 'string' || (d[k] as string).length > max) return null;
  return Object.freeze({ ...d }) as unknown as BlockOrderDraft;
}
export function readBlockOrderState(value: unknown, limits: BlockOrderLimits): BlockOrderSavedState | null {
  try {
    if (!Object.values(limits).every(n => integer(n,1,64))) return null;
    const d = data(value,['input','draft','step','motion','guide','explain','answers','display']);
    if (!d || !integer(d.step,0,limits.steps-1) || !integer(d.guide,0,limits.guides-1) || typeof d.motion !== 'boolean' || typeof d.explain !== 'boolean') return null;
    const input = readBlockOrderInput(d.input), draft = readBlockOrderDraft(d.draft), display = data(d.display,['mode','example','flat','page']);
    if (!input || !draft || !display || !['configuration','conditional','profile','ensemble'].includes(display.mode as string) || !integer(display.example,0,7) || typeof display.flat !== 'boolean') return null;
    const n = input.shape.reduce((a,b) => a*b,1), v = input.blockShape.reduce((a,b) => a*b,1), rows = (v+1)**(n/v);
    if (!integer(display.page,0,Math.ceil(rows/25)-1)) return null;
    const answer = d.answers;
    if (!Array.isArray(answer) || Object.getPrototypeOf(answer) !== Array.prototype || answer.length !== limits.questions || Reflect.ownKeys(answer).length !== limits.questions+1) return null;
    const answers: (number | null)[] = [];
    for (let i = 0; i < limits.questions; i++) {
      const e = Object.getOwnPropertyDescriptor(answer,String(i));
      if (!e || !Object.hasOwn(e,'value') || e.value !== null && !integer(e.value,0,limits.options-1)) return null;
      answers.push(e.value);
    }
    return Object.freeze({ input,draft,step: d.step,motion: d.motion,guide: d.guide,explain: d.explain,
      answers: Object.freeze(answers) as (number | null)[], display: Object.freeze({ ...display }) as unknown as BlockOrderDisplay });
  } catch { return null; }
}
export function emptyBlockOrderState(limits: BlockOrderLimits): BlockOrderSavedState {
  return readBlockOrderState({ input: blockOrderDefault,draft: blockOrderDraft(blockOrderDefault),step: 0,motion: true,guide: 0,explain: false,
    answers: Array(limits.questions).fill(null),display: { mode: 'configuration',example: 0,flat: true,page: 0 } },limits)!;
}
export function parseStoredBlockOrderState(raw: string, limits: BlockOrderLimits): BlockOrderSavedState | null {
  try { if (typeof raw !== 'string' || raw.length > 8000) return null; const d = data(JSON.parse(raw),['version','model']); return d?.version === 1 ? readBlockOrderState(d.model,limits) : null; } catch { return null; }
}
