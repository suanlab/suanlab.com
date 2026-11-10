/** Reader-safe persistence. Store drafts and choices, never numerical responses or run flags. */
import { readThermalFieldInput, thermalFieldData as data, thermalFieldDefault, type ThermalFieldInput } from './learning-thermal-field-input';
import { thermalDraft, type ThermalDraft, type ThermalMatrix, type ThermalScale } from './learning-thermal-field-view';
export const thermalFieldStorageKey = 'suanlab-thermal-field-v1';
export interface ThermalLearningLimits { steps: number; guides: number; questions: number; options: number }
export interface ThermalDisplay { kind: ThermalMatrix; scale: ThermalScale; selected: number; flat: boolean }
export interface ThermalSavedState {
  input: ThermalFieldInput; draft: ThermalDraft; step: number; motion: boolean; guide: number; explain: boolean;
  answers: (number | null)[]; display: ThermalDisplay;
}
const integer = (x: unknown, min: number, max: number): x is number => typeof x === 'number' && Number.isInteger(x) && !Object.is(x, -0) && x >= min && x <= max;
function textJet(value: unknown, length: number) {
  const own = data(value, ['value', 'first', 'second']);
  return own && Object.values(own).every(x => typeof x === 'string' && x.length <= length) ? Object.freeze({ ...own }) as unknown as ThermalDraft['source'] : null;
}
export function readThermalDraft(value: unknown): ThermalDraft | null {
  const own = data(value, ['shape', 'boundary', 'constraint', 'spacing', 'temperature', 'referenceScale', 'mass', 'stiffness', 'backgroundDensity', 'source']);
  if (!own || typeof own.shape !== 'string' || own.shape.length > 12 || !['open', 'periodic', 'dirichlet'].includes(own.boundary as string) || !['none', 'mean-zero'].includes(own.constraint as string)) return null;
  if (['spacing', 'temperature', 'referenceScale'].some(k => typeof own[k] !== 'string' || (own[k] as string).length > 38)) return null;
  const mass = textJet(own.mass, 38), stiffness = textJet(own.stiffness, 38), backgroundDensity = textJet(own.backgroundDensity, 38), source = textJet(own.source, 1280);
  return mass && stiffness && backgroundDensity && source ? Object.freeze({ ...own, mass, stiffness, backgroundDensity, source }) as unknown as ThermalDraft : null;
}
export function emptyThermalState(limits: ThermalLearningLimits): ThermalSavedState {
  return readThermalState({ input: thermalFieldDefault, draft: thermalDraft(thermalFieldDefault), step: 0, motion: true, guide: 0, explain: false,
    answers: Array(limits.questions).fill(null), display: { kind: 'covariance', scale: 'linear', selected: 0, flat: true } }, limits)!;
}
export function readThermalState(value: unknown, limits: ThermalLearningLimits): ThermalSavedState | null {
  try {
    if (!Object.values(limits).every(n => Number.isInteger(n) && n >= 1 && n <= 64)) return null;
    const own = data(value, ['input', 'draft', 'step', 'motion', 'guide', 'explain', 'answers', 'display']);
    if (!own || !integer(own.step, 0, limits.steps - 1) || !integer(own.guide, 0, limits.guides - 1) || typeof own.motion !== 'boolean' || typeof own.explain !== 'boolean') return null;
    const input = readThermalFieldInput(own.input), draft = readThermalDraft(own.draft), display = data(own.display, ['kind', 'scale', 'selected', 'flat']);
    if (!input || !draft || !display || !['covariance', 'inverseOnSpace', 'precision'].includes(display.kind as string) || !['linear', 'signed-log'].includes(display.scale as string) || typeof display.flat !== 'boolean') return null;
    const n = input.shape.reduce((a, b) => a * b, 1);
    if (!integer(display.selected, 0, n * n - 1)) return null;
    const answer = own.answers;
    if (!Array.isArray(answer) || Object.getPrototypeOf(answer) !== Array.prototype || answer.length !== limits.questions || Reflect.ownKeys(answer).length !== limits.questions + 1) return null;
    const answers: (number | null)[] = [];
    for (let i = 0; i < limits.questions; i++) {
      const d = Object.getOwnPropertyDescriptor(answer, String(i));
      if (!d || !Object.hasOwn(d, 'value') || d.value !== null && !integer(d.value, 0, limits.options - 1)) return null;
      answers.push(d.value);
    }
    return Object.freeze({ input, draft, step: own.step, motion: own.motion, guide: own.guide, explain: own.explain,
      answers: Object.freeze(answers) as unknown as (number | null)[], display: Object.freeze({ ...display }) as unknown as ThermalDisplay });
  } catch { return null; }
}
export function parseStoredThermalState(raw: string, limits: ThermalLearningLimits): ThermalSavedState | null {
  try { if (typeof raw !== 'string' || raw.length > 24000) return null; const own = data(JSON.parse(raw), ['version', 'model']); return own?.version === 1 ? readThermalState(own.model, limits) : null; } catch { return null; }
}
