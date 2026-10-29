import { formatNumber as number, type Vec3 } from '@/lib/learning-simulations';
export function NumberField({ label, value, onChange, min, max, step = .5 }: { label: string; value: number; onChange: (n: number) => void; min: number; max: number; step?: number | 'any' }) {
  return <label className="ai-number-field"><span>{label}</span><input type="number" min={min} max={max} step={step} value={value} onChange={e => {
    const n = e.target.valueAsNumber; if (Number.isFinite(n) && n >= min && n <= max) onChange(n);
  }} /></label>;
}
export function VectorFields({ label, value, change, min = -5, max = 5 }: { label: string; value: Vec3; change: (v: Vec3) => void; min?: number; max?: number }) {
  return <fieldset className="ai-vector-fields"><legend>{label}</legend><div className="ai-three-fields">{value.map((v, i) => <NumberField key={i} label={`${label} · ${['x', 'y', 'z'][i]}`} value={v} min={min} max={max} onChange={n => change(value.map((x, j) => j === i ? n : x) as Vec3)} />)}</div></fieldset>;
}
export function Tuple({ value }: { value: number[] }) { return <>({value.map(number).join(', ')})</>; }
