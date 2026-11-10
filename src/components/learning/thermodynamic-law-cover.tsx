import type { ThermodynamicLawKind } from '@/lib/learning-thermodynamic-laws';

/** Decorative motifs; the lab contains the calculated physical diagrams. */
export default function ThermodynamicLawCover({ kind }: { kind: ThermodynamicLawKind }) {
  return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art">
    {kind === 'carnot-cycle-engine-refrigerator' ? <>
      <path d="M55 35V195H315" fill="none" stroke="var(--learning-neutral)" strokeWidth="2" />
      <path d="M85 70Q130 65 185 92Q205 120 265 157Q210 177 160 150Q135 110 85 70Z" fill="var(--learning-blue)" fillOpacity=".12" stroke="var(--learning-blue)" strokeWidth="4" />
      <path d="M85 70Q130 65 185 92M160 150Q210 177 265 157" fill="none" stroke="var(--learning-orange)" strokeWidth="5" />
      <circle cx="185" cy="92" r="6" fill="var(--learning-green)" />
    </> : kind === 'two-reservoir-cycle-entropy-admissibility' ? <>
      <g fill="var(--learning-blue)" fillOpacity=".12" stroke="var(--learning-blue)" strokeWidth="3"><rect x="55" y="20" width="130" height="45" rx="12" /><rect x="55" y="95" width="130" height="55" rx="12" /><rect x="55" y="185" width="130" height="40" rx="12" /></g>
      <path d="M120 65V90m-7-8 7 8 7-8M120 150V180m-7-8 7 8 7-8M185 122H290m-9-7 9 7-9 7" fill="none" stroke="var(--learning-orange)" strokeWidth="4" />
    </> : <>
      <path d="M45 175H160M200 70H315" stroke="var(--learning-blue)" strokeWidth="4" />
      {[65, 90, 115, 140].map(x => <circle key={x} cx={x} cy="162" r="7" fill="var(--learning-green)" />)}
      {[220, 245, 270].map(x => <circle key={x} cx={x} cy="57" r="7" fill="var(--learning-orange)" />)}
      <path d="M175 165V80m-6 8 6-8 6 8m-6 77-6-8m6 8 6-8" fill="none" stroke="var(--learning-neutral)" strokeWidth="2" />
    </>}
  </svg>;
}
