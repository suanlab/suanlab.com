import React from 'react';
import type { BoltzmannGibbsKind } from '@/lib/learning-boltzmann-gibbs';
/** Decorative ideal hand examples; actual observations come from workers. */
export default function BoltzmannGibbsCover({ kind }: { kind: BoltzmannGibbsKind }) {
  return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art">{kind === 'reference-measure-boltzmann-softmax-identifiability' ? <>
    <text x="180" y="28" textAnchor="middle" fontSize="15" fill="currentColor">pᵢ ∝ mᵢ exp(−βEᵢ)</text>
    {[4 / 9, 4 / 9, 1 / 9].map((p, i) => <g key={i}><rect x={45 + 105 * i} y={186 - 200 * p} width="50" height={200 * p} fill="var(--learning-blue)" /><text x={70 + 105 * i} y="208" textAnchor="middle" fontSize="13" fill="currentColor">m={i === 1 ? 2 : 1}</text></g>)}
    <text x="180" y="233" textAnchor="middle" fontSize="12" fill="currentColor">H(label) + E[ln m]</text>
  </> : kind === 'finite-gibbs-conditionals-scan-support' ? <>
    <text x="180" y="27" textAnchor="middle" fontSize="15" fill="currentColor">πK₁K₂=π · flow ≠ reverse flow</text>
    <path d="M86 80H274V180H86Z" fill="none" stroke="var(--learning-blue)" strokeWidth="2" />{['−−', '−+', '+−', '++'].map((s, i) => <g key={s}><circle cx={86 + 188 * (i % 2)} cy={80 + 100 * Math.floor(i / 2)} r="25" stroke="var(--learning-orange)" fill="hsl(var(--background))" /><text x={86 + 188 * (i % 2)} y={86 + 100 * Math.floor(i / 2)} textAnchor="middle" fontSize="18" fill="currentColor">{s}</text></g>)}
    <text x="180" y="232" textAnchor="middle" fontSize="13" fill="currentColor">K₁ · K₂ · random · block</text>
  </> : <>
    <text x="180" y="27" textAnchor="middle" fontSize="15" fill="currentColor">A ≻ 0 · Σ=τA⁻¹</text>
    {[1, 2, 3].map(r => <ellipse key={r} cx="180" cy="128" rx={23 * r} ry={14 * r} transform="rotate(-35 180 128)" fill="var(--learning-blue)" fillOpacity=".04" stroke={r === 2 ? 'var(--learning-orange)' : 'var(--learning-blue)'} strokeWidth="2" />)}
    <text x="180" y="229" textAnchor="middle" fontSize="13" fill="currentColor">density ≠ probability mass</text>
  </>}</svg>;
}
