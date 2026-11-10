import React from 'react';
import type { PartitionKind } from '@/lib/learning-partition';
/** Decorative fixed hand examples; actual numerical observations live in labs. */
export default function PartitionCover({ kind }: { kind: PartitionKind }) {
  return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art">
    {kind === 'finite-spectrum-partition-cumulant-generators' ? <>
      <text x="180" y="28" textAnchor="middle" fill="currentColor" fontSize="15">Z → U, Var(E), κ₃(E)</text>
      {[4 / 7, 2 / 7, 1 / 7].map((p, i) => <g key={i}><line x1={30 + 103 * i} x2={94 + 103 * i} y1={150 - 40 * i} y2={150 - 40 * i} stroke="var(--learning-orange)" strokeWidth="3" /><rect x={43 + 103 * i} y={205 - 130 * p} width="38" height={130 * p} fill="var(--learning-blue)" /><text x={62 + 103 * i} y="225" textAnchor="middle" fill="currentColor" fontSize="12">i={i}</text></g>)}
    </> : kind === 'coupled-binary-sites-partition-factorization' ? <>
      <text x="180" y="28" textAnchor="middle" fill="currentColor" fontSize="15">E=−Jσ₁σ₂−h(σ₁+σ₂)</text>
      {['−−', '−+', '+−', '++'].map((s, i) => <g key={s}><rect x={40 + 150 * (i % 2)} y={48 + 85 * Math.floor(i / 2)} width="128" height="72" rx="8" fill="var(--learning-blue)" fillOpacity={i === 0 || i === 3 ? '.4' : '.1'} stroke="var(--learning-blue)" /><text x={104 + 150 * (i % 2)} y={78 + 85 * Math.floor(i / 2)} textAnchor="middle" fill="currentColor" fontSize="19">{s}</text><text x={104 + 150 * (i % 2)} y={102 + 85 * Math.floor(i / 2)} textAnchor="middle" fill="currentColor" fontSize="12">{i === 0 || i === 3 ? '2/5' : '1/10'}</text></g>)}
      <text x="180" y="226" textAnchor="middle" fill="currentColor" fontSize="13">E[σ₁]=E[σ₂]=0 · Cov=3/5</text>
    </> : <>
      <text x="180" y="28" textAnchor="middle" fill="currentColor" fontSize="15">q=1/2 · L=2 · δ=1/8</text>
      {[.5, .25, .125, .0625, .03125, .015625].map((p, i) => <g key={i}><rect x={38 + i * 48} y={185 - 250 * p} width="30" height={250 * p} fill={i <= 2 ? 'var(--learning-blue)' : 'var(--learning-orange)'} fillOpacity={i <= 2 ? '.9' : '.5'} /><text x={53 + i * 48} y="205" textAnchor="middle" fill="currentColor" fontSize="12">{i}</text></g>)}
      <path d="M179 45V190" fill="none" stroke="var(--learning-neutral)" strokeDasharray="5 4" /><text x="180" y="230" textAnchor="middle" fill="currentColor" fontSize="13">ZL / Z∞ = 1−δ</text>
    </>}
  </svg>;
}
