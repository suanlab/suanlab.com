import React from 'react';
import type { EnsembleKind } from '@/lib/learning-ensembles';
/** Decorative small finite examples. Measured values belong in the lab. */
export default function EnsembleCover({ kind }: { kind: EnsembleKind }) {
  return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art">
    {kind === 'finite-binary-reservoir-canonical-comparison' ? <>
      <rect x="24" y="25" width="115" height="62" rx="10" fill="var(--learning-blue)" fillOpacity=".12" stroke="var(--learning-blue)" />
      <rect x="219" y="25" width="115" height="62" rx="10" fill="var(--learning-orange)" fillOpacity=".12" stroke="var(--learning-orange)" />
      <text x="81" y="62" textAnchor="middle" fill="currentColor" fontSize="16">n=2</text><text x="276" y="62" textAnchor="middle" fill="currentColor" fontSize="16">b=2</text>
      <path d="M145 56h67m-9-6l9 6-9 6" stroke="var(--learning-neutral)" fill="none" strokeWidth="2" />
      {[1 / 6, 4 / 6, 1 / 6].map((p, i) => <rect key={`p${i}`} x={63 + 90 * i} y={205 - 130 * p} width="22" height={130 * p} fill="var(--learning-blue)" />)}
      {[.25, .5, .25].map((p, i) => <rect key={`q${i}`} x={87 + 90 * i} y={205 - 130 * p} width="22" height={130 * p} fill="var(--learning-orange)" fillOpacity=".55" stroke="var(--learning-orange)" strokeDasharray="2 2" />)}
      <text x="180" y="228" textAnchor="middle" fill="currentColor" fontSize="13">K=2 · P(A|K), Q(A)</text>
    </> : kind === 'canonical-binary-energy-fluctuations' ? <>
      <text x="180" y="30" textAnchor="middle" fill="currentColor" fontSize="15">n=2 · ε/(kBT)=ln 2</text>
      {[4 / 9, 4 / 9, 1 / 9].map((p, i) => <g key={i}><rect x={53 + i * 100} y={193 - 240 * p} width="52" height={240 * p} rx="4" fill="var(--learning-blue)" /><text x={79 + i * 100} y="215" textAnchor="middle" fill="currentColor" fontSize="13">K={i}</text></g>)}
      <path d="M34 193h292" stroke="var(--learning-neutral)" /><text x="180" y="239" textAnchor="middle" fill="currentColor" fontSize="12">Var(E)=ε² Var(K)</text>
    </> : <>
      <text x="180" y="25" textAnchor="middle" fill="currentColor" fontSize="15">M=2 · μ=ε · T&gt;0</text>
      {['00', '01', '10', '11'].map((pattern, i) => <g key={pattern}><rect x={36 + i * 77} y="55" width="57" height="43" rx="7" fill="var(--learning-green)" fillOpacity=".12" stroke="var(--learning-green)" /><text x={64 + i * 77} y="82" textAnchor="middle" fill="currentColor" fontSize="15">{pattern}</text></g>)}
      {[.25, .5, .25].map((p, i) => <g key={i}><rect x={61 + 95 * i} y={207 - 170 * p} width="45" height={170 * p} fill="var(--learning-green)" /><text x={83 + 95 * i} y="227" textAnchor="middle" fill="currentColor" fontSize="13">N={i}</text></g>)}
      <path d="M34 207h292" stroke="var(--learning-neutral)" />
    </>}
  </svg>;
}
