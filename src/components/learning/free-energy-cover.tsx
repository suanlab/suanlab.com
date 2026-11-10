import React from 'react';
import type { FreeEnergyKind } from '@/lib/learning-free-energy';
/** Decorative cover motifs; calculated curves and observations live in the lab. */
export default function FreeEnergyCover({ kind }: { kind: FreeEnergyKind }) {
  return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art">
    <path d="M40 30V200H325" stroke="var(--learning-neutral)" strokeWidth="2" fill="none" />
    {kind === 'finite-spectrum-helmholtz-kl-variational' ? <>
      {[0, 1, 2].map(i => <g key={i}><rect x={65 + 85 * i} y={80 + i * 30} width="24" height={120 - i * 30} rx="4" fill="var(--learning-blue)" /><rect x={94 + 85 * i} y={105 + i * 15} width="20" height={95 - i * 15} rx="4" fill="var(--learning-orange)" /></g>)}
      <path d="M58 55Q180 180 310 50" fill="none" stroke="var(--learning-green)" strokeWidth="3" />
    </> : <>
      <path d="M55 50Q155 125 310 155" fill="none" stroke="var(--learning-blue)" strokeWidth="4" />
      <path d="M55 80Q140 220 210 140T310 60" fill="none" stroke="var(--learning-green)" strokeWidth="4" />
      <path d="M55 159H315" fill="none" stroke="var(--learning-neutral)" strokeDasharray="5 5" strokeWidth="2" />
      <circle cx="145" cy="157" r="7" fill="var(--learning-orange)" />
    </>}
  </svg>;
}
