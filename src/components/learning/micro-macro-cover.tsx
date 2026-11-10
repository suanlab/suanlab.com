import React from 'react';
import type { MicroMacroKind } from '@/lib/learning-micro-macro';
/** Decorative finite examples. Exact interactive values live in the lab. */
export default function MicroMacroCover({ kind }: { kind: MicroMacroKind }) {
  return <svg viewBox="0 0 360 240" aria-hidden="true" className="ai-book-art">
    {kind === 'binary-shell-multiplicity-block-counts' ? <>
      {['1100', '1010', '1001', '0110', '0101', '0011'].map((pattern, i) => {
        const x = 32 + (i % 2) * 95, y = 32 + Math.floor(i / 2) * 65;
        return <g key={pattern}><path d={`M${x + 66} ${y + 19}L270 112`} stroke="var(--learning-neutral)" fill="none" strokeWidth="1.2" />
          <rect x={x} y={y} width="66" height="38" rx="7" fill="var(--learning-blue)" fillOpacity=".12" stroke="var(--learning-blue)" /><text x={x + 33} y={y + 24} textAnchor="middle" fill="currentColor" fontSize="15">{pattern}</text></g>;
      })}<circle cx="282" cy="112" r="39" fill="var(--learning-orange)" fillOpacity=".13" stroke="var(--learning-orange)" strokeWidth="2" />
      <text x="282" y="109" textAnchor="middle" fill="currentColor" fontSize="17">K=2</text><text x="282" y="132" textAnchor="middle" fill="currentColor" fontSize="13">Ω=6</text><text x="180" y="224" textAnchor="middle" fill="currentColor" fontSize="14">1/16 · 6/16 · 1/6</text>
    </> : kind === 'finite-macro-partition-conditional-entropy' ? <>
      {[.6, .2, .1, .1].map((p, i) => <rect key={i} x={32 + i * 38} y={104 - 100 * p} width="23" height={100 * p} rx="3" fill="var(--learning-blue)" />)}
      {[.8, .2].map((p, i) => <rect key={i} x={236 + i * 44} y={104 - 100 * p} width="27" height={100 * p} rx="3" fill="var(--learning-orange)" />)}
      <path d="M190 70H221M213 63L221 70L213 77M270 132V153" fill="none" stroke="var(--learning-neutral)" strokeWidth="2" />
      <text x="95" y="126" textAnchor="middle" fill="currentColor" fontSize="14">p</text><text x="273" y="126" textAnchor="middle" fill="currentColor" fontSize="14">P</text>
      {[.4, .4, .1, .1].map((p, i) => <rect key={i} x={130 + i * 38} y={213 - 100 * p} width="23" height={100 * p} rx="3" fill="var(--learning-green)" />)}
      <text x="83" y="199" textAnchor="middle" fill="currentColor" fontSize="14">pbar</text>
    </> : <>
      <text x="180" y="15" textAnchor="middle" fill="currentColor" fontSize="11">p=(½,½), e=¼</text>
      <path d="M75 62L270 40M75 62L270 198M75 165L270 118M75 165L270 198" fill="none" stroke="var(--learning-neutral)" strokeWidth="2" />
      {[62, 165].map((y, i) => <g key={i}><circle cx="61" cy={y} r="24" fill="var(--learning-blue)" fillOpacity=".13" stroke="var(--learning-blue)" strokeWidth="2" /><text x="61" y={y + 5} textAnchor="middle" fill="currentColor" fontSize="14">X={i}</text></g>)}
      {[40, 118, 198].map((y, i) => <g key={i}><rect x="267" y={y - 20} width="60" height="40" rx="8" fill={i === 2 ? 'var(--learning-orange)' : 'var(--learning-green)'} fillOpacity=".13" stroke={i === 2 ? 'var(--learning-orange)' : 'var(--learning-green)'} strokeWidth="2" /><text x="297" y={y + 5} textAnchor="middle" fill="currentColor" fontSize="14">{i === 2 ? '?' : `Y=${i}`}</text></g>)}
      <text x="180" y="235" textAnchor="middle" fill="currentColor" fontSize="14">I(X;Y) = ¾ ln 2</text>
    </>}
  </svg>;
}
