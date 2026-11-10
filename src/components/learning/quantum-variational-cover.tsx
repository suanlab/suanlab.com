import type { QuantumVariationalKind } from '@/lib/learning-quantum-variational';

/** Decorative chapter artwork, not an eigenvalue or probability calculation. */
export default function QuantumVariationalCover({ kind }: { kind: QuantumVariationalKind }) {
  const rayleigh = kind === 'finite-state-rayleigh-energy-variance';
  return <svg data-qv-cover={kind} viewBox="0 0 320 180" aria-hidden="true" style={{ width: '100%', height: '100%' }}>
    <rect width={320} height={180} fill="hsl(var(--muted))" />
    {rayleigh ? <>
      {[45, 85, 135].map((y, i) => <g key={y}>
        <path d={`M30 ${y}H145`} stroke="var(--learning-neutral)" strokeWidth={2} />
        <circle cx={50 + i * 35} cy={y} r={6} fill="var(--learning-orange)" />
        <path d={`M185 ${y}h${[80, 45, 25][i]}`} stroke="var(--learning-blue)" strokeWidth={9} />
      </g>)}
      <path d="M95 25V155" stroke="var(--learning-blue)" strokeWidth={2} strokeDasharray="5 4" />
    </> : <>
      <path d="M45 120 140 155 270 90 175 55Z" fill="var(--learning-green)" fillOpacity={0.14} stroke="var(--learning-green)" strokeWidth={2} />
      <path d="M150 105 225 110" stroke="var(--learning-blue)" strokeWidth={4} />
      <path d="M150 105 150 32M144 40 150 32 156 40" fill="none" stroke="var(--learning-orange)" strokeWidth={4} />
      <path d="M150 105 95 150M150 105 260 150M150 105V20" stroke="var(--learning-neutral)" strokeDasharray="4 4" />
      <circle cx={150} cy={105} r={5} fill="var(--learning-blue)" />
    </>}
  </svg>;
}
