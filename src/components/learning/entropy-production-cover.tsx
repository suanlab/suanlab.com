/** Decorative cover; numerical comparisons are computed in the dedicated worker. */
export default function EntropyProductionCover({ kind }: { kind: string }) {
  return <svg data-entropy-cover={kind} viewBox="0 0 480 320" aria-hidden="true" style={{ width: '100%', height: '100%' }}><g fill="none" stroke="currentColor" strokeWidth={3}><circle cx={120} cy={80} r={27} /><circle cx={240} cy={170} r={27} /><circle cx={360} cy={80} r={27} /><path d="M148 95L215 145M208 131L215 145L199 145M265 145L335 95M319 96L335 95L331 110M328 65H152M166 56L152 65L166 74" /><path d="M45 220V285H435M60 278L410 230" /><path d="M60 250H410" strokeDasharray="7 6" /><path d="M97 120L210 202M270 202L383 120M325 38H155" strokeDasharray="5 6" opacity={.45} /></g></svg>;
}
