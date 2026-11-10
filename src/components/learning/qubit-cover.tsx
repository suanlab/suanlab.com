'use client';
import { useId } from 'react';
import { useLanguage } from '@/components/language-provider';
import type { QubitTeachingId } from '@/lib/learning-qubit-teaching-input';
export default function QubitCover({ kind }: { kind: QubitTeachingId }) {
  const id = useId(), { language } = useLanguage();
  const blue = 'var(--learning-blue)', orange = 'var(--learning-orange)', neutral = 'var(--learning-neutral)';
  const circuit = kind === 'finite-qubit-circuit-order-interference', channel = kind === 'qubit-channel-kraus-choi-verification', environment = kind === 'system-environment-stinespring-reference';
  return <svg viewBox="0 0 640 340" role="img" aria-labelledby={id} data-qubit-cover={kind} style={{ width: '100%', height: 'auto', maxHeight: 340, color: 'hsl(var(--foreground))' }}>
    <title id={id}>{language === 'ko' ? '양자 회로·채널 개념 모식도 · 현재 입력의 수치 결과가 아닙니다.' : 'Quantum circuit and channel schematic; not numerical results for the current input.'}</title>
    <g fill="none" stroke={neutral} strokeWidth={3}>
      {circuit ? <>{[125, 225].map(y => <path key={y} d={`M65 ${y}H575`} />)}<rect x={130} y={90} width={80} height={70} rx={8} stroke={blue} /><rect x={285} y={90} width={80} height={70} rx={8} stroke={orange} /><path d="M480 125V225" stroke={blue} /><circle cx={480} cy={125} r={9} fill={blue} stroke="none" /><circle cx={480} cy={225} r={25} stroke={blue} /><path d="M455 225H505M480 200V250" stroke={blue} /><g fill="currentColor" stroke="none" fontSize={42} textAnchor="middle"><text x={170} y={140}>H</text><text x={325} y={140}>T</text></g></>
        : <><path d="M65 170H215M425 170H575" /><rect x={215} y={95} width={210} height={150} rx={15} stroke={blue} /><path d={environment ? 'M320 245V300H525' : 'M425 170L470 125M425 170L470 215'} stroke={orange} /><g fill="currentColor" stroke="none" fontSize={42} textAnchor="middle"><text x={105} y={145}>ρ</text><text x={320} y={185}>{channel ? 'Φ' : environment ? 'V' : '𝓘₀, 𝓘₁'}</text><text x={530} y={145}>{environment ? 'B' : channel ? 'Φ(ρ)' : 'r'}</text>{environment && <text x={555} y={315}>E</text>}</g></>}
    </g>
  </svg>;
}
