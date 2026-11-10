'use client';
import { useLanguage } from '@/components/language-provider';
export default function BlockOrderCover() {
  const { language } = useLanguage();
  return <svg viewBox="0 0 480 250" role="img" aria-label={language === 'ko' ? '서로 다른 미시 배치와 같은 블록 평균' : 'Different microscopic arrangements with the same block averages'}>
    <desc>{language === 'ko' ? '네 스핀의 서로 다른 두 배치가 블록 평균(0,0)으로 갑니다. 숨긴 배치들의 확률은 같지 않을 수 있습니다.' : 'Two arrangements of four spins map to block means(0,0). Hidden arrangements need not have equal probabilities.'}</desc>
    {[['+','−','+','−'],['+','−','−','+']].map((spins,row) => <g key={row}>{[0,1].map(b => <rect key={b} x={25+b*115} y={35+row*110} width={108} height={60} rx={9} fill="none" stroke="var(--learning-blue)" strokeWidth={2}/>)}
      {spins.map((spin,i) => <text key={i} x={45+i*56} y={74+row*110} fill={spin === '+' ? 'var(--learning-blue)' : 'var(--learning-orange)'} fontSize={23}>{spin}</text>)}
      <path d={`M270 ${65+row*110}H310`} stroke="currentColor" strokeWidth={2}/><text x={330} y={72+row*110} fill="currentColor" fontSize={18}>m=(0,0)</text>
    </g>)}<text x={35} y={230} fill="currentColor" fontSize={17}>Q_k(x) = Σ_d n(k,d) x^d</text>
  </svg>;
}
