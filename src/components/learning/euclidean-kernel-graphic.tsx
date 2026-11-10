import React, { useId } from 'react';
import { euclideanKernelSteps } from '@/data/euclidean-kernel-teaching';
export default function EuclideanKernelGraphic({ step = 0, language, compact = false }: { step?: number; language: 'ko' | 'en'; compact?: boolean }) {
  const id = useId(), s = Math.max(0, Math.min(7, step)), t = (ko: string, en: string) => language === 'ko' ? ko : en;
  const label = (x: number, y: number, ko: string, en: string, middle = false) => <text x={x} y={y} textAnchor={middle ? 'middle' : 'start'} fill="currentColor">{t(ko, en)}</text>;
  const box = (x: number, title: string, value: string, detail: string) => {
    const lines=[''];for(const word of detail.split(' ')){const i=lines.length-1;if(lines[i]&&lines[i].length+word.length+1>24)lines.push(word);else lines[i]+=(lines[i]?' ':'')+word;}
    return <g><rect x={x} y={125} width={200} height={150} rx={10} fill="none" stroke="var(--learning-blue)" strokeWidth={2}/><text x={x+100} y={165} textAnchor="middle" fill="currentColor" fontSize={16}>{title}</text><text x={x+100} y={210} textAnchor="middle" fill="currentColor">{value}</text><text x={x+100} textAnchor="middle" fill="currentColor" fontSize={15}>{lines.map((line,i)=><tspan key={i} x={x+100} y={lines.length===1?250:241+i*21}>{line}</tspan>)}</text></g>;
  };
  return <svg viewBox="0 0 720 400" role="img" aria-labelledby={`${id}-title ${id}-desc`} data-euclidean-reference={s} style={{ width: '100%', minWidth: compact ? 0 : 720, fontSize: 17 }}>
    <title id={`${id}-title`}>{euclideanKernelSteps[s].title[language]}</title><desc id={`${id}-desc`}>{euclideanKernelSteps[s].diagram[language]}</desc>
    {label(28, 30, '설명용 기준 · 현재 입력과 별개', 'Explanatory reference; separate from current input')}
    {s === 0 && <>
      {label(55, 70, 'N=2; m=ℏ=τ=ω²=1; a=0; b=1', 'N=2; m=hbar=τ=ω²=1; a=0; b=1')}
      <path d="M85 285L635 125" fill="none" stroke="var(--learning-neutral)" strokeWidth={3} strokeDasharray="6 5" /><polyline points="85,285 360,213.889 635,125" fill="none" stroke="var(--learning-blue)" strokeWidth={4} />
      {[['85','285'],['360','213.889'],['635','125']].map(([x,y],i)=><circle key={i} cx={x} cy={y} r={8} fill={i===1?'var(--learning-blue)':'var(--learning-orange)'} />)}
      {label(330, 190, '자유 1/2', 'Free 1/2')}{label(375, 245, '유한 4/9', 'Finite 4/9')}
      {label(85, 325, 't=0', 't=0')}{label(360, 325, 't=1/2', 't=1/2', true)}{label(635, 325, 't=1', 't=1', true)}
      {label(360, 375, '평균의 선분 연결 · 무작위 표본 경로 아님', 'Interpolated means; not random sample paths', true)}
    </>}
    {s === 1 && <>
      {label(360, 95, 'U_N=δ[V(a)/2+Σ내부 V(xᵢ)+V(b)/2]/ℏ', 'U_N=δ[V(a)/2+Σinterior V(xᵢ)+V(b)/2]/hbar', true)}
      {box(25,t('경로 가중치', 'Path weight'),'∏ Gδ × exp(−U_N)',t('내부 좌표 적분', 'Integrate interior'))}
      {box(260,t('끝점 커널', 'Endpoint kernel'),'K_N(b,a)',t('끝점 밀도 단위', 'Inverse-length units'))}
      {box(495,t('고정끝 내부 법칙', 'Pinned interior law'),'p=weight/K_N','∫p dx₁…dx_N−1=1')}
      {label(360, 335, 'Gaussian 앞 인자 N개를 모두 유지', 'Keep every one of the N Gaussian prefactors', true)}
      {label(360, 375, '내부 밀도의 적분1과 끝점 커널을 구별', 'Separate a unit interior integral from the endpoint kernel', true)}
    </>}
    {s === 2 && <>
      {label(55, 80, 'N=2 기준 · 내부 좌표 하나', 'N=2 reference; one interior coordinate')}
      {box(25,'Q μ=f','(9/2)(4/9)=2',t('두 끝 forcing 합산', 'Add both endpoint terms'))}
      {box(260,'Q C=1','(9/2)(2/9)=1',t('평균과 공분산 구분', 'Separate mean and covariance'))}
      {box(495,'s=2; v=1/4','Q=s(2+v)=9/2','C_endpoints=0')}
      {label(360, 355, '고정끝 간선도 대각에 기여합니다.', 'Pinned endpoint edges also contribute to the diagonal.', true)}
    </>}
    {s === 3 && <>
      {label(45, 75, 'A₂(x)=9(x−4/9)²/4+49/72', 'A₂(x)=9(x−4/9)²/4+49/72')}
      <polyline points={Array.from({length:41},(_,i)=>{const x=-.5+i*1.9/40;return `${85+550*i/40},${320-75*(9/4*(x-4/9)**2+49/72)}`;}).join(' ')} fill="none" stroke="var(--learning-blue)" strokeWidth={3}/>
      <circle cx={85+550*(4/9+.5)/1.9} cy={320-75*49/72} r={8} fill="var(--learning-green)"/>
      {label(360, 125, '최소 작용 49/72 · det Q=9/2', 'Minimum action49/72; det Q=9/2', true)}
      {label(360, 355, 'K₂(1,0)=2 exp(−49/72)/(3√π)', 'K₂(1,0)=2 exp(−49/72)/(3√π)', true)}
      {label(360, 385, '지수와 요동의 앞 인자를 함께 유지', 'Retain both the exponent and fluctuation prefactor', true)}
    </>}
    {s === 4 && <>
      {label(45, 80, 'v=1/4 고정 · 주행렬식의 점화식', 'Fixed v=1/4; leading-determinant recurrence')}
      {['D₀=1','D₁=9/4','D₂=65/16','D₃=441/64'].map((v,i)=><g key={i}><rect x={35+i*172} y={130} width={155} height={75} fill="var(--learning-blue)" opacity={.16}/><text x={112+i*172} y={175} textAnchor="middle" fill="currentColor">{v}</text></g>)}
      {label(360, 255, 'D_k=(2+v)D_(k−1)−D_(k−2)', 'D_k=(2+v)D_(k−1)−D_(k−2)', true)}
      {label(360, 300, 'θ=2asinh(ωδ/2); ω_eff=θ/δ; m_eff=m sinhθ/θ', 'θ=2asinh(ωδ/2); ω_eff=θ/δ; m_eff=m sinhθ/θ', true)}
      {label(360, 360, '이 행렬식 나열은 고정시간 정련이 아닙니다.', 'These leading minors are not fixed-time refinement.', true)}
    </>}
    {s === 5 && <>
      {label(45, 70, 'm=ℏ=τ=ω²=1; a=1/3; b=4/3', 'm=hbar=τ=ω²=1; a=1/3; b=4/3')}
      {label(45, 100, '세로 log₁₀|log(K_N/K_cont)| · 근사', 'Vertical log₁₀|log(K_N/K_cont)|; approximate')}
      <path d="M65 120V330H640" fill="none" stroke="currentColor"/>{[-1,-3,-5].map(v=><text key={v} x={52} y={105-45*v} textAnchor="end" fill="currentColor">{v}</text>)}
      {[[.001623467342291,.000402481260428,.000100406242253,.0000250881418669],[.105790134009,.052485814594,.026142072909,.0130459214752]].map((values,p)=><g key={p}><polyline points={values.map((v,i)=>`${100+170*i},${100-45*Math.log10(v)}`).join(' ')} fill="none" stroke={p?'var(--learning-orange)':'var(--learning-blue)'} strokeWidth={3} strokeDasharray={p?'7 4':undefined}/>{values.map((v,i)=><circle key={i} cx={100+170*i} cy={100-45*Math.log10(v)} r={6} fill={p?'var(--learning-orange)':'var(--learning-blue)'}/>)}</g>)}
      {[4,8,16,32].map((n,i)=><text key={n} x={100+170*i} y={345} textAnchor="middle" fill="currentColor">N={n}</text>)}
      {label(360, 380, '파랑: 대칭 · 주황: 오른쪽 끝점 · 예제의 관찰', 'Blue:symmetric; orange:right endpoint; example-specific', true)}
    </>}
    {s === 6 && <>
      {label(45, 75, 'N=2; m=ℏ=τ=ω²=1; K₂(x,x)=2 exp(−17x²/36)/(3√π)', 'N=2; m=hbar=τ=ω²=1; K₂(x,x)=2 exp(−17x²/36)/(3√π)')}
      <polygon points={`85,300 ${Array.from({length:61},(_,i)=>{const x=-3+i/10;return `${85+550*i/60},${300-400*2/(3*Math.sqrt(Math.PI))*Math.exp(-17*x*x/36)}`;}).join(' ')} 635,300`} fill="var(--learning-blue)" opacity={.22}/>
      <path d="M85 300H635" stroke="currentColor"/>
      {label(360, 125, '전체 직선 Z₂=4/√17 · 그림은 −3…3', 'Full-line Z₂=4/√17; drawing spans −3…3', true)}
      {label(85, 330, 'x=−3', 'x=−3')}{label(360, 330, 'x=0', 'x=0', true)}{label(635, 330, 'x=3', 'x=3', true)}
      {label(360, 375, '공통 끝점을 적분 · τ=βℏ', 'Integrate the common endpoint; τ=βhbar', true)}
    </>}
    {s === 7 && <>
      {box(25,t('고정끝 내부 밀도', 'Pinned density'),'∫p dx=1',t('내부 위치 정규화', 'Normalize interior values'))}
      {box(260,t('끝점 커널', 'Endpoint kernel'),'∫K db=√(32/49)',t('기준 예 a=0', 'Reference a=0'))}
      {box(495,t('열적 위치 밀도', 'Thermal position density'),'∫K(x,x)/Z dx=1',t('trace로 정규화', 'Normalize by the trace'))}
      {label(360, 335, '무한 직선: tr K_free=∞', 'Infinite line: tr K_free=∞', true)}
      {label(360, 375, '양성만으로 Born 확률이나 전이 법칙이 되지 않음', 'Positivity alone gives neither Born nor transition probabilities', true)}
    </>}
  </svg>;
}
