import React, {useId} from 'react';
import {realTimePhaseSteps} from '@/data/real-time-phase-teaching';
import {realTimeReferenceSquare,realTimeReferencePhases,realTimeReferenceAction,realTimeReferenceDamping,
  realTimeReferenceDensities,realTimeReferenceVariances,realTimeReferenceFolds} from '@/lib/learning-real-time-phase-reference';

export default function RealTimePhaseGraphic({step=0,language,compact=false}:{step?:number;language:'ko'|'en';compact?:boolean}) {
  const id=useId(),s=!compact&&Number.isInteger(step)&&step>=0&&step<realTimePhaseSteps.length?step:0;
  const t=(ko:string,en:string)=>language==='ko'?ko:en,blue='var(--learning-blue)',orange='var(--learning-orange)',green='var(--learning-green)',neutral='var(--learning-neutral)';
  const label=(x:number,y:number,text:string,middle=false,size=17)=><text x={x} y={y} textAnchor={middle?'middle':'start'} fill="currentColor" fontSize={size}>{text}</text>;
  const arrow=(x:number,y:number,dx:number,dy:number,color=blue,dashed=false)=>{
    const angle=Math.atan2(dy,dx),endX=x+dx,endY=y+dy;
    return <g><line x1={x} y1={y} x2={endX} y2={endY} stroke={color} strokeWidth={3} strokeDasharray={dashed?'7 5':undefined}/><path d={`M${endX},${endY}L${endX-10*Math.cos(angle-.45)},${endY-10*Math.sin(angle-.45)}L${endX-10*Math.cos(angle+.45)},${endY-10*Math.sin(angle+.45)}Z`} fill={color}/></g>;
  };
  const axes=(x:number,y:number,width:number,height:number)=><g stroke={neutral} strokeWidth={1.5}><line x1={x} x2={x+width} y1={y} y2={y}/><line x1={x} x2={x} y1={y-height} y2={y+12}/></g>;
  const box=(x:number,title:string,lines:string[])=> <g><rect x={x} y={130} width={210} height={150} rx={12} fill="none" stroke={blue} strokeWidth={2}/>{label(x+105,167,title,true)}{lines.map((line,i)=><g key={i}>{label(x+105,206+i*35,line,true)}</g>)}</g>;
  // The small cover uses a simpler composition; detailed captions belong to the full view.
  if(compact)return <svg viewBox="0 0 360 220" role="img" aria-labelledby={`${id}-title ${id}-desc`} data-real-time-reference={0} data-reference-compact="true" style={{display:'block',width:'100%'}}>
    <title id={`${id}-title`}>{realTimePhaseSteps[0].title[language]}</title><desc id={`${id}-desc`}>{realTimePhaseSteps[0].diagram[language]}</desc>
    <circle cx={85} cy={110} r={48} fill="none" stroke={neutral}/>
    {[[48,0],[0,-48],[-48,0],[0,48]].map(([dx,dy],i)=><g key={i}>{arrow(85,110,dx,dy,[blue,orange,green,neutral][i])}</g>)}
    {label(148,116,'1',true,24)}{label(85,47,'i',true,24)}{label(20,116,'−1',true,24)}{label(85,184,'−i',true,24)}
    {realTimeReferenceSquare.slice(0,-1).map(([x,y],i)=>{const next=realTimeReferenceSquare[i+1];return <g key={i}>{arrow(235+75*x,160-75*y,75*(next[0]-x),-75*(next[1]-y),[blue,orange,green,neutral][i])}</g>;})}
    {label(272,53,'Σ=0',true,30)}<circle cx={235} cy={160} r={4} fill={blue}/>
  </svg>;
  return <svg viewBox="0 0 720 440" role="img" aria-labelledby={`${id}-title ${id}-desc`} data-real-time-reference={s} data-reference-compact={compact?'true':'false'} style={{display:'block',width:'100%',minWidth:compact?0:720}}>
    <title id={`${id}-title`}>{realTimePhaseSteps[s].title[language]}</title><desc id={`${id}-desc`}>{realTimePhaseSteps[s].diagram[language]}</desc>
    {label(24,32,compact?t('실시간 위상 · 간섭','Real-time phase · interference'):t('설명용 고정 기준 · 현재 입력과 독립','Fixed teaching reference; independent of your input'),false,compact?25:17)}
    {s===0&&<>
      <circle cx={190} cy={220} r={90} fill="none" stroke={neutral} strokeWidth={2}/>
      {[[90,0], [0,-90], [-90,0], [0,90]].map(([dx,dy],i)=><g key={i}>{arrow(190,220,dx,dy,[blue,orange,green,neutral][i])}</g>)}
      {label(300,226,'1',false,compact?28:17)}{label(190,104,'i',true,compact?28:17)}{label(75,226,'−1',true,compact?28:17)}{label(190,338,'−i',true,compact?28:17)}
      {realTimeReferenceSquare.slice(0,-1).map(([x,y],i)=>{const next=realTimeReferenceSquare[i+1];return <g key={i}>{arrow(450+130*x,285-130*y,130*(next[0]-x),-130*(next[1]-y),[blue,orange,green,neutral][i])}</g>;})}
      <circle cx={450} cy={285} r={6} fill={blue}/>{label(515,119,'1+i−1−i=0',true,compact?32:23)}
      {label(360,390,t('네 단위 진폭 · 닫힌 합 · 경로적분의 표본 아님','Four unit amplitudes; closed sum; not sampled paths'),true,compact?22:18)}
    </>}
    {s===1&&<>
      {label(28,90,'φ=x')}{label(28,238,'φ=x²/2')}
      {realTimeReferencePhases.map((p,i)=><g key={i}>
        {[{angle:p.linear,y:140,color:orange},{angle:p.quadratic,y:285,color:blue}].map(({angle,y,color},j)=><g key={j}><circle cx={125+i*125} cy={y} r={35} stroke={neutral} fill="none"/>{arrow(125+i*125,y,32*Math.cos(angle),-32*Math.sin(angle),color)}</g>)}
        {label(125+i*125,355,`x=${p.x}`,true)}
      </g>)}
      {label(360,405,t('ℏ=1/4 · 단위 화살표5개 · 적분값 아님','hbar=1/4; five unit arrows; not integral values'),true)}
    </>}
    {s===2&&<>
      {label(35,88,'S(q)')}{label(38,235,'0')}{label(673,235,'q')}{axes(65,230,590,150)}<line x1={65} x2={65} y1={230} y2={350} stroke={neutral}/>
      <polyline points={realTimeReferenceAction.map(p=>`${65+(p.x+3)/4*590},${230-p.y*65}`).join(' ')} fill="none" stroke={blue} strokeWidth={4}/>
      <circle cx={360} cy={213.75} r={7} fill={orange}/>{label(380,195,'q*=−1; S*=1/4')}
      {label(65,355,'q=−3')}{label(655,355,'q=1',true)}{label(360,395,t('H=−1 · 최소점이 아닌 최대점','H=−1; a maximum, not a minimum'),true)}
    </>}
    {s===3&&<>
      {label(360,90,'H=[[0,−1],[−1,0]]; det H=−1',true,22)}
      {arrow(360,235,115,-115,blue)}{arrow(360,235,-115,115,blue)}
      {arrow(360,235,115,115,orange)}{arrow(360,235,-115,-115,orange)}
      {label(495,120,'(1,1): λ=−1')}{label(495,354,'(1,−1): λ=+1')}
      {label(360,405,t('첫 pivot0 · 역행렬 존재 · 음의 모드 유지','First pivot zero; inverse exists; retain the negative mode'),true)}
    </>}
    {s===4&&<>
      <circle cx={360} cy={235} r={105} fill="none" stroke={neutral}/>
      {arrow(360,235,-74.25,74.25,blue)}{arrow(360,235,74.25,-74.25,orange,true)}
      {label(30,175,t('각 sqrt의 곱','Product of roots'))}{label(30,208,'arg=−3π/4')}
      {label(485,265,t('곱의 주 sqrt','Principal root of product'))}{label(485,298,'arg=+π/4')}
      {label(360,385,t('자유 N=4 · ε→0 · 두 branch의 부호 반대','Free N=4; ε→0; opposite branch signs'),true)}
    </>}
    {s===5&&<>
      {box(30,t('호환','Compatible'),['a=1; b=−1','Hq=f: 0=0'])}{box(480,t('비호환','Incompatible'),['a=1; b=0','Hq=f: 0=1'])}
      <line x1={360} x2={360} y1={100} y2={300} stroke={orange} strokeWidth={3} strokeDasharray="7 5"/>
      {label(360,330,'b=−a',true)}{label(360,377,'K_N(b,a)=−iδ(b+a)',true,22)}
      {label(360,413,t('점선은 지지 조건 · delta의 유한 높이가 아님','Dashed support condition; no finite delta height'),true)}
    </>}
    {s===6&&<>
      {axes(70,310,580,210)}
      {realTimeReferenceDamping.map((p,i)=><g key={i}><line x1={360+130*p.x} x2={360+130*p.x} y1={310} y2={310-180*p.y} stroke={neutral} strokeDasharray="4 4"/><circle cx={360+130*p.x} cy={310-180*p.y} r={7} fill={blue}/>{label(360+130*p.x,343,`q=${p.x}`,true)}</g>)}
      {label(360,85,'|e^(iq²/2−q²/2)|=e^(−q²/2)',true,22)}
      {label(360,390,t('ε=ℏ=1 · 비정규화 감쇠 · 복소 위상은 남음','ε=hbar=1; unnormalized damping; phase remains'),true)}
    </>}
    {s===7&&<>
      {label(32,90,'ρ')}{label(32,330,'0')}{label(682,330,'x')}{axes(55,325,610,230)}
      <polyline points={realTimeReferenceDensities.map(p=>`${55+(p.x+4)/9*610},${325-460*p.y}`).join(' ')} fill="none" stroke={blue} strokeWidth={4}/>
      <polyline points={realTimeReferenceDensities.map(p=>`${55+(p.x+4)/9*610},${325-460*p.propagated}`).join(' ')} fill="none" stroke={orange} strokeWidth={3} strokeDasharray="7 5"/>
      {label(145,90,'ρ₀=N(0,1)')}{label(440,90,'ρ_t=N(1,5/4)')}
      {label(55,355,'x=−4')}{label(665,355,'x=5',true)}{label(360,400,t('정규화된 실제 위치 밀도 · 창 밖 꼬리는 생략','Normalized position densities; tails extend beyond the window'),true)}
    </>}
    {s===8&&<><line x1={235} x2={235} y1={95} y2={320} stroke={neutral}/><line x1={570} x2={570} y1={95} y2={320} stroke={neutral} strokeDasharray="3 5"/>{label(235,335,'0',true)}{label(570,335,'Σ_qq=1',true)}
      {realTimeReferenceVariances.map((r,i)=><g key={i}>
        {label(32,130+i*135,`ℏ=${r.hbar}`)}
        <line x1={235} x2={235+335*r.fixed} y1={117+i*135} y2={117+i*135} stroke={blue} strokeWidth={9}/>
        <line x1={235} x2={235+335*r.scaled} y1={162+i*135} y2={162+i*135} stroke={orange} strokeWidth={7} strokeDasharray="9 4"/>
        {label(650,124+i*135,r.fixedText,true)}{label(650,169+i*135,r.scaledText,true)}
      </g>)}
      {label(360,358,t('실선 σ²=1 · 점선 σ²=ℏ · 길이는 분산','Solid: σ²=1; dashed: σ²=hbar; length is variance'),true)}
      {label(360,405,'m=τ=1; χ=0; Σ_qq=σ²+ℏ²/(4σ²)',true)}
    </>}
    {s===9&&<>
      {arrow(125,305,425,0,blue)}{arrow(125,305,0,-190,orange)}
      {label(360,350,'N↑; τ fixed; δ↓',true)}{label(160,140,'ℏ↓; N fixed')}
      {label(235,210,'ω²<0: '+t('물리 역진동자','physical inversion'))}
      {label(235,255,'ω²>0, v>4: '+t('거친 격자','coarse grid'))}
      {label(360,408,t('유니터리 유한 곱 ≠ 연속 전파의 오차0','A unitary finite product does not imply zero continuum error'),true)}
    </>}
    {(s===10||s===11)&&<>
      {label(32,110,'φ_u(x)')}{label(680,235,'x')}{axes(60,230,600,145)}
      {(s===10?realTimeReferenceFolds.slice(0,1):realTimeReferenceFolds).map((curve,i)=><g key={curve.control}>
        <polyline points={curve.points.map(p=>`${65+(p.x+2)*147.5},${230-28*p.y}`).join(' ')} fill="none" stroke={[blue,orange,green][i]} strokeWidth={3} strokeDasharray={i===1?'7 5':i===2?'2 5':undefined}/>
        {curve.stationary.map(x=><circle key={x} cx={65+(x+2)*147.5} cy={230-28*(x*x*x/3-curve.control*x)} r={6} fill={[blue,orange,green][i]}/>)}
        <line x1={85+i*210} x2={150+i*210} y1={56} y2={56} stroke={[blue,orange,green][i]} strokeWidth={3} strokeDasharray={i===1?'7 5':i===2?'2 5':undefined}/>{label(85+i*210,80,`u=${curve.control}`)}
      </g>)}
      {s===10?<>{label(112,327,'x=−1; φ=2/3; φ″=−2')}{label(393,327,'x=+1; φ=−2/3; φ″=+2')}{label(360,405,t('진폭과 Fresnel 위상을 함께 합산','Sum amplitudes with their Fresnel phases'),true)}</>:<>{label(360,368,t('u>0: 2개 · u=0: 퇴화1개 · u<0: 0개','u>0: two; u=0: one degenerate; u<0: none'),true)}{label(360,410,'J=2πℏ^(1/3)Ai(−u/ℏ^(2/3))',true,22)}</>}
    </>}
    {s===12&&<>
      {box(20,t('양의 posterior','Positive posterior'),[t('Laplace 근사','Laplace approximation'),t('확률 모멘트','Probability moments')])}
      {box(255,t('복소 진폭','Complex amplitude'),[t('정상 위상','Stationary phase'),t('간섭·branch','Interference; branches')])}
      {box(490,t('정규화된 상태','Normalized state'),['ρ=|ψ|²',t('양의 공분산','Positive covariance')])}
      {label(360,352,t('공통 도구: Hessian · 모드 · 선형 응답','Shared tools: Hessians, modes, linear response'),true)}
      {label(360,402,t('서로 다른 대상 · 일반 sign problem의 해결 아님','Different objects; no general sign-problem solution'),true)}
    </>}
  </svg>;
}
