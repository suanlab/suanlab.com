'use client';
import {useId,useState,type ComponentProps,type ReactNode} from 'react';
import {Button as BaseButton} from '@/components/ui/button';
import type {PhaseSuccess} from '@/lib/learning-phase-transitions-protocol';
import type {PartitionObservation as Q} from '@/lib/learning-partition-exact-core';
import {partitionNumber as number,partitionQuantity as originalQuantity,partitionQuantityStatus as originalStatus} from './partition-view';
export function PhaseButton({className,...props}:ComponentProps<typeof BaseButton>){return <BaseButton {...props} className={`h-auto min-h-9 max-w-full whitespace-normal break-words ${className??''}`}/>;}
const Button=PhaseButton;
export type PhaseTextFn=(ko:string,en:string)=>string;
export const phaseQuantity=(q:Q|null,t:PhaseTextFn)=>q===null?t('미제공','Not available'):originalQuantity(q,t);
const quantity=phaseQuantity;
export const phaseMagnetization=(q:Q,t:PhaseTextFn)=>q.value===null?quantity(q,t):String(q.value);
const status=(q:Q|null,t:PhaseTextFn)=>q===null?t('이 준비에서 계산값이 제공되지 않음','No computed value provided for this preparation'):originalStatus(q,t);
export function PhaseTable({title,headers,rows}:{title:string;headers:string[];rows:(string|number)[][]}) {
 return <div className="ai-discrete-table" role="region" tabIndex={0} aria-label={title}><table><caption style={{textAlign:'left',padding:'.5rem'}}>{title}</caption><thead><tr>{headers.map((h,i)=><th scope="col" key={i}>{h}</th>)}</tr></thead><tbody>{rows.map((row,i)=><tr key={i}>{row.map((cell,j)=>j===0?<th scope="row" key={j}>{cell}</th>:<td key={j} style={{overflowWrap:'anywhere',maxWidth:'19rem'}}>{cell}</td>)}</tr>)}</tbody></table></div>;
}
function PhasePlot({label,t,children}:{label:string;t:PhaseTextFn;children:ReactNode}){
 const hint=useId();
 return <><div className="ai-foundation-scroll" role="region" tabIndex={0} aria-label={label} aria-describedby={hint}>{children}</div><p id={hint} className="ai-small">{t('좁은 화면에서는 그림을 가로로 스크롤하세요. 그림 영역에 초점을 두고 방향키를 사용할 수 있습니다.','On narrow screens, scroll the plot horizontally. Focus the plot region to use the arrow keys.')}</p></>;
}
function Readout({title,rows,t}:{title:string;rows:[string,Q|null,string][];t:PhaseTextFn}){
 return <PhaseTable title={title} headers={[t('관측량','Observation'),t('값','Value'),t('단위·의미','Units / meaning'),t('표현 상태','Representation status'),t('근사 ln|q|','Approximate ln|q|')]} rows={rows.map(([name,q,unit])=>[name,name==='m'&&q!==null?phaseMagnetization(q,t):quantity(q,t),unit,status(q,t),number(q?.logMagnitude??null)])}/>;
}
export const phaseRole=(role:string,t:PhaseTextFn)=>({
 'global-minimum':t('전역 최소','Global minimum'),'metastable-minimum':t('준안정 최소','Metastable minimum'),'unstable-maximum':t('불안정 극대','Unstable maximum'),'critical-minimum':t('임계 최소','Critical minimum'),'stationary-inflection':t('정지 변곡점','Stationary inflection'),'minimum-global-unresolved':t('최소·전역 비교 미해결','Minimum, global comparison unresolved'),'unresolved':t('국소 분류 미해결','Local classification unresolved'),
}[role]??t('미해결','Unresolved'));
function FiniteView({response,t}:{response:Extract<PhaseSuccess,{kind:'finite-curie-weiss-magnetization-symmetry'}>;t:PhaseTextFn}){
 const r=response.result,[page,setPage]=useState(0),last=Math.floor((r.states.length-1)/16),current=Math.min(page,last),step=280/r.states.length;
 const rows:[string,Q|null,string][]=[['E[m]',r.meanMagnetization,'m'],['E[|m|]',r.meanAbsoluteMagnetization,'|m|'],['E[m²]',r.meanSquareMagnetization,'m²'],['Var(m)',r.magnetizationVariance,'m²'],[t('평형 반응 N Var(m)','Equilibrium response N Var(m)'),r.fieldSusceptibility,'dE[m]/db'],['ln Z',r.logPartition,'nat'],[t('거시 엔트로피','Macro entropy'),r.macroEntropyNats,'nat'],[t('조건부 ln g','Conditional ln g'),r.conditionalMicrostateEntropyNats,'nat'],[t('미시 엔트로피','Micro entropy'),r.microstateEntropyNats,'nat'],[t('자기 항 포함 에너지','Energy with self interaction'),r.meanEnergyInThermalUnits,'E/(kBT)'],[t('자기 항 제외 에너지','Energy without self interaction'),r.meanEnergyWithoutSelfInteractionInThermalUnits,'E_pair/(kBT)'],['ln Z_pair',r.logPartitionWithoutSelfInteraction,'nat']];
 return <>
  <p>{t('유한 확률분포 · 정확한 미시 배치 수','Finite probability law · exact microstate count')}: <strong>{r.microstateCount}</strong>. {t('인증한 최빈 M','Certified modal M')}: {r.modes.certified?r.modes.candidateTotalMagnetizations.join(', '):t('아직 미해결인 후보','Unresolved candidates')}. {t('유한 두 봉우리는 상전이 특이점의 증명이 아닙니다.','Finite two-peak structure does not prove a transition singularity.')}</p>
  <figure className="ai-foundation-figure" data-phase-bars><h3>{t('자화의 전체 확률분포','Full magnetization distribution')}</h3><PhasePlot label={t('자화 확률 그림','Magnetization probability plot')} t={t}><svg style={{minWidth:576,maxWidth:760,margin:'0 auto'}} viewBox="0 0 360 260" role="img" aria-label={t('가로축 m=M/N, 세로축 확률0부터1','Horizontal m=M/N, vertical probability zero to one')}>
   <path d="M45 20V215H335" stroke="var(--learning-neutral)" fill="none"/>{[0,.5,1].map(v=><text key={v} x="40" y={219-190*v} textAnchor="end" style={{fontSize:11}} fill="currentColor">{v}</text>)}
   {r.states.map((s,i)=>{const v=s.probability.value;return <g key={i}><title>{`M=${s.totalMagnetization}; p=${quantity(s.probability,t)}; ${status(s.probability,t)}`}</title>{v!==null&&v>0?<rect x={48+i*step} y={215-190*v} width={Math.max(.6,step*.8)} height={190*v} fill="var(--learning-blue)"/>:<circle cx={48+(i+.4)*step} cy="215" r="2" fill="none" stroke="var(--learning-orange)"/>}{i%Math.max(1,Math.ceil(r.states.length/6))===0&&<text x={48+i*step} y="235" textAnchor="middle" style={{fontSize:10}} fill="currentColor">{number(s.magnetizationPerSpin.value)}</text>}</g>;})}<text x="325" y="252" style={{fontSize:12}} fill="currentColor">m</text>
  </svg></PhasePlot><figcaption>{t('그림은 정확한 분포의 반올림 표시입니다. 너무 작은 양의 확률은 그려지지 않을 수 있습니다. 아래 표의 부호·표현 상태를 함께 읽으세요.','The figure displays rounded values of the exact law. Tiny positive probabilities may be too small to draw. Read their signs and representation status in the table.')}</figcaption></figure>
  <Readout title={t('평균·요동·엔트로피·자기 항','Means, fluctuations, entropy and self energy')} rows={rows} t={t}/>
  <div className="ai-lab-actions"><Button type="button" variant="outline" disabled={!current} onClick={()=>setPage(current-1)}>{t('이전 상태 행','Previous state rows')}</Button><output>{current+1}/{last+1}</output><Button type="button" variant="outline" disabled={current===last} onClick={()=>setPage(current+1)}>{t('다음 상태 행','Next state rows')}</Button></div>
  <PhaseTable title={t('유한 거시상태의 확률과 다중도','Finite macrostate probabilities and multiplicities')} headers={['M','m','g_N(M)','p(M)','ln p(M)',t('확률 표현','Probability representation')]} rows={r.states.slice(current*16,current*16+16).map(s=>[s.totalMagnetization,quantity(s.magnetizationPerSpin,t),s.multiplicity,quantity(s.probability,t),quantity(s.logProbability,t),status(s.probability,t)])}/>
 </>;
}
type RootResponse=Exclude<PhaseSuccess,{kind:'finite-curie-weiss-magnetization-symmetry'}>;
/** Float samples are drawing data only. Certified roots come from the worker. */
export function phaseDrawingValue(response:RootResponse,m:number,mode:'potential'|'derivative'):number|null {
 if(response.kind==='curie-weiss-variational-stability-limits'){
  if(m< -1||m>1||mode==='derivative'&&Math.abs(m)>=1)return null;
  const {couplingK:K,fieldB:b}=response.input.parameters;
  if(mode==='derivative'){
   if(Math.abs(m)<=.125){let A=0,power=1;for(let j=0;j<28;j++){A+=power/(2*j+3);power*=m*m;}return (1-K)*m-b+m*m*m*A;}
   return Math.atanh(m)-K*m-b;
  }
  const entropy=Math.abs(m)===1?0:-((1+m)/2)*Math.log((1+m)/2)-((1-m)/2)*Math.log((1-m)/2);
  return -K*m*m/2-b*m-entropy;
 }
 const {quadraticA:a,quarticB:b,sexticC:c,fieldH:H}=response.input.parameters;
 return mode==='derivative'?((c*m*m+b)*m*m+a)*m-H:((c*m*m/6+b/4)*m*m+a/2)*m*m-H*m;
}
/** A floating-point drawing can be unresolved even when exact roots are certified.
 * The scale estimates cancellation in the displayed expression, never root count.
 */
export function phaseDrawingResolved(response:RootResponse,mode:'potential'|'derivative',samples:readonly {x:number;y:number}[]):boolean {
 if(samples.length<2)return false;const extent=Math.max(...samples.map(p=>Math.abs(p.x))),ys=samples.map(p=>p.y);
 let scale:number;
 if(response.kind==='curie-weiss-variational-stability-limits'){const {couplingK:K,fieldB:b}=response.input.parameters;scale=mode==='potential'?Math.abs(K)*extent*extent/2+Math.abs(b)*extent+Math.log(2):Math.abs(K)*extent+Math.abs(b)+Math.abs(Math.atanh(Math.min(1-Number.EPSILON,extent)));}
 else{const {quadraticA:a,quarticB:b,sexticC:c,fieldH:H}=response.input.parameters;scale=mode==='potential'?Math.abs(a)*extent**2/2+Math.abs(b)*extent**4/4+Math.abs(c)*extent**6/6+Math.abs(H)*extent:Math.abs(a)*extent+Math.abs(b)*extent**3+Math.abs(c)*extent**5+Math.abs(H);}
 return Math.max(...ys)-Math.min(...ys)>Math.max(Number.MIN_VALUE,64*Number.EPSILON*scale);
}
function Landscape({response,t}:{response:RootResponse;t:PhaseTextFn}){
 const [mode,setMode]=useState<'potential'|'derivative'>('potential'),[focus,setFocus]=useState<number|null>(null),id=useId();
 const roots=response.result.roots,values=roots.map(x=>x.magnetization.value),finite=values.filter((v):v is number=>v!==null&&Number.isFinite(v));
 let lo=response.kind==='curie-weiss-variational-stability-limits'?-1:Math.min(-1,...finite)-.4,hi=response.kind==='curie-weiss-variational-stability-limits'?1:Math.max(1,...finite)+.4;
 if(focus!==null&&values[focus]!==null&&values[focus]!==undefined){const center=values[focus]!,distances=finite.map(v=>Math.abs(v-center)).filter(d=>d>0),nearest=Math.min(...distances),radius=Number.isFinite(nearest)?Math.max(nearest*1.8,Math.abs(center)*1e-14,Number.MIN_VALUE):center!==0?Math.max(Math.abs(center)*.8,Number.MIN_VALUE):.05;lo=center-radius;hi=center+radius;if(response.kind==='curie-weiss-variational-stability-limits'){lo=Math.max(-1,lo);hi=Math.min(1,hi);}}
 const samples=Array.from({length:321},(_,i)=>{const x=lo+(hi-lo)*i/320;return {x,y:phaseDrawingValue(response,x,mode)};}).filter((p):p is {x:number;y:number}=>p.y!==null&&Number.isFinite(p.y));
 const ys=samples.map(p=>p.y),low=Math.min(...ys),high=Math.max(...ys),resolved=phaseDrawingResolved(response,mode,samples);
 const padding=resolved?(high-low)*.1:1,ylo=resolved?low-padding:low-1,yhi=resolved?high+padding:high+1;
 const px=(x:number)=>82+248*(x-lo)/(hi-lo),py=(y:number)=>215-190*(y-ylo)/(yhi-ylo);
 const path=samples.map((p,i)=>`${i?'L':'M'}${px(p.x).toFixed(3)} ${py(p.y).toFixed(3)}`).join(' ');
 const title=mode==='potential'?t('제약·현상론 자유에너지','Constrained / phenomenological free energy'):t('퍼텐셜의 도함수','Potential derivative');
 return <figure className="ai-foundation-figure" data-phase-landscape data-phase-drawing={resolved?'sampled':'unresolved'}><h3 id={id}>{title}</h3><div className="ai-lab-actions" role="group" aria-label={t('그림 선택','Choose drawing')}><Button type="button" variant="outline" aria-pressed={mode==='potential'} onClick={()=>setMode('potential')}>{t('자유에너지','Free energy')}</Button><Button type="button" variant="outline" aria-pressed={mode==='derivative'} onClick={()=>setMode('derivative')}>{t('도함수','Derivative')}</Button><Button type="button" variant="outline" aria-pressed={focus===null} onClick={()=>setFocus(null)}>{t('전체 그림','Full view')}</Button>{roots.map((_,i)=><Button type="button" variant="outline" key={i} disabled={values[i]===null} aria-pressed={focus===i} onClick={()=>setFocus(i)}>{t('근 확대','Zoom root')} {i+1}</Button>)}</div>
 <PhasePlot label={title} t={t}><svg style={{minWidth:576,maxWidth:760,margin:'0 auto'}} viewBox="0 0 360 270" role="img" aria-label={t('가로축 질서변수, 세로축은 선택한 퍼텐셜 또는 도함수. 마커 번호는 근 표와 같습니다.','Horizontal order parameter, vertical selected potential or derivative. Marker numbers match the root table.')}><path d="M82 20V215H335" fill="none" stroke="var(--learning-neutral)"/>{resolved&&<path d={path} fill="none" stroke="var(--learning-blue)" strokeWidth="2"/>}{[lo,(lo+hi)/2,hi].map((x,i)=><text key={i} x={px(x)} y="239" textAnchor={i===0?'start':i===2?'end':'middle'} style={{fontSize:10}} fill="currentColor">{number(x)}</text>)}{(resolved?[low,high]:[low]).map((y,i)=><text key={i} x="77" y={py(y)+3} textAnchor="end" style={{fontSize:10}} fill="currentColor">{number(y)}</text>)}
 {roots.map((r,i)=>{const x=r.magnetization.value;if(x===null||x<lo||x>hi)return null;const y=phaseDrawingValue(response,x,mode);if(y===null||!Number.isFinite(y))return null;return <g key={i}><title>{`${i+1}: ${phaseRole(r.role,t)}; m=${quantity(r.magnetization,t)}`}</title><circle cx={px(x)} cy={py(y)} r="4" fill={r.globalMinimum?'var(--learning-orange)':'hsl(var(--background))'} stroke="var(--learning-orange)" strokeWidth="2"/><text x={px(x)+6} y={py(y)-8} style={{fontSize:11}} fill="currentColor">{i+1}</text></g>;})}<text x="330" y="260" style={{fontSize:12}} fill="currentColor">m</text></svg></PhasePlot>
 {!resolved&&<p role="status">{t('이 범위의 변화는 그림 정밀도로 해결되지 않습니다. 정확한0으로 해석하지 마세요.','Changes in this range are unresolved at drawing precision. Do not interpret this as exact zero.')}</p>}
 <figcaption>{t('선은 반올림한 표본 그림이며 근의 개수·안정성을 판정하지 않습니다. 마커의 근 위치와 역할은 계산된 구간에서 가져옵니다. 확대에서도 같은 숫자로 보이면 근 표의 부호·중복도·상태를 읽으세요.','Lines are rounded samples and do not decide root count or stability. Marker locations and roles come from computed enclosures. If numbers still look identical when zoomed, read signs, multiplicities and status in the root table.')}</figcaption></figure>;
}
function RootView({response,t}:{response:RootResponse;t:PhaseTextFn}){
 const r=response.result,[selected,setSelected]=useState(0),actual=Math.min(selected,Math.max(0,r.roots.length-1)),root=r.roots[actual];
 const rows:[string,Q|null,string][]=[];
 if(response.kind==='curie-weiss-variational-stability-limits'){
  rows.push([t('극한 압력','Limiting pressure'),response.result.thermodynamicLimitPressure,'ln Z/N'],[t('두 최소의 자유에너지 차이','Free-energy gap between minima'),response.result.minimumFreeEnergyGap,'psi_meta−psi_global']);
  if(response.result.spinodal)rows.push(['s',response.result.spinodal.turningMagnetization,'m'],['B_sp',response.result.spinodal.fieldMagnitude,'b']);
 }else rows.push([t('최소 자유에너지','Minimum free energy'),response.result.minimumFreeEnergy,'Phi']);
 if(root)rows.push(['m',root.magnetization,'m'],[t('곡률','Curvature'),root.curvature,response.kind==='landau-sixth-order-coexistence-stability'?'d²Phi/dm²':'d²psi/dm²'],[t('자유에너지','Free energy'),root.constrainedFreeEnergy,response.kind==='landau-sixth-order-coexistence-stability'?'Phi':'psi'],[t('가지의 미분','Branch derivative'),root.branchFieldDerivative,response.kind==='landau-sixth-order-coexistence-stability'?'dm/dH':'dm/db']);
 return <>
  <p>{t('서로 다른 정지근','Distinct stationary roots')}: <strong>{r.rootCount??t('개수 비교 미해결','Count comparison unresolved')}</strong>. {t('전역 최소 행','Global-minimum rows')}: {r.globalMinimumIndices.length?r.globalMinimumIndices.map(i=>i+1).join(', '):t('확정하지 않음','Not certified')}.</p>
  {response.kind==='landau-sixth-order-coexistence-stability'&&!response.result.globalSelectionResolved&&<p role="status">{t('근 집합은 완전하지만 전역 비교는 미해결입니다. 후보 행','The root set is complete but global selection is unresolved. Candidate rows')}: {response.result.globalMinimumCandidates.map(i=>i+1).join(', ')}.</p>}
  {response.kind==='curie-weiss-variational-stability-limits'&&!response.result.fullyResolved&&<p role="status">{t('현재 정밀도에서 완전한 근 집합을 발표하지 않습니다. 가능한 근 개수','A complete root set is not published at this precision. Possible root counts')}: {response.result.rootCountCandidates.join(', ')}.</p>}
  <Landscape key={response.inputKey} response={response} t={t}/>
  <PhaseTable title={t('모든 정지근의 역할','Roles of all stationary roots')} headers={[t('행','Row'),'m',t('역할','Role'),t('중복도','Multiplicity'),t('곡률','Curvature'),t('자유에너지','Free energy'),t('가지 미분','Branch derivative'),t('m 표현','m representation')]} rows={r.roots.map((x,i)=>[i+1,phaseMagnetization(x.magnetization,t),phaseRole(x.role,t),x.stationaryMultiplicity,quantity(x.curvature,t),quantity(x.constrainedFreeEnergy,t),x.branchFieldDerivative===null&&x.stationaryMultiplicity%2===0?t('단일값 미분 없음','No single-valued derivative'):quantity(x.branchFieldDerivative,t),status(x.magnetization,t)])}/>
  <div className="ai-lab-actions" role="group" aria-label={t('상세 근 선택','Choose a root for details')}>{r.roots.map((_,i)=><Button key={i} type="button" variant="outline" aria-pressed={actual===i} onClick={()=>setSelected(i)}>{t('근 상세','Root details')} {i+1}</Button>)}</div>
  <Readout title={t('선택한 근과 비교 경계','Selected root and comparison thresholds')} rows={rows} t={t}/>
  {response.kind==='curie-weiss-variational-stability-limits'&&response.result.roots[actual]&&<Readout title={t('엔트로피와 영점 대비 차이','Entropy and difference from zero')} rows={[[t('이진 엔트로피','Binary entropy'),response.result.roots[actual].entropyNats,'nat'],['psi(m)−psi(0)',response.result.roots[actual].freeEnergyDifferenceToZero,'psi']]} t={t}/>}
  {response.kind==='landau-sixth-order-coexistence-stability'&&response.result.roots[actual]&&<Readout title={t('고차 분류와 전역 최소 대비','Higher-order classification and excess above the global minimum')} rows={[[t('처음 비영인 정지 도함수','Leading nonzero stationary derivative'),response.result.roots[actual].leadingStationaryDerivative,'g^(multiplicity)'],[t('전역 최소 대비 에너지 초과','Energy excess above the global minimum'),response.result.roots[actual].freeEnergyExcess,'Phi−Phi_min']]} t={t}/>}
  <PhaseTable title={t('인접 극대까지의 정적 장벽','Static barriers to adjacent maxima')} headers={[t('최소 행','Minimum row'),t('극대 행','Maximum row'),t('장벽','Barrier'),t('표현 상태','Representation status')]} rows={response.kind==='landau-sixth-order-coexistence-stability'?response.result.barriers.map(b=>[b.minimumIndex+1,b.maximumIndex+1,quantity(b.barrier,t),status(b.barrier,t)]):response.result.barriers.map(b=>[b.minimumIndex+1,2,quantity(b.barrier,t),status(b.barrier,t)])}/>
  <p className="ai-small">{t('가지 미분은 불안정한 가지에서 음수일 수 있으며 유한 평형의 요동 감수율이 아닙니다. 접힘에는 단일값 미분이 없습니다. 장벽만으로 전환 시간이나 실험 이력을 계산하지 않습니다.','A branch derivative can be negative on an unstable branch; it is not finite-equilibrium fluctuation susceptibility. A fold has no single-valued derivative. Barriers alone do not determine switching times or experimental hysteresis.')}</p>
 </>;
}
export function PhaseResultView({response,language}:{response:PhaseSuccess;language:'ko'|'en'}){
 const t:PhaseTextFn=(ko,en)=>language==='ko'?ko:en;
 return <div data-phase-result={response.kind} data-phase-resolved={String(response.result.fullyResolved)}><p className="ai-small">{response.result.fullyResolved?t('현재 준비의 관측량 비교를 해결했습니다.','Observations for this preparation are resolved.'):t('정밀도 상한에서 미해결인 항목을 그대로 남깁니다.','Unresolved items at the precision ceiling are retained.')}{' '}{t('계산 정밀도','Arithmetic precision')}: {response.result.precisionBits} bit.</p>{response.kind==='finite-curie-weiss-magnetization-symmetry'?<FiniteView response={response} t={t}/>:<RootView response={response} t={t}/>}<p className="ai-small">{t('표의 로그 크기는 근사 표시입니다. 수치 인증은 방향성 유리 구간을 따릅니다.','Magnitude logarithms in tables are approximate displays. Numerical certification uses directed rational enclosures.')}</p></div>;
}
