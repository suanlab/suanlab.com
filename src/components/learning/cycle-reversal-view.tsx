'use client';
import React, { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { useSiteMotion } from '@/components/motion/motion-provider';
import { useSceneActivity } from '@/components/motion/use-scene-activity';
import type { ReversalInput, ReversalSuccess } from '@/lib/learning-cycle-reversal-protocol';
import type { PartitionObservation as Q } from '@/lib/learning-partition-exact-core';
import { MasterButton, MasterTable, MasterObservations } from './master-equation-view';
import { partitionQuantity as quantity } from './partition-view';
export const ReversalButton = MasterButton, ReversalTable = MasterTable;
type Text = (ko: string, en: string) => string;
const blue='var(--learning-blue)',orange='var(--learning-orange)',neutral='var(--learning-neutral)';
export function reversalScope(input: ReversalInput, t: Text) {
  const p=input.parameters;
  return `${t('전이율 분자','Rate numerators')} (a,b,c,d,e,f)=(${[p.clockwise01,p.clockwise12,p.clockwise20,p.counter10,p.counter21,p.counter02].join(',')}) / ${p.rateDenominator} s⁻¹ · ${t('상태','States')}=[${p.states.join(',')}] · ${t('대기','Dwells')}=[${p.dwellNumerators.join(',')}] / ${p.timeDenominator} s`;
}
function Diagram({title,caption,children,t}:{title:string;caption:string;children:ReactNode;t:Text}) {
  const id=useId();
  return <figure style={{margin:'1rem 0'}}><figcaption>{caption}</figcaption><p id={id} className="ai-small">{t('좁은 화면에서는 좌우로 밀거나 그림 영역에 초점을 두고 좌우 방향키로 이동하세요.','On narrow screens swipe horizontally or focus the diagram region and use left/right arrow keys.')}</p><div data-reversal-diagram role="region" tabIndex={0} aria-label={title} aria-describedby={id} style={{maxWidth:'100%',overflowX:'auto',overscrollBehaviorX:'contain'}} className="rounded focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">{children}</div></figure>;
}
function Arrow({from,to,offset=0,color}:{from:number[];to:number[];offset?:number;color:string}) {
  const dx=to[0]-from[0],dy=to[1]-from[1],length=Math.hypot(dx,dy),ux=dx/length,uy=dy/length;
  const x=from[0]+ux*64-uy*offset,y=from[1]+uy*64+ux*offset,bx=to[0]-ux*64-uy*offset,by=to[1]-uy*64+ux*offset;
  return <g fill="none" stroke={color} strokeWidth={3}><path d={`M${x} ${y}L${bx} ${by}`}/><path d={`M${bx-10*ux+6*uy} ${by-10*uy-6*ux}L${bx} ${by}L${bx-10*ux-6*uy} ${by-10*uy+6*ux}`}/></g>;
}
function Rooms({response,t}:{response:ReversalSuccess;t:Text}) {
  const [flow,setFlow]=useState<'gross'|'net'>('gross'),[reverse,setReverse]=useState(false),r=response.result;
  const positions=[[320,100],[530,400],[110,400]],title=t('세 상태의 점유와 문별 흐름','Three-state occupancy and door flows');
  const matrix=reverse?r.reversedGenerator:r.generator;
  return <><div className="ai-lab-actions" role="group" aria-label={t('생성자 보기','Choose generator view')}>{[false,true].map(v=><ReversalButton key={String(v)} variant="outline" aria-pressed={reverse===v} onClick={()=>setReverse(v)}>{v?t('정상 반전 Q*','Stationary reversal Q*'):t('원래 생성자 Q','Original generator Q')}</ReversalButton>)}</div><div className="ai-lab-actions" role="group" aria-label={t('흐름 표시 선택','Choose flow display')}>{(['gross','net'] as const).map(v=><ReversalButton key={v} variant="outline" aria-pressed={flow===v} onClick={()=>setFlow(v)}>{v==='gross'?t('양방향 한 방향 흐름','Both one-way flows'):t('순흐름만','Net current only')}</ReversalButton>)}</div>
  <Diagram title={title} t={t} caption={t('원은 같은 크기이며 숫자가 정상 점유율입니다. 같은 길이의 화살표는 방향만 나타냅니다. 파랑은0→1→2→0, 주황은 그 반대입니다. 크기는 아래 문별 표에서 읽으세요.','Equal-size rooms show stationary occupancy numerically. Equal-length arrows indicate direction only. Blue denotes0→1→2→0 and orange the opposite orientation. Read magnitudes in the door table.')}><svg data-reversal-rooms viewBox="0 0 640 530" role="img" aria-label={title} style={{minWidth:640,width:'100%',maxWidth:800,fontSize:16}}>
  <text x={35} y={35} fill="currentColor">{reverse?'Q*':'Q'} · {flow==='gross'?'Fij=πi qij':'Jij=Fij−Fji'} [s⁻¹]</text>
  {r.orientedEdges.map((edge,i)=>{const j=(i+1)%3,sign=(edge.netCurrent.sign??0)*(reverse?-1:1);return flow==='gross'?<g key={i}><Arrow from={positions[i]} to={positions[j]} offset={-10} color={blue}/><Arrow from={positions[j]} to={positions[i]} offset={-10} color={orange}/></g>:sign===0?<g key={i}><path d={`M${positions[i][0]} ${positions[i][1]}L${positions[j][0]} ${positions[j][1]}`} stroke={neutral} strokeDasharray="5 5"/><text x={(positions[i][0]+positions[j][0])/2+8} y={(positions[i][1]+positions[j][1])/2-10} fill="currentColor">J=0</text></g>:<Arrow key={i} from={positions[sign>0?i:j]} to={positions[sign>0?j:i]} color={sign>0?blue:orange}/>;})}
  {positions.map(([x,y],i)=><g key={i}><circle cx={x} cy={y} r={53} fill="hsl(var(--background))" stroke={blue} strokeWidth={2}/><text x={x} y={y-12} fill="currentColor" textAnchor="middle">{t('방','Room')} {i}</text><text x={x} y={y+15} fill="currentColor" textAnchor="middle">π={quantity(r.stationary[i],t)}</text></g>)}
  <text x={35} y={505} fill="currentColor">{t('대기 위험률은 Q와 Q*에서 같습니다.','Escape rates agree for Q and Q*.')}</text></svg></Diagram>
  <MasterTable title={t('선택 생성자의 문별 정상 흐름 [1/s]','Door flows for the selected generator [1/s]')} headers={[t('문','Door'),'Fij','Fji','Jij']} rows={r.orientedEdges.map(e=>[`${e.from}↔${e.to}`,quantity(reverse?e.backwardFlow:e.forwardFlow,t),quantity(reverse?e.forwardFlow:e.backwardFlow,t),`${reverse&&e.netCurrent.sign!==0?'−(' :''}${quantity(e.netCurrent,t)}${reverse&&e.netCurrent.sign!==0?')':''}`])}/>
  <MasterObservations title={t('선택 생성자 [1/s]','Selected generator [1/s]')} t={t} rows={matrix.flatMap((row,i)=>row.map((q,j):[string,Q]=>[`${reverse?'Q*':'Q'}[${i},${j}]`,q]))}/></>;
}
function Strip({response,index,reverse,t}:{response:ReversalSuccess;index:number;reverse:boolean;t:Text}) {
  const p=response.input.parameters,indices=Array.from({length:index+1},(_,i)=>i);if(reverse)indices.reverse();
  const title=reverse?t('선택 접두 경로의 역재생','Reversed selected prefix path'):t('선택 접두 경로의 정방향','Forward selected prefix path');
  const width=Math.max(640,indices.length*180+30);
  return <Diagram title={title} t={t} caption={t('상태와 대기시간을 함께 뒤집습니다. 칸 너비는 시간에 비례하지 않습니다. 지정 경로이며 무작위 표본이 아닙니다.','States and waits reverse together. Cell widths are not proportional to time. This is a specified path, not a random sample.')}><svg data-reversal-strip data-reversal-reversed={String(reverse)} viewBox={`0 0 ${width} 165`} role="img" aria-label={title} style={{minWidth:width,width:'100%',fontSize:16}}>{indices.map((source,i)=><g key={i}><rect x={20+i*180} y={20} width={150} height={110} rx={8} fill="none" stroke={source===index?orange:blue} strokeWidth={2}/><text x={35+i*180} y={52} fill="currentColor">{t('방','Room')} {p.states[source]}</text><text x={35+i*180} y={85} fill="currentColor">δ={p.dwellNumerators[source]}/{p.timeDenominator}s</text><text x={35+i*180} y={115} fill="currentColor">{t('원래 칸','Original cell')} {source}</text>{i<indices.length-1&&<text x={177+i*180} y={80} fill="currentColor">→</text>}</g>)}</svg></Diagram>;
}
function Paged({response,t}:{response:ReversalSuccess;t:Text}) {
  const [page,setPage]=useState(0),r=response.result,total=Math.ceil(r.path.length/24),i=Math.min(page,total-1);
  return <div data-reversal-paged><MasterTable title={t('경로 접두부별 기록','Path-prefix records')} headers={[t('칸','Cell'),t('방','Room'),'δ[s]','T[s]',t('끝점 로그','Endpoint log'),t('점프 로그','Jump log'),'R']} rows={r.path.slice(i*24,(i+1)*24).map(row=>[row.index,row.state,quantity(row.dwell,t),quantity(row.elapsed,t),quantity(row.endpointLogRatio,t),quantity(row.jumpLogRatio,t),quantity(row.totalLogRatio,t)])}/><div className="ai-lab-actions"><ReversalButton variant="outline" disabled={i===0} onClick={()=>setPage(i-1)}>{t('이전 기록','Previous records')}</ReversalButton><span>{i+1}/{total}</span><ReversalButton variant="outline" disabled={i===total-1} onClick={()=>setPage(i+1)}>{t('다음 기록','Next records')}</ReversalButton></div></div>;
}
function Panel({response,language,animate=false}:{response:ReversalSuccess;language:'ko'|'en';animate?:boolean}) {
  const t:Text=(ko,en)=>language==='ko'?ko:en,r=response.result,[record,setRecord]=useState(r.path.length-1),[reverse,setReverse]=useState(false),[playing,setPlaying]=useState(false),root=useRef<HTMLDivElement>(null),motion=useSiteMotion(),active=useSceneActivity(root),id=useId();
  const i=Math.min(record,r.path.length-1),row=r.path[i];
  useEffect(()=>{
    const lab=root.current?.closest('[data-reversal-lab]');
    if(!lab)return;
    const pause=()=>setPlaying(false);
    lab.addEventListener('focusin',pause);
    return()=>lab.removeEventListener('focusin',pause);
  },[]);
  useEffect(()=>{if(!active||!motion.canAnimate)setPlaying(false);},[active,motion.canAnimate]);
  useEffect(()=>{if(!playing||!active||!motion.canAnimate)return;const timer=setTimeout(()=>{if(i<r.path.length-1)setRecord(i+1);else setPlaying(false);},1000);return()=>clearTimeout(timer);},[playing,active,motion.canAnimate,i,r.path.length]);
  const observations:[string,Q|null][]=[
    [t('관측 시간 [s]','Observation time [s]'),r.totalTime],[t('순환 순흐름 J [1/s]','Cycle current J [1/s]'),r.cycleCurrent],
    [t('순환율 곱 비율 abc/def','Cycle product ratio abc/def'),r.cycleRatio],[t('순환 affinity A','Cycle affinity A'),r.cycleAffinity],
    [t('정상 경로 상대엔트로피율 σ [1/s]','Stationary path relative-entropy rate σ [1/s]'),r.pathRelativeEntropyRate],
    [t('정상 점프 활동률 [1/s]','Stationary jump activity [1/s]'),r.stationaryActivity],
  ];
  return <div ref={root} data-reversal-result={response.kind} data-reversal-result-key={response.inputKey} data-reversal-result-input={JSON.stringify(response.input)} data-reversal-playing={String(playing)} style={{minWidth:0}} onFocusCapture={()=>setPlaying(false)}>
  <p style={{overflowWrap:'anywhere'}}>{reversalScope(response.input,t)}</p><p className="ai-small">{t('정확 유리수와 방향성 로그 구간을 구별합니다. Number 표시 불가는 영 값이 아닙니다. 실제 열·온도·국소 상세 균형을 지정하지 않았으며 경로는 선택한 예제입니다.','Exact rationals and directed log enclosures are distinct. An unavailable Number is not a zero. Heat, temperature and local detailed balance are not specified; the path is a chosen example.')}</p>
  <MasterTable title={t('현재 모형의 분류','Current model classifications')} headers={[t('항목','Item'),t('결과','Result')]} rows={[[t('상세 균형','Detailed balance'),r.classification.detailedBalance?t('만족','Satisfied'):t('위반','Broken')],[t('균일 정상분포','Uniform stationary law'),r.classification.stationaryIsUniform?t('예','Yes'):t('아니오','No')],[t('현재 경로비가 정확히1','Current path ratio is exactly1'),r.classification.pathRatioIsExactlyOne?t('예','Yes'):t('아니오','No')],[t('물리적 열 해석','Physical heat interpretation'),t('가정을 지정하지 않음','Assumptions not specified')]]}/>
  <Rooms response={response} t={t}/><MasterObservations title={t('정상 확률과 보존식','Stationary probabilities and conservation')} t={t} rows={[...r.stationary.map((q,j):[string,Q]=>[`π${j}`,q]),...r.escapeRates.map((q,j):[string,Q]=>[`r${j} [1/s]`,q]),...r.stationaryResidual.map((q,j):[string,Q]=>[`[πQ]${j}`,q]),...r.reversedRowResidual.map((q,j):[string,Q]=>[`Σ Q*${j}j`,q])]} />
  <MasterObservations title={t('순환과 앙상블 평균','Circulation and ensemble means')} rows={observations} t={t}/>
  <p className="ai-small">{t('σ≥0은 정상 앙상블 평균율입니다. 개별 경로의 R은 음수이거나0일 수 있습니다. 무방향 문을 한 번씩 세면1/2를 추가하지 않습니다.','σ≥0 is a stationary ensemble mean rate. An individual path’s R may be negative or zero. Counting each undirected door once introduces no additional1/2.')}</p>
  <MasterObservations title={t('문별 흐름 비율과 평균율 기여','Door flow ratios and mean-rate contributions')} t={t} rows={r.orientedEdges.flatMap(e=>[[`${e.from}→${e.to} · Fij/Fji`,e.stationaryFlowRatio],[`${e.from}→${e.to} · ln(Fij/Fji)`,e.logStationaryFlowRatio],[`${e.from}↔${e.to} · σ [1/s]`,e.relativeEntropyContribution]] as [string,Q][])} />
  <label className="ai-number-field"><span id={id}>{t('선택 접두부의 마지막 칸','Last cell of the selected prefix')}: {i}/{r.path.length-1}</span><input data-reversal-record type="range" min={0} max={r.path.length-1} step={1} value={i} aria-labelledby={id} onChange={e=>{setPlaying(false);setRecord(Number(e.target.value));}}/></label>
  {animate&&<><div className="ai-lab-actions"><ReversalButton variant="outline" disabled={!active||!motion.canAnimate||r.path.length<2} aria-pressed={playing} onClick={()=>{if(playing){setPlaying(false);return;}if(i===r.path.length-1)setRecord(0);setPlaying(true);}}>{playing?t(' 경로 설명 재생 중지','Pause path explanation'):t('경로 설명 자동 재생','Play path explanation')}</ReversalButton><ReversalButton variant="outline" disabled={i===r.path.length-1} onClick={()=>{setPlaying(false);setRecord(i+1);}}>{t('경로 한 칸','One path cell')}</ReversalButton></div><p className="ai-small">{t('재생은 칸을1초 간격으로 설명합니다. 실제 대기시간 배율이나 무작위 경로 생성이 아니며 계산을 다시 하지 않습니다.','Playback explains one cell each second. It is neither a physical waiting-time scale nor random path generation, and it does not recalculate the model.')}{!motion.canAnimate&&t(' 모션 설정으로 자동 재생을 멈췄습니다.',' Motion preferences pause automatic playback.')}</p></>}
  <div className="ai-lab-actions" role="group" aria-label={t('경로 방향 보기','Choose path orientation')}>{[false,true].map(v=><ReversalButton key={String(v)} variant="outline" aria-pressed={reverse===v} onClick={()=>setReverse(v)}>{v?t('대기와 함께 역재생','Reverse with waits'):t('정방향 경로','Forward path')}</ReversalButton>)}</div><Strip response={response} index={i} reverse={reverse} t={t}/>
  <MasterObservations title={t('선택 접두 경로의 비교','Selected prefix-path comparison')} t={t} rows={[[t('접두부 경과시간 [s]','Prefix elapsed time [s]'),row.elapsed],[t('유출 적분 Σri δi','Escape integral Σri δi'),row.integratedEscape],[t('끝점 비율 πstart/πend','Endpoint ratio πstart/πend'),row.endpointRatio],[t('점프율 곱 비율','Jump-rate product ratio'),row.jumpRateRatio],[t('전체 경로 밀도비','Complete path-density ratio'),row.pathRatio],[t('끝점 로그 항','Endpoint log term'),row.endpointLogRatio],[t('점프 로그 항','Jump log term'),row.jumpLogRatio],[t('총 로그 경로비 R','Total log path ratio R'),row.totalLogRatio],[t('정방향 로그 밀도·1초 기준','Forward log density;1s reference'),row.logForwardDensity],[t('역재생 로그 밀도·1초 기준','Reversed log density;1s reference'),row.logReverseDensity]]}/>
  <p className="ai-small">{t('각 기록은 그 접두부 자체의 역재생과 비교합니다. 상태 순서와 대기 순서를 함께 뒤집고 정상 끝점 인자를 보존합니다. 로그 밀도는 순서 있는 점프 시각에 대한 밀도이며 연속 경로 하나의 점확률이 아닙니다.','Each record compares that prefix with its own reversal. Reverse both state and dwell sequences and retain the stationary endpoint factor. Log density is with respect to ordered jump times, not the point probability of one continuous path.')}</p>
  <MasterObservations title={t('시간 관측·점프 관측·선택 경로','Time observations, jump observations and the chosen path')} t={t} rows={r.stationary.flatMap((q,j):[string,Q|null][]=>[[`π${j}`,q],[`${t('점프 시각 기준','Jump-epoch reference')} ${j}`,r.jumpEpochStationary[j]],[`${t('선택 경로의 시간 점유','Chosen-path time occupation')} ${j}`,r.chosenPathOccupation?.[j]??null]])}/>
  <p className="ai-small">{t('선택 경로의 점유는 정상분포 추정값이 아닙니다. 영 시간에서는 미정의입니다. 긴 정상 경로를 점프 시각만 관찰한 기준은 πi ri에 비례합니다.','Chosen-path occupation is not a stationary-law estimate and is undefined at zero duration. For a long stationary path, jump-epoch reference occupancy is proportional to πi ri.')}</p><Paged response={response} t={t}/>
  </div>;
}
export function ReversalResultView(props:{response:ReversalSuccess;language:'ko'|'en';animate?:boolean}) { return <Panel key={props.response.inputKey} {...props}/>; }
