'use client';
import { useEffect,useId,useRef,useState } from 'react';
import type { ManyBodySuccess } from '@/lib/learning-many-body-protocol';
import { MasterButton as Button } from './master-equation-view';
import { readLabPalette } from './lab-palette';
import type { SpinGraphGeometry,SpinGraphRenderer } from './many-body-spin-graph-engine';
export type SpinTeachingResult=Extract<ManyBodySuccess,{kind:'spin-chain'}>['result'];
export default function SpinGraph({result,selected,language}:{result:SpinTeachingResult;selected:number;language:'ko'|'en'}){
 const t=(ko:string,en:string)=>language==='ko'?ko:en,id=useId(),frame=useRef<HTMLElement>(null),host=useRef<HTMLDivElement>(null),renderer=useRef<SpinGraphRenderer>();
 const[requested,setRequested]=useState(false),[mode,setMode]=useState<'loading'|'3d'|'2d'>('2d'),[attempt,setAttempt]=useState(0);
 const active=new Set(result.activeEdges.map(e=>e.join('-')));
 const geometry:SpinGraphGeometry={selected,sites:result.localStates.map(s=>({index:s.site,position:result.parameters.graph===0?[2.6*(s.site-(result.sites-1)/2),0,0]:[2.5*Math.cos(2*Math.PI*s.site/result.sites),0,2.5*Math.sin(2*Math.PI*s.site/result.sites)],bloch:s.bloch.map(q=>q.value!==null&&q.resolution!=='unresolved-at-precision-ceiling'?q.value:null) as SpinGraphGeometry['sites'][number]['bloch']})),edges:result.graphEdges.map(([from,to])=>({from,to,active:active.has(`${from}-${to}`)}))};
 const latest=useRef(geometry);latest.current=geometry;
 useEffect(()=>{if(!requested)return;let cancelled=false;setMode('loading');import('./many-body-spin-graph-engine').then(({createSpinGraphRenderer})=>{if(cancelled||!host.current||!frame.current)return;try{renderer.current=createSpinGraphRenderer(host.current,latest.current,readLabPalette(frame.current),()=>{renderer.current?.dispose();renderer.current=undefined;if(!cancelled)setMode('2d');});setMode('3d');}catch{if(!cancelled)setMode('2d');}}).catch(()=>{if(!cancelled)setMode('2d');});return()=>{cancelled=true;renderer.current?.dispose();renderer.current=undefined;};},[requested,attempt]);
 useEffect(()=>{if(frame.current)renderer.current?.update(latest.current,readLabPalette(frame.current));},[selected,result]);
 useEffect(()=>{const observer=new MutationObserver(()=>{if(frame.current)renderer.current?.update(latest.current,readLabPalette(frame.current));});observer.observe(document.documentElement,{attributes:true,attributeFilter:['class']});return()=>observer.disconnect();},[]);
 const plotHeight=result.parameters.graph===0?180:360;
 const point=(i:number)=>{const s=geometry.sites[i].position;return[300+50*s[0],plotHeight/2+40*s[2]];};
 return <figure ref={frame} aria-labelledby={id} data-spin-graph={mode}>
  <figcaption id={id}>{t('위치·결합 그래프와 국소 Bloch 기대값','Site/bond graph and local Bloch expectations')}</figcaption>
  <p>{t('주황 선은 적어도 한 결합 계수가 비영인 쌍입니다. 회색 선은 허용되지만 결합이0인 쌍입니다. 위치 배치는 그래프 설명용이며 실제 거리나 공간 차원을 뜻하지 않습니다.','Orange lines connect pairs with at least one nonzero coupling. Grey lines are allowed pairs with zero coupling. The layout explains a graph, not actual distances or spatial dimension.')}</p>
  <div hidden={requested&&mode==='3d'} role="region" tabIndex={0} aria-label={t('2D 위치·결합 그래프','2D site and bond graph')} style={{overflowX:'auto',maxWidth:'100%'}}><svg viewBox={`0 0 600 ${plotHeight}`} role="img" aria-label={t('위치 번호와 활성 결합','Site indices and active bonds')} style={{width:'100%',minWidth:280,maxWidth:720,maxHeight:320,display:'block',margin:'auto'}}><title>{t('위치 번호와 활성 결합','Site indices and active bonds')}</title>{geometry.edges.map(e=>{const[a,b]=point(e.from),[c,d]=point(e.to);return <line key={`${e.from}-${e.to}`} x1={a} y1={b} x2={c} y2={d} stroke={e.active?'var(--learning-orange)':'var(--learning-neutral)'} strokeWidth={e.active?4:1} strokeDasharray={e.active?undefined:'5 5'}/>;})}{geometry.sites.map(s=>{const[x,y]=point(s.index);return <g key={s.index}><circle cx={x} cy={y} r={23} fill="hsl(var(--background))" stroke={s.index===selected?'var(--learning-blue)':'var(--learning-neutral)'} strokeWidth={s.index===selected?4:1}/><text x={x} y={y+6} textAnchor="middle" fill="currentColor">{s.index}</text></g>;})}</svg></div>
  <div ref={host} hidden={!requested||mode!=='3d'} role="img" aria-label={t('3D 국소 Bloch 벡터와 결합','3D local Bloch vectors and bonds')} style={{width:'100%',height:300}}/>
  <div className="ai-lab-actions"><Button variant="outline" aria-pressed={requested} onClick={()=>setRequested(v=>!v)}>{requested?t('3D 닫기','Close 3D'):t('3D Bloch 벡터 보기','Show 3D Bloch vectors')}</Button>{requested&&mode==='3d'&&(['perspective','front','top'] as const).map((view,i)=><Button key={view} variant="outline" onClick={()=>renderer.current?.view(view)}>{[t('시점 초기화','Reset view'),t('정면','Front'),t('위에서','Top')][i]}</Button>)}{requested&&mode==='2d'&&<Button variant="outline" onClick={()=>setAttempt(v=>v+1)}>{t('3D 다시 시도','Retry 3D')}</Button>}</div>
  {requested&&mode!=='3d'&&<p role="status">{mode==='loading'?t('3D 준비 중','Preparing 3D'):t('3D를 사용할 수 없습니다. 2D 그래프와 아래의 전체 XYZ 표를 이용하세요.','3D is unavailable. Use the 2D graph and complete XYZ table below.')}</p>}
  <p className="ai-small">{t('3D의 파란 화살표는 각 위치의 (〈X〉,〈Y〉,〈Z〉)입니다. GHZ의 국소 벡터는0이어도 공동 상태는 비영입니다. 드래그·시점 버튼은 카메라만 바꾸며 물리적 시간 진화는 없습니다. 모든 좌표는 아래 표에서도 확인할 수 있습니다.','Blue 3D arrows show local (〈X〉,〈Y〉,〈Z〉). GHZ can have zero local vectors and a nonzero joint state. Drag/view buttons change only the camera; no physical-time evolution occurs. Every coordinate is also available in the table below.')}</p>
 </figure>;
}
