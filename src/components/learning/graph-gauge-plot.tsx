'use client';
import {useEffect,useId,useRef,useState} from 'react';
import {Button} from '@/components/ui/button';
import {graphGaugeView,type GraphGaugeMode} from '@/lib/learning-graph-gauge-view';
import type {GraphGaugeResult} from '@/lib/learning-graph-gauge';
import {readLabPalette} from './lab-palette';
import type {GraphGaugePlotRenderer} from './graph-gauge-plot-engine';
export default function GraphGaugePlot({result,mode,walk,language}:{result:GraphGaugeResult;mode:GraphGaugeMode;walk:number;language:'ko'|'en'}){
 const t=(ko:string,en:string)=>language==='ko'?ko:en,id=useId(),frame=useRef<HTMLElement>(null),host=useRef<HTMLDivElement>(null),renderer=useRef<GraphGaugePlotRenderer>(),model=graphGaugeView(result,mode,walk),latest=useRef(model);latest.current=model;
 const[visible,setVisible]=useState(false),[flat,setFlat]=useState(false),[state,setState]=useState<'loading'|'3d'|'2d'>('loading'),[attempt,setAttempt]=useState(0);
 const label=t('추상 그래프와 국소 복소 화살표 · 화살표 키로 시점 조작, Home으로 초기화','Abstract graph and local complex arrows; arrow keys rotate, Home resets');
 useEffect(()=>{if(!frame.current)return;if(!('IntersectionObserver'in window)){setVisible(true);return;}const observer=new IntersectionObserver(([entry])=>setVisible(entry.isIntersecting),{rootMargin:'160px'});observer.observe(frame.current);return()=>observer.disconnect();},[]);
 useEffect(()=>{if(!visible||flat||!latest.current)return;let cancelled=false;setState('loading');import('./graph-gauge-plot-engine').then(({createGraphGaugePlot})=>{if(cancelled||!host.current||!frame.current||!latest.current)return;try{renderer.current=createGraphGaugePlot(host.current,latest.current,readLabPalette(frame.current),label,()=>{renderer.current?.dispose();renderer.current=undefined;setState('2d');});setState('3d');}catch{setState('2d');}}).catch(()=>{if(!cancelled)setState('2d');});return()=>{cancelled=true;renderer.current?.dispose();renderer.current=undefined;};},[visible,flat,attempt,label]);
 useEffect(()=>{if(frame.current&&model)renderer.current?.update(model,readLabPalette(frame.current));},[model]);
 useEffect(()=>{const observer=new MutationObserver(()=>{if(frame.current&&latest.current)renderer.current?.update(latest.current,readLabPalette(frame.current));});observer.observe(document.documentElement,{attributes:true,attributeFilter:['class']});return()=>observer.disconnect();},[]);
 const fallback=flat||!visible||state!=='3d',scale=150/(model?.extent??3.45),point=(p:{x:number;z:number})=>({x:300+scale*p.x,y:180-scale*p.z});
 return <figure ref={frame} aria-labelledby={id} data-graph-gauge-plot={fallback?'2d':'3d'}><figcaption id={id}>{t('추상 연결과 물질 성분','Abstract connections and matter components')}</figcaption>
  <div style={{minHeight:360}}><div ref={host} hidden={fallback} style={{width:'100%',height:360}}/>
  {fallback&&model&&<div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('그래프2D 그림 · 좌우 이동 가능','Graph 2D diagram; scroll horizontally')}><svg viewBox="0 0 600 360" role="img" aria-label={label} style={{width:'100%',minWidth:600,height:360}}>
   <defs><marker id={`${id}-arrow`} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="var(--learning-orange)"/></marker></defs>
   {model.edges.map(e=>{const a=point(e.points[18]),b=point(e.points[19]),mid=point(e.points[12]);return <g key={e.index}><polyline points={e.points.map(p=>{const q=point(p);return`${q.x},${q.y}`;}).join(' ')} fill="none" stroke={e.highlight?'var(--learning-green)':'var(--learning-neutral)'} strokeWidth={e.highlight?3:1.5}/><path d={`M${a.x} ${a.y}L${b.x} ${b.y}`} stroke="var(--learning-orange)" markerEnd={`url(#${id}-arrow)`}/><path d={`M${mid.x} ${mid.y}L${mid.x+.4*scale*e.re!} ${mid.y-.4*scale*e.im!}`} stroke="var(--learning-blue)" strokeWidth={3}/><circle cx={mid.x+.4*scale*e.re!} cy={mid.y-.4*scale*e.im!} r={3} fill="var(--learning-blue)"/></g>;})}
   {model.nodes.map(n=>{const p=point(n);return <g key={n.site}><circle cx={p.x} cy={p.y} r={5} fill="var(--learning-blue)"/><path d={`M${p.x} ${p.y}L${p.x+.65*scale*n.re} ${p.y-.65*scale*n.im}`} stroke="var(--learning-orange)" strokeWidth={3}/><text x={p.x+10} y={p.y+18} fill="currentColor" fontSize={18}>{n.site}</text></g>;})}
  </svg></div>}
  {!model&&<p>{t('그림의 수치 범위를 벗어났습니다. 아래 정확한 표를 확인하세요.','Values exceed the drawing range. Inspect the exact tables below.')}</p>}</div>
  <div className="qubit-actions"><Button variant="outline" onClick={()=>setFlat(v=>!v)}>{flat?t('3D 보기','Show 3D'):t('2D 보기','Show 2D')}</Button><Button variant="outline" disabled={fallback} onClick={()=>renderer.current?.view('perspective')}>{t('시점 초기화','Reset view')}</Button><Button variant="outline" disabled={fallback} onClick={()=>renderer.current?.view('front')}>{t('정면','Front')}</Button>{!flat&&state==='2d'&&<Button variant="outline" onClick={()=>setAttempt(n=>n+1)}>{t('3D 다시 시도','Retry 3D')}</Button>}</div>
  <p>{t('원형 배치는 노드 번호를 구분하는 추상 배치입니다. 파란 점은 노드, 간선 중간의 짧은 파란 표시는 링크의 Re/Im 위상입니다. 주황 선은 물질 성분과 저장 간선 순회 방향, 초록 선은 선택 경로가 쓰는 간선입니다. 순회 횟수·방향은 정확한 경로 표에서 읽으세요. 화살표는 에너지 흐름이 아닙니다.','The circle is an abstract layout for site indices. Blue dots are sites; short blue marks at edge midpoints show link Re/Im phases. Orange marks show matter components and stored edge traversal directions; green edges occur in the selected walk. Read multiplicities and traversal directions in the exact path table. Arrows are not energy currents.')}</p>
  <p>{t('복소 화살표의 가로/세로는 Re/Im이며 모든 모드에서 같은 최대 진폭으로 나눕니다. 링크 위상은 척도1입니다. 3D의 높이는 밀도를 최대 밀도로 나눈 값의1.2배이며 위상 화살표는 수평면에 그립니다. 공간 좌표·시간 변화가 아닙니다.','Complex arrows use Re/Im divided by the same maximum amplitude in every mode. Link phases use scale1. In 3D,height is1.2 times normalized density and phase arrows lie in horizontal planes. These are not spatial coordinates or time evolution.')}</p>
  <p className="ai-small">{t('그리기 진폭 척도(근사)','Drawing amplitude scale (approximate)')}: {model?.maximum.toExponential(5)??'—'}</p>
 </figure>;
}
