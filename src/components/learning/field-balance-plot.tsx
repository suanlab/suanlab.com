'use client';
import {useEffect,useId,useRef,useState} from 'react';
import {Button} from '@/components/ui/button';
import {fieldBalanceView,type FieldBalanceQuantity} from '@/lib/learning-field-balance-view';
import type {FieldBalanceResult} from '@/lib/learning-field-balance';
import {readLabPalette} from './lab-palette';
import type {ScalarFieldPlotRenderer} from './scalar-field-plot-engine';
export default function FieldBalancePlot({result,quantity,language}:{result:FieldBalanceResult;quantity:FieldBalanceQuantity;language:'ko'|'en'}){
 const t=(ko:string,en:string)=>language==='ko'?ko:en,id=useId(),frame=useRef<HTMLElement>(null),host=useRef<HTMLDivElement>(null),renderer=useRef<ScalarFieldPlotRenderer>();
 const model=fieldBalanceView(result,quantity),latest=useRef(model);latest.current=model;
 const[visible,setVisible]=useState(false),[flat,setFlat]=useState(false),[mode,setMode]=useState<'loading'|'3d'|'2d'>('loading'),[attempt,setAttempt]=useState(0);
 const label=t('점별 수지 그림 · 화살표 키로 시점 조작, Home으로 초기화','Site balance diagram; arrow keys rotate, Home resets');
 useEffect(()=>{if(!frame.current)return;if(!('IntersectionObserver'in window)){setVisible(true);return;}const observer=new IntersectionObserver(([entry])=>setVisible(entry.isIntersecting),{rootMargin:'160px'});observer.observe(frame.current);return()=>observer.disconnect();},[]);
 useEffect(()=>{if(!visible||flat||!latest.current)return;let cancelled=false;setMode('loading');import('./scalar-field-plot-engine').then(({createScalarFieldPlot})=>{if(cancelled||!host.current||!frame.current||!latest.current)return;try{renderer.current=createScalarFieldPlot(host.current,latest.current,readLabPalette(frame.current),label,()=>{renderer.current?.dispose();renderer.current=undefined;setMode('2d');});setMode('3d');}catch{setMode('2d');}}).catch(()=>{if(!cancelled)setMode('2d');});return()=>{cancelled=true;renderer.current?.dispose();renderer.current=undefined;};},[visible,flat,attempt,label]);
 useEffect(()=>{if(frame.current&&model)renderer.current?.update(model,readLabPalette(frame.current));},[model]);
 useEffect(()=>{const observer=new MutationObserver(()=>{if(frame.current&&latest.current)renderer.current?.update(latest.current,readLabPalette(frame.current));});observer.observe(document.documentElement,{attributes:true,attributeFilter:['class']});return()=>observer.disconnect();},[]);
 const fallback=flat||!visible||mode!=='3d';
 return <figure ref={frame} aria-labelledby={id} data-balance-plot={fallback?'2d':'3d'}><figcaption id={id}>{t('현재 점별 값 · 시간은 적분하지 않았습니다','Current site values; time has not been integrated')}</figcaption>
  <div ref={host} hidden={fallback} style={{width:'100%',height:300}}/>
  {fallback&&model&&<div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('공간 상태2D 그림 · 좌우 이동 가능','Spatial state 2D diagram; scroll horizontally')}><svg viewBox="0 0 600 280" role="img" aria-label={label} style={{width:'100%',minWidth:440,height:'auto'}}>
   <path d="M45 130H565M45 25V240" stroke="var(--learning-neutral)" fill="none"/>
   <polyline points={model.curve.map(p=>`${45+520*p.x},${130-95*p.y}`).join(' ')} fill="none" stroke="var(--learning-blue)" strokeWidth={3}/>
   {model.points.map(p=><g key={p.site}><path d={`M${45+520*p.x} 130V${130-95*p.y}`} stroke="var(--learning-orange)"/><circle cx={45+520*p.x} cy={130-95*p.y} r={4} fill="var(--learning-blue)"/></g>)}
   <text fontSize={20} x={45} y={265} fill="currentColor">x/L=0</text><text fontSize={20} x={505} y={265} fill="currentColor">x/L=1</text><text fontSize={20} x={50} y={22} fill="currentColor">{quantity==='energy'?'E':quantity==='energyDerivative'?'dE/dt':'source'} / max|·|</text>
  </svg></div>}
  {!model&&<p>{t('이 입력은 그림의 수치 범위를 벗어났습니다. 정확한 표를 확인하세요.','This input exceeds the drawing range. Inspect the exact table.')}</p>}
  <div className="qubit-actions"><Button variant="outline" onClick={()=>setFlat(v=>!v)}>{flat?t('3D 보기','Show 3D'):t('2D 보기','Show 2D')}</Button>{!fallback&&<><Button variant="outline" onClick={()=>renderer.current?.view('perspective')}>{t('시점 초기화','Reset view')}</Button><Button variant="outline" onClick={()=>renderer.current?.view('front')}>{t('정면','Front')}</Button></>}{!flat&&mode==='2d'&&<Button variant="outline" onClick={()=>setAttempt(n=>n+1)}>{t('3D 다시 시도','Retry 3D')}</Button>}</div>
  <p>{t('가로축은x/L, 세로축은 선택한 양의 최대 절댓값으로 나눈 그리기 좌표입니다. 양마다 척도가 다릅니다. 주기 경계의 끝에 반복한 점은 추가 자유도가 아닙니다. 고정 끝점의 에너지를0으로 그리지 않습니다. 정확한 값과 좌표는 아래 표에서 확인하세요.','Horizontal coordinates are x/L; heights are divided by the maximum absolute selected value for drawing. Each quantity has a separate scale. Repeated periodic endpoints are not additional degrees of freedom. Fixed endpoint energies are not drawn as zero. Inspect exact values and positions in the table.')}</p>
  <p className="ai-small">{t('현재 세로 척도(근사)','Current vertical scale (approximate)')}: {model?.maximum.toExponential(5)??'—'} {model?.exactZero&&t('· 모두 정확히0','· all exactly zero')}</p>
 </figure>;
}
