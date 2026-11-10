'use client';
import React, { useEffect, useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import type { BoundaryLayerView } from '@/lib/learning-boundary-layer-view';
import { readLabPalette } from './lab-palette';
import type { BoundaryComparisonRenderer } from './boundary-layer-comparison-engine';
export default function BoundaryLayerPlots({ model: m, language, selected }: { model: BoundaryLayerView; language: 'ko' | 'en'; selected:number }) {
  const t = (ko: string, en: string) => language === 'ko' ? ko : en, id = useId(), frame = useRef<HTMLElement>(null), host = useRef<HTMLDivElement>(null), renderer = useRef<BoundaryComparisonRenderer>();
  const latest = useRef(m); latest.current = m;
  const [flat, setFlat] = useState(true), [visible, setVisible] = useState(false), [state, setState] = useState<'loading' | '2d' | '3d'>('loading'), [attempt, setAttempt] = useState(0);
  const label=t('해 비교 띠: 가로는 선택 좌표, 높이는 해 값, 깊이는 해 종류입니다. 화살표 키로 회전, Home으로 초기화','Solution comparison ribbons: horizontal selected coordinate, height solution value, depth solution method. Arrow keys rotate; Home resets');
  useEffect(() => { if (!frame.current) return; if (!('IntersectionObserver' in window)) { setVisible(true); return; } const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: '160px' }); observer.observe(frame.current); return () => observer.disconnect(); }, []);
  useEffect(() => {
    if (!visible || flat) return;
    let cancelled = false; setState('loading');
    import('./boundary-layer-comparison-engine').then(({ createBoundaryComparisonPlot }) => {
      if (cancelled || !host.current || !frame.current) return;
      try { renderer.current = createBoundaryComparisonPlot(host.current, latest.current, readLabPalette(frame.current), label, () => { renderer.current?.dispose(); renderer.current = undefined; setState('2d'); }); setState('3d'); }
      catch { setState('2d'); }
    }).catch(() => { if (!cancelled) setState('2d'); });
    return () => { cancelled = true; renderer.current?.dispose(); renderer.current = undefined; };
  }, [visible, flat, attempt, label]);
  useEffect(() => { if (frame.current) renderer.current?.update(m, readLabPalette(frame.current)); }, [m]);
  useEffect(() => { const observer = new MutationObserver(() => { if (frame.current) renderer.current?.update(latest.current, readLabPalette(frame.current)); }); observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] }); return () => observer.disconnect(); }, []);
  const fallback=flat||!visible||state!=='3d',x=(v:number)=>65+510*v/m.xMax,y=(v:number)=>250-195*(m.transform(v)-m.yMin)/(m.yMax-m.yMin);
  const colors={exact:'var(--learning-blue)',outer:'var(--learning-orange)',composite:'var(--learning-green)',inner:'currentColor'},names={exact:t('정확해','Exact'),outer:t('바깥해','Outer'),composite:t('합성해','Composite'),inner:t('선도 안쪽해','Leading inner')};
  const chosen=m.points[Math.max(0,Math.min(64,selected))];
  return <figure ref={frame} data-boundary-comparison={fallback?'2d':'3d'} aria-labelledby={`${id}-profile`}>
   <figcaption id={`${id}-profile`}>{t('같은 경계값 문제의 해와 근사 비교','Compare solutions and approximations of the same boundary problem')}</figcaption>
   <div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('해 비교 그림 · 좌우 이동 가능','Solution plot; scroll horizontally')} style={{display:fallback?'block':'none'}}>
    <svg viewBox="0 0 640 320" role="img" aria-label={t('정확해·바깥해·합성해의 검증된 표본 투영','Verified sample projections of exact, outer and composite solutions')} style={{width:'100%',minWidth:640,height:320,fontSize:18}}>
     <path d="M65 50V250H575" stroke="currentColor" fill="none" opacity=".4"/>
     {[0,.5,1].map(f=><g key={f}><text x={65+510*f} y={280} textAnchor="middle" fill="currentColor">{(m.xMax*f).toPrecision(3)}</text></g>)}
     {[0,.5,1].map(f=><text key={f} x={60} y={255-195*f} textAnchor="end" fill="currentColor" fontSize={14}>{(m.yMin+(m.yMax-m.yMin)*f).toPrecision(3)}</text>)}
     {m.kinds.map(kind=><polyline key={kind} fill="none" stroke={colors[kind]} strokeWidth={kind==='exact'?3:2} strokeDasharray={kind==='outer'?'8 5':kind==='composite'?'3 4':undefined} points={m.points.map(p=>`${x(p.coordinate)},${y(p[kind]!)}`).join(' ')}/>)}
     <line x1={x(chosen.coordinate)} x2={x(chosen.coordinate)} y1={50} y2={250} stroke="currentColor" opacity=".3"/><circle cx={x(chosen.coordinate)} cy={y(chosen.exact)} r={5} fill={colors.exact}/>
     <text x={320} y={310} fill="currentColor" textAnchor="middle">{m.coordinate==='inner'?'X=y/δ':'x'}</text>
    </svg>
   </div>
   <div ref={host} className="scalar-three-host" style={{display:fallback?'none':'block'}}/>
   <p className="qubit-legend">{m.kinds.map(kind=><span key={kind} style={{color:colors[kind],marginRight:16,display:'inline-block'}}>{names[kind]}</span>)}</p>
   <div className="qubit-actions"><Button variant="outline" onClick={()=>{setFlat(v=>!v);if(!flat)setState('loading');}}>{flat?t('3D 보기','Show 3D'):t('2D 보기','Show 2D')}</Button>{!flat&&state==='2d'&&<Button variant="outline" onClick={()=>setAttempt(v=>v+1)}>{t('3D 다시 시도','Retry 3D')}</Button>}{!fallback&&<><Button variant="outline" onClick={()=>renderer.current?.view('perspective')}>{t('시점 초기화','Reset view')}</Button><Button variant="outline" onClick={()=>renderer.current?.view('front')}>{t('정면','Front')}</Button></>}</div>
   {!flat&&state==='loading'&&<p role="status">{t('3D 비교를 준비합니다.','Preparing the 3D comparison.')}</p>}
   <p>{t('그림은 검증된 포함 구간의 중간값을 부동소수점으로 투영합니다. 3D 깊이는 해의 종류이고, 연결 띠는 비교를 위한 보간입니다. 물리적 공간이나 시간 축이 아닙니다.','Drawings project midpoints of verified intervals into floating point. 3D depth labels solution methods; connecting ribbons interpolate for comparison. Depth is not a physical space or time coordinate.')}</p>
   <p>{m.coordinate==='inner'?t('늘린 좌표 창은 X≤min(8,1/δ)입니다. 오른쪽 층에서는 X가 커질수록 물리 x는 작아집니다. 창 바깥을 지수가 정확히0인 영역으로 해석하지 마세요.','The stretched window is X≤min(8,1/δ). For a right layer, physical x decreases as X increases. The region outside this window is not an exactly zero exponential.'):t('물리 좌표65점은 아주 얇은 층을 놓칠 수 있습니다. 비영 흐름에서는 늘린 좌표도 비교하세요.','The 65 physical samples can miss a very thin layer. At nonzero drift, compare the stretched-coordinate view too.')}</p>
   {m.zeroDrift&&<p data-boundary-zero-drift>{t('흐름0: 정확한 이차식만 표시하며 안쪽 좌표나 비교 깊이를 만들지 않습니다.','Zero drift: only the exact quadratic is displayed; no stretched coordinate or comparison depth is manufactured.')}</p>}
  </figure>;
}
