'use client';
import React,{useEffect,useId,useRef,useState} from 'react';
import {Button} from '@/components/ui/button';
import type {DuffingOrbitView} from '@/lib/learning-duffing-secular-orbit-view';
import {readLabPalette} from './lab-palette';
import type {DuffingOrbitRenderer} from './duffing-secular-orbit-engine';
export default function DuffingSecularOrbit({model:m,language}:{model:DuffingOrbitView;language:'ko'|'en'}){
 const t=(ko:string,en:string)=>language==='ko'?ko:en,id=useId(),frame=useRef<HTMLElement>(null),host=useRef<HTMLDivElement>(null),renderer=useRef<DuffingOrbitRenderer>();
 const latest=useRef(m);latest.current=m;
 const[flat,setFlat]=useState(true),[visible,setVisible]=useState(false),[state,setState]=useState<'loading'|'2d'|'3d'>('loading'),[attempt,setAttempt]=useState(0);
 const label=t('정확 기준해: 가로 위치 x, 세로 속도 v, 깊이 시간 t. 화살표 키로 회전, Home으로 초기화','Exact reference: horizontal position x, vertical velocity v, depth time t. Arrow keys rotate; Home resets');
  useEffect(() => { if (!frame.current) return; if (!('IntersectionObserver' in window)) { setVisible(true); return; } const observer = new IntersectionObserver(([entry]) => { if(entry.isIntersecting){setVisible(true);observer.disconnect();} }, { rootMargin: '160px' }); observer.observe(frame.current); return () => observer.disconnect(); }, []);
  useEffect(() => {
    if (!visible || flat) return;
    let cancelled = false; setState('loading');
    import('./duffing-secular-orbit-engine').then(({ createDuffingOrbitPlot }) => {
      if (cancelled || !host.current || !frame.current) return;
      try { renderer.current = createDuffingOrbitPlot(host.current, latest.current, readLabPalette(frame.current), label, () => { renderer.current?.dispose(); renderer.current = undefined; setState('2d'); }); setState('3d'); }
      catch { setState('2d'); }
    }).catch(() => { if (!cancelled) setState('2d'); });
    return () => { cancelled = true; renderer.current?.dispose(); renderer.current = undefined; };
  }, [visible, flat, attempt, label]);
  useEffect(() => { if (frame.current) renderer.current?.update(m, readLabPalette(frame.current)); }, [m]);
  useEffect(() => { const observer = new MutationObserver(() => { if (frame.current) renderer.current?.update(latest.current, readLabPalette(frame.current)); }); observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] }); return () => observer.disconnect(); }, []);
 const fallback=flat||!visible||state!=='3d';
 return <figure ref={frame} aria-labelledby={`${id}-title`} data-duffing-orbit={fallback?'2d':'3d'}><figcaption id={`${id}-title`}>{t('정확 기준해의 위치·속도·시간','Exact-reference position, velocity and time')}</figcaption>
 <p>{t('3D는 관측점 주변 정확 기준해의 (x,v,t) 곡선입니다. 근사해의 속도 표본을 생성한 그림이 아닙니다. 네 방법의 비교는 위 2D 파형을 사용하세요.','The 3D curve is the local exact reference in (x,v,t). It does not invent sampled approximation velocities. Use the 2D waveforms above to compare the four methods.')}</p>
 <div ref={host} className="scalar-three-host" style={{display:fallback?'none':'block',height:360,width:'100%'}}/>
 <div className="qubit-actions"><Button variant="outline" onClick={()=>setFlat(v=>!v)}>{flat?t('3D 궤적 보기','Show 3D orbit'):t('3D 궤적 닫기','Close 3D orbit')}</Button>{!flat&&state==='2d'&&<Button variant="outline" onClick={()=>setAttempt(v=>v+1)}>{t('3D 다시 시도','Retry 3D')}</Button>}{!fallback&&<><Button variant="outline" onClick={()=>renderer.current?.view('perspective')}>{t('시점 초기화','Reset view')}</Button><Button variant="outline" onClick={()=>renderer.current?.view('front')}>{t('정면','Front')}</Button></>}</div>
 {!flat&&state==='loading'&&<p role="status">{t('3D 궤적을 준비합니다.','Preparing the 3D orbit.')}</p>}
 {!flat&&state==='2d'&&<p role="status">{t('3D를 사용할 수 없습니다. 위 파형과 표에서 계속 비교할 수 있습니다.','3D is unavailable. Continue comparing with the waveforms and tables above.')}</p>}
 <p>{t('가로 x는 주황 축, 세로 v는 초록 축, 깊이 t는 중립색 축입니다. 각 축을 독립적으로 정규화하므로 그림 각도나 길이로 물리 단위를 비교하지 마세요. 시간은 국소 창 안에서 선형입니다.','Position x is the orange axis, velocity v the green axis and time t the neutral axis. Axes are scaled independently: angles and lengths cannot compare physical units. Time is linear within the local window.')}</p>
 <p style={{overflowWrap:'anywhere'}}>{t('표시 정규화','Display normalization')}: x/{m.positionScale.toPrecision(6)}, v/{m.velocityScale.toPrecision(6)}, t∈[{m.start.toString()}, {m.end.toString()}]. {t('위치·속도 척도는 표본 절댓값의 최대이며 모두0이면 표시 분모1을 씁니다. 연결은 표본 보간입니다.','Position/velocity scales are sampled absolute maxima, with display denominator 1 when all values are zero. Connections interpolate samples.')}</p>
 {m.equilibrium&&<p>{t('평형에서는 x=v=0입니다. 깊이 방향의 선은 시간 경과만 나타내며 진동이 아닙니다.','At equilibrium x=v=0. A line in depth records elapsed time only; it is not oscillation.')}</p>}
 </figure>;
}
