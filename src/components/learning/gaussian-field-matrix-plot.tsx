'use client';
import React,{useEffect,useId,useRef,useState} from 'react';
import {Button} from '@/components/ui/button';
import type {GaussianFieldView} from '@/lib/learning-gaussian-field-view';
import {readLabPalette} from './lab-palette';
import type {GaussianMatrixRenderer} from './gaussian-field-matrix-engine';
export default function GaussianFieldMatrixPlot({model:m,language}:{model:Pick<GaussianFieldView,'n'|'points'|'maximum'>;language:'ko'|'en'}){
 const t=(ko:string,en:string)=>language==='ko'?ko:en,id=useId(),frame=useRef<HTMLElement>(null),host=useRef<HTMLDivElement>(null),renderer=useRef<GaussianMatrixRenderer>();
 const latest=useRef(m);latest.current=m;
 const[flat,setFlat]=useState(true),[visible,setVisible]=useState(false),[state,setState]=useState<'loading'|'2d'|'3d'>('loading'),[attempt,setAttempt]=useState(0);
 const label=t('이산 행렬 막대: i는 가로, j는 깊이, 높이는 표시한 행렬 값. 화살표 키로 회전, Home으로 초기화','Discrete matrix bars: horizontal i, depth j, height displayed matrix value. Arrow keys rotate; Home resets');
  useEffect(() => { if (!frame.current) return; if (!('IntersectionObserver' in window)) { setVisible(true); return; } const observer = new IntersectionObserver(([entry]) => { if(entry.isIntersecting){setVisible(true);observer.disconnect();} }, { rootMargin: '160px' }); observer.observe(frame.current); return () => observer.disconnect(); }, []);
  useEffect(() => {
    if (!visible || flat) return;
    let cancelled = false; setState('loading');
    import('./gaussian-field-matrix-engine').then(({ createGaussianMatrixPlot }) => {
      if (cancelled || !host.current || !frame.current) return;
      try { renderer.current = createGaussianMatrixPlot(host.current, latest.current, readLabPalette(frame.current), label, () => { renderer.current?.dispose(); renderer.current = undefined; setState('2d'); }); setState('3d'); }
      catch { setState('2d'); }
    }).catch(() => { if (!cancelled) setState('2d'); });
    return () => { cancelled = true; renderer.current?.dispose(); renderer.current = undefined; };
  }, [visible, flat, attempt, label]);
  useEffect(() => { if (frame.current) renderer.current?.update(m, readLabPalette(frame.current)); }, [m]);
  useEffect(() => { const observer = new MutationObserver(() => { if (frame.current) renderer.current?.update(latest.current, readLabPalette(frame.current)); }); observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] }); return () => observer.disconnect(); }, []);
 const fallback=flat||!visible||state!=='3d';
 return <figure ref={frame} aria-labelledby={`${id}-title`} data-gaussian-matrix-plot={fallback?'2d':'3d'}><figcaption id={`${id}-title`}>{t('이산 위치 쌍의3D 행렬 막대','3D matrix bars for discrete site pairs')}</figcaption>
 <div ref={host} className="scalar-three-host" style={{display:fallback?'none':'block',height:360,width:'100%'}}/>
 <div className="qubit-actions"><Button variant="outline" onClick={()=>setFlat(v=>!v)}>{flat?t('3D 행렬 보기','Show 3D matrix'):t('3D 행렬 닫기','Close 3D matrix')}</Button>{!flat&&state==='2d'&&<Button variant="outline" onClick={()=>setAttempt(v=>v+1)}>{t('3D 다시 시도','Retry 3D')}</Button>}{!fallback&&<><Button variant="outline" onClick={()=>renderer.current?.view('perspective')}>{t('시점 초기화','Reset view')}</Button><Button variant="outline" onClick={()=>renderer.current?.view('front')}>{t('정면','Front')}</Button></>}</div>
 {!flat&&state==='loading'&&<p role="status">{t('3D 행렬을 준비합니다.','Preparing the 3D matrix.')}</p>}
 {!flat&&state==='2d'&&<p role="status">{t('3D를 사용할 수 없습니다. 열지도와 정확한 표에서 계속 비교할 수 있습니다.','3D is unavailable. Continue comparing with the heat map and exact tables.')}</p>}
 <p>{t('가로 i는 주황 축, 깊이 j는 중립색 축, 세로는 선택한 행렬 값입니다. 양수는 파랑, 음수는 주황, 선택한 성분은 초록 표식입니다. 0은 기준면의 점으로 표시합니다.','Horizontal i is the orange axis, depth j the neutral axis; vertical height is the selected matrix value. Positive bars are blue, negative bars orange and the selected entry has a green marker. Zero entries are points on the baseline.')}</p>
 <p style={{overflowWrap:'anywhere'}}>{t('높이의 양·음 최대 표시 크기','Symmetric height display maximum')}: {m.maximum.toPrecision(6)}. {t('선택 행렬과 척도에 맞춰 자동 조정합니다. 이산 두 첨자는 실제 공간의 두 축이나 시간이 아니며 연속 곡면을 만들지 않습니다.','Rescales for the selected matrix and value transform. The two discrete indices are not two physical spatial axes or time; no continuous surface is constructed.')}</p>
 </figure>;
}
