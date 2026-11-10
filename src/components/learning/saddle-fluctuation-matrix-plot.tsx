'use client';
import React,{useEffect,useId,useRef,useState} from 'react';
import {Button} from '@/components/ui/button';
import type {SaddleFluctuationMatrixView} from '@/lib/learning-saddle-fluctuation-view';
import {readLabPalette} from './lab-palette';
import type {SaddleFluctuationMatrixRenderer} from './saddle-fluctuation-matrix-engine';
export default function SaddleFluctuationMatrixPlot({model:m,language}:{model:SaddleFluctuationMatrixView;language:'ko'|'en'}){
 const t=(ko:string,en:string)=>language==='ko'?ko:en,id=useId(),frame=useRef<HTMLElement>(null),host=useRef<HTMLDivElement>(null),renderer=useRef<SaddleFluctuationMatrixRenderer>();
 const latest=useRef(m);latest.current=m;
 const[flat,setFlat]=useState(true),[visible,setVisible]=useState(false),[state,setState]=useState<'loading'|'2d'|'3d'>('loading'),[attempt,setAttempt]=useState(0);
 const label=t('이산 행렬 막대: i는 가로, j는 깊이, 높이는 표시한 행렬 값. 화살표 키로 회전, Home으로 초기화', 'Discrete matrix bars: horizontal i, depth j, height displayed matrix value. Arrow keys rotate; Home resets');
  useEffect(() => { if (!frame.current) return; if (!('IntersectionObserver' in window)) { setVisible(true); return; } const observer = new IntersectionObserver(([entry]) => { if(entry.isIntersecting){setVisible(true);observer.disconnect();} }, { rootMargin: '160px' }); observer.observe(frame.current); return () => observer.disconnect(); }, []);
  useEffect(() => {
    if (!visible || flat) return;
    let cancelled = false; setState('loading');
    import('./saddle-fluctuation-matrix-engine').then(({ createSaddleFluctuationMatrixPlot }) => {
      if (cancelled || !host.current || !frame.current) return;
      try { renderer.current = createSaddleFluctuationMatrixPlot(host.current, latest.current, readLabPalette(frame.current), label, () => { renderer.current?.dispose(); renderer.current = undefined; setState('2d'); }); setState('3d'); }
      catch { setState('2d'); }
    }).catch(() => { if (!cancelled) setState('2d'); });
    return () => { cancelled = true; renderer.current?.dispose(); renderer.current = undefined; };
  }, [visible, flat, attempt, label]);
  useEffect(() => { if (frame.current) renderer.current?.update(m, readLabPalette(frame.current)); }, [m]);
  useEffect(() => { const observer = new MutationObserver(() => { if (frame.current) renderer.current?.update(latest.current, readLabPalette(frame.current)); }); observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] }); return () => observer.disconnect(); }, []);
 const fallback=flat||!visible||state!=='3d';
 return <figure ref={frame} aria-labelledby={`${id}-title`} data-saddle-matrix-plot={fallback?'2d':'3d'}><figcaption id={`${id}-title`}>{t('시간 첨자 쌍의3D 행렬 막대', '3D matrix bars for time-index pairs')}</figcaption>
 <div ref={host} className="scalar-three-host" style={{display:fallback?'none':'block',height:360,width:'100%'}}/>
 <div className="qubit-actions"><Button variant="outline" onClick={()=>setFlat(v=>!v)}>{flat?t('3D 행렬 보기', 'Show 3D matrix'):t('3D 행렬 닫기', 'Close 3D matrix')}</Button>{!flat&&state==='2d'&&<Button variant="outline" onClick={()=>setAttempt(v=>v+1)}>{t('3D 다시 시도', 'Retry 3D')}</Button>}{!fallback&&<><Button variant="outline" onClick={()=>renderer.current?.view('perspective')}>{t('시점 초기화', 'Reset view')}</Button><Button variant="outline" onClick={()=>renderer.current?.view('front')}>{t('정면', 'Front')}</Button></>}</div>
 {!flat&&state==='loading'&&<p role="status">{t('3D 행렬을 준비합니다.', 'Preparing the 3D matrix.')}</p>}
 {!flat&&state==='2d'&&<p role="status">{t('3D를 사용할 수 없습니다. 열지도와 정확한 표에서 계속 비교할 수 있습니다.', '3D is unavailable. Continue comparing with the heat map and exact tables.')}</p>}
 <p>{t('가로 i는 주황 축, 깊이 j는 중립색 축, 세로는 선택한 행렬 값입니다. 양수는 파랑, 음수는 주황, 선택한 성분은 초록 표식입니다. 0은 기준면의 점으로 표시합니다.', 'Horizontal i is the orange axis, depth j the neutral axis; vertical height is the selected matrix value. Positive bars are blue, negative bars orange and the selected entry has a green marker. Zero entries are points on the baseline.')}</p>
 <p style={{overflowWrap:'anywhere'}}>{t('높이의 양·음 최대 표시 크기', 'Symmetric height display maximum')}: {m.maximum.toPrecision(6)}. {t('선택 행렬과 척도에 맞춰 자동 조정합니다. 가로 두 축은 두 관측 시각의 첨자이며 실제 공간의 두 좌표가 아닙니다. 높이는 행렬 성분의 표시값이지 경로 곡면이 아닙니다.', 'Rescales for the selected matrix and value transform. The horizontal axes index two observation times, not two physical spatial coordinates. Heights are scaled matrix entries, not a sample-path surface.')}</p>
 </figure>;
}
