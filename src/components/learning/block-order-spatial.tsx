'use client';
import React, { useEffect, useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import type { BlockOrderInput } from '@/lib/learning-block-order-input';
import { blockOrderApprox, type BlockOrderSpatialModel } from '@/lib/learning-block-order-view';
import { readLabPalette } from './lab-palette';
import type { BlockOrderSpatialRenderer } from './block-order-spatial-engine';

export default function BlockOrderSpatial({ model,input,language,flat,onFlat }: { model: BlockOrderSpatialModel; input: BlockOrderInput; language: 'ko' | 'en'; flat: boolean; onFlat: (flat: boolean) => void }) {
  const id = useId(), frame = useRef<HTMLElement>(null), host = useRef<HTMLDivElement>(null), renderer = useRef<BlockOrderSpatialRenderer>();
  const latest = useRef(model); latest.current = model;
  const [visible,setVisible] = useState(false), [state,setState] = useState<'loading' | '2d' | '3d'>('loading'), [attempt,setAttempt] = useState(0);
  const t = (ko: string,en: string) => language === 'ko' ? ko : en;
  const label = t('공간 격자: 가로와 깊이는 실제 사이트 위치, 높이는 스핀 단위의 선택 값. 화살표로 시점 회전, Home으로 초기화.','Spatial lattice: horizontal and depth coordinates are physical sites; height is the selected value in spin units. Arrows rotate; Home resets.');
  useEffect(() => { if (!frame.current) return; if (!('IntersectionObserver' in window)) { setVisible(true); return; } const observer = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setVisible(true); observer.disconnect(); } },{ rootMargin: '160px' }); observer.observe(frame.current); return () => observer.disconnect(); },[]);
  useEffect(() => {
    if (!visible || flat || !latest.current.available) return;
    let cancelled = false; setState('loading');
    import('./block-order-spatial-engine').then(({ createBlockOrderSpatialPlot }) => {
      if (cancelled || !host.current || !frame.current) return;
      try { renderer.current = createBlockOrderSpatialPlot(host.current,latest.current,readLabPalette(frame.current),label,() => { renderer.current?.dispose(); renderer.current = undefined; setState('2d'); }); setState('3d'); }
      catch { setState('2d'); }
    }).catch(() => { if (!cancelled) setState('2d'); });
    return () => { cancelled = true; renderer.current?.dispose(); renderer.current = undefined; };
  },[visible,flat,attempt,label]);
  useEffect(() => { if (frame.current) renderer.current?.update(model,readLabPalette(frame.current)); },[model]);
  useEffect(() => { const observer = new MutationObserver(() => { if (frame.current) renderer.current?.update(latest.current,readLabPalette(frame.current)); }); observer.observe(document.documentElement,{ attributes: true,attributeFilter: ['class'] }); return () => observer.disconnect(); },[]);
  const nx = input.shape[0], ny = input.shape[1] ?? 1, width = Math.max(460,nx*72+60), height = ny*100+150;
  const pos = (i: number) => ({ x: 60+(i%nx)*72,y: 95+Math.floor(i/nx)*100 });
  const blocksX = nx/input.blockShape[0], blocks = new Set(model.points.map(p => p.block));
  const fallback = flat || !visible || state !== '3d' || !model.available;
  return <figure ref={frame} aria-labelledby={`${id}-title`} data-block-spatial-plot={fallback ? '2d' : '3d'}>
    <figcaption id={`${id}-title`}>{t('같은 격자의 미시 배치와 평균','Microscopic arrangements and means on the same lattice')}</figcaption>
    <div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('사이트와 블록의2D 지도 · 좌우 이동 가능','2D site and block map; scroll horizontally')}>
      <svg className="block-order-grid-svg" viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={t('점은 사이트, 테두리는 블록. 점의 숫자는 선택한 종류의 값이며 아래 표에 정확한 값이 있습니다.','Points are sites and boxes are blocks. Point labels show the selected quantity; exact values are in the table below.')}>
        {[...blocks].map(b => <g key={b}><rect x={30+(b%blocksX)*input.blockShape[0]*72} y={40+Math.floor(b/blocksX)*(input.blockShape[1] ?? 1)*100} width={input.blockShape[0]*72-12} height={(input.blockShape[1] ?? 1)*100-2} rx={10} fill="none" stroke="var(--learning-blue)" strokeWidth={2}/><text x={38+(b%blocksX)*input.blockShape[0]*72} y={62+Math.floor(b/blocksX)*(input.blockShape[1] ?? 1)*100} fill="currentColor" fontSize={14}>B{b}</text></g>)}
        {model.edges.map(([i,j],edge) => {
          const a = pos(i), b = pos(j), wrap = j < i, different = model.mode === 'configuration' && model.points[i].value !== model.points[j].value, cross = model.points[i].block !== model.points[j].block;
          const d = !wrap ? `M${a.x} ${a.y}L${b.x} ${b.y}` : a.y === b.y ? `M${a.x} ${a.y+20}C${a.x} ${a.y+72},${b.x} ${b.y+72},${b.x} ${b.y+20}` : `M${a.x+20} ${a.y}C${a.x+54} ${a.y},${b.x+54} ${b.y},${b.x+20} ${b.y}`;
          return <path key={edge} d={d} fill="none" stroke={different ? 'var(--learning-orange)' : 'var(--learning-neutral)'} strokeWidth={different ? 3 : 1.5} strokeDasharray={cross ? '5 4' : undefined}/>;
        })}
        {model.points.map(p => { const at = pos(p.site); return <g key={p.site}><circle cx={at.x} cy={at.y} r={22} fill="hsl(var(--background))" stroke={p.value !== null && p.value < 0 ? 'var(--learning-orange)' : p.value === 0 ? 'var(--learning-neutral)' : 'var(--learning-blue)'} strokeWidth={2}/><text x={at.x} y={at.y+5} textAnchor="middle" fill="currentColor" fontSize={14}>{model.mode === 'configuration' ? p.value === 1 ? '+1' : '−1' : blockOrderApprox(p.exact)}</text><text x={at.x} y={at.y+39} textAnchor="middle" fill="currentColor" fontSize={14}>i={p.site}</text></g>; })}
      </svg>
    </div>
    <p>{t('지도에서 파선은 블록 사이 edge, 주황 edge는 선택한 미시 배치의 불일치입니다. 평균을 표시할 때는 불일치 여부를 지정하지 않습니다.','On the map, dashed edges cross blocks and orange edges mark disagreements in the selected microscopic arrangement. Disagreement is not assigned to an average field.')}</p>
    <div ref={host} className="scalar-three-host" style={{ display: fallback ? 'none' : 'block',height: 360,width: '100%' }}/>
    <div className="qubit-actions"><Button variant="outline" disabled={!model.available} onClick={() => onFlat(!flat)}>{flat ? t('3D 격자 보기','Show 3D lattice') : t('3D 격자 닫기','Close 3D lattice')}</Button>{!flat && state === '2d' && <Button variant="outline" onClick={() => setAttempt(n => n+1)}>{t('3D 다시 시도','Retry 3D')}</Button>}{!fallback && <><Button variant="outline" onClick={() => renderer.current?.view('perspective')}>{t('시점 초기화','Reset view')}</Button><Button variant="outline" onClick={() => renderer.current?.view('front')}>{t('정면','Front')}</Button></>}</div>
    {!flat && state === 'loading' && <p role="status">{t('3D 격자를 준비합니다.','Preparing the 3D lattice.')}</p>}
    {!flat && state === '2d' && <p role="status">{t('3D를 사용할 수 없습니다.2D 지도와 정확한 표에서 계속 비교하세요.','3D is unavailable. Continue comparing with the2D map and exact tables.')}</p>}
    <p>{t('그림은 현재 배치 또는 평균의 근사 표시입니다.3D 가로·깊이는 공간의 두 축이며 높이는 고정 스핀 단위−1…+1입니다. 입력마다 높이를 재정규화하지 않습니다. 값0은 기준면의 점입니다. 곡선 edge는 주기 연결이며 추가 사이트가 아닙니다.','The drawing approximates the current arrangement or mean. The3D horizontal and depth axes are spatial coordinates; height uses fixed spin units−1…+1 without per-input rescaling. Zero is a baseline point. Curved edges are periodic connections, not additional sites.')}</p>
  </figure>;
}
