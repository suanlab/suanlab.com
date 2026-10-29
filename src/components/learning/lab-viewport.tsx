'use client';
import { Button } from '@/components/ui/button';
import { useEffect, useRef, useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { formatNumber } from '@/lib/learning-simulations';
import type { LabScene } from './lab-model';
import type { LabRenderer } from './scene-engine';
import { readLabPalette } from './lab-palette';

function Projection({ scene, label }: { scene: LabScene; label: string }) {
  const planar = scene.kind === 'inner-products' || scene.kind === 'eigenvalues';
  const project = (p: number[]) => [190 + (p[0] - p[2] * .55) * 29, 175 - (p[1] - p[2] * .3) * 29];
  return <svg viewBox="0 0 380 300" role="img" aria-label={label} className="ai-lab-projection">
    <defs><marker id={`arrow-${scene.kind}`} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="context-stroke" /></marker></defs>
    {Array.from({ length: 9 }, (_, i) => i - 4).map(i => <path key={i} d={planar ? `M${project([-4, i, 0])}L${project([4, i, 0])}M${project([i, -4, 0])}L${project([i, 4, 0])}` : `M${project([-4, 0, i])}L${project([4, 0, i])}M${project([i, 0, -4])}L${project([i, 0, 4])}`} stroke="var(--learning-grid)" strokeOpacity=".35" fill="none" />)}
    {([[4.5, 0, 0], [0, 4.5, 0], [0, 0, 4.5]]).map((p, i) => <g key={i}><path d={`M${project([0, 0, 0])}L${project(p)}`} stroke="var(--learning-neutral)" /><text x={project(p)[0]} y={project(p)[1] - 5} fill="hsl(var(--foreground))" fontSize="12">{(scene.axisLabels || ['x', scene.kind === 'gradients' ? 'L' : 'y', scene.kind === 'gradients' ? 'y' : 'z'])[i]}</text></g>)}
    {scene.surface && Array.from({ length: 25 }, (_, i) => <polyline key={i} points={Array.from({ length: 25 }, (_, j) => project(scene.surface!.vertices.slice((i * 25 + j) * 3, (i * 25 + j) * 3 + 3)).join(',')).join(' ')} stroke="var(--learning-green)" fill="none" strokeWidth=".6" />)}
    {scene.lines.map((l, i) => <polyline key={i} points={l.points.map(p => project(p).join(',')).join(' ')} fill="none" stroke={`var(--learning-${l.color})`} strokeWidth="2" strokeDasharray={l.dashed ? '5 4' : undefined} />)}
    {scene.arrows.map((a, i) => <path key={i} d={`M${project(a.start)}L${project(a.end)}`} stroke={`var(--learning-${a.color})`} strokeWidth="2.5" strokeDasharray={a.dashed ? '5 4' : undefined} markerEnd={`url(#arrow-${scene.kind})`} />)}
    {scene.dots.map((d, i) => <circle key={i} cx={project(d.position)[0]} cy={project(d.position)[1]} r="5" fill={`var(--learning-${d.color})`} />)}
  </svg>;
}
export default function LabViewport({ scene }: { scene: LabScene }) {
  const { language } = useLanguage(); const ko = language === 'ko';
  const frame = useRef<HTMLDivElement>(null); const host = useRef<HTMLDivElement>(null); const renderer = useRef<LabRenderer>(); const latest = useRef(scene); latest.current = scene;
  const [visible, setVisible] = useState(false); const [mode, setMode] = useState<'loading' | '3d' | '2d'>('loading'); const [attempt, setAttempt] = useState(0);
  const [flat, setFlat] = useState(false);
  useEffect(() => {
    if (!frame.current) return;
    if (!('IntersectionObserver' in window)) { setVisible(true); return; }
    const observer = new IntersectionObserver(([entry]) => { if (entry.isIntersecting) { setVisible(true); observer.disconnect(); } }, { rootMargin: '160px' });
    observer.observe(frame.current); return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible || flat) return;
    let cancelled = false;
    setMode('loading');
    import('./scene-engine').then(({ createLabRenderer }) => {
      if (cancelled || !host.current) return;
      try {
        renderer.current = createLabRenderer(host.current, latest.current, readLabPalette(frame.current!), () => {
          renderer.current?.dispose(); renderer.current = undefined; setMode('2d');
        });
        setMode('3d');
      } catch { setMode('2d'); }
    }).catch(() => { if (!cancelled) setMode('2d'); });
    return () => { cancelled = true; renderer.current?.dispose(); renderer.current = undefined; };
  }, [visible, flat, attempt]);
  useEffect(() => {
    if (frame.current) renderer.current?.update(scene, readLabPalette(frame.current));
  }, [scene]);
  useEffect(() => {
    // Update materials in place so a theme change preserves the camera and model.
    const observer = new MutationObserver(() => {
      if (frame.current) renderer.current?.update(latest.current, readLabPalette(frame.current));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);
  const title = scene.kind === 'vectors-and-forces' ? (ko ? '힘과 운동의 공간 그림' : 'Spatial view of forces and motion')
    : scene.kind === 'linear-transforms' ? (ko ? '기저와 정육면체의 변환' : 'Transformed basis and unit cube') : scene.kind === 'inner-products' ? (ko ? '두 벡터의 내적과 직교투영' : 'Dot product and orthogonal projection') : scene.kind === 'least-squares' ? (ko ? '세 관측의 응답 공간에서 예측과 잔차' : 'Prediction and residual in the response space of three observations') : scene.kind === 'eigenvalues' ? (ko ? '대칭 행렬의 고유방향과 공분산' : 'Eigendirections and covariance of a symmetric matrix') : (ko ? '손실 곡면과 경사하강 경로' : 'Loss surface and gradient descent path');
  const fallback = flat || mode !== '3d';
  return <div className="ai-viewport-block">
    <div ref={frame} className="ai-viewport" data-renderer={fallback ? '2d' : '3d'}>
      <div ref={host} role="img" aria-label={title} className="ai-canvas-host" hidden={fallback} />
      {fallback && <Projection scene={scene} label={title} />}
      <span className="ai-view-badge">{fallback ? '2D' : '3D'} · {scene.axisLabels ? scene.axisLabels.join(' / ') : scene.kind === 'gradients' ? 'x / L / y' : 'x / y / z'}</span>
    </div>
    <div className="ai-view-controls">
      <Button variant="outline" size="sm" type="button" onClick={() => { setFlat(!flat); if (flat) setAttempt(n => n + 1); }}>{flat ? (ko ? '3D 보기' : 'Show 3D') : (ko ? '2D 보기' : 'Show 2D')}</Button>
      {!fallback && <><Button variant="outline" size="sm" type="button" onClick={() => renderer.current?.view('perspective')}>{ko ? '시점 초기화' : 'Reset view'}</Button><Button variant="outline" size="sm" type="button" onClick={() => renderer.current?.view('front')}>{ko ? '정면' : 'Front'}</Button><Button variant="outline" size="sm" type="button" onClick={() => renderer.current?.view('top')}>{ko ? '위에서' : 'Top'}</Button></>}
      {!flat && mode === '2d' && <Button variant="outline" size="sm" type="button" onClick={() => setAttempt(n => n + 1)}>{ko ? '3D 다시 시도' : 'Retry 3D'}</Button>}
    </div>
    <p className="ai-small">{!flat && mode === '2d' ? (ko ? '3D를 사용할 수 없어 같은 데이터의 2D 그림을 표시합니다.' : '3D is unavailable. This 2D projection uses the same data.') : fallback ? (ko ? '같은 데이터의 2D 투영도입니다.' : 'A 2D projection of the same data.') : (ko ? '드래그하여 회전 · 버튼으로 시점 변경' : 'Drag to rotate · Use buttons to change the view')}
      {' · '}{ko ? '격자 1칸' : '1 grid interval'} = {formatNumber(scene.gridUnit)} {scene.kind === 'vectors-and-forces' ? '' : (ko ? '좌표 단위' : 'coordinate units')}
      {scene.kind === 'gradients' && <>. {ko ? '높이 = L ×' : 'Height = L ×'} {scene.heightScale.toPrecision(3)}. {ko ? '높이와 좌표의 시각적 축척은 다릅니다.' : 'Height and coordinates use different visual scales.'}</>}
    </p>
  </div>;
}
