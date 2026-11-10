'use client';

import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { quantumRitzGeometry, type RitzGeometry, type RitzResidualComponent } from '@/lib/learning-quantum-variational-geometry';
import type { QuantumRitzResult } from './quantum-variational-view';
import { MasterButton as Button } from './master-equation-view';
import { readLabPalette } from './lab-palette';
import type { RitzSpaceRenderer } from './quantum-variational-space-engine';

export default function QuantumVariationalSpace({ result, component, language, scope, energyScale }: {
  result: QuantumRitzResult; component: RitzResidualComponent; language: 'ko' | 'en'; scope: string; energyScale: number;
}) {
  const t = (ko: string, en: string) => language === 'ko' ? ko : en;
  const geometry = useMemo(() => quantumRitzGeometry(result, component, energyScale), [result, component, energyScale]);
  if (!geometry) return <p data-qv-space="unavailable">{t(
    '이 공간 그림은 모든 좌표가 실수인 3차원 사례에만 적용됩니다. 복소수·다른 차원·해결되지 않은 좌표는 위의 전체 성분 그림과 행렬에서 확인하세요.',
    'This space diagram applies only to real three-dimensional cases. For complex cases, other dimensions or unresolved coordinates, use the complete component diagrams and matrices above.',
  )}</p>;
  return <Spatial geometry={geometry} language={language} scope={scope} />;
}

function Spatial({ geometry, language, scope }: { geometry: RitzGeometry; language: 'ko' | 'en'; scope: string }) {
  const t = (ko: string, en: string) => language === 'ko' ? ko : en;
  const [projection, setProjection] = useState<'xy' | 'xz'>('xy');
  const [visible, setVisible] = useState(false), [flat, setFlat] = useState(false);
  const [mode, setMode] = useState<'loading' | '3d' | '2d'>('loading'), [attempt, setAttempt] = useState(0);
  const frame = useRef<HTMLElement>(null), host = useRef<HTMLDivElement>(null), renderer = useRef<RitzSpaceRenderer>();
  const latest = useRef(geometry); latest.current = geometry;
  const id = useId();
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
    import('./quantum-variational-space-engine').then(({ createRitzSpaceRenderer }) => {
      if (cancelled || !host.current || !frame.current) return;
      try {
        renderer.current = createRitzSpaceRenderer(host.current, latest.current, readLabPalette(frame.current), () => {
          renderer.current?.dispose(); renderer.current = undefined;
          if (!cancelled) setMode('2d');
        });
        setMode('3d');
      } catch { if (!cancelled) setMode('2d'); }
    }).catch(() => { if (!cancelled) setMode('2d'); });
    return () => { cancelled = true; renderer.current?.dispose(); renderer.current = undefined; };
  }, [visible, flat, attempt]);
  useEffect(() => { if (frame.current) renderer.current?.update(geometry, readLabPalette(frame.current)); }, [geometry]);
  useEffect(() => {
    const observer = new MutationObserver(() => { if (frame.current) renderer.current?.update(latest.current, readLabPalette(frame.current)); });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);
  const fallback = flat || mode !== '3d';
  const title = `${scope} · ${t('시험 공간과 선택 잔차', 'Trial space and selected residual')}`;
  const second = projection === 'xy' ? 1 : 2;
  const extent = Math.max(1.2, ...geometry.residual.map(Math.abs));
  const scale = 105 / extent;
  const point = (v: number[]) => `${160 + scale * v[0]},${145 - scale * v[second]}`;
  const patch = geometry.axes.length === 2 ? Array.from({ length: 65 }, (_, i) => {
    const a = 2 * Math.PI * i / 64;
    return point(geometry.axes[0].map((v, j) => v * Math.cos(a) + geometry.axes[1][j] * Math.sin(a)));
  }).join(' ') : null;
  return <figure ref={frame} aria-labelledby={id} data-qv-space={fallback ? projection : '3d'} data-qv-space-status={flat ? 'flat' : mode} style={{ margin: '1.5rem 0' }}>
    <figcaption id={id}>{title}</figcaption>
    <div ref={host} hidden={fallback} role="img" aria-label={`${title} · 3D`} style={{ width: '100%', height: 340 }} />
    {fallback && <svg viewBox="0 0 320 300" role="img" aria-label={`${title} · ${projection}`} style={{ width: '100%', maxWidth: 600 }}>
      <title>{title}</title>
      <desc>{t('파랑은 단위 상태, 주황은 공통 에너지 눈금으로 나눈 잔차, 초록은 시험 공간의 단위 노름 절단입니다.', 'Blue: unit state. Orange: residual divided by a common energy scale. Green: unit-norm section of the trial space.')}</desc>
      <path d="M25 145H295M160 15V275" stroke="var(--learning-neutral)" />
      <circle cx={160} cy={145} r={scale} fill="none" stroke="var(--learning-neutral)" strokeDasharray="4 4" />
      {patch && <polyline points={patch} fill="var(--learning-green)" fillOpacity={0.15} stroke="var(--learning-green)" />}
      {geometry.columns === 1 && <line x1={160 - scale * geometry.axes[0][0]} y1={145 + scale * geometry.axes[0][second]} x2={160 + scale * geometry.axes[0][0]} y2={145 - scale * geometry.axes[0][second]} stroke="var(--learning-green)" strokeWidth={5} />}
      {geometry.columns === 3 && <circle cx={160} cy={145} r={scale} fill="var(--learning-green)" fillOpacity={0.12} stroke="var(--learning-green)" />}
      {[geometry.state, geometry.residual].map((v, i) => <g key={i}>
        <polyline points={`160,145 ${point(v)}`} fill="none" stroke={i ? 'var(--learning-orange)' : 'var(--learning-blue)'} strokeWidth={3} />
        <circle cx={160 + scale * v[0]} cy={145 - scale * v[second]} r={4} fill={i ? 'var(--learning-orange)' : 'var(--learning-blue)'} />
      </g>)}
      <text x={285} y={165} fill="currentColor">e₀</text>
      <text x={170} y={25} fill="currentColor">{second === 1 ? 'e₁' : 'e₂'}</text>
    </svg>}
    <div className="ai-lab-actions" style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
      <Button variant="outline" onClick={() => setFlat(value => !value)}>{flat ? t('3D 보기', 'Show 3D') : t('2D 보기', 'Show 2D')}</Button>
      {!fallback && (['perspective', 'front', 'top'] as const).map((view, i) => <Button key={view} variant="outline" onClick={() => renderer.current?.view(view)}>{[t('시점 초기화', 'Reset view'), t('정면', 'Front'), t('위에서', 'Top')][i]}</Button>)}
      {!flat && mode === '2d' && <Button variant="outline" onClick={() => setAttempt(value => value + 1)}>{t('3D 다시 시도', 'Retry 3D')}</Button>}
    </div>
    {mode === '2d' && !flat && <p role="status">{t('3D를 표시할 수 없어 같은 좌표의 2D 투영을 표시합니다.', '3D is unavailable; the same coordinates are shown as 2D projections.')}</p>}
    {fallback &&
    <div className="ai-lab-actions" role="group" aria-label={t('투영 평면', 'Projection plane')}>
      {(['xy', 'xz'] as const).map(v => <Button key={v} variant="outline" aria-pressed={projection === v} onClick={() => setProjection(v)}>{v === 'xy' ? 'e₀ / e₁' : 'e₀ / e₂'}</Button>)}
    </div>}
    <p className="ai-small">{t('파랑은 단위 상태, 주황은 r/(√N E*)입니다. 드래그와 시점 버튼은 카메라만 움직입니다. 자동 회전이나 물리적 시간 진화는 없습니다. 키보드로는 시점 버튼 또는 두 2D 투영을 이용하세요.', 'Blue is the unit state; orange is r/(√N E*). Drag and view buttons move only the camera. There is no automatic rotation or physical time evolution. For keyboard access, use the view buttons or the two 2D projections.')}</p>
    <p className="ai-small">{fallback ? t(
      '같은 상태를 서로 다른 평면에 투영합니다. 투영에서 화살표가 사라져도 실제 벡터가 0이라는 뜻은 아닙니다. 초록 영역은 무한한 선형 공간 자체가 아니라 단위 노름 절단이며, 점선 원은 표시 눈금입니다. 상태나 물리적 시간은 바뀌지 않습니다.',
      'These are projections of the same state onto different planes. An invisible projected arrow does not imply a zero vector. Green shows a unit-norm section, not the entire unbounded linear space; the dashed circle is a scale reference. Neither the state nor physical time changes.',
    ) : t(
      '초록은 무한한 시험 공간의 단위 노름 절단입니다. 선은 양 끝을 잘라 표시하고 평면은 원판으로, 전체 공간은 단위 구의 초록 그물로 표시합니다. 회색 구는 노름 1의 기준이며 공간의 경계가 아닙니다.',
      'Green marks a unit-norm section of the unbounded trial space: a bounded line segment, a disk for a plane, or a green unit-sphere wireframe for the full space. The grey sphere references norm one; it is not a boundary of the space.',
    )}</p>
  </figure>;
}
