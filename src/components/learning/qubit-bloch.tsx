'use client';
import { useEffect, useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { readLabPalette } from './lab-palette';
import type { BlochRenderer } from './hilbert-bloch-engine';

export default function QubitBloch({ vector, language, scope, description }: { description?: string; vector: number[]; language: 'ko' | 'en'; scope: string }) {
  const t = (ko: string, en: string) => language === 'ko' ? ko : en;
  const id = useId(), frame = useRef<HTMLElement>(null), host = useRef<HTMLDivElement>(null);
  const renderer = useRef<BlochRenderer>(), latest = useRef(vector); latest.current = vector;
  const [visible, setVisible] = useState(false), [flat, setFlat] = useState(false);
  const [mode, setMode] = useState<'loading' | '3d' | '2d'>('loading'), [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!frame.current) return;
    if (!('IntersectionObserver' in window)) { setVisible(true); return; }
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: '160px' });
    observer.observe(frame.current); return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!visible || flat) return;
    let cancelled = false; setMode('loading');
    import('./hilbert-bloch-engine').then(({ createBlochRenderer }) => {
      if (cancelled || !host.current || !frame.current) return;
      try {
        renderer.current = createBlochRenderer(host.current, latest.current, readLabPalette(frame.current), () => {
          renderer.current?.dispose(); renderer.current = undefined; setMode('2d');
        });
        setMode('3d');
      } catch { setMode('2d'); }
    }).catch(() => { if (!cancelled) setMode('2d'); });
    return () => { cancelled = true; renderer.current?.dispose(); renderer.current = undefined; setMode('2d'); };
  }, [visible, flat, attempt]);
  useEffect(() => { if (frame.current) renderer.current?.update(vector, readLabPalette(frame.current)); }, [vector]);
  useEffect(() => {
    const observer = new MutationObserver(() => { if (frame.current) renderer.current?.update(latest.current, readLabPalette(frame.current)); });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);
  const fallback = flat || !visible || mode !== '3d';
  return <figure ref={frame} aria-labelledby={id} data-qubit-bloch={fallback ? '2d' : '3d'}>
    <figcaption id={id}>{scope} · {t('단일 큐비트 Bloch ball', 'Single-qubit Bloch ball')}</figcaption>
    <div ref={host} hidden={fallback} role="img" aria-label={`${scope} · Bloch 3D`} style={{ width: '100%', height: 300 }} />
    {fallback && <div className="qubit-scroll" role="region" tabIndex={0} aria-label={`${scope} · ${t('Bloch 투영 그림 · 좌우로 이동 가능', 'Bloch projections; scroll horizontally')}`}><svg viewBox="0 0 560 280" role="img" aria-label={`${scope} · ${t('XY·XZ 투영', 'XY and XZ projections')}`} style={{ width: '100%', height: 'auto' }}>
      {[[0, 1, 'X / Y'], [0, 2, 'X / Z']].map(([a, b, label], i) => <g key={i}>
        <circle cx={140 + i * 280} cy={135} r={90} stroke="var(--learning-neutral)" fill="none" />
        <path d={`M${40 + i * 280} 135H${240 + i * 280}M${140 + i * 280} 35V235`} stroke="var(--learning-neutral)" />
        <path d={`M${140 + i * 280} 135L${140 + i * 280 + 90 * vector[Number(a)]} ${135 - 90 * vector[Number(b)]}`} stroke="var(--learning-blue)" strokeWidth={3} />
        <circle cx={140 + i * 280 + 90 * vector[Number(a)]} cy={135 - 90 * vector[Number(b)]} r={5} fill="var(--learning-blue)" />
        <text x={140 + i * 280} y={265} fontSize={24} textAnchor="middle" fill="currentColor">{label}</text>
      </g>)}
    </svg></div>}
    <div className="qubit-actions"><Button variant="outline" onClick={() => setFlat(value => !value)}>{flat ? t('3D 보기', 'Show 3D') : t('2D 보기', 'Show 2D')}</Button>
      {!fallback && (['perspective', 'front', 'top'] as const).map((view, i) => <Button key={view} variant="outline" onClick={() => renderer.current?.view(view)}>{[t('시점 초기화', 'Reset view'), t('정면', 'Front'), t('위에서', 'Top')][i]}</Button>)}
      {!flat && mode === '2d' && <Button variant="outline" onClick={() => setAttempt(n => n + 1)}>{t('3D 다시 시도', 'Retry 3D')}</Button>}
    </div>
    <p>{description ?? t('입자 위치가 아닌 상태 좌표입니다. 길이를1로 보정하지 않습니다. 영벡터는 단일 큐비트의 최대 혼합 상태이며, 결합 상태 전체의 정보는 밀도행렬과 상관 표에서 확인하세요. 카메라는 자동 회전하지 않고 상태 변화는 즉시 반영됩니다. 3D가 불가능하면 같은 좌표의2D 투영을 사용합니다.', 'These are state coordinates, not particle positions. The length is not normalized to one. A zero vector represents a maximally mixed single qubit; inspect the joint density matrix and correlations for the full state. The camera does not rotate automatically and state updates are immediate. If 3D is unavailable, the same coordinates remain available in 2D.')}</p>
  </figure>;
}
