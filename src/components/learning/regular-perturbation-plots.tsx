'use client';
import React, { useEffect, useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import type { RegularPerturbationView } from '@/lib/learning-regular-perturbation-view';
import { readLabPalette } from './lab-palette';
import type { PerturbationEnergyRenderer } from './regular-perturbation-energy-engine';

const scientific = (x: number) => x === 0 ? '0' : x.toExponential(3);
export default function RegularPerturbationPlots({ model, language }: { model: RegularPerturbationView; language: 'ko' | 'en' }) {
  const t = (ko: string, en: string) => language === 'ko' ? ko : en, id = useId();
  const frame = useRef<HTMLElement>(null), host = useRef<HTMLDivElement>(null), renderer = useRef<PerturbationEnergyRenderer>();
  const latest = useRef(model); latest.current = model;
  const [flat, setFlat] = useState(true), [visible, setVisible] = useState(false), [state, setState] = useState<'loading' | '3d' | '2d'>('loading'), [attempt, setAttempt] = useState(0);
  const label = t('에너지 곡면: 가로 변위, 깊이 결합, 높이 에너지. 화살표 키로 회전, Home으로 초기화', 'Energy surface: horizontal displacement, depth coupling, height energy. Arrow keys rotate; Home resets');
  useEffect(() => { if (!frame.current) return; if (!('IntersectionObserver' in window)) { setVisible(true); return; } const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: '160px' }); observer.observe(frame.current); return () => observer.disconnect(); }, []);
  useEffect(() => {
    if (!visible || flat) return;
    let cancelled = false; setState('loading');
    import('./regular-perturbation-energy-engine').then(({ createPerturbationEnergyPlot }) => {
      if (cancelled || !host.current || !frame.current) return;
      try { renderer.current = createPerturbationEnergyPlot(host.current, latest.current, readLabPalette(frame.current), label, () => { renderer.current?.dispose(); renderer.current = undefined; setState('2d'); }); setState('3d'); }
      catch { setState('2d'); }
    }).catch(() => { if (!cancelled) setState('2d'); });
    return () => { cancelled = true; renderer.current?.dispose(); renderer.current = undefined; };
  }, [visible, flat, attempt, label]);
  useEffect(() => { if (frame.current) renderer.current?.update(model, readLabPalette(frame.current)); }, [model]);
  useEffect(() => { const observer = new MutationObserver(() => { if (frame.current) renderer.current?.update(latest.current, readLabPalette(frame.current)); }); observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] }); return () => observer.disconnect(); }, []);
  const x = (value: number) => 60 + 500 * (value / model.width + 1) / 2;
  const fy = (value: number) => 150 - 105 * value / model.forceMax;
  const ey = (value: number) => 245 - 200 * (value - model.energyMin) / (model.energyMax - model.energyMin || 1);
  const line = (points: { x: number; force: number; energy: number }[], kind: 'force' | 'energy') => points.map(p => `${x(p.x)},${kind === 'force' ? fy(p.force) : ey(p.energy)}`).join(' ');
  const rootLeft = x(model.lower), rootRight = x(model.upper), fallback = flat || !visible || state !== '3d';
  const plotStyle = { width: '100%', minWidth: 620, height: 310 };
  const logY = (value: number | null) => value === null ? 268 : 245 - 200 * (value - model.logMin) / (model.logMax - model.logMin);
  const series = [{ key: 'term', color: 'var(--learning-blue)', name: t('보정항 절댓값', 'Absolute term') }, { key: 'residual', color: 'var(--learning-orange)', name: t('잔차 절댓값', 'Absolute residual') }, { key: 'bound', color: 'var(--learning-green)', name: t('오차 보증', 'Error certificate') }, { key: 'distance', color: 'var(--learning-neutral)', name: t('근 거리 상한', 'Root-distance upper bound') }] as const;
  return <>
    <figure aria-labelledby={`${id}-force`} data-perturbation-force><figcaption id={`${id}-force`}>{t('방정식의 영점과 근사점', 'Equation zero and approximation')}</figcaption>
      <div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('힘 그래프. 좌우 이동 가능', 'Force graph; scroll horizontally')}><svg viewBox="0 0 620 310" role="img" aria-label={t('단조 힘과 근 포함 구간, 선택한 부분합', 'Monotone force, certified root bracket and selected partial sum')} style={plotStyle}>
        <path d="M60 45V255H560M60 150H560" fill="none" stroke="currentColor" opacity=".4" />
        <rect x={rootLeft - 2} y={45} width={Math.max(4, rootRight - rootLeft)} height={210} fill="var(--learning-green)" opacity=".18" />
        <polyline points={line(model.samples, 'force')} fill="none" stroke="var(--learning-blue)" strokeWidth={3} />
        {model.partialInside && <><path d={`M${x(model.partial!)} 150V${fy(model.force(model.partial!))}`} stroke="var(--learning-orange)" strokeWidth={3} /><circle cx={x(model.partial!)} cy={fy(model.force(model.partial!))} r={6} fill="var(--learning-orange)" /></>}
        <text x={60} y={285} fontSize={18} fill="currentColor">{scientific(-model.width)}</text><text x={560} y={285} textAnchor="end" fontSize={18} fill="currentColor">{scientific(model.width)}</text>
        <text x={60} y={25} fontSize={18} fill="currentColor">F(x) · ±{scientific(model.forceMax)}</text><text x={580} y={155} fontSize={18} fill="currentColor">x</text>
      </svg></div>
      <p>{t('파랑은 F(x), 초록은 근 구간, 주황은 선택한 부분합과 잔차입니다. 한 픽셀보다 좁은 구간은 보이도록 확장했습니다. 곡선과 점은 근사 그림이며 정확한 부호와 끝점은 표에서 확인하세요.', 'Blue is F(x), green the root bracket, orange the selected partial sum and residual. Subpixel brackets are enlarged for visibility. Curves and markers are approximate; use the tables for exact signs and endpoints.')}</p>
      {!model.partialInside && <p role="status" data-perturbation-off-window>{t('선택한 부분합은 현재 표시 구간 밖입니다. 값을 잘라 표시하지 않으며 아래 정확한 값과전체 차수 그래프에서 확인할 수 있습니다.', 'The selected partial sum lies outside this plotting window. Its value is not clipped; inspect the exact value and full order plot below.')}</p>}
    </figure>
    <figure ref={frame} aria-labelledby={`${id}-energy`} data-perturbation-energy={fallback ? '2d' : '3d'}><figcaption id={`${id}-energy`}>{t('에너지와 결합의 변화', 'Energy as coupling changes')}</figcaption>
      <div ref={host} hidden={fallback} style={{ width: '100%', height: 360 }} />
      {fallback && <div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('에너지 그래프. 좌우 이동 가능', 'Energy graph; scroll horizontally')}><svg viewBox="0 0 620 310" role="img" aria-label={t('현재 결합의 볼록 에너지 곡선', 'Convex energy curve at the current coupling')} style={plotStyle}>
        <path d="M60 45V255H560" fill="none" stroke="currentColor" opacity=".4" /><polyline points={line(model.samples, 'energy')} fill="none" stroke="var(--learning-orange)" strokeWidth={3} />
        <circle cx={x(model.root)} cy={ey(model.rootEnergy)} r={6} fill="var(--learning-green)" />
        {model.partialInside && <circle cx={x(model.partial!)} cy={ey(model.energy(model.partial!))} r={6} fill="var(--learning-blue)" />}
        <text x={60} y={25} fontSize={18} fill="currentColor">V · [{scientific(model.energyMin)}, {scientific(model.energyMax)}]</text><text x={60} y={285} fontSize={18} fill="currentColor">{scientific(-model.width)}</text><text x={560} y={285} textAnchor="end" fontSize={18} fill="currentColor">{scientific(model.width)}</text>
      </svg></div>}
      <div className="qubit-actions"><Button variant="outline" onClick={() => setFlat(v => !v)}>{flat ? t('3D 보기', 'Show 3D') : t('2D 보기', 'Show 2D')}</Button>{!fallback && <><Button variant="outline" onClick={() => renderer.current?.view('perspective')}>{t('시점 초기화', 'Reset view')}</Button><Button variant="outline" onClick={() => renderer.current?.view('front')}>{t('정면', 'Front')}</Button></>}{!flat && state === '2d' && <Button variant="outline" onClick={() => setAttempt(n => n + 1)}>{t('3D 다시 시도', 'Retry 3D')}</Button>}</div>
      <p>{t('2D는 현재 λ의 에너지입니다. 3D의 가로축은 x, 깊이는 0부터 현재 λ, 높이는 같은 선형 척도의 실제 에너지 V입니다. 파랑은 λ=0, 주황은 현재 λ, 초록 점은 현재 평형입니다. 양의 λ에서도 항상 볼록하며, 이는 Taylor 급수 수렴과 별개입니다.', '2D shows energy at the current λ. In 3D, horizontal position is x, depth ranges from 0 to the current λ, and height is actual V on one linear scale. Blue is λ=0, orange the current λ, and the green point the current equilibrium. Convexity at positive λ is separate from Taylor-series convergence.')}</p>
      <p className="ai-small">x ∈ [{scientific(-model.width)}, {scientific(model.width)}]; λ ∈ [0, {scientific(model.lambda)}]; V ∈ [{scientific(model.energyMin)}, {scientific(model.energyMax)}]</p>
    </figure>
    <figure aria-labelledby={`${id}-order`} data-perturbation-order><figcaption id={`${id}-order`}>{t('모든 계산 차수의 크기 비교', 'Magnitudes across every calculated order')}</figcaption>
      <div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('로그 차수 그래프. 좌우 이동 가능', 'Logarithmic order graph; scroll horizontally')}><svg viewBox="0 0 620 310" role="img" aria-label={t('항·잔차·오차 보증·근 거리 상한의 상용로그', 'Base-ten logarithms of term, residual, certificate and root-distance upper bound')} style={plotStyle}>
        <path d="M60 45V255H560" fill="none" stroke="currentColor" opacity=".4" />
        <text x={60} y={25} fontSize={18} fill="currentColor">log₁₀ |value| · [{model.logMin}, {model.logMax}]</text><text x={60} y={300} fontSize={18} fill="currentColor">{t('0: 빈 기호', 'Zero: hollow marker')}</text><text x={560} y={300} textAnchor="end" fontSize={18} fill="currentColor">n=0…{model.orderRows.length - 1}</text>
        {series.map(series => <g key={series.key}><title>{series.name}</title>{model.orderRows.map((row, i) => { const px = 60 + 500 * i / Math.max(1, model.orderRows.length - 1), value = row[series.key], previous = model.orderRows[i - 1]; return <g key={i}>{i > 0 && value !== null && previous[series.key] !== null && <path d={`M${60 + 500 * (i - 1) / Math.max(1, model.orderRows.length - 1)} ${logY(previous[series.key])}L${px} ${logY(value)}`} stroke={series.color} strokeWidth={2} />}<circle cx={px} cy={logY(value)} r={i === model.chosen.order ? 5 : 3} fill={value === null ? 'hsl(var(--background))' : series.color} stroke={series.color} strokeWidth={2} /></g>; })}</g>)}
      </svg></div>
      <p>{series.map(s => <span key={s.key} style={{ color: s.color, marginInlineEnd: 16 }}>● {s.name} </span>)}</p>
      <p>{t('비영 값은 절댓값의 log₁₀를 사용해 아주 작거나 큰 유리수도 표시합니다. 정확한 0만 아래 빈 기호로 분리하며 0의 로그값이 아닙니다. 부호와 모든 자릿수는 표에 보존합니다. 차수가 높을수록 오차가 항상 작아지는 것은 아닙니다.', 'Nonzero values use log₁₀ magnitudes so very small or large fractions remain visible. Exact zeros alone use hollow markers below the axis; these are not logarithms of zero. Signs and all digits remain in the tables. Higher order does not always mean smaller error.')}</p>
    </figure>
  </>;
}
