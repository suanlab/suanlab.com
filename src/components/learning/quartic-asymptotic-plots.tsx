'use client';
import React, { useEffect, useId, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import type { QuarticAsymptoticView } from '@/lib/learning-quartic-asymptotic-view';
import { readLabPalette } from './lab-palette';
import type { QuarticIntegrandRenderer } from './quartic-asymptotic-integrand-engine';
export default function QuarticAsymptoticPlots({ model: m, language }: { model: QuarticAsymptoticView; language: 'ko' | 'en' }) {
  const t = (ko: string, en: string) => language === 'ko' ? ko : en, id = useId(), frame = useRef<HTMLElement>(null), host = useRef<HTMLDivElement>(null), renderer = useRef<QuarticIntegrandRenderer>();
  const latest = useRef(m); latest.current = m;
  const [flat, setFlat] = useState(true), [visible, setVisible] = useState(false), [state, setState] = useState<'loading' | '2d' | '3d'>('loading'), [attempt, setAttempt] = useState(0), [logIntegrand, setLogIntegrand] = useState(false);
  const label = t('사차 적분함수 곡면: 가로 x, 깊이 결합, 높이 지수 인자. 화살표 키로 회전, Home으로 초기화', 'Quartic kernel surface: horizontal x, depth coupling, height exponential factor. Arrow keys rotate; Home resets');
  useEffect(() => { if (!frame.current) return; if (!('IntersectionObserver' in window)) { setVisible(true); return; } const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: '160px' }); observer.observe(frame.current); return () => observer.disconnect(); }, []);
  useEffect(() => {
    if (!visible || flat) return;
    let cancelled = false; setState('loading');
    import('./quartic-asymptotic-integrand-engine').then(({ createQuarticIntegrandPlot }) => {
      if (cancelled || !host.current || !frame.current) return;
      try { renderer.current = createQuarticIntegrandPlot(host.current, latest.current, readLabPalette(frame.current), label, () => { renderer.current?.dispose(); renderer.current = undefined; setState('2d'); }); setState('3d'); }
      catch { setState('2d'); }
    }).catch(() => { if (!cancelled) setState('2d'); });
    return () => { cancelled = true; renderer.current?.dispose(); renderer.current = undefined; };
  }, [visible, flat, attempt, label]);
  useEffect(() => { if (frame.current) renderer.current?.update(m, readLabPalette(frame.current)); }, [m]);
  useEffect(() => { const observer = new MutationObserver(() => { if (frame.current) renderer.current?.update(latest.current, readLabPalette(frame.current)); }); observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] }); return () => observer.disconnect(); }, []);
  const fallback = flat || !visible || state !== '3d', style = { width: '100%', minWidth: 640, height: 320 };
  const x = (value: number) => 70 + 500 * (value / m.width + 1) / 2, logFloor = Math.min(...m.samples.map(p => p.logValue));
  const iy = (value: number, log: number) => logIntegrand ? 50 - 195 * log / (-logFloor || 1) : 245 - 195 * value;
  const nx = (n: number) => 70 + 500 * n / Math.max(1, m.rows.length - 1), py = (value: number) => 245 - 195 * (m.partialTransform(value) - m.yMin) / (m.yMax - m.yMin);
  const ly = (value: number | null) => value === null ? 270 : 245 - 195 * (value - m.logMin) / (m.logMax - m.logMin);
  const inWindow = (value: number) => m.partialScale === 'all' || value >= 0 && value <= 1.05;
  const series = [ { key: 'term', color: 'var(--learning-blue)', name: t('항 절댓값', 'Absolute term') }, { key: 'bound', color: 'var(--learning-orange)', name: t('첫 누락 항 상계', 'First-omitted bound') }, { key: 'distance', color: 'var(--learning-neutral)', name: t('순수 적분 거리 상한', 'Raw quadrature distance upper') }, { key: 'error', color: 'var(--learning-green)', name: t('결합한 오차 상한', 'Combined error upper') } ] as const;
  return <>
    <figure ref={frame} data-quartic-integrand={fallback ? '2d' : '3d'} aria-labelledby={`${id}-integrand`}><figcaption id={`${id}-integrand`}>{t('큰 요동을 억제하는 사차 결합', 'Quartic coupling suppresses large fluctuations')}</figcaption>
      <div ref={host} hidden={fallback} style={{ width: '100%', height: 360 }} />
      {fallback && <><label className="scalar-check"><input type="checkbox" aria-label={t('적분함수의 로그 높이', 'Log integrand height')} checked={logIntegrand} onChange={e => setLogIntegrand(e.target.checked)} />{t('적분함수의 로그 높이', 'Log integrand height')}</label><div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('적분함수 그림. 좌우 이동 가능', 'Kernel plot; scroll horizontally')}><svg viewBox="0 0 640 320" role="img" aria-label={t('같은 높이 척도의 가우스 기준과 현재 적분함수', 'Gaussian and current kernels on one height scale')} style={style}>
        <path d="M70 50V250H570" fill="none" stroke="currentColor" opacity=".4" />
        <polyline points={m.samples.map(p => `${x(p.x)},${iy(p.gaussian, -p.x * p.x / (2 * Math.LN10))}`).join(' ')} fill="none" stroke="var(--learning-blue)" strokeWidth={3} />
        <polyline points={m.samples.map(p => `${x(p.x)},${iy(p.value, p.logValue)}`).join(' ')} fill="none" stroke="var(--learning-orange)" strokeWidth={3} />
        <text x={70} y={28} fontSize={18} fill="currentColor">{logIntegrand ? `log₁₀ kernel: [${logFloor.toFixed(2)},0]` : 'kernel: [0,1]'}</text><text x={70} y={285} fontSize={18} fill="currentColor">−{m.width}</text><text x={570} y={285} textAnchor="end" fontSize={18} fill="currentColor">+{m.width}</text>
      </svg></div></>}
      <div className="qubit-actions"><Button variant="outline" onClick={() => setFlat(v => !v)}>{flat ? t('3D 보기', 'Show 3D') : t('2D 보기', 'Show 2D')}</Button>{!fallback && <><Button variant="outline" onClick={() => renderer.current?.view('perspective')}>{t('시점 초기화', 'Reset view')}</Button><Button variant="outline" onClick={() => renderer.current?.view('front')}>{t('정면', 'Front')}</Button></>}{!flat && state === '2d' && <Button variant="outline" onClick={() => setAttempt(n => n + 1)}>{t('3D 다시 시도', 'Retry 3D')}</Button>}</div>
      <p>{t('파랑은 g=0, 주황은 현재 g의 e^(−x²/2−gx⁴/4)입니다. 확률밀도로 정규화하지 않은 적분 인자이며 이 그림의 면적을 그대로 Z라고 읽지 않습니다. 3D는 x·0부터 현재 g까지의 결합·선형 높이입니다. 시간 진화가 아닙니다.', 'Blue is g=0; orange is the current e^(−x²/2−gx⁴/4). These are unnormalized kernels, so the plotted area is not directly Z. The 3D axes are x, coupling from 0 to the current g, and linear height. This is not time evolution.')}</p>
      {m.coupling === 0 && <p data-quartic-zero-coupling>{t('g=0에서는 결합 축의 길이가0이므로 모든 단면을 한 가우스 곡선으로 겹칩니다. 깊이를 가진 별도의 곡면이 아닙니다.', 'At g=0 the coupling axis has zero extent, so all sections coincide as one Gaussian curve rather than a surface with artificial depth.')}</p>}
      <p className="ai-small">{t('그림의 범위 밖과 작은 꼬리는 정확한 적분 구간·꼬리 보증에 포함됩니다. 그려진0은 부동소수점 해상도 때문일 수 있습니다. 2D 로그 높이는 작은 꼬리를 직접 표시합니다.', 'The exact integral interval and tail certificate include the region beyond this drawing. A drawn zero may reflect floating resolution. The 2D log height shows small tails directly.')}</p>
    </figure>
    <figure data-quartic-partials aria-labelledby={`${id}-partials`}><figcaption id={`${id}-partials`}>{t('부분합과 적분 포함 구간', 'Partial sums and the integral enclosure')}</figcaption>
      <div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('부분합 그림. 좌우 이동 가능', 'Partial-sum plot; scroll horizontally')}><svg viewBox="0 0 640 320" role="img" aria-label={t('유지 차수별 부분합과 별도 적분 구간', 'Partial sums by retained order and the separate quadrature interval')} style={style}>
        <path d="M70 50V250H570" fill="none" stroke="currentColor" opacity=".4" />
        <rect x={70} y={py(m.upper) - 2} width={500} height={Math.max(4, py(m.lower) - py(m.upper))} fill="var(--learning-green)" opacity=".22" />
        {m.rows.map((row, i) => <g key={i}>{i > 0 && inWindow(row.partial) && inWindow(m.rows[i - 1].partial) && <path d={`M${nx(i - 1)} ${py(m.rows[i - 1].partial)}L${nx(i)} ${py(row.partial)}`} stroke="var(--learning-blue)" strokeWidth={2} />}{inWindow(row.partial) && <circle cx={nx(i)} cy={py(row.partial)} r={i === m.chosen.order ? 6 : 3} fill={row.order % 2 ? 'var(--learning-orange)' : 'var(--learning-blue)'} />}</g>)}
        <text x={70} y={28} fontSize={18} fill="currentColor">{m.partialScale === 'physical' ? 'S_N: [0,1.05]' : `sign(S) log₁₀(1+|S|): [${m.yMin.toFixed(2)},${m.yMax.toFixed(2)}]`}</text><text x={70} y={290} fontSize={18} fill="currentColor">N=0…{m.rows.length - 1}</text>
      </svg></div>
      <p>{t('짝수 부분합은 파랑, 홀수는 주황, 초록 띠는 순수 적분 구간입니다. 한 픽셀보다 좁은 띠는 보이도록 확장했습니다. 값의 판정은 아래 정확한 표를 사용하세요.', 'Even partial sums are blue, odd ones orange, and the green band is raw quadrature. Subpixel bands are enlarged for visibility. Use exact tables for numerical decisions.')}</p>
      {m.partialScale === 'physical' && m.outsidePhysicalWindow > 0 && <p data-quartic-off-window role="status">{t(`전체 ${m.rows.length}개 중 ${m.outsidePhysicalWindow}개 부분합은 이 표시 범위 밖입니다. 전체 값 척도에서 모두 비교할 수 있습니다.`, `${m.outsidePhysicalWindow} of ${m.rows.length} partial sums lie outside this window. Switch to the full-value scale to compare all of them.`)}</p>}
    </figure>
    <figure data-quartic-errors aria-labelledby={`${id}-errors`}><figcaption id={`${id}-errors`}>{t('차수별 항·오차 보증·거리', 'Terms, certificates and distances by order')}</figcaption>
      <div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('로그 오차 그림. 좌우 이동 가능', 'Log error plot; scroll horizontally')}><svg viewBox="0 0 640 320" role="img" aria-label={t('항·누락 항·수치 거리·결합한 오차 상한의 로그 크기', 'Log magnitudes of terms, omitted bounds, quadrature distance and combined error upper bounds')} style={style}>
        <path d="M70 50V250H570" fill="none" stroke="currentColor" opacity=".4" /><text x={70} y={28} fontSize={18} fill="currentColor">log₁₀ |value|: [{m.logMin},{m.logMax}]</text>
        {series.map(series => <g key={series.key}><title>{series.name}</title>{m.rows.map((row, i) => { const value = row[series.key], previous = m.rows[i - 1]; return <g key={i}>{i > 0 && value !== null && previous[series.key] !== null && <path d={`M${nx(i - 1)} ${ly(previous[series.key])}L${nx(i)} ${ly(value)}`} stroke={series.color} strokeWidth={2} />}<circle cx={nx(i)} cy={ly(value)} r={i === m.chosen.order ? 5 : 3} fill={value === null ? 'hsl(var(--background))' : series.color} stroke={series.color} strokeWidth={2} /></g>; })}</g>)}
        <text x={70} y={310} fontSize={18} fill="currentColor">{t('0: 아래 빈 기호', 'Zero: hollow marker below')}</text><text x={570} y={310} textAnchor="end" fontSize={18} fill="currentColor">N</text>
      </svg></div>
      <p>{series.map(s => <span key={s.key} style={{ color: s.color, marginInlineEnd: 12 }}>● {s.name} </span>)}</p><p>{t('로그는 비영 값의 절댓값에만 적용합니다. 정확한0은 별도 표시하며 작은 양수를0으로 바꾸지 않습니다. 거리와 증명된 상계가 겹쳐 보일 때도 표에서 출처를 구별하세요.', 'Logarithms apply only to nonzero magnitudes. Exact zeros are marked separately; tiny positives are not replaced by zero. Use the table to distinguish sources even when distance and certificate curves overlap.')}</p>
    </figure>
  </>;
}
