'use client';
import React, { useEffect, useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import type { RegularPerturbationResult } from '@/lib/learning-regular-perturbation-protocol';
import { regularPerturbationInputKey } from '@/lib/learning-regular-perturbation-protocol';
import { regularPerturbationView, type PerturbationWindow } from '@/lib/learning-regular-perturbation-view';
import RegularPerturbationPlots from './regular-perturbation-plots';

export default function RegularPerturbationResultView({ result: r, language }: { result: RegularPerturbationResult; language: 'ko' | 'en' }) {
  const t = (ko: string, en: string) => language === 'ko' ? ko : en, id = useId();
  const [order, setOrder] = useState(r.input.order), [window, setWindow] = useState<PerturbationWindow>('equilibrium'), [playing, setPlaying] = useState(false), [reduced, setReduced] = useState(true);
  useEffect(() => { const media = matchMedia('(prefers-reduced-motion: reduce)'), update = () => { setReduced(media.matches); if (media.matches) setPlaying(false); }; update(); media.addEventListener('change', update); return () => media.removeEventListener('change', update); }, []);
  useEffect(() => { if (!playing || reduced) return; const timer = setInterval(() => setOrder(n => (n + 1) % (r.input.order + 1)), 900); const stop = () => { if (document.visibilityState === 'hidden') setPlaying(false); }; document.addEventListener('visibilitychange', stop); return () => { clearInterval(timer); document.removeEventListener('visibilitychange', stop); }; }, [playing, reduced, r.input.order]);
  const model = regularPerturbationView(r, window, order), chosen = model.chosen;
  const region = r.convergenceRegion === 'inside-radius' ? t('Taylor 원판 내부', 'Inside the Taylor disk') : r.convergenceRegion === 'outside-radius' ? t('Taylor 원판 외부: 차수 증가 시 발산', 'Outside the Taylor disk: divergent with increasing order') : r.convergenceRegion === 'radius-boundary' ? t('수렴 반경 경계: 별도 판정', 'Radius boundary: classified separately') : t('정확한 영 해', 'Identically zero branch');
  const metric = (key: string, label: string, value: string) => <div key={key}><span id={`${id}-${key}`}>{label}</span><strong data-perturbation-metric={key} role="region" tabIndex={0} aria-labelledby={`${id}-${key}`} style={{ maxHeight: 160, overflow: 'auto', fontSize: '1.1rem' }}>{value}</strong></div>;
  const table = (caption: string, headers: string[], rows: (string | number)[][]) => <div className="qubit-scroll" role="region" tabIndex={0} aria-label={caption} style={{ maxHeight: 520, overflow: 'auto' }}><table className="scalar-table"><caption>{caption}</caption><thead style={{ position: 'sticky', top: 0, background: 'hsl(var(--card))', zIndex: 1 }}><tr>{headers.map((header, i) => <th scope="col" key={i}>{header}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}><div role="region" tabIndex={0} aria-label={`${caption}, ${headers[j]}, ${i}`} style={{ maxHeight: 160, overflow: 'auto', minWidth: 100 }}>{cell}</div></td>)}</tr>)}</tbody></table></div>;
  return <div data-perturbation-result data-perturbation-result-key={regularPerturbationInputKey(r.input)}>
    <h3>{t('검증한 결과 탐색', 'Explore the verified result')}</h3>
    <p data-perturbation-region>{region}. {t('이는 선택 입력의 급수 분류이며 유한 차수 오차 보증과 구분합니다.', 'This classifies the series at the selected input, separately from finite-order error certificates.')}</p>
    <label className="qubit-field" htmlFor={`${id}-order`}>{t('선택 차수', 'Selected order')}: {order}<input id={`${id}-order`} aria-label={t('선택 차수', 'Selected order')} type="range" min={0} max={r.input.order} step={1} value={order} onChange={e => { setPlaying(false); setOrder(Number(e.target.value)); }} /></label>
    <div className="qubit-actions"><Button variant="outline" disabled={reduced || r.input.order === 0} aria-pressed={playing} onClick={() => setPlaying(v => !v)}>{playing ? t('차수 재생 멈춤', 'Pause order playback') : t('차수 변화 재생', 'Play order changes')}</Button></div>
    <p className="ai-small">{t('이미 계산·검증한 부분합을 전환합니다. 새 계산이나 물리적 시간 진행이 아닙니다. 모션 감소 설정에서는 자동 재생을 사용하지 않습니다.', 'Playback switches already calculated and verified partial sums. It is neither a new calculation nor physical time evolution. Automatic playback is disabled when reduced motion is requested.')}</p>
    <div className="information-metrics">
      {metric('g', t('무차원 결합 g', 'Dimensionless coupling g'), r.dimensionlessCoupling)}
      {metric('scale', t('선형 변위 h/k', 'Linear displacement h/k'), r.linearScale)}
      {metric('partial', `P${order}`, chosen.approximation)}{metric('residual', 'F(P)', chosen.residual)}
      {metric('bound', t('변위 오차 상계 |F|/k', 'Displacement bound |F|/k'), chosen.errorBound)}
      {metric('energy-bound', t('에너지 차이 상계 F²/(2k)', 'Energy-gap bound F²/(2k)'), chosen.energyGapBound)}
      {metric('distance-lower', t('근 거리 하한', 'Root-distance lower bound'), chosen.rootDistance.lower)}
      {metric('distance-upper', t('근 거리 상한', 'Root-distance upper bound'), chosen.rootDistance.upper)}
    </div>
    <label className="qubit-field" htmlFor={`${id}-window`}>{t('그림의 변위 구간', 'Displacement plotting window')}<select id={`${id}-window`} aria-label={t('그림의 변위 구간', 'Displacement plotting window')} value={window} onChange={e => setWindow(e.target.value as PerturbationWindow)}><option value="equilibrium">{t('평형 주변', 'Around equilibrium')}</option><option value="linear">{t('선형 변위 척도', 'Linear-response scale')}</option></select></label>
    <RegularPerturbationPlots model={model} language={language} />
    <h3>{t('정확한 근 구간과 감수율', 'Exact root bracket and susceptibility')}</h3>
    {table(t('유리수 검산 구간', 'Rational verification intervals'), [t('대상', 'Quantity'), t('하한', 'Lower'), t('상한', 'Upper')], [[t('유일한 실근', 'Unique real root'), r.root.lower, r.root.upper], [t('끝점의 방정식 잔차', 'Equation residual at endpoints'), r.root.lowerResidual, r.root.upperResidual], ['dx*/dh', r.susceptibility.lower, r.susceptibility.upper]])}
    <p>{t('근 구간의 폭은', 'The root-bracket width is')} <span data-perturbation-root-width style={{ overflowWrap: 'anywhere' }}>{r.root.width}</span>. {t('기울기 하한 k>0과 끝점 부호가 포함을 보장합니다. 거리 구간은 이 근 구간으로부터 계산하며 실제 오차와 동일한 숫자라고 주장하지 않습니다.', 'The positive slope lower bound k and endpoint signs certify enclosure. Distance intervals come from this bracket and are not claimed to equal the exact error.')}</p>
    <h3>{t('전체 계수·항·부분합·보증', 'All coefficients, terms, partial sums and certificates')}</h3>
    {table(t('차수별 정확한 유리수', 'Exact fractions by order'), ['n', 'aₙ', 'aₙλⁿ', 'Pₙ', 'F(Pₙ)', '|F|/k', 'F²/(2k)', t('거리 하한', 'Distance lower'), t('거리 상한', 'Distance upper')], r.approximations.map(row => [row.order, row.coefficient, row.term, row.approximation, row.residual, row.errorBound, row.energyGapBound, row.rootDistance.lower, row.rootDistance.upper]))}
    <h3>{t('기호 잔차의 차수 소거', 'Order cancellation in the symbolic residual')}</h3>
    <p>{t('수치 λ를 선택하기 전의 R_N(λ)=kP_N+λP_N³−h입니다. 0부터 입력 N차까지의 계수는 정확히 0이어야 합니다. 아래 모든 계수는 λ의 오름차순입니다.', 'Before selecting numerical λ, R_N(λ)=kP_N+λP_N³−h. Coefficients from order 0 through input N must be exactly zero. All coefficients below are in increasing powers of λ.')}</p>
    {table(t('잔차 다항식 계수', 'Residual polynomial coefficients'), [t('λ의 차수', 'Power of λ'), t('계수', 'Coefficient')], r.residualPolynomial.map((value, i) => [i, value]))}
  </div>;
}
