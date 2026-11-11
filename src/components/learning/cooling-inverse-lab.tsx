'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { coolingInverseGuide } from '@/data/cooling-inverse-reader';
import { fitOneCoolingReading, fitTwoCoolingReadings, predictCooling } from '@/lib/learning-cooling-inverse';
import { WavePlot, waveNumber, type Mark, type Series } from './waves-plot';

const firstTime = 2;
const first = predictCooling(40, .2, firstTime);

export default function CoolingInverseLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [secondTime, setSecondTime] = useState(6);
  const [error, setError] = useState(0);
  const [includeSecond, setIncludeSecond] = useState(false);
  const second = predictCooling(40, .2, secondTime) + error;
  const fit = includeSecond ? fitTwoCoolingReadings(first, second, firstTime, secondTime) : null;
  const curve = (initial: number, rate: number) => Array.from({ length: 81 }, (_, i): [number, number] => [
    i / 8, predictCooling(initial, rate, i / 8),
  ]);
  const series: Series[] = [];
  if (!includeSecond) {
    for (const [rate, color] of [[.1, 'blue'], [.2, 'neutral'], [.4, 'orange']] as const) {
      const candidate = fitOneCoolingReading(first, firstTime, rate);
      series.push({ name: `k = ${rate}`, color, segments: [curve(candidate.initial, rate)] });
    }
  } else if (fit) {
    series.push({ name: ko ? '두 관측으로 구한 곡선' : 'Curve fitted to two readings', color: 'blue', segments: [curve(fit.initial, fit.rate)] });
  }
  const dots: Mark[] = [{ color: 'neutral', point: [firstTime, first] }];
  if (includeSecond) dots.push({ color: 'orange', point: [secondTime, second] });
  const status = !includeSecond
    ? (ko ? '첫 관측을 만족하는 여러 후보' : 'Multiple curves fit the first reading')
    : fit
      ? (ko ? '두 관측을 만족하는 유일한 곡선' : 'A unique curve fits both readings')
      : (ko ? '두 관측을 만족하는 양의 냉각 모형이 없습니다.' : 'No positive cooling model fits both readings.');
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="cooling-inverse-title">
    <h2 id="cooling-inverse-title">{ko ? '온도에서 냉각률 복원하기' : 'Recover the cooling rate from temperatures'}</h2>
    <label className="flex items-center gap-2">
      <input type="checkbox" checked={includeSecond} onChange={e => setIncludeSecond(e.target.checked)}/>
      {ko ? '두 번째 관측 포함' : 'Include the second reading'}
    </label>
    <label>t₂ = {secondTime} min
      <input className="block w-full" type="range" min={2.25} max={10} step={.25} value={secondTime} disabled={!includeSecond} onChange={e => setSecondTime(Number(e.target.value))}/>
    </label>
    <label>{ko ? '두 번째 관측 오차' : 'Second reading error'} = {error} K
      <input className="block w-full" type="range" min={-10} max={10} step={.5} value={error} disabled={!includeSecond} onChange={e => setError(Number(e.target.value))}/>
    </label>
    <WavePlot marker="cooling-inverse" title={ko ? '냉각 곡선과 온도 관측' : 'Cooling curves and readings'}
      xLabel="t (min)" yLabel="y (K)" xDomain={[0, 10]}
      series={series} dots={dots}
      caption={ko ? '회색 점은 첫 관측입니다. 두 번째 관측을 포함하면 주황 점을 표시합니다.' : 'Gray marks the first reading. Orange marks the second reading when included.'}/>
    <p role="status">{status}</p>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <div className="contents"><dt>A (K)</dt><dd>{waveNumber(fit?.initial)}</dd></div>
      <div className="contents"><dt>k (min⁻¹)</dt><dd>{waveNumber(fit?.rate)}</dd></div>
    </dl>
    {coolingInverseGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function CoolingInverseConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="cooling-inverse-concept-title">
    <h2 id="cooling-inverse-concept-title">{ko ? '같은 관측을 설명하는 곡선들' : 'Curves explaining the same reading'}</h2>
    <p>{coolingInverseGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{coolingInverseGuide.steps[0].formula}</p>
    <p>{coolingInverseGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '관측을 추가해 보기' : 'Add an observation'}</a>
  </section>;
}
