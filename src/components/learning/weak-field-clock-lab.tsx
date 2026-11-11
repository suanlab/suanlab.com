'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { weakFieldClockGuide } from '@/data/weak-field-clock-reader';
import { calculateWeakFieldClock } from '@/lib/learning-weak-field-clock';
import { WavePlot, waveNumber } from './waves-plot';

export default function WeakFieldClockLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [radius, setRadius] = useState(12);
  const [tolerance, setTolerance] = useState(.01);
  const result = calculateWeakFieldClock(radius);
  const points = Array.from({ length: 55 }, (_, i): [number, number] => [6 + i, calculateWeakFieldClock(6 + i).error]);
  const readings = [
    [ko ? '정지 시계 진행률' : 'Static clock rate', result.staticRate],
    [ko ? '정확한 궤도 시계 진행률' : 'Exact orbital clock rate', result.exact],
    [ko ? '일차 근사 진행률' : 'First-order clock rate', result.linear],
    [ko ? '절대 진행률 오차' : 'Absolute rate error', result.error],
  ] as const;
  const meets = result.error <= tolerance;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="weak-field-title">
    <h2 id="weak-field-title">{ko ? '궤도 시계의 근사 오차' : 'Approximation error of an orbital clock'}</h2>
    <label>r/M = {radius}
      <input className="block w-full" type="range" min={6} max={60} value={radius} onChange={e => setRadius(Number(e.target.value))}/>
    </label>
    <label>{ko ? '허용오차' : 'Tolerance'}
      <select className="ml-2 rounded border border-border bg-background p-1" value={tolerance} onChange={e => setTolerance(Number(e.target.value))}>
        {[.001, .005, .01, .02].map(value => <option key={value}>{value}</option>)}
      </select>
    </label>
    <WavePlot marker="weak-field-clock" title={ko ? '반지름에 따른 절단 오차' : 'Truncation error versus radius'}
      xLabel="r/M" yLabel={ko ? '진행률 오차' : 'Rate error'} xDomain={[6, 60]}
      series={[
        { name: ko ? '일차식의 오차' : 'First-order error', color: 'blue', segments: [points] },
        { name: ko ? '허용오차' : 'Tolerance', color: 'neutral', segments: [[[6, tolerance], [60, tolerance]]] },
      ]}
      dots={[{ color: 'orange', point: [radius, result.error] }]}
      caption={ko ? '범위의 왼쪽 끝 6M은 원형 궤도 안정성의 경계입니다.' : 'The left endpoint at 6M marks the boundary of circular-orbit stability.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    <p role="status">{meets ? (ko ? '허용오차 충족' : 'Tolerance met') : (ko ? '허용오차 초과' : 'Tolerance exceeded')}</p>
    {weakFieldClockGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function WeakFieldClockConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="weak-field-concept-title">
    <h2 id="weak-field-concept-title">{ko ? '근사의 차수와 정확도' : 'Order and accuracy of an approximation'}</h2>
    <p>{weakFieldClockGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{weakFieldClockGuide.steps[0].formula}</p>
    <p>{weakFieldClockGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '허용오차와 비교하기' : 'Compare with a tolerance'}</a>
  </section>;
}
