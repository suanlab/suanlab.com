'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { availabilityLeakageGuide } from '@/data/availability-leakage-reader';
import { calculateAvailabilityLeakage } from '@/lib/learning-availability-leakage';
import { WavePlot, waveNumber } from './waves-plot';

export default function AvailabilityLeakageLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [correlation, setCorrelation] = useState(0);
  const result = calculateAvailabilityLeakage(correlation);
  const curve = Array.from({ length: 41 }, (_, i): [number, number] => [
    i / 40, calculateAvailabilityLeakage(i / 40).availableMse,
  ]);
  const readings = [
    [ko ? '현재 정보 MSE' : 'Available information MSE', result.availableMse],
    [ko ? '미래 특성 누출 MSE' : 'Leaked future feature MSE', result.futureMse],
    [ko ? '목표 누출 MSE' : 'Leaked target MSE', result.targetMse],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="availability-leakage-title">
    <h2 id="availability-leakage-title">{ko ? '판단 시각의 정보 경계' : 'The decision-time information boundary'}</h2>
    <label>Corr(X,F) = {correlation}
      <input className="block w-full" type="range" min={0} max={1} step={.05} value={correlation} onChange={e => setCorrelation(Number(e.target.value))}/>
    </label>
    <WavePlot marker="availability-leakage" title={ko ? '예측 가능성과 정보별 오차' : 'Predictability and error by information set'}
      xLabel="ρ" yLabel="MSE" xDomain={[0, 1]} yDomain={[0, 1.05]}
      series={[
        { name: ko ? '현재 정보' : 'Available information', color: 'blue', segments: [curve] },
        { name: ko ? '미래 특성' : 'Future feature', color: 'orange', segments: [[[0, .36], [1, .36]]] },
        { name: ko ? '목표 자체' : 'The target itself', color: 'neutral', segments: [[[0, 0], [1, 0]]] },
      ]}
      dots={[{ color: 'blue', point: [correlation, result.availableMse] }]}
      caption={ko ? '합성 과정에서 현재 시각에 사용할 수 있는 변수는 X입니다.' : 'In the synthetic process, X is available at decision time.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {availabilityLeakageGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function AvailabilityLeakageConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="availability-leakage-concept-title">
    <h2 id="availability-leakage-concept-title">{ko ? '판단할 때 알 수 있었던 것' : 'What was known at the time of the decision'}</h2>
    <p>{availabilityLeakageGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{availabilityLeakageGuide.steps[0].formula}</p>
    <p>{availabilityLeakageGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '가용 정보와 누출된 정보 비교하기' : 'Compare available and leaked information'}</a>
  </section>;
}
