'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { sensorIdentifiabilityGuide } from '@/data/sensor-identifiability-reader';
import { calculateSensorIdentifiability } from '@/lib/learning-sensor-identifiability';
import { WavePlot, waveNumber } from './waves-plot';

export default function SensorIdentifiabilityLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [coefficient, setCoefficient] = useState(-1);
  const [sigma, setSigma] = useState(.1);
  const [error, setError] = useState(0);
  const result = calculateSensorIdentifiability(coefficient, sigma, 0, error);
  const states = {
    unique: ko ? '두 성분을 유일하게 추정' : 'Both amplitudes have unique estimates',
    nonunique: ko ? '합만 관측됨' : 'Only the sum is identifiable',
    inconsistent: ko ? '두 관측을 동시에 만족하는 해가 없음' : 'No solution matches both readings',
  };
  const readings = [
    ['y₁', result.first], ['y₂', result.second],
    ['â', result.estimate?.a], ['b̂', result.estimate?.b],
    [ko ? 'b̂의 표준편차' : 'Standard deviation of b̂', result.bDeviation],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="sensor-identifiability-title">
    <h2 id="sensor-identifiability-title">{ko ? '센서 방향과 해의 불확실성' : 'Sensor directions and uncertainty'}</h2>
    <label>r = {coefficient}
      <input className="block w-full" type="range" min={-1} max={1} step={.1} value={coefficient} onChange={e => setCoefficient(Number(e.target.value))}/>
    </label>
    <label>{ko ? '잡음 표준편차' : 'Noise standard deviation'} σ = {sigma}
      <input className="block w-full" type="range" min={0} max={.5} step={.05} value={sigma} onChange={e => setSigma(Number(e.target.value))}/>
    </label>
    <label>{ko ? '둘째 관측 오차' : 'Second reading error'} = {error}
      <input className="block w-full" type="range" min={-.2} max={.2} step={.05} value={error} onChange={e => setError(Number(e.target.value))}/>
    </label>
    <WavePlot marker="sensor-identifiability" title={ko ? '관측을 만족하는 매개변수 직선' : 'Parameter lines compatible with the readings'}
      xLabel="b" yLabel="a" xDomain={[0, 6]}
      series={[
        { name: 'y₁ = a + b', color: 'blue', segments: [[[0, result.first], [6, result.first - 6]]] },
        { name: 'y₂ = a + rb', color: 'orange', segments: [[[0, result.second], [6, result.second - 6 * coefficient]]] },
      ]}
      dots={result.estimate ? [{ color: 'neutral', point: [result.estimate.b, result.estimate.a] }] : []}
      caption={ko ? '두 관측을 만족하는 교점이 추정값입니다.' : 'The intersection satisfies both observed equations.'}/>
    <p role="status">{states[result.state]}</p>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {sensorIdentifiabilityGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function SensorIdentifiabilityConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="sensor-identifiability-concept-title">
    <h2 id="sensor-identifiability-concept-title">{ko ? '두 관측선이 만나는 곳' : 'Where the two observation lines meet'}</h2>
    <p>{sensorIdentifiabilityGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{sensorIdentifiabilityGuide.steps[0].formula}</p>
    <p>{sensorIdentifiabilityGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '센서 방향을 바꾸기' : 'Change the sensor direction'}</a>
  </section>;
}
