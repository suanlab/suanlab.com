'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { rindlerObservablesGuide } from '@/data/rindler-observables-reader';
import { calculateRindlerReadings } from '@/lib/learning-rindler-observables';
import { WavePlot, waveNumber } from './waves-plot';

export default function RindlerObservablesLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [rho, setRho] = useState(1);
  const [eta, setEta] = useState(1);
  const result = calculateRindlerReadings(rho, eta);
  const endpoint = calculateRindlerReadings(rho, 2);
  const limit = Math.ceil(endpoint.position);
  const points = Array.from({ length: 81 }, (_, i): [number, number] => {
    const { position, time } = calculateRindlerReadings(rho, i / 40);
    return [position, time];
  });
  const readings = [
    ['T', result.time], ['X', result.position],
    [ko ? '가속 시계의 고유 시간' : 'Accelerated proper time', result.acceleratedProperTime],
    [ko ? '관성 직선 경로의 고유 시간' : 'Inertial straight-path proper time', result.inertialProperTime],
    [ko ? '고유 가속도' : 'Proper acceleration', result.acceleration],
    [ko ? '좌표 빛 속력' : 'Coordinate light speed', result.coordinateLightSpeed],
    [ko ? '국소 빛 속력' : 'Local light speed', result.measuredLightSpeed],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="rindler-title">
    <h2 id="rindler-title">{ko ? '가속 관측자의 시계와 좌표' : 'An accelerated observer’s clock and coordinates'}</h2>
    <label>ρ = {rho}
      <input className="block w-full" type="range" min={.5} max={4} step={.25} value={rho} onChange={e => setRho(Number(e.target.value))}/>
    </label>
    <label>η = {eta}
      <input className="block w-full" type="range" min={0} max={2} step={.05} value={eta} onChange={e => setEta(Number(e.target.value))}/>
    </label>
    <WavePlot marker="rindler-observables" title={ko ? '관성 좌표에서 본 가속 세계선' : 'Accelerated worldline in inertial coordinates'}
      xLabel="X" yLabel="T" xDomain={[0, limit]} yDomain={[0, limit]}
      series={[
        { name: ko ? '가속 세계선' : 'Accelerated worldline', color: 'blue', segments: [points] },
        { name: ko ? '쐐기 경계' : 'Wedge boundary', color: 'neutral', segments: [[[0, 0], [limit, limit]]] },
        { name: ko ? '관성 경로' : 'Inertial path', color: 'orange', segments: [[[rho, 0], [result.position, result.time]]] },
      ]}
      dots={[{ color: 'orange', point: [result.position, result.time] }]}
      caption={ko ? '두 경로는 선택한 끝점을 공유합니다. 시계는 각 경로의 고유 시간을 잽니다.' : 'The paths share the selected endpoints. Each clock measures proper time along its own path.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {rindlerObservablesGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function RindlerObservablesConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="rindler-concept-title">
    <h2 id="rindler-concept-title">{ko ? '좌표를 바꾸고 측정량을 읽기' : 'Change coordinates and read observables'}</h2>
    <p>{rindlerObservablesGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{rindlerObservablesGuide.steps[0].formula}</p>
    <p>{rindlerObservablesGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '두 시계 읽기 비교' : 'Compare both clock readings'}</a>
  </section>;
}
