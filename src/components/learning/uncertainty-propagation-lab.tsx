'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { uncertaintyPropagationGuide } from '@/data/uncertainty-propagation-reader';
import { calculateCorrelatedVariance } from '@/lib/learning-uncertainty-propagation';
import { WavePlot, waveNumber } from './waves-plot';

export default function UncertaintyPropagationLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [correlation, setCorrelation] = useState(.75);
  const [scale, setScale] = useState(1);
  const result = calculateCorrelatedVariance(1, scale, correlation);
  const ellipse = Array.from({ length: 81 }, (_, i): [number, number] => {
    const angle = Math.PI * i / 40;
    return [Math.cos(angle), scale * (correlation * Math.cos(angle) + Math.sqrt(1 - correlation ** 2) * Math.sin(angle))];
  });
  const readings = [
    [ko ? '공분산' : 'Covariance', result.covariance],
    [ko ? '합의 분산' : 'Sum variance', result.sumVariance],
    [ko ? '차의 분산' : 'Difference variance', result.differenceVariance],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="uncertainty-propagation-title">
    <h2 id="uncertainty-propagation-title">{ko ? '상관에 따른 합과 차의 분산' : 'Sum and difference variance under correlation'}</h2>
    <label>ρ = {correlation}
      <input className="block w-full" type="range" min={-1} max={1} step={.05} value={correlation} onChange={e => setCorrelation(Number(e.target.value))}/>
    </label>
    <label>σ₂ = {scale}
      <input className="block w-full" type="range" min={.5} max={2} step={.05} value={scale} onChange={e => setScale(Number(e.target.value))}/>
    </label>
    <WavePlot marker="uncertainty-propagation" title={ko ? '공분산 타원과 투영 방향' : 'Covariance ellipse and projection directions'}
      xLabel="x₁" yLabel="x₂" xDomain={[-2.1, 2.1]} yDomain={[-2.1, 2.1]}
      series={[
        { name: ko ? '오차 타원' : 'Error ellipse', color: 'blue', segments: [ellipse] },
        { name: ko ? '합 방향' : 'Sum direction', color: 'orange', segments: [[[-1.5, -1.5], [1.5, 1.5]]] },
        { name: ko ? '차 방향' : 'Difference direction', color: 'neutral', segments: [[[-1.5, 1.5], [1.5, -1.5]]] },
      ]}
      caption={ko ? 'σ₁은 1입니다. 두 축의 수치 범위가 같습니다.' : 'The first standard deviation is one. The numerical ranges of both axes match.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {uncertaintyPropagationGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function UncertaintyPropagationConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="uncertainty-propagation-concept-title">
    <h2 id="uncertainty-propagation-concept-title">{ko ? '함께 움직이는 측정 오차' : 'Measurement errors that move together'}</h2>
    <p>{uncertaintyPropagationGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{uncertaintyPropagationGuide.steps[0].formula}</p>
    <p>{uncertaintyPropagationGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '상관과 오차 척도 조절하기' : 'Vary correlation and error scale'}</a>
  </section>;
}
