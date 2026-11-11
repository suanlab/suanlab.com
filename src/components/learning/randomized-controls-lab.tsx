'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { randomizedControlsGuide } from '@/data/randomized-controls-reader';
import { calculateRandomizedControls } from '@/lib/learning-randomized-controls';
import { WavePlot, waveNumber } from './waves-plot';

export default function RandomizedControlsLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [coefficient, setCoefficient] = useState(.3);
  const [correlation, setCorrelation] = useState(0);
  const result = calculateRandomizedControls(coefficient, correlation);
  const curve = Array.from({ length: 41 }, (_, i) => {
    const rho = i / 40;
    return { rho, ...calculateRandomizedControls(coefficient, rho) };
  });
  const readings = [
    [ko ? '전체 모형 MSE' : 'Full model MSE', result.full],
    [ko ? '특성 제거 MSE' : 'Feature removed MSE', result.removed],
    [ko ? '고정 무작위화 MSE' : 'Frozen randomization MSE', result.frozenRandom],
    [ko ? '재학습 무작위화 MSE' : 'Refitted randomization MSE', result.refittedRandom],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="randomized-controls-title">
    <h2 id="randomized-controls-title">{ko ? '고정 대조와 재학습 대조' : 'Frozen and refitted controls'}</h2>
    <label>c = {coefficient}
      <input className="block w-full" type="range" min={0} max={.6} step={.05} value={coefficient} onChange={e => setCoefficient(Number(e.target.value))}/>
    </label>
    <label>Corr(Z,Z′) = {correlation}
      <input className="block w-full" type="range" min={0} max={1} step={.05} value={correlation} onChange={e => setCorrelation(Number(e.target.value))}/>
    </label>
    <WavePlot marker="randomized-controls" title={ko ? '교란에 남은 정보와 MSE' : 'Retained information and MSE'}
      xLabel="Corr(Z,Z′)" yLabel="MSE" xDomain={[0, 1]}
      series={[
        { name: ko ? '고정' : 'Frozen', color: 'blue', segments: [curve.map(row => [row.rho, row.frozenRandom])] },
        { name: ko ? '재학습' : 'Refitted', color: 'orange', segments: [curve.map(row => [row.rho, row.refittedRandom])] },
      ]}
      dots={[{ color: 'blue', point: [correlation, result.frozenRandom] }, { color: 'orange', point: [correlation, result.refittedRandom] }]}
      caption={ko ? '독립 대조는 상관 0에 해당합니다. 잡음 분산은 0.01입니다.' : 'The independent control is at zero correlation. Noise variance is 0.01.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {randomizedControlsGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function RandomizedControlsConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="randomized-controls-concept-title">
    <h2 id="randomized-controls-concept-title">{ko ? '원인을 겨냥하는 대조' : 'A control targeting the proposed mechanism'}</h2>
    <p>{randomizedControlsGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{randomizedControlsGuide.steps[0].formula}</p>
    <p>{randomizedControlsGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '교란과 재학습의 효과 비교하기' : 'Compare perturbation and refitting'}</a>
  </section>;
}
