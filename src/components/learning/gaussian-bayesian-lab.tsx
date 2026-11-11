'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { gaussianBayesianGuide } from '@/data/gaussian-bayesian-reader';
import { calculateGaussianPosterior, gaussianDensity } from '@/lib/learning-gaussian-bayesian';
import { WavePlot, waveNumber } from './waves-plot';

export default function GaussianBayesianLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [strength, setStrength] = useState(.5);
  const [noise, setNoise] = useState(.5);
  const [observation, setObservation] = useState(1);
  const result = calculateGaussianPosterior(0, 1, strength, noise ** 2, observation);
  const extent = Math.max(4, Math.abs(result.mean) + 4 * Math.sqrt(result.variance));
  const density = (mean: number, variance: number) => Array.from({ length: 201 }, (_, i): [number, number] => {
    const x = (i / 100 - 1) * extent;
    return [x, gaussianDensity(x, mean, variance)];
  });
  const readings = [
    [ko ? '사후 평균' : 'Posterior mean', result.mean],
    [ko ? '사후 분산' : 'Posterior variance', result.variance],
    [ko ? '새 관측의 예측 평균' : 'Predictive mean of a new reading', result.predictiveMean],
    [ko ? '새 관측의 예측 분산' : 'Predictive variance of a new reading', result.predictiveVariance],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="gaussian-bayesian-title">
    <h2 id="gaussian-bayesian-title">{ko ? '관측에 따른 사후 밀도' : 'Posterior density after an observation'}</h2>
    <label>{ko ? '센서 강도' : 'Sensor strength'} s = {strength}
      <input className="block w-full" type="range" min={0} max={1} step={.05} value={strength} onChange={e => setStrength(Number(e.target.value))}/>
    </label>
    <label>{ko ? '관측 잡음 표준편차' : 'Observation noise standard deviation'} √r = {noise}
      <input className="block w-full" type="range" min={.1} max={1} step={.05} value={noise} onChange={e => setNoise(Number(e.target.value))}/>
    </label>
    <label>{ko ? '관측값' : 'Observation'} y = {observation}
      <input className="block w-full" type="range" min={-2} max={2} step={.1} value={observation} onChange={e => setObservation(Number(e.target.value))}/>
    </label>
    <WavePlot marker="gaussian-bayesian" title={ko ? '사전 밀도와 사후 밀도' : 'Prior and posterior densities'}
      xLabel="x" yLabel={ko ? '확률 밀도' : 'Probability density'} xDomain={[-extent, extent]}
      series={[
        { name: ko ? '사전' : 'Prior', color: 'neutral', segments: [density(0, 1)] },
        { name: ko ? '사후' : 'Posterior', color: 'blue', segments: [density(result.mean, result.variance)] },
      ]}
      caption={ko ? '사전 분포의 평균은 0, 분산은 1입니다. 예측은 같은 센서의 독립 후속 관측을 사용합니다.' : 'The prior has mean zero and variance one. Prediction assumes an independent follow-up using the same sensor.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {gaussianBayesianGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function GaussianBayesianConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="gaussian-bayesian-concept-title">
    <h2 id="gaussian-bayesian-concept-title">{ko ? '정밀도와 사후 분포' : 'Precision and the posterior'}</h2>
    <p>{gaussianBayesianGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{gaussianBayesianGuide.steps[0].formula}</p>
    <p>{gaussianBayesianGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '관측의 정보량 바꾸기' : 'Vary the information in the observation'}</a>
  </section>;
}
