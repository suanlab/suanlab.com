'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { tikhonovNoiseGainGuide } from '@/data/tikhonov-noise-gain-reader';
import { calculateTikhonovMode } from '@/lib/learning-tikhonov-noise-gain';
import { WavePlot, waveNumber } from './waves-plot';

export default function TikhonovNoiseGainLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [strength, setStrength] = useState(.05);
  const [penalty, setPenalty] = useState(.0025);
  const [sigma, setSigma] = useState(.05);
  const at = (weight: number) => calculateTikhonovMode(strength, weight, 1, sigma, sigma);
  const result = at(penalty);
  const curve = Array.from({ length: 51 }, (_, i) => {
    const weight = i * .0005;
    const model = at(weight);
    return model ? [{ weight, model }] : [];
  }).flat();
  const readings = [
    [ko ? '신호 보존율' : 'Signal retention', result?.signalRetention],
    [ko ? '잡음 표준편차' : 'Noise standard deviation', result?.noiseDeviation],
    [ko ? '예시 추정값' : 'Example estimate', result?.estimate],
    [ko ? '평균제곱오차' : 'Mean squared error', result?.meanSquaredError],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="tikhonov-title">
    <h2 id="tikhonov-title">{ko ? '정규화로 편향과 분산 조절하기' : 'Adjust bias and variance with regularization'}</h2>
    <label>s = {strength}
      <input className="block w-full" type="range" min={0} max={.5} step={.05} value={strength} onChange={e => setStrength(Number(e.target.value))}/>
    </label>
    <label>λ = {penalty}
      <input className="block w-full" type="range" min={0} max={.025} step={.0005} value={penalty} onChange={e => setPenalty(Number(e.target.value))}/>
    </label>
    <label>{ko ? '관측 잡음' : 'Observation noise'} σ = {sigma}
      <input className="block w-full" type="range" min={0} max={.2} step={.01} value={sigma} onChange={e => setSigma(Number(e.target.value))}/>
    </label>
    <WavePlot marker="tikhonov-noise-gain" title={ko ? '벌점에 따른 편향 제곱과 분산' : 'Squared bias and variance versus penalty'}
      xLabel="λ" yLabel={ko ? '오차의 제곱' : 'Squared error'} xDomain={[0, .025]}
      series={[
        { name: ko ? '편향 제곱' : 'Squared bias', color: 'blue', segments: [curve.map(row => [row.weight, row.model.bias ** 2])] },
        { name: ko ? '분산' : 'Variance', color: 'orange', segments: [curve.map(row => [row.weight, row.model.noiseDeviation ** 2])] },
        { name: ko ? '평균제곱오차' : 'Mean squared error', color: 'neutral', segments: [curve.map(row => [row.weight, row.model.meanSquaredError])] },
      ]}
      dots={result ? [{ color: 'neutral', point: [penalty, result.meanSquaredError] }] : []}
      caption={ko ? '참 신호는 1입니다. 예시 추정에는 +σ의 관측 오차를 사용합니다.' : 'The true signal is one. The example estimate uses a +σ observation error.'}/>
    {!result && <p role="status">{ko ? '관측과 벌점이 모두 없는 방향의 해는 유일하지 않습니다.' : 'A mode with zero strength and zero penalty has no unique estimate.'}</p>}
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {tikhonovNoiseGainGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function TikhonovNoiseGainConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="tikhonov-concept-title">
    <h2 id="tikhonov-concept-title">{ko ? '정규화가 신호와 잡음을 줄이는 방식' : 'How regularization shrinks signal and noise'}</h2>
    <p>{tikhonovNoiseGainGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{tikhonovNoiseGainGuide.steps[0].formula}</p>
    <p>{tikhonovNoiseGainGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '벌점을 바꿔 오차 비교하기' : 'Change the penalty and compare errors'}</a>
  </section>;
}
