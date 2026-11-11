'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { landauerErasureGuide } from '@/data/landauer-erasure-reader';
import { calculateLandauerErasure } from '@/lib/learning-landauer-erasure';
import { WavePlot, waveNumber } from './waves-plot';

export default function LandauerErasureLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [probability, setProbability] = useState(.5);
  const [error, setError] = useState(0);
  const [temperature, setTemperature] = useState(300);
  const result = calculateLandauerErasure(probability, error, temperature);
  const curve = Array.from({ length: 101 }, (_, i): [number, number] => [
    i / 100, calculateLandauerErasure(i / 100, 0, temperature).initialEntropy,
  ]);
  const readings = [
    [ko ? '초기 엔트로피 (nat)' : 'Initial entropy (nat)', result.initialEntropy],
    [ko ? '최종 엔트로피 (nat)' : 'Final entropy (nat)', result.finalEntropy],
    [ko ? '엔트로피 감소량 (nat)' : 'Entropy decrease (nat)', result.entropyDecrease],
    [ko ? '열 하한 (10⁻²¹ J)' : 'Heat lower bound (10⁻²¹ J)', result.heatFloor * 1e21],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="landauer-title">
    <h2 id="landauer-title">{ko ? '초기화 전후의 엔트로피' : 'Entropy before and after reset'}</h2>
    <label>{ko ? '초기 확률' : 'Initial probability'} p = {probability}
      <input className="block w-full" type="range" min={0} max={1} step={.05} value={probability} onChange={e => setProbability(Number(e.target.value))}/>
    </label>
    <label>{ko ? '초기화 오차' : 'Reset error'} ε = {error}
      <input className="block w-full" type="range" min={0} max={.5} step={.05} value={error} onChange={e => setError(Number(e.target.value))}/>
    </label>
    <label>T = {temperature} K
      <input className="block w-full" type="range" min={150} max={600} step={25} value={temperature} onChange={e => setTemperature(Number(e.target.value))}/>
    </label>
    <WavePlot marker="landauer-erasure" title={ko ? '이진 엔트로피와 두 확률' : 'Binary entropy and the two probabilities'}
      xLabel={ko ? '상태 1의 확률' : 'Probability of state 1'} yLabel="h (nat)" xDomain={[0, 1]} yDomain={[0, .75]}
      series={[{ name: ko ? '이진 엔트로피' : 'Binary entropy', color: 'neutral', segments: [curve] }]}
      dots={[
        { color: 'blue', point: [probability, result.initialEntropy] },
        { color: 'orange', point: [error, result.finalEntropy] },
      ]}
      caption={ko ? '파란 점은 초기 상태, 주황 점은 최종 상태입니다.' : 'Blue marks the initial state and orange marks the final state.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {result.entropyDecrease < 0 && <p role="status">{ko ? '최종 엔트로피가 더 큽니다. 이 부등식에서 양의 열 하한이 나오지 않습니다.' : 'The final entropy is larger. This inequality gives a negative heat lower bound.'}</p>}
    {landauerErasureGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function LandauerErasureConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="landauer-concept-title">
    <h2 id="landauer-concept-title">{ko ? '메모리 엔트로피에서 열 하한까지' : 'From memory entropy to a heat bound'}</h2>
    <p>{landauerErasureGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{landauerErasureGuide.steps[0].formula}</p>
    <p>{landauerErasureGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '초기 확률과 오차 바꾸기' : 'Vary the initial probability and error'}</a>
  </section>;
}
