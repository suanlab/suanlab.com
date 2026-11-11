'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { aggregationProofGuide } from '@/data/aggregation-proof-reader';
import { calculateAggregationEffects } from '@/lib/learning-aggregation-proof';
import { WavePlot, waveNumber } from './waves-plot';

export default function AggregationProofLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [secondCount, setSecondCount] = useState(9);
  const result = calculateAggregationEffects([1, secondCount], [.4, -.1]);
  const curve = Array.from({ length: 20 }, (_, i): [number, number] => [
    i + 1, calculateAggregationEffects([1, i + 1], [.4, -.1]).observationMean,
  ]);
  const readings = [
    [ko ? '사건 가중 평균 (K)' : 'Episode mean (K)', result.episodeMean],
    [ko ? '시점 가중 평균 (K)' : 'Point mean (K)', result.observationMean],
    [ko ? '횟수와 효과의 공분산' : 'Covariance of count and effect', result.covariance],
    [ko ? '평균 횟수' : 'Mean count', result.meanCount],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="aggregation-proof-title">
    <h2 id="aggregation-proof-title">{ko ? '같은 사건, 다른 집계' : 'The same episodes with different aggregation'}</h2>
    <label>n₂ = {secondCount}
      <input className="block w-full" type="range" min={1} max={20} value={secondCount} onChange={e => setSecondCount(Number(e.target.value))}/>
    </label>
    <WavePlot marker="aggregation-proof" title={ko ? '관측 횟수에 따른 가중평균' : 'Weighted mean by observation count'}
      xLabel="n₂" yLabel="K" xDomain={[1, 20]} yDomain={[-.12, .2]}
      series={[
        { name: ko ? '시점 가중' : 'Point weighted', color: 'blue', segments: [curve] },
        { name: ko ? '사건 가중' : 'Episode weighted', color: 'orange', segments: [[[1, .15], [20, .15]]] },
      ]}
      dots={[{ color: 'blue', point: [secondCount, result.observationMean] }]}
      caption={ko ? '첫 사건 횟수는 1, 사건별 효과는 0.4 K와 −0.1 K입니다.' : 'The first count is one and the episode effects are 0.4 K and −0.1 K.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {aggregationProofGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function AggregationProofConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="aggregation-proof-concept-title">
    <h2 id="aggregation-proof-concept-title">{ko ? '가정이 결론을 바꾸는 예' : 'How assumptions change a conclusion'}</h2>
    <p>{aggregationProofGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{aggregationProofGuide.steps[0].formula}</p>
    <p>{aggregationProofGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '횟수 가중치 바꾸기' : 'Change the observation weights'}</a>
  </section>;
}
