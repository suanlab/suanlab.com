'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { pairedSignTestGuide } from '@/data/paired-sign-test-reader';
import { calculatePairedSignTail } from '@/lib/learning-paired-sign-test';
import { WavePlot, waveNumber } from './waves-plot';

export default function PairedSignTestLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [total, setTotal] = useState(6);
  const [positive, setPositive] = useState(6);
  const result = calculatePairedSignTail(total, positive);
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="paired-sign-title">
    <h2 id="paired-sign-title">{ko ? '귀무 부호의 정확한 꼬리확률' : 'Exact tail probability under fair signs'}</h2>
    <label>{ko ? '독립 쌍' : 'Independent pairs'} n = {total}
      <input className="block w-full" type="range" min={2} max={12} value={total} onChange={e => {
        const count = Number(e.target.value);
        setTotal(count);
        setPositive(Math.min(positive, count));
      }}/>
    </label>
    <label>{ko ? '양의 차이' : 'Positive differences'} k = {positive}
      <input className="block w-full" type="range" min={0} max={total} value={positive} onChange={e => setPositive(Number(e.target.value))}/>
    </label>
    <WavePlot marker="paired-sign-test" title={ko ? '공정 부호에서의 양수 개수' : 'Positive count under fair signs'}
      xLabel="k" yLabel={ko ? '확률' : 'Probability'} xDomain={[0, total]}
      series={[
        { name: ko ? '분포 전체' : 'Full distribution', color: 'neutral', segments: result.probabilities.map((p, i) => [[i, 0], [i, p]]) },
        { name: ko ? '꼬리에 포함' : 'Included tail', color: 'orange', segments: result.probabilities.slice(positive).map((p, i) => [[i + positive, 0], [i + positive, p]]) },
      ]}
      caption={ko ? '동률을 제외한 독립 쌍의 부호 확률을 1/2로 둡니다.' : 'Signs of independent non-tied pairs are assigned probability one-half.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <dt>{ko ? '단측 꼬리확률' : 'One-sided tail probability'}</dt><dd>{waveNumber(result.tail)}</dd>
      <dt>{ko ? '양수 비율' : 'Positive proportion'}</dt><dd>{waveNumber(positive / total)}</dd>
    </dl>
    {pairedSignTestGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function PairedSignTestConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="paired-sign-concept-title">
    <h2 id="paired-sign-concept-title">{ko ? '독립 반복과 부호의 증거' : 'Independent repetitions and sign evidence'}</h2>
    <p>{pairedSignTestGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{pairedSignTestGuide.steps[0].formula}</p>
    <p>{pairedSignTestGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '관측한 부호 수 바꾸기' : 'Vary the observed sign count'}</a>
  </section>;
}
