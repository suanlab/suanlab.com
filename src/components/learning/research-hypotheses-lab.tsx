'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { researchHypothesesGuide } from '@/data/research-hypotheses-reader';
import { calculateResearchEffect } from '@/lib/learning-research-hypotheses';
import { WavePlot, waveNumber } from './waves-plot';

const baseline = [.65, .52, .44, .60, .55, .50];
const examples = [
  [.52, .38, .36, .42, .48, .47],
  [.34, .58, .26, .38, .65, .42],
];

export default function ResearchHypothesesLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [example, setExample] = useState(0);
  const [threshold, setThreshold] = useState(.1);
  const result = calculateResearchEffect(baseline, examples[example]);
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="research-hypotheses-title">
    <h2 id="research-hypotheses-title">{ko ? '실용 기준과 쌍별 개선량' : 'Paired improvement and the practical threshold'}</h2>
    <label>{ko ? '합성 평가' : 'Synthetic evaluation'}{' '}
      <select className="rounded border border-border bg-background p-1" value={example} onChange={e => setExample(Number(e.target.value))}>
        <option value={0}>{ko ? '작은 사건별 변동' : 'Lower episode variability'}</option>
        <option value={1}>{ko ? '큰 사건별 변동' : 'Higher episode variability'}</option>
      </select>
    </label>
    <label>{ko ? '실용 기준' : 'Practical threshold'} = {threshold} K
      <input className="block w-full" type="range" min={0} max={.2} step={.01} value={threshold} onChange={e => setThreshold(Number(e.target.value))}/>
    </label>
    <WavePlot marker="research-hypotheses" title={ko ? '재시작 사건별 개선량' : 'Improvement by restart episode'}
      xLabel={ko ? '사건' : 'Episode'} yLabel="K" xDomain={[1, 6]} yDomain={[-.15, .35]}
      series={[
        { name: ko ? '사건별 차이' : 'Paired effect', color: 'blue', segments: result.differences.map((value, i) => [[i + 1, value]]) },
        { name: ko ? '실용 기준' : 'Practical threshold', color: 'orange', segments: [[[1, threshold], [6, threshold]]] },
      ]}
      caption={ko ? '실제 연구에서는 최종 결과를 보기 전에 기준을 정합니다.' : 'Set the threshold before observing final outcomes in a real evaluation.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <dt>{ko ? '평균 개선량 (K)' : 'Mean improvement (K)'}</dt><dd>{waveNumber(result.average)}</dd>
      <dt>{ko ? '쌍별 표준오차 (K)' : 'Paired standard error (K)'}</dt><dd>{waveNumber(result.standardError)}</dd>
    </dl>
    <p role="status">{result.average > threshold
      ? (ko ? '점추정값이 기준을 넘습니다.' : 'The point estimate exceeds the threshold.')
      : (ko ? '점추정값이 기준 이하입니다.' : 'The point estimate is at or below the threshold.')}</p>
    {researchHypothesesGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function ResearchHypothesesConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="research-hypotheses-concept-title">
    <h2 id="research-hypotheses-concept-title">{ko ? '검증 가능한 개선 주장' : 'A testable improvement claim'}</h2>
    <p>{researchHypothesesGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{researchHypothesesGuide.steps[0].formula}</p>
    <p>{researchHypothesesGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '점추정과 실용 기준 비교하기' : 'Compare estimates and practical thresholds'}</a>
  </section>;
}
