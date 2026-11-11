'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { contributionEvidenceGuide } from '@/data/contribution-evidence-reader';
import { calculateContributionEvidence } from '@/lib/learning-contribution-evidence';
import { WavePlot, waveNumber } from './waves-plot';

export default function ContributionEvidenceLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [longPhysics, setLongPhysics] = useState(.4);
  const result = calculateContributionEvidence(.6, .5, .54, longPhysics);
  const readings = [
    [ko ? '짧은 구간 개선 (K)' : 'Short-context gain (K)', result.shortGain],
    [ko ? '긴 구간 개선 (K)' : 'Long-context gain (K)', result.longGain],
    [ko ? '개선량의 차 (K)' : 'Difference of effects (K)', Math.abs(result.interaction) < 1e-12 ? 0 : result.interaction],
    [ko ? '두 변경의 전체 개선 (K)' : 'Combined gain (K)', result.combinedGain],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="contribution-evidence-title">
    <h2 id="contribution-evidence-title">{ko ? '관측 구간을 맞춘 비교' : 'Compare at matched context lengths'}</h2>
    <label>{ko ? '긴 구간의 열수지 모형 오차' : 'Long-context physical model error'} = {longPhysics} K
      <input className="block w-full" type="range" min={.3} max={.6} step={.01} value={longPhysics} onChange={e => setLongPhysics(Number(e.target.value))}/>
    </label>
    <WavePlot marker="contribution-evidence" title={ko ? '합성 평가 오차 비교' : 'Synthetic evaluation errors'}
      xLabel={ko ? '관측 구간 (1 짧음, 2 긺)' : 'Context (1 short, 2 long)'} yLabel="K" xDomain={[1, 2]} yDomain={[.25, .65]}
      series={[
        { name: ko ? '기준선' : 'Baseline', color: 'blue', segments: [[[1, .6], [2, .5]]] },
        { name: ko ? '열수지 항 포함' : 'With physical term', color: 'orange', segments: [[[1, .54], [2, longPhysics]]] },
      ]}
      caption={ko ? '같은 관측 구간에 놓인 두 점의 차가 조건을 맞춘 개선량입니다.' : 'The gap within each context is its matched improvement.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {contributionEvidenceGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function ContributionEvidenceConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="contribution-evidence-concept-title">
    <h2 id="contribution-evidence-concept-title">{ko ? '주장에 맞는 비교 고르기' : 'Choose a comparison that tests the claim'}</h2>
    <p>{contributionEvidenceGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{contributionEvidenceGuide.steps[0].formula}</p>
    <p>{contributionEvidenceGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '조건을 맞춰 개선량 비교하기' : 'Compare gains under matched conditions'}</a>
  </section>;
}
