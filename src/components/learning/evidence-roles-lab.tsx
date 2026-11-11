'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { evidenceRolesGuide } from '@/data/evidence-roles-reader';
import { calculateHoeffdingEvidence } from '@/lib/learning-evidence-roles';
import { WavePlot, waveNumber } from './waves-plot';

export default function EvidenceRolesLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [samples, setSamples] = useState(100);
  const [candidates, setCandidates] = useState(1);
  const result = calculateHoeffdingEvidence(samples, candidates, .05, .18);
  const curve = Array.from({ length: 39 }, (_, i) => ({
    samples: 50 + i * 25,
    one: calculateHoeffdingEvidence(50 + i * 25, 1, .05, .18).radius,
    many: calculateHoeffdingEvidence(50 + i * 25, candidates, .05, .18).radius,
  }));
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="evidence-roles-title">
    <h2 id="evidence-roles-title">{ko ? '관측 평균과 가정에 따른 반지름' : 'The observed mean and a conditional bound'}</h2>
    <label>{ko ? '독립 관측 수' : 'Independent observations'} m = {samples}
      <input className="block w-full" type="range" min={50} max={1000} step={25} value={samples} onChange={e => setSamples(Number(e.target.value))}/>
    </label>
    <label>{ko ? '고정 후보 수' : 'Fixed candidates'} M = {candidates}
      <input className="block w-full" type="range" min={1} max={100} value={candidates} onChange={e => setCandidates(Number(e.target.value))}/>
    </label>
    <WavePlot marker="evidence-roles" title={ko ? '후보 수에 따른 보장 반지름' : 'Bound radius for fixed model families'}
      xLabel="m" yLabel="ε" xDomain={[50, 1000]}
      series={[
        { name: 'M = 1', color: 'neutral', segments: [curve.map(row => [row.samples, row.one])] },
        { name: `M = ${candidates}`, color: 'blue', segments: [curve.map(row => [row.samples, row.many])] },
      ]}
      dots={[{ color: 'blue', point: [samples, result.radius] }]}
      caption={ko ? 'δ는 0.05, 관측 평균은 0.18로 고정합니다.' : 'Delta is fixed at 0.05 and the observed mean at 0.18.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <dt>{ko ? '반지름' : 'Radius'}</dt><dd>{waveNumber(result.radius)}</dd>
      <dt>{ko ? '위험도 범위와 겹친 구간' : 'Interval clipped to the risk range'}</dt><dd>[{waveNumber(result.lower)}, {waveNumber(result.upper)}]</dd>
    </dl>
    {evidenceRolesGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function EvidenceRolesConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="evidence-roles-concept-title">
    <h2 id="evidence-roles-concept-title">{ko ? '가정과 함께 전달하는 보장' : 'A guarantee stated with its assumptions'}</h2>
    <p>{evidenceRolesGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{evidenceRolesGuide.steps[0].formula}</p>
    <p>{evidenceRolesGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '보장할 후보 수 바꾸기' : 'Change the number of models covered'}</a>
  </section>;
}
