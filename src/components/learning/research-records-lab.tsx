'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { researchRecordsGuide } from '@/data/research-records-reader';
import { calculateResearchRecordSummary } from '@/lib/learning-research-records';
import { WavePlot, waveNumber } from './waves-plot';

const losses = [.42, .55, .50, .60, .48, .57, .45, .65, .53, .59];

export default function ResearchRecordsLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [selected, setSelected] = useState(1);
  const result = calculateResearchRecordSummary(losses, selected);
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="research-records-title">
    <h2 id="research-records-title">{ko ? '보고할 결과와 보존한 실행' : 'Reported results and retained runs'}</h2>
    <label>{ko ? '선택한 완료 실행 수' : 'Selected completed runs'} = {selected}
      <input className="block w-full" type="range" min={1} max={10} value={selected} onChange={e => setSelected(Number(e.target.value))}/>
    </label>
    <WavePlot marker="research-records" title={ko ? '오차순으로 정렬한 합성 검증 실행' : 'Synthetic validation runs sorted by loss'}
      xLabel={ko ? '손실 순위' : 'Loss rank'} yLabel={ko ? '손실' : 'Loss'} xDomain={[1, 10]} yDomain={[.35, .7]}
      series={[
        { name: ko ? '완료 실행' : 'Completed runs', color: 'neutral', segments: result.ordered.map((value, i) => [[i + 1, value]]) },
        { name: ko ? '선택 실행' : 'Selected runs', color: 'orange', segments: result.selected.map((value, i) => [[i + 1, value]]) },
      ]}
      caption={ko ? '모든 실행은 보존하고 선택한 결과의 요약을 따로 계산합니다.' : 'All runs are retained and the selected subset is summarized separately.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <dt>{ko ? '선택 평균' : 'Selected mean'}</dt><dd>{waveNumber(result.selectedMean)}</dd>
      <dt>{ko ? '전체 평균' : 'Overall mean'}</dt><dd>{waveNumber(result.allMean)}</dd>
      <dt>{ko ? '보존한 완료 실행 수' : 'Retained completed runs'}</dt><dd>{result.completedCount}</dd>
    </dl>
    {researchRecordsGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function ResearchRecordsConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="research-records-concept-title">
    <h2 id="research-records-concept-title">{ko ? '결과에서 근거로 돌아가는 경로' : 'A path from the result back to its evidence'}</h2>
    <p>{researchRecordsGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{researchRecordsGuide.steps[0].formula}</p>
    <p>{researchRecordsGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '선택 기준과 결과 요약 비교하기' : 'Compare selection and reported summaries'}</a>
  </section>;
}
