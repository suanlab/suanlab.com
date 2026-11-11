'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { baselineBudgetsGuide } from '@/data/baseline-budgets-reader';
import { calculateBaselineBudget } from '@/lib/learning-baseline-budgets';
import { WavePlot, waveNumber } from './waves-plot';

export default function BaselineBudgetsLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [budget, setBudget] = useState(4);
  const result = calculateBaselineBudget(budget);
  const curve = Array.from({ length: 64 }, (_, i) => ({ budget: i + 1, ...calculateBaselineBudget(i + 1) }));
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="baseline-budgets-title">
    <h2 id="baseline-budgets-title">{ko ? '같은 예산에서의 두 손실' : 'Compare losses at the same budget'}</h2>
    <label>{ko ? '공통 예산' : 'Shared budget'} b = {budget}
      <input className="block w-full" type="range" min={1} max={64} value={budget} onChange={e => setBudget(Number(e.target.value))}/>
    </label>
    <WavePlot marker="baseline-budgets" title={ko ? '합성 학습 곡선' : 'Synthetic learning curves'}
      xLabel={ko ? '예산' : 'Budget'} yLabel={ko ? '손실' : 'Loss'} xDomain={[1, 64]} yDomain={[.2, 1.05]}
      series={[
        { name: 'A', color: 'blue', segments: [curve.map(row => [row.budget, row.first])] },
        { name: 'B', color: 'orange', segments: [curve.map(row => [row.budget, row.second])] },
      ]}
      dots={[{ color: 'blue', point: [budget, result.first] }, { color: 'orange', point: [budget, result.second] }]}
      caption={ko ? '실제 비교에서는 같은 자원 단위로 예산을 측정합니다.' : 'Measure the budget in the same resource unit for a real comparison.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <dt>{ko ? 'A의 손실' : 'A loss'}</dt><dd>{waveNumber(result.first)}</dd>
      <dt>{ko ? 'B의 손실' : 'B loss'}</dt><dd>{waveNumber(result.second)}</dd>
    </dl>
    {baselineBudgetsGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function BaselineBudgetsConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="baseline-budgets-concept-title">
    <h2 id="baseline-budgets-concept-title">{ko ? '예산과 함께 보고하는 성능' : 'Performance reported with its budget'}</h2>
    <p>{baselineBudgetsGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{baselineBudgetsGuide.steps[0].formula}</p>
    <p>{baselineBudgetsGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '같은 예산에서 비교하기' : 'Compare at a shared budget'}</a>
  </section>;
}
