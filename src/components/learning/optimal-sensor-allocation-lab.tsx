'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { optimalSensorAllocationGuide } from '@/data/optimal-sensor-allocation-reader';
import { calculateSensorAllocations } from '@/lib/learning-optimal-sensor-allocation';
import { WavePlot, waveNumber } from './waves-plot';

export default function OptimalSensorAllocationLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [budget, setBudget] = useState(12);
  const [first, setFirst] = useState(6);
  const [noise, setNoise] = useState(3);
  const [criterion, setCriterion] = useState<'trace' | 'determinant'>('trace');
  const { plans, traceBest, determinantBest } = calculateSensorAllocations(budget, 1, noise);
  const best = criterion === 'trace' ? traceBest : determinantBest;
  const selected = plans[first - 1];
  const label = criterion === 'trace' ? (ko ? '대각합' : 'Trace') : (ko ? '행렬식' : 'Determinant');
  const readings = [
    ['Var(â)', selected.varianceA], ['Var(b̂)', selected.varianceB],
    [ko ? '대각합' : 'Trace', selected.trace], [ko ? '행렬식' : 'Determinant', selected.determinant],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="sensor-allocation-title">
    <h2 id="sensor-allocation-title">{ko ? '관측 횟수를 어떻게 나눌까요?' : 'How should we allocate readings?'}</h2>
    <label>N = {budget}
      <input className="block w-full" type="range" min={4} max={30} value={budget} onChange={e => {
        const total = Number(e.target.value);
        setBudget(total);
        setFirst(Math.min(first, total - 1));
      }}/>
    </label>
    <label>nₐ = {first}
      <input className="block w-full" type="range" min={1} max={budget - 1} value={first} onChange={e => setFirst(Number(e.target.value))}/>
    </label>
    <label>σᵦ = {noise}
      <input className="block w-full" type="range" min={.5} max={4} step={.5} value={noise} onChange={e => setNoise(Number(e.target.value))}/>
    </label>
    <div className="flex flex-wrap items-center gap-3">
      <label>{ko ? '설계 기준' : 'Design criterion'}{' '}
        <select className="rounded border border-border bg-background p-1" value={criterion} onChange={e => setCriterion(e.target.value === 'trace' ? 'trace' : 'determinant')}>
          <option value="trace">{ko ? '대각합' : 'Trace'}</option>
          <option value="determinant">{ko ? '행렬식' : 'Determinant'}</option>
        </select>
      </label>
      <button type="button" className="rounded border border-border px-2 py-1" onClick={() => setFirst(best.first)}>
        {ko ? '최적 배분 적용' : 'Apply optimal allocation'}
      </button>
    </div>
    <WavePlot marker="optimal-sensor-allocation" title={ko ? '관측 배분과 설계 기준' : 'Design criterion by measurement count'}
      xLabel="nₐ" yLabel={label} xDomain={[1, budget - 1]}
      series={[{ name: label, color: 'blue', segments: [plans.map(plan => [plan.first, plan[criterion]])] }]}
      dots={[{ color: 'orange', point: [first, selected[criterion]] }]}
      caption={ko ? '센서 A의 표준편차는 1입니다.' : 'Sensor A has noise standard deviation one.'}/>
    <p role="status">{ko ? '현재 배분' : 'Current allocation'} ({first}, {budget - first}) · {ko ? '최적 배분 한 가지' : 'An optimal allocation'} ({best.first}, {best.second})</p>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {optimalSensorAllocationGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function OptimalSensorAllocationConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="sensor-allocation-concept-title">
    <h2 id="sensor-allocation-concept-title">{ko ? '무엇을 최소화할까요?' : 'What should we minimize?'}</h2>
    <p>{optimalSensorAllocationGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{optimalSensorAllocationGuide.steps[0].formula}</p>
    <p>{optimalSensorAllocationGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '두 기준의 최적 배분 비교하기' : 'Compare optimal allocations'}</a>
  </section>;
}
