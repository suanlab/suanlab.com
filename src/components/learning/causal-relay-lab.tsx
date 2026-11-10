'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { causalRelayGuide } from '@/data/causal-relay-reader';
import { calculateCausalRelay } from '@/lib/learning-causal-relay';
import { WavePlot, waveNumber, type Series } from './waves-plot';

const controls = [
  { key: 'deadline', ko: '도착 위치 b', en: 'Deadline position b', min: -7, max: 7, step: .25 },
  { key: 'time', ko: '중계 시각 t', en: 'Relay time t', min: 0, max: 6, step: .25 },
  { key: 'position', ko: '중계 위치 x', en: 'Relay position x', min: -7, max: 7, step: .25 },
  { key: 'beta', ko: '비교 관성계 β', en: 'Comparison frame β', min: -.9, max: .9, step: .05 },
] as const;

export default function CausalRelayLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [input, setInput] = useState({ deadline: 2, time: 3, position: 1, beta: 0 });
  const { deadline, time, position, beta } = input;
  const result = calculateCausalRelay(deadline, time, position, beta);
  const series: Series[] = [
    { name: ko ? '송신의 미래 원뿔' : 'Emission future cone', color: 'blue', segments: [[[-6, 6], [0, 0], [6, 6]]] },
    { name: ko ? '기한의 과거 원뿔' : 'Deadline past cone', color: 'orange', segments: [[[deadline - 6, 0], [deadline, 6], [deadline + 6, 0]]] },
  ];
  if (result.interval) {
    const { lower, upper } = result.interval;
    series.push({ name: ko ? '가능한 중계 범위' : 'Feasible relay interval', color: 'green', segments: [lower === upper ? [[lower, time]] : [[lower, time], [upper, time]]] });
  }
  const verdict = (ok: boolean) => ok ? (ko ? '가능' : 'Feasible') : (ko ? '불가' : 'Infeasible');
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="relay-experiment-title">
    <h2 id="relay-experiment-title">{ko ? '중계 가능한 위치 찾기' : 'Find feasible relay positions'}</h2>
    <div className="flex flex-wrap gap-2">
      {[2, 6, 7].map(value => <button key={value} type="button" className="rounded border border-border px-2 py-1"
        onClick={() => setInput(current => ({ ...current, deadline: value }))}>b = {value}</button>)}
    </div>
    {controls.map(({ key, min, max, step, ...labels }) => <label key={key}>{labels[language]} = {input[key]}
      <input className="block w-full" type="range" min={min} max={max} step={step} value={input[key]}
        onChange={e => setInput(current => ({ ...current, [key]: Number(e.target.value) }))}/>
    </label>)}
    <WavePlot marker="causal-relay" title={ko ? '송신과 도착을 잇는 원뿔' : 'Cones for emission and deadline'}
      xLabel="x" yLabel="ct" xDomain={[-13, 13]} yDomain={[0, 6]} series={series}
      dots={[{ color: 'neutral', point: [position, time] }]}
      caption={ko ? '그림은 S의 좌표입니다. 회색 점이 선택한 중계기이며 β는 기한의 비교 관성계 시각에만 적용합니다.' : 'The diagram uses frame S. The gray dot is the selected relay, and β changes the reported deadline time in the comparison frame.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <div className="contents"><dt>{ko ? '가능한 중계 위치' : 'Feasible relay positions'}</dt><dd>{result.interval ? `[${waveNumber(result.interval.lower)}, ${waveNumber(result.interval.upper)}]` : '∅'}</dd></div>
      <div className="contents"><dt>{ko ? '송신 → 중계' : 'Emission → relay'}</dt><dd>{verdict(result.reachableFromSender)}</dd></div>
      <div className="contents"><dt>{ko ? '중계 → 기한' : 'Relay → deadline'}</dt><dd>{verdict(result.reachesDeadline)}</dd></div>
      <div className="contents"><dt>{ko ? '비교 관성계의 기한' : 'Deadline in comparison frame'} ct′</dt><dd>{waveNumber(result.deadlineTimeInBoostedFrame)}</dd></div>
    </dl>
    {causalRelayGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function CausalRelayConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="relay-concept-title">
    <h2 id="relay-concept-title">{ko ? '두 원뿔에서 동시에 허용되는 사건' : 'Events allowed by both cones'}</h2>
    <p>{causalRelayGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{causalRelayGuide.steps[0].formula}</p>
    <p>{causalRelayGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '교집합 찾기' : 'Find the intersection'}</a>
  </section>;
}
