'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { minkowskiPathGuide } from '@/data/minkowski-path-reader';
import { compareMinkowskiPaths } from '@/lib/learning-minkowski-path';
import { WavePlot, waveNumber } from './waves-plot';

const labels = {
  timelike: { ko: '시간꼴', en: 'Timelike' },
  null: { ko: '빛꼴', en: 'Null' },
  spacelike: { ko: '공간꼴', en: 'Spacelike' },
};

export default function MinkowskiPathLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [turn, setTurn] = useState({ ct: 3, x: 2 });
  const result = compareMinkowskiPaths(turn);
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="minkowski-experiment-title">
    <h2 id="minkowski-experiment-title">{ko ? '재회하는 두 시계' : 'Two clocks that reunite'}</h2>
    <label>{ko ? '방향 전환 시각' : 'Turnaround time'} ct = {turn.ct}
      <input className="block w-full" type="range" min={.5} max={5.5} step={.25} value={turn.ct}
        onChange={e => setTurn(current => ({ ...current, ct: Number(e.target.value) }))}/>
    </label>
    <label>{ko ? '방향 전환 위치' : 'Turnaround position'} x = {turn.x}
      <input className="block w-full" type="range" min={-4} max={4} step={.25} value={turn.x}
        onChange={e => setTurn(current => ({ ...current, x: Number(e.target.value) }))}/>
    </label>
    <WavePlot marker="minkowski-path" title={ko ? '같은 끝점을 잇는 세계선' : 'Worldlines between common endpoints'}
      xLabel="x" yLabel="ct" xDomain={[-4, 4]} yDomain={[0, 6]}
      series={[
        { name: ko ? '빛꼴 경계' : 'Null boundary', color: 'neutral', segments: [[[0, 0], [-3, 3], [0, 6], [3, 3], [0, 0]]] },
        { name: ko ? '관성 시계' : 'Inertial clock', color: 'blue', segments: [[[0, 0], [0, 6]]] },
        { name: ko ? '선택한 경로' : 'Selected path', color: 'orange', segments: [[[0, 0], [turn.x, turn.ct], [0, 6]]] },
      ]}
      dots={[{ color: 'orange', point: [turn.x, turn.ct] }]}
      caption={ko ? '회색 경계 안의 방향 전환 사건은 두 끝점과 시간꼴로 이어집니다. 경계에서는 한 구간 이상이 빛꼴입니다.' : 'A turnaround inside the gray boundary connects to both endpoints by timelike legs. On the boundary at least one leg is null.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <div className="contents"><dt>{ko ? '가는 구간' : 'Outgoing leg'}</dt><dd>{labels[result.outgoing.kind][language]} · s²={waveNumber(result.outgoing.squared)}</dd></div>
      <div className="contents"><dt>{ko ? '돌아오는 구간' : 'Return leg'}</dt><dd>{labels[result.returning.kind][language]} · s²={waveNumber(result.returning.squared)}</dd></div>
      <div className="contents"><dt>{ko ? '관성 경로의 고유 시간' : 'Inertial proper time'}</dt><dd>{result.inertialProperTime}</dd></div>
      <div className="contents"><dt>{ko ? '선택한 경로의 고유 시간' : 'Selected path proper time'}</dt><dd>{waveNumber(result.properTime)}</dd></div>
    </dl>
    {minkowskiPathGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function MinkowskiPathConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="minkowski-concept-title">
    <h2 id="minkowski-concept-title">{ko ? '간격의 부호와 시계의 경로' : 'The sign of an interval and the clock path'}</h2>
    <p>{minkowskiPathGuide.intuition[language]}</p>
    <p className="ai-concept-equation">s² = (Δct)² − (Δx)²</p>
    <p>{minkowskiPathGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '경로별 시계 읽기 비교' : 'Compare clock readings by path'}</a>
  </section>;
}
