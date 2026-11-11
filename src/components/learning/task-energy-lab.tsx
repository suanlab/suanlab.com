'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { taskEnergyGuide } from '@/data/task-energy-reader';
import { calculateTaskEnergy } from '@/lib/learning-task-energy';
import { WavePlot, waveNumber } from './waves-plot';

export default function TaskEnergyLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [idle, setIdle] = useState(20);
  const [tasks, setTasks] = useState(100);
  const slow = calculateTaskEnergy([0, .03, .20], [40, 40, 40], tasks, idle);
  const fast = calculateTaskEnergy([0, .06, .10], [60, 60, 60], tasks, idle);
  const slowLabel = ko ? '긴 실행' : 'Long run';
  const fastLabel = ko ? '짧은 실행' : 'Short run';
  const readings = [
    [ko ? '총 에너지 (J)' : 'Gross energy (J)', slow.gross, fast.gross],
    [ko ? '유휴 대비 에너지 (J)' : 'Incremental energy (J)', slow.incremental, fast.incremental],
    [ko ? '작업당 총 에너지 (mJ)' : 'Gross energy per task (mJ)', 1e3 * slow.grossPerTask, 1e3 * fast.grossPerTask],
    [ko ? '작업당 유휴 대비 에너지 (mJ)' : 'Incremental energy per task (mJ)', 1e3 * slow.incrementalPerTask, 1e3 * fast.incrementalPerTask],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="task-energy-title">
    <h2 id="task-energy-title">{ko ? '두 실행의 전력과 에너지 비교' : 'Power and energy for two runs'}</h2>
    <label>{ko ? '유휴 기준' : 'Idle baseline'} = {idle} W
      <input className="block w-full" type="range" min={0} max={40} value={idle} onChange={e => setIdle(Number(e.target.value))}/>
    </label>
    <label>{ko ? '완료 작업' : 'Completed tasks'} = {tasks}
      <input className="block w-full" type="range" min={10} max={200} step={10} value={tasks} onChange={e => setTasks(Number(e.target.value))}/>
    </label>
    <WavePlot marker="task-energy" title={ko ? '각 실행에서 측정한 전력' : 'Power measured during each run'}
      xLabel={ko ? '시간 (s)' : 'Time (s)'} yLabel={ko ? '전력 (W)' : 'Power (W)'} xDomain={[0, .20]} yDomain={[0, 65]}
      series={[
        { name: slowLabel, color: 'blue', segments: [[[0, 40], [.20, 40]]] },
        { name: fastLabel, color: 'orange', segments: [[[0, 60], [.10, 60]]] },
        { name: ko ? '유휴 기준' : 'Idle baseline', color: 'neutral', segments: [[[0, idle], [.20, idle]]] },
      ]}
      caption={ko ? '실행별 전력선을 실행 종료 시점까지 표시합니다.' : 'Each power trace is shown through the end of its run.'}/>
    <table className="w-full text-left text-sm tabular-nums">
      <caption>{ko ? '실행과 작업 단위의 에너지' : 'Energy per run and per task'}</caption>
      <thead><tr><th scope="col">{ko ? '측정량' : 'Measure'}</th><th scope="col">{slowLabel}</th><th scope="col">{fastLabel}</th></tr></thead>
      <tbody>
        {readings.map(([label, a, b]) => <tr key={label}>
          <th scope="row">{label}</th><td>{waveNumber(a)}</td><td>{waveNumber(b)}</td>
        </tr>)}
      </tbody>
    </table>
    {taskEnergyGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function TaskEnergyConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="task-energy-concept-title">
    <h2 id="task-energy-concept-title">{ko ? '전력 그래프의 넓이' : 'Area under a power graph'}</h2>
    <p>{taskEnergyGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{taskEnergyGuide.steps[0].formula}</p>
    <p>{taskEnergyGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '에너지와 유휴 기준 비교하기' : 'Compare energy and idle baselines'}</a>
  </section>;
}
