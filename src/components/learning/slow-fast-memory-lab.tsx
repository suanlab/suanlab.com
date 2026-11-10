'use client';

import { useState } from 'react';
import { slowFastMemoryGuide } from '@/data/slow-fast-memory-reader';
import { useLanguage } from '@/components/language-provider';
import { calculateSlowFastMemory } from '@/lib/learning-slow-fast-memory';
import { WavePlot, waveNumber } from './waves-plot';

export default function SlowFastMemoryLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [fastRate, setFastRate] = useState(4);
  const [mixing, setMixing] = useState(.6);
  const [preparation, setPreparation] = useState('cold');
  const [time, setTime] = useState(.5);
  const [zoom, setZoom] = useState(false);
  const initialFast = preparation === 'cold' ? 0 : mixing + (preparation === 'excess' ? 1 : 0);
  const observe = (time: number) => calculateSlowFastMemory({ fastRate, mixing, initialFast, time });
  const point = observe(time);
  const plotLimit = zoom ? 1 : 6;
  const samples = Array.from({ length: 81 }, (_, i) => {
    const t = plotLimit * i / 80;
    return { t, ...observe(t) };
  });
  const stats = [
    [ko ? '전체 계의 느린 감쇠율' : 'Slow decay rate of the full system', -point.slowEigenvalue],
    [ko ? '축소 감쇠율' : 'Reduced decay rate', 1 - mixing],
    [ko ? '빠른 감쇠율과 느린 감쇠율의 비' : 'Ratio of fast to slow decay rates', point.fastEigenvalue / point.slowEigenvalue],
    [ko ? '선택한 시점의 궤적 오차' : 'Trajectory error at the selected time', point.reductionError],
    [ko ? '잔차 |y − γx|' : 'Residual |y − γx|', point.closureResidual],
    [ko ? '잔차 상계' : 'Residual bound', point.closureBound],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="slow-fast-experiment-title">
    <h2 id="slow-fast-experiment-title">{ko ? '빠른 변수의 기억과 초기 조건' : 'Memory and initialization of the fast variable'}</h2>
    <label>{ko ? '빠른 변수의 감쇠 계수' : 'Fast variable decay coefficient'} D = {fastRate}
      <input className="block w-full" type="range" min={2} max={64} value={fastRate} onChange={e => setFastRate(Number(e.target.value))}/>
    </label>
    <label>{ko ? '혼합 계수' : 'Mixing coefficient'} γ = {mixing}
      <input className="block w-full" type="range" min={0} max={.95} step={.05} value={mixing} onChange={e => setMixing(Number(e.target.value))}/>
    </label>
    <label>{ko ? '숨은 초기값' : 'Hidden initial value'} y₀
      <select className="ml-2 rounded border border-border bg-background p-1" value={preparation} onChange={e => setPreparation(e.target.value)}>
        <option value="cold">0</option><option value="prepared">γ</option><option value="excess">γ + 1</option>
      </select>
    </label>
    <label>{ko ? '관측 시점' : 'Observation time'} t = {time}
      <input className="block w-full" type="range" min={0} max={6} step={.025} value={time} onChange={e => setTime(Number(e.target.value))}/>
    </label>
    <label className="flex items-center gap-2">
      <input type="checkbox" checked={zoom} onChange={e => setZoom(e.target.checked)}/>
      {ko ? '초기 구간 확대' : 'Zoom into the initial interval'}
    </label>
    <WavePlot marker="slow-fast-memory" title={ko ? '느린 변수의 전체 궤적과 축소 궤적' : 'Full and reduced slow-variable trajectories'}
      xLabel="t" yLabel="x(t)" xDomain={[0, plotLimit]} vertical={[1 / fastRate]}
      series={[
        { name: ko ? '전체 계' : 'Full system', color: 'blue', segments: [samples.map(row => [row.t, row.slow])] },
        { name: ko ? '순간 응답 근사' : 'Instantaneous approximation', color: 'orange', segments: [samples.map(row => [row.t, row.reduced])] },
      ]}
      dots={time > plotLimit ? [] : [{ color: 'blue', point: [time, point.slow] }, { color: 'orange', point: [time, point.reduced] }]}
      caption={ko ? '점선은 기억 커널의 시간 폭 1/D입니다. 전체 계의 느린 감쇠율은 고윳값에서 계산합니다.' : 'The dashed line marks the memory-kernel width 1/D. The full system’s slow decay rate comes from its eigenvalues.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {stats.map(([name, value]) => <div className="contents" key={name}><dt>{name}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {slowFastMemoryGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function SlowFastMemoryConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  const [hidden, setHidden] = useState(0);
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="slow-fast-concept-title">
    <h2 id="slow-fast-concept-title">{ko ? '같은 느린 값에서 다른 기울기' : 'Different slopes at the same slow value'}</h2>
    <p>{slowFastMemoryGuide.intuition[language]}</p>
    <label>{ko ? '숨은 초기값' : 'Hidden initial value'} y₀
      <select className="ml-2 rounded border border-border bg-background p-1" value={hidden} onChange={e => setHidden(Number(e.target.value))}>
        <option>0</option><option>2</option>
      </select>
    </label>
    <p className="ai-concept-equation" aria-live="polite">x₀ = 1, ẋ(0) = {hidden - 1}</p>
    <p>{slowFastMemoryGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '전체 궤적과 근사 비교하기' : 'Compare the full and reduced trajectories'}</a>
  </section>;
}
