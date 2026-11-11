'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { memoryRooflineGuide } from '@/data/memory-roofline-reader';
import { calculateMemoryRoofline } from '@/lib/learning-memory-roofline';
import { WavePlot, waveNumber } from './waves-plot';

export default function MemoryRooflineLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [traffic, setTraffic] = useState(100);
  const [peak, setPeak] = useState(8);
  const [bandwidth, setBandwidth] = useState(200);
  const result = calculateMemoryRoofline(1e9, traffic * 1e6, peak * 1e12, bandwidth * 1e9);
  const names = {
    memory: ko ? '메모리 대역폭 제한' : 'Memory bandwidth limited',
    compute: ko ? '연산 처리율 제한' : 'Compute throughput limited',
    balanced: ko ? '두 상한이 만나는 지점' : 'At the ridge',
  };
  const readings = [
    [ko ? '연산 집약도 (FLOP/byte)' : 'Operational intensity (FLOP/byte)', result.intensity],
    [ko ? '교차점 (FLOP/byte)' : 'Ridge (FLOP/byte)', result.ridge],
    [ko ? '연산 시간 (ms)' : 'Compute time (ms)', 1e3 * result.computeTime],
    [ko ? '전송 시간 (ms)' : 'Transfer time (ms)', 1e3 * result.transferTime],
    [ko ? '실행 시간 하한 (ms)' : 'Runtime floor (ms)', 1e3 * result.runtimeFloor],
    [ko ? '처리율 상한 (TFLOP/s)' : 'Throughput ceiling (TFLOP/s)', result.throughputCeiling / 1e12],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="roofline-title">
    <h2 id="roofline-title">{ko ? '연산과 메모리 전송의 성능 상한' : 'Performance ceilings from compute and memory'}</h2>
    <label>{ko ? '전송량' : 'Traffic'} D = {traffic} MB
      <input className="block w-full" type="range" min={10} max={200} step={5} value={traffic} onChange={e => setTraffic(Number(e.target.value))}/>
    </label>
    <label>P = {peak} TFLOP/s
      <input className="block w-full" type="range" min={2} max={12} value={peak} onChange={e => setPeak(Number(e.target.value))}/>
    </label>
    <label>B = {bandwidth} GB/s
      <input className="block w-full" type="range" min={100} max={400} step={25} value={bandwidth} onChange={e => setBandwidth(Number(e.target.value))}/>
    </label>
    <WavePlot marker="memory-roofline" title={ko ? '연산 집약도와 처리율 상한' : 'Operational intensity and throughput ceiling'}
      xLabel="FLOP/byte" yLabel="TFLOP/s" xDomain={[0, 140]} yDomain={[0, 13]}
      series={[
        { name: ko ? '대역폭 구간' : 'Bandwidth-limited region', color: 'blue', segments: [[[0, 0], [result.ridge, peak]]] },
        { name: ko ? '연산 구간' : 'Compute-limited region', color: 'orange', segments: [[[result.ridge, peak], [140, peak]]] },
      ]}
      dots={[{ color: 'neutral', point: [result.intensity, result.throughputCeiling / 1e12] }]}
      caption={ko ? '연산 수는 10억 FLOP으로 고정합니다. 두 축은 선형 눈금입니다.' : 'Operation count is fixed at one billion FLOPs. Both axes use linear scales.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    <p role="status">{names[result.bottleneck]}</p>
    {memoryRooflineGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function MemoryRooflineConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="roofline-concept-title">
    <h2 id="roofline-concept-title">{ko ? '두 과정 중 더 오래 걸리는 시간' : 'The time required by the slower process'}</h2>
    <p>{memoryRooflineGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{memoryRooflineGuide.steps[0].formula}</p>
    <p>{memoryRooflineGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '전송량을 바꿔 병목 확인하기' : 'Vary traffic and identify the bottleneck'}</a>
  </section>;
}
