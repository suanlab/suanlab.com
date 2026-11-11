'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { photodiodeNoiseGuide } from '@/data/photodiode-noise-reader';
import { calculatePhotodiodeNoise } from '@/lib/learning-photodiode-noise';
import { WavePlot, waveNumber } from './waves-plot';

export default function PhotodiodeNoiseLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [resistance, setResistance] = useState(100);
  const [current, setCurrent] = useState(1);
  const [temperature, setTemperature] = useState(300);
  const result = calculatePhotodiodeNoise(resistance * 1e3, 50e-12, current * 1e-6, temperature);
  const cutoff = result.cutoff / 1e3;
  const bandwidth = result.equivalentBandwidth / 1e3;
  const curve = Array.from({ length: 81 }, (_, i): [number, number] => [
    4 * cutoff * i / 80, 1 / (1 + (4 * i / 80) ** 2),
  ]);
  const readings = [
    [ko ? '차단 주파수 (kHz)' : 'Cutoff frequency (kHz)', cutoff],
    [ko ? '등가 대역폭 (kHz)' : 'Equivalent bandwidth (kHz)', bandwidth],
    [ko ? '산탄 잡음 분산 (µV²)' : 'Shot variance (µV²)', result.shotVariance * 1e12],
    [ko ? '열잡음 분산 (µV²)' : 'Thermal variance (µV²)', result.thermalVariance * 1e12],
    [ko ? '합산 RMS 전압 (µV)' : 'Combined RMS voltage (µV)', result.rmsVoltage * 1e6],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="photodiode-noise-title">
    <h2 id="photodiode-noise-title">{ko ? '측정 필터를 통과하는 잡음' : 'Noise through the measurement filter'}</h2>
    <label>R = {resistance} kΩ
      <input className="block w-full" type="range" min={50} max={200} step={25} value={resistance} onChange={e => setResistance(Number(e.target.value))}/>
    </label>
    <label>{ko ? '전체 전류' : 'Total current'} I = {current} µA
      <input className="block w-full" type="range" min={0} max={4} step={.25} value={current} onChange={e => setCurrent(Number(e.target.value))}/>
    </label>
    <label>T = {temperature} K
      <input className="block w-full" type="range" min={200} max={400} step={10} value={temperature} onChange={e => setTemperature(Number(e.target.value))}/>
    </label>
    <WavePlot marker="photodiode-noise" title={ko ? '정규화한 잡음 전달과 등가 대역' : 'Normalized noise transfer and equivalent passband'}
      xLabel="f (kHz)" yLabel="|Z(f)/Z(0)|²" xDomain={[0, 4 * cutoff]} yDomain={[0, 1.05]}
      series={[
        { name: ko ? 'RC 필터' : 'RC filter', color: 'blue', segments: [curve] },
        { name: ko ? '등가 대역' : 'Equivalent passband', color: 'orange', segments: [[[0, 1], [bandwidth, 1], [bandwidth, 0], [4 * cutoff, 0]]] },
      ]}
      dots={[{ color: 'blue', point: [cutoff, .5] }]}
      caption={ko ? '정전용량은 50 pF입니다. 등가 대역은 그래프 바깥의 필터 꼬리까지 포함해 계산합니다.' : 'Capacitance is 50 pF. The equivalent bandwidth includes the filter tail beyond the plot.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {photodiodeNoiseGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function PhotodiodeNoiseConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="photodiode-noise-concept-title">
    <h2 id="photodiode-noise-concept-title">{ko ? '잡음 밀도에서 분산까지' : 'From noise density to variance'}</h2>
    <p>{photodiodeNoiseGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{photodiodeNoiseGuide.steps[0].formula}</p>
    <p>{photodiodeNoiseGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '잡음 성분 비교하기' : 'Compare noise contributions'}</a>
  </section>;
}
