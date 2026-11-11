'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { photodiodeBandwidthGuide } from '@/data/photodiode-bandwidth-reader';
import { calculatePhotodiodeResponse, calculatePhotodiodeStep } from '@/lib/learning-photodiode-bandwidth';
import { WavePlot, waveNumber } from './waves-plot';

export default function PhotodiodeBandwidthLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [resistance, setResistance] = useState(100);
  const [capacitance, setCapacitance] = useState(50);
  const [frequency, setFrequency] = useState(32);
  const readout = calculatePhotodiodeResponse(resistance * 1e3, capacitance * 1e-12, .5, 10e-6, frequency * 1e3);
  const step = Array.from({ length: 81 }, (_, i): [number, number] => [
    i, calculatePhotodiodeStep(readout.finalVoltage, readout.tau, i * 1e-6),
  ]);
  const response = Array.from({ length: 101 }, (_, i): [number, number] => [
    2 * i, 1 / Math.hypot(1, 2e3 * i / readout.cutoff),
  ]);
  const readings = [
    [ko ? '최종 전압 (V)' : 'Final voltage (V)', readout.finalVoltage],
    [ko ? '차단 주파수 (kHz)' : 'Cutoff frequency (kHz)', readout.cutoff / 1e3],
    [ko ? '10–90% 상승 시간 (µs)' : '10–90% rise time (µs)', readout.riseTime * 1e6],
    [ko ? '상대 진폭' : 'Relative amplitude', readout.relativeAmplitude],
    [ko ? '위상 (°)' : 'Phase (°)', readout.phaseDegrees],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="photodiode-title">
    <h2 id="photodiode-title">{ko ? '수동 광다이오드 읽기 회로' : 'Passive photodiode readout'}</h2>
    <label>R = {resistance} kΩ
      <input className="block w-full" type="range" min={50} max={200} step={25} value={resistance} onChange={e => setResistance(Number(e.target.value))}/>
    </label>
    <label>C = {capacitance} pF
      <input className="block w-full" type="range" min={25} max={100} step={5} value={capacitance} onChange={e => setCapacitance(Number(e.target.value))}/>
    </label>
    <label>f = {frequency} kHz
      <input className="block w-full" type="range" min={0} max={200} value={frequency} onChange={e => setFrequency(Number(e.target.value))}/>
    </label>
    <div className="grid min-w-0 grid-cols-1 gap-4 md:grid-cols-2">
      <WavePlot marker="photodiode-step" title={ko ? '광 파워 계단 응답' : 'Optical power step response'}
        xLabel={ko ? '시간 (µs)' : 'Time (µs)'} yLabel="V" xDomain={[0, 80]} yDomain={[0, 1.05]}
        series={[{ name: ko ? '출력 전압' : 'Output voltage', color: 'blue', segments: [step] }]}
        caption={ko ? '계단 입력은 10 µW, 응답도는 0.5 A/W입니다.' : 'The step is 10 µW and the responsivity is 0.5 A/W.'}/>
      <WavePlot marker="photodiode-frequency" title={ko ? '주파수 응답의 크기' : 'Frequency response magnitude'}
        xLabel="f (kHz)" yLabel="|G| / |G(0)|" xDomain={[0, 200]} yDomain={[0, 1.05]}
        series={[{ name: ko ? '상대 진폭' : 'Relative amplitude', color: 'orange', segments: [response] }]}
        dots={[{ color: 'orange', point: [frequency, readout.relativeAmplitude] }]}
        caption={ko ? '차단 주파수의 상대 진폭은 약 0.707입니다.' : 'The relative amplitude at cutoff is about 0.707.'}/>
    </div>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {photodiodeBandwidthGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function PhotodiodeBandwidthConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="photodiode-concept-title">
    <h2 id="photodiode-concept-title">{ko ? '이득과 응답 속도' : 'Gain and response speed'}</h2>
    <p>{photodiodeBandwidthGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{photodiodeBandwidthGuide.steps[0].formula}</p>
    <p>{photodiodeBandwidthGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '회로의 시간 상수 바꾸기' : 'Change the circuit time constant'}</a>
  </section>;
}
