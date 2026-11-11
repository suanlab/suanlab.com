'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { energyMeasurementGuide } from '@/data/energy-measurement-reader';
import { calculatePairedEnergy } from '@/lib/learning-energy-measurement';
import { WavePlot, waveNumber } from './waves-plot';

const baselineA = [60, 62, 59, 61, 63, 60];
const baselineB = [55, 58, 56, 56, 59, 55];
const sharedOffsets = [2, -3, 1, 4, -2, 3];

export default function EnergyMeasurementLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [offset, setOffset] = useState(0);
  const [blocks, setBlocks] = useState(6);
  const [calibration, setCalibration] = useState(1);
  const sample = (values: number[]) => values.slice(0, blocks).map((value, i) => value + offset * sharedOffsets[i]);
  const a = sample(baselineA);
  const b = sample(baselineB);
  const result = calculatePairedEnergy(a, b, calibration / 100);
  const points = (values: number[]) => values.map((value, i): [number, number] => [i + 1, value]);
  const readings = [
    [ko ? '쌍별 평균 차이 (mJ)' : 'Mean paired difference (mJ)', result.mean],
    [ko ? '반복 표준오차 (mJ)' : 'Repeatability standard error (mJ)', result.standardError],
    [ko ? '공통 교정 불확실성 (mJ)' : 'Shared calibration uncertainty (mJ)', result.calibrationUncertainty],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="energy-measurement-title">
    <h2 id="energy-measurement-title">{ko ? '같은 조건에서 두 구현을 재기' : 'Measure two implementations in matched blocks'}</h2>
    <label>{ko ? '가산 오프셋 배율' : 'Additive offset multiplier'} = {offset}
      <input className="block w-full" type="range" min={0} max={3} value={offset} onChange={e => setOffset(Number(e.target.value))}/>
    </label>
    <label>{ko ? '측정 블록 수' : 'Measurement blocks'} = {blocks}
      <input className="block w-full" type="range" min={2} max={6} value={blocks} onChange={e => setBlocks(Number(e.target.value))}/>
    </label>
    <label>{ko ? '교정 불확실성' : 'Calibration uncertainty'} = {calibration}%
      <input className="block w-full" type="range" min={0} max={5} step={.5} value={calibration} onChange={e => setCalibration(Number(e.target.value))}/>
    </label>
    <WavePlot marker="energy-measurement" title={ko ? '블록별 가상 에너지 측정값' : 'Synthetic energy observations by block'}
      xLabel={ko ? '측정 블록' : 'Measurement block'} yLabel="mJ" xDomain={[1, 6]} yDomain={[45, 75]}
      series={[
        { name: 'A', color: 'blue', segments: [points(a)] },
        { name: 'B', color: 'orange', segments: [points(b)] },
      ]}
      caption={ko ? '각 블록의 같은 오프셋이 두 실행의 수치에 더해집니다.' : 'The same block offset is added to both observations.'}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {energyMeasurementGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function EnergyMeasurementConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="energy-measurement-concept-title">
    <h2 id="energy-measurement-concept-title">{ko ? '차이를 먼저 계산하는 실험' : 'An experiment that starts with paired differences'}</h2>
    <p>{energyMeasurementGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{energyMeasurementGuide.steps[0].formula}</p>
    <p>{energyMeasurementGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '공통 오프셋과 교정 오차 조절하기' : 'Adjust the shared offset and calibration'}</a>
  </section>;
}
