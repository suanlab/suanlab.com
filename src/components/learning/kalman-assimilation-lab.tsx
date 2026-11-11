'use client';

import { useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { kalmanAssimilationGuide } from '@/data/kalman-assimilation-reader';
import { calculateKalmanAssimilation } from '@/lib/learning-kalman-assimilation';
import { WavePlot, waveNumber } from './waves-plot';

export default function KalmanAssimilationLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [variance, setVariance] = useState(.25);
  const [observed, setObserved] = useState(false);
  const [step, setStep] = useState(1);
  const rows = calculateKalmanAssimilation(0, 1, .9, .2, variance, [1, observed ? .8 : null, .2]);
  const current = rows[step - 1];
  const readings = [
    [ko ? '관측값' : 'Observation', current.observation],
    [ko ? '예측 평균' : 'Forecast mean', current.forecastMean],
    [ko ? '분석 평균' : 'Analysis mean', current.mean],
    [ko ? '예측 분산' : 'Forecast variance', current.forecastVariance],
    [ko ? '분석 분산' : 'Analysis variance', current.variance],
    [ko ? '칼만 이득' : 'Kalman gain', current.gain],
  ] as const;
  return <section id="experiment" className="grid min-w-0 grid-cols-1 gap-4" aria-labelledby="kalman-assimilation-title">
    <h2 id="kalman-assimilation-title">{ko ? '관측이 빠진 단계에서의 예측과 분석' : 'Forecast and analysis through a missing reading'}</h2>
    <label>{ko ? '관측 잡음 분산' : 'Observation noise variance'} r = {variance}
      <input className="block w-full" type="range" min={.05} max={1} step={.05} value={variance} onChange={e => setVariance(Number(e.target.value))}/>
    </label>
    <label>{ko ? '표시할 단계' : 'Step to inspect'} = {step}
      <input className="block w-full" type="range" min={1} max={3} value={step} onChange={e => setStep(Number(e.target.value))}/>
    </label>
    <label className="flex items-center gap-2">
      <input type="checkbox" checked={observed} onChange={e => setObserved(e.target.checked)}/>
      {ko ? '두 번째 관측 포함' : 'Include the second reading'}
    </label>
    <WavePlot marker="kalman-assimilation" title={ko ? '단계별 예측 분산과 분석 분산' : 'Forecast and analysis variance by step'}
      xLabel={ko ? '단계' : 'Step'} yLabel={ko ? '분산' : 'Variance'} xDomain={[1, 3]} yDomain={[0, 1.1]}
      series={[
        { name: ko ? '예측' : 'Forecast', color: 'blue', segments: [rows.map((row, i) => [i + 1, row.forecastVariance])] },
        { name: ko ? '분석' : 'Analysis', color: 'orange', segments: [rows.map((row, i) => [i + 1, row.variance])] },
      ]}
      dots={[{ color: 'blue', point: [step, current.forecastVariance] }, { color: 'orange', point: [step, current.variance] }]}
      caption={ko ? '동역학의 a는 0.9, 과정 잡음 분산 q는 0.2입니다.' : 'The dynamics use a=0.9 and process variance q=0.2.'}/>
    {current.observation === null && <p role="status">{ko ? '이 단계는 관측 없이 예측을 전달합니다.' : 'This step carries the forecast forward without an observation.'}</p>}
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      {readings.map(([key, value]) => <div className="contents" key={key}><dt>{key}</dt><dd>{waveNumber(value)}</dd></div>)}
    </dl>
    {kalmanAssimilationGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function KalmanAssimilationConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="kalman-assimilation-concept-title">
    <h2 id="kalman-assimilation-concept-title">{ko ? '관측이 예측을 교정하는 정도' : 'How much a reading corrects the forecast'}</h2>
    <p>{kalmanAssimilationGuide.intuition[language]}</p>
    <p className="ai-concept-equation">{kalmanAssimilationGuide.steps[0].formula}</p>
    <p>{kalmanAssimilationGuide.steps[0].explanation[language]}</p>
    <a href="#experiment">{ko ? '관측 공백 확인하기' : 'Inspect a missing reading'}</a>
  </section>;
}
