'use client';

import { useEffect, useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { legendreResponseGuide } from '@/data/legendre-response-reader';
import { WavePlot, waveNumber } from './waves-plot';
import type { LegendreResponseModel } from '@/lib/learning-legendre-response';
import type { LegendreResponseReply } from '@/lib/learning-legendre-response.worker';
import '@/styles/scalar-gibbs-lab.css';

const models = {
  gaussian: { ko: '가우시안', en: 'Gaussian' },
  spin: { ko: '스핀 하나', en: 'Single spin' },
  quartic: { ko: '사차 깁스 장', en: 'Quartic Gibbs field' },
};
type Rows = Extract<LegendreResponseReply, { rows: unknown }>['rows'];
type Bounds = Rows[number]['mean'];

function number(text: string) {
  const [n, d = '1'] = text.split('/');
  return Number(n) / Number(d);
}

const midpoint = (value: Bounds) => (number(value.lower) + number(value.upper)) / 2;
const rounded = (value: Bounds | null) => value
  ? value.lower === value.upper
    ? waveNumber(number(value.lower))
    : `[${waveNumber(number(value.lower))}, ${waveNumber(number(value.upper))}]`
  : '—';

function ResponsePlots({ rows, ko }: { rows: Rows; ko: boolean }) {
  const [index, setIndex] = useState(rows.length >> 1);
  const selected = rows[index];
  const w = midpoint(selected.generator), m = midpoint(selected.mean);
  const lower = rows[0].source, upper = rows[rows.length - 1].source;
  const tangent = (h: number) => w + m * (h - selected.source);
  const fields = [
    ['m', selected.mean],
    ['W(h)', selected.generator],
    ['χ(h)', selected.susceptibility],
    ['Γ(m)', selected.effectiveAction],
    ['Γ″(m)', selected.effectiveCurvature],
  ] as const;
  return <>
    <label>
      {ko ? '외부장 h' : 'Source h'} = {selected.source}
      <input className="block w-full" aria-label={ko ? '외부장 h' : 'Source h'} aria-valuetext={String(selected.source)}
        type="range" min={0} max={rows.length - 1} value={index} onChange={e => setIndex(Number(e.target.value))}/>
    </label>
    <WavePlot marker="legendre-generator" title={ko ? '연결 생성함수와 접선' : 'Connected generator and tangent'}
      xLabel="h" yLabel="W(h)" vertical={[selected.source]} dots={[{ point: [selected.source, w], color: 'orange' }]}
      series={[
        { name: 'W(h)', color: 'blue', segments: [rows.map(row => [row.source, midpoint(row.generator)])] },
        { name: ko ? '기울기 m인 접선' : 'Tangent with slope m', color: 'orange', segments: [[[lower, tangent(lower)], [upper, tangent(upper)]]] },
      ]}
      caption={ko ? '구간 중점을 이어 그렸습니다. 접선의 기울기는 선택한 외부장의 평균입니다.' : 'The plot joins interval midpoints. The tangent slope is the mean at the selected source.'}/>
    <WavePlot marker="legendre-effective-action" title={ko ? '평균 좌표에서의 유효작용' : 'Effective action in mean coordinates'}
      xLabel="m" yLabel="Γ(m)" vertical={[m]} dots={[{ point: [m, midpoint(selected.effectiveAction)], color: 'orange' }]}
      series={[{ name: 'Γ(m)', color: 'blue', segments: [rows.map(row => [midpoint(row.mean), midpoint(row.effectiveAction)])] }]}
      caption={ko ? 'Γ의 기울기는 선택한 외부장 h이고, 곡률은 감수율의 역수입니다.' : 'The slope of Γ is the selected source h. Its curvature is the reciprocal susceptibility.'}/>
    <dl className="scalar-gibbs-results">
      {fields.map(([label, value]) => <div key={label}>
        <dt>{label}</dt><dd title={value ? `[${value.lower}, ${value.upper}]` : undefined}>{rounded(value)}</dd>
      </div>)}
    </dl>
    <p>{ko ? '수치 표시는 구간 끝점을 반올림했습니다.' : 'The displayed interval endpoints are rounded.'}</p>
  </>;
}

export default function LegendreResponseLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [model, setModel] = useState<LegendreResponseModel>('spin');
  const [attempt, setAttempt] = useState(0);
  const [response, setResponse] = useState<{ model: LegendreResponseModel; attempt: number; reply: LegendreResponseReply }>();
  const reply = response?.model === model && response.attempt === attempt ? response.reply : null;

  useEffect(() => {
    let worker: Worker | undefined;
    let active = true;
    const settle = (value: LegendreResponseReply) => {
      worker?.terminate();
      if (active) setResponse({ model, attempt, reply: value });
    };
    try {
      worker = new Worker(new URL('../../lib/learning-legendre-response.worker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = ({ data }: MessageEvent<LegendreResponseReply>) => settle(data);
      worker.onerror = () => settle({ error: true });
      worker.onmessageerror = () => settle({ error: true });
      worker.postMessage(model);
    } catch {
      settle({ error: true });
    }
    return () => {
      active = false;
      worker?.terminate();
    };
  }, [model, attempt]);

  return <section id="experiment" className="scalar-gibbs" aria-labelledby="legendre-response-title">
    <h2 id="legendre-response-title">{ko ? '외부장과 평균 사이의 르장드르 변환' : 'Legendre transform between source and mean'}</h2>
    <label>{ko ? '확률 모형' : 'Probability model'}
      <select className="ml-2 rounded border border-border bg-background px-2 py-1" value={model} onChange={e => {
        setModel(e.target.value as LegendreResponseModel);
      }}>
        {Object.entries(models).map(([key, label]) => <option key={key} value={key}>{label[language]}</option>)}
      </select>
    </label>
    {!reply && <p role="status">{ko ? '응답 곡선을 계산하고 있습니다.' : 'Computing the response curves.'}</p>}
    {reply && ('rows' in reply ? <ResponsePlots key={model} rows={reply.rows} ko={ko}/> :
      <div role="status">
        <p>{ko ? '계산에 실패했습니다.' : 'Calculation failed.'}</p>
        <button type="button" onClick={() => setAttempt(value => value + 1)}>
          {ko ? '다시 계산' : 'Try again'}
        </button>
      </div>)}
    {legendreResponseGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}


export function LegendreResponseConcept({ language }: { language: 'ko' | 'en' }) {
  const [source, setSource] = useState(1);
  const ko = language === 'ko';
  const tangent = (h: number) => source ** 2 / 4 + source / 2 * (h - source);
  return <section className="ai-intuition scalar-gibbs" id="visual-intuition" aria-labelledby="legendre-concept-title">
    <h2 id="legendre-concept-title">{ko ? '접선의 기울기를 바꿔 보기' : 'Change the tangent slope'}</h2>
    <p>{legendreResponseGuide.intuition[language]}</p>
    <label>{ko ? '외부장 h' : 'Source h'} = {source}
      <input className="block w-full" type="range" min={-2} max={2} step={.125} value={source} onChange={e => setSource(Number(e.target.value))}/>
    </label>
    <WavePlot marker="legendre-concept" title={ko ? '가우시안의 생성함수' : 'Gaussian generator'}
      xLabel="h" yLabel="W(h)" vertical={[source]} dots={[{ point: [source, source ** 2 / 4], color: 'orange' }]}
      series={[
        { name: 'W(h) = h²/4', color: 'blue', segments: [Array.from({ length: 33 }, (_, i) => {
          const h = i / 8 - 2;
          return [h, h ** 2 / 4];
        })] },
        { name: ko ? '접선' : 'Tangent', color: 'orange', segments: [[[-2, tangent(-2)], [2, tangent(2)]]] },
      ]}
      caption={ko ? `현재 기울기 m = ${source / 2}입니다. 접선의 절편에 음수를 붙이면 Γ(m) = ${source ** 2 / 4}입니다.` : `The slope is m = ${source / 2}. The negative tangent intercept gives Γ(m) = ${source ** 2 / 4}.`}/>
    <a href="#experiment">{ko ? '세 모형으로 비교하기' : 'Compare the three models'}</a>
  </section>;
}
