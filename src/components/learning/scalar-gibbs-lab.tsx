'use client';

import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { scalarGibbsGuide, scalarGibbsPresetLabels } from '@/data/scalar-gibbs-reader';
import { readScalarGibbsInput, scalarGibbsPresets, type ScalarGibbsInput } from '@/lib/learning-scalar-gibbs-input';
import type { ScalarGibbsResult } from '@/lib/learning-scalar-gibbs';
import type { ScalarGibbsReply } from '@/lib/learning-scalar-gibbs.worker';
import '@/styles/scalar-gibbs-lab.css';

const size = 128;
const weight = (p: ScalarGibbsInput, x: number, y: number) => {
  const local = (v: number, h: number) => p.quartic * v ** 4 + p.quadratic * v ** 2 - h * v;
  const action = local(x, p.sources[0]) + (p.sources.length === 2 ? local(y, p.sources[1]) + p.coupling * (x - y) ** 2 / 2 : 0);
  return Math.exp(-action);
};

function GibbsPlot({ input, ko }: { input: ScalarGibbsInput; ko: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const pair = input.sources.length === 2;
  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const coordinate = (i: number) => input.radius * (2 * (i + .5) / size - 1);
    const samples = Array.from({ length: pair ? size * size : size }, (_, i) =>
      weight(input, coordinate(i % size), -coordinate(Math.floor(i / size))));
    const peak = Math.max(...samples);
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, size, size);
    if (pair) {
      const image = ctx.createImageData(size, size);
      samples.forEach((value, i) => {
        const level = Math.sqrt(value / peak);
        image.data.set([17 + 163 * level, 24 + 209 * level, 39 + 208 * level, 255], i * 4);
      });
      ctx.putImageData(image, 0, 0);
    } else {
      ctx.strokeStyle = '#b4e9f7';
      ctx.lineWidth = 2;
      ctx.beginPath();
      samples.forEach((value, i) => {
        const y = size - 6 - (size - 12) * value / peak;
        if (i === 0) ctx.moveTo(i, y);
        else ctx.lineTo(i, y);
      });
      ctx.stroke();
    }
  }, [input, pair]);

  return <figure className="scalar-gibbs-plot">
    <div className="scalar-gibbs-axes">
      <span>{pair ? `x₂ ∈ [−${input.radius}, ${input.radius}]` : 'e⁻ᴬ / max e⁻ᴬ'}</span>
      <canvas ref={ref} width={size} height={size} aria-hidden="true"/>
      <span>x₁ ∈ [−{input.radius}, {input.radius}]</span>
    </div>
    <figcaption>{pair
      ? (ko ? '가로 x₁은 오른쪽으로, 세로 x₂는 위로 증가합니다. 밝을수록 상대 볼츠만 가중치가 큽니다.' : 'x₁ increases to the right and x₂ increases upward. Brighter cells have larger relative Boltzmann weights.')
      : (ko ? '한 격자점의 볼츠만 가중치를 표본 최댓값으로 나눈 곡선입니다.' : 'The curve shows one-site Boltzmann weights divided by their sampled maximum.')}</figcaption>
  </figure>;
}

export function ScalarGibbsConcept({ language }: { language: 'ko' | 'en' }) {
  const [coupled, setCoupled] = useState(true);
  const ko = language === 'ko';
  return <section className="ai-intuition scalar-gibbs" id="visual-intuition" aria-labelledby="scalar-gibbs-concept-title">
    <h2 id="scalar-gibbs-concept-title">{ko ? '두 격자점의 결합을 그림으로 보기' : 'Visualize coupling between two sites'}</h2>
    <p>{scalarGibbsGuide.intuition[language]}</p>
    <div className="scalar-gibbs-actions" role="group" aria-label={ko ? '결합 상태' : 'Coupling state'}>
      {[true, false].map((value, i) => <Button key={i} variant="outline" type="button" aria-pressed={coupled === value} onClick={() => setCoupled(value)}>{scalarGibbsPresetLabels[i][language]}</Button>)}
    </div>
    <GibbsPlot input={scalarGibbsPresets[coupled ? 0 : 1]} ko={ko}/>
    <p>{scalarGibbsGuide.steps[0].explanation[language]}</p>
  </section>;
}

function decimal(rational: string) {
  const [n, d = '1'] = rational.split('/');
  return Number(n) / Number(d);
}

type Range = ScalarGibbsResult['logPartition'];

function RangeValue({ value }: { value: Range }) {
  const format = (x: string) => decimal(x).toPrecision(6);
  return <span title={`[${value.lower}, ${value.upper}]`}>
    {value.lower === value.upper ? format(value.lower) : `[${format(value.lower)}, ${format(value.upper)}]`}
  </span>;
}

function ResultTable({ result, ko }: { result: ScalarGibbsResult; ko: boolean }) {
  const mass = result.integrals[0];
  const statistics = [
    ['log Z', result.logPartition],
    ['F/T', result.freeEnergyOverTemperature],
    ...result.means.map((value, i) => [`E[x${i + 1}]`, value] as const),
    ...result.secondMoments.map((value, i) => [`E[x${i + 1}²]`, value] as const),
    ...result.fourthMoments.map((value, i) => [`E[x${i + 1}⁴]`, value] as const),
    ...result.covariance.flatMap((row, i) => row.slice(i).map((value, j) => [`Cov(x${i + 1}, x${i + j + 1})`, value] as const)),
    [ko ? '상자 안의 이동된 질량' : 'Shifted mass inside box', mass.boxIntegral],
  ] as const;

  return <div data-gibbs-result>
    <p>{ko ? '구간 끝점의 표시값은 반올림했습니다. 상자 안 질량과 꼬리 상계는 A에 상수 C를 더한 가중치 e⁻⁽ᴬ⁺ᶜ⁾ 기준입니다.' : 'Displayed interval endpoints are rounded. The in-box mass and tail bound use the shifted weight e⁻⁽ᴬ⁺ᶜ⁾.'}</p>
    <dl className="scalar-gibbs-results">
      {statistics.map(([label, value]) => <div key={label}><dt>{label}</dt><dd><RangeValue value={value}/></dd></div>)}
      <div><dt>{ko ? '상자 밖 질량 상계' : 'Outside-box mass upper bound'}</dt><dd title={mass.outsideAbsoluteUpper}>{decimal(mass.outsideAbsoluteUpper).toExponential(5)}</dd></div>
      <div><dt>{ko ? '원점의 헤시안' : 'Hessian at the origin'}</dt><dd>{result.originHessian.map(row => `[${row.join(', ')}]`).join(' ')}</dd></div>
    </dl>
    <details>
      <summary>{ko ? '정확한 유리수 경계' : 'Exact rational bounds'}</summary>
      <dl className="scalar-gibbs-results">
        {statistics.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>[{value.lower}, {value.upper}]</dd></div>)}
        <div><dt>{ko ? '상자 밖 질량 상계' : 'Outside-box mass upper bound'}</dt><dd>{mass.outsideAbsoluteUpper}</dd></div>
      </dl>
    </details>
  </div>;
}

const controls = [
  { name: 'quartic', symbol: 'a', ko: '사차항', en: 'Quartic', min: .125, max: 2, step: .125 },
  { name: 'quadratic', symbol: 'b', ko: '이차항', en: 'Quadratic', min: -2, max: 2, step: .125 },
  { name: 'coupling', symbol: 'c', ko: '결합', en: 'Coupling', min: 0, max: 2, step: .125 },
  { name: 'radius', symbol: 'R', ko: '적분 반경', en: 'Integration radius', min: 2, max: 8, step: 1 },
] as const;

type State = { status: 'idle' | 'running' | 'error' } | { status: 'ready'; result: ScalarGibbsResult };

export default function ScalarGibbsLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [preset, setPreset] = useState({ index: 0, version: 0 });
  const [siteCount, setSiteCount] = useState(2);
  const [input, setInput] = useState(scalarGibbsPresets[0]);
  const workerRef = useRef<Worker>();
  const [state, setState] = useState<State>({ status: 'idle' });
  const defaults = scalarGibbsPresets[preset.index];

  function stopCalculation() {
    workerRef.current?.terminate();
    workerRef.current = undefined;
  }

  useEffect(() => stopCalculation, []);

  function calculate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const field = (name: string) => Number(data.get(name));
    const next = readScalarGibbsInput({
      quartic: field('quartic'), quadratic: field('quadratic'), coupling: field('coupling'),
      radius: field('radius'), panels: field('panels'),
      sources: Array.from({ length: siteCount }, (_, i) => field(`h${i}`)),
    });
    stopCalculation();
    if (!next) {
      setState({ status: 'error' });
      return;
    }
    setInput(next);
    setState({ status: 'running' });
    try {
      const worker = new Worker(new URL('../../lib/learning-scalar-gibbs.worker.ts', import.meta.url), { type: 'module' });
      workerRef.current = worker;
      const settle = (value: State) => {
        if (workerRef.current !== worker) return;
        stopCalculation();
        setState(value);
      };
      worker.onmessage = ({ data }: MessageEvent<ScalarGibbsReply>) => settle('result' in data
        ? { status: 'ready', result: data.result } : { status: 'error' });
      worker.onerror = () => settle({ status: 'error' });
      worker.onmessageerror = () => settle({ status: 'error' });
      worker.postMessage(next);
    } catch {
      stopCalculation();
      setState({ status: 'error' });
    }
  }

  return <section id="experiment" className="scalar-gibbs" aria-labelledby="scalar-gibbs-lab-title">
    <h2 id="scalar-gibbs-lab-title">{ko ? '사차 깁스 장 실험' : 'Quartic Gibbs field experiment'}</h2>
    <div className="scalar-gibbs-actions" role="group" aria-label={ko ? '예제 불러오기' : 'Load an example'}>
      {scalarGibbsPresetLabels.map((label, index) => <Button key={index} type="button" size="sm" variant="outline" onClick={() => {
        setPreset(old => ({ index, version: old.version + 1 }));
        const example = scalarGibbsPresets[index];
        setSiteCount(example.sources.length);
        setInput(example);
        stopCalculation();
        setState({ status: 'idle' });
      }}>{label[language]}</Button>)}
    </div>
    <form onSubmit={calculate} key={preset.version}>
      <div className="scalar-gibbs-controls">
        <label>{ko ? '격자점 수' : 'Number of sites'}<select value={siteCount} onChange={e => setSiteCount(Number(e.target.value))}><option value={1}>1</option><option value={2}>2</option></select></label>
        {controls.filter(field => field.name !== 'coupling' || siteCount === 2).map(field => <label key={field.name}>
          {field[language]} ({field.symbol})
          <input name={field.name} type="number" required min={field.min} max={field.max} step={field.step} defaultValue={defaults[field.name]}/>
        </label>)}
        {Array.from({ length: siteCount }, (_, i) => <label key={i}>{ko ? '외부장' : 'Source'} h{i + 1}
          <input name={`h${i}`} type="number" required min={-1} max={1} step={.125} defaultValue={defaults.sources[i] ?? 0}/>
        </label>)}
        <label>{ko ? '축당 분할 수' : 'Panels per axis'}
          <select name="panels" defaultValue={defaults.panels}>{[16, 32, 64, 128].map(n => <option key={n}>{n}</option>)}</select>
        </label>
      </div>
      <div className="scalar-gibbs-actions">
        <Button type="submit">{ko ? '적용하고 계산' : 'Apply and calculate'}</Button>
        {state.status === 'running' && <Button type="button" variant="outline" onClick={() => {
          stopCalculation();
          setState({ status: 'idle' });
        }}>{ko ? '취소' : 'Cancel'}</Button>}
      </div>
    </form>
    <GibbsPlot input={input} ko={ko}/>
    <div className="scalar-gibbs-actions" aria-label={ko ? '그림과 계산의 매개변수' : 'Parameters used for the plot and calculation'}>
      <span>a = {input.quartic}</span><span>b = {input.quadratic}</span><span>c = {input.coupling}</span>
      <span>h = [{input.sources.join(', ')}]</span><span>R = {input.radius}</span>
      <span>{ko ? '축당 분할 수' : 'Panels per axis'} = {input.panels}</span>
    </div>
    <p role="status">{{
      idle: ko ? '설정을 적용하면 적분 구간을 계산합니다.' : 'Apply the settings to compute the integral bounds.',
      running: ko ? '적분 구간을 계산 중입니다.' : 'Computing integral bounds.',
      ready: ko ? '적분 구간 계산 완료' : 'Integral bounds ready',
      error: ko ? '계산에 실패했습니다. 입력을 확인한 후 다시 계산하세요.' : 'Calculation failed. Check the inputs and try again.',
    }[state.status]}</p>
    {state.status === 'ready' && <ResultTable result={state.result} ko={ko}/>}
    {scalarGibbsGuide.experiments.map(lesson => <details key={lesson.title.en}>
      <summary>{lesson.title[language]}</summary>
      <p>{lesson.predict[language]}</p><p>{lesson.action[language]}</p><p>{lesson.explain[language]}</p>
    </details>)}
  </section>;
}
