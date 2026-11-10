'use client';

import { useEffect, useState } from 'react';
import { useLanguage } from '@/components/language-provider';
import { isingCriticalityGuide } from '@/data/ising-criticality-reader';
import type { IsingSweepReply } from '@/lib/learning-ising-cylinder.worker';
import { WavePlot } from './waves-plot';

const colors = ['blue', 'orange', 'green', 'neutral'];
const layout = 'grid min-w-0 grid-cols-1 gap-4';
type Sweep = Extract<IsingSweepReply, { sweep: unknown }>['sweep'];

function IsingPlot({ sweep, ko }: { sweep: Sweep; ko: boolean }) {
  const [index, setIndex] = useState(22);
  const [observable, setObservable] = useState<'binderRatio' | 'susceptibilityPerSite'>('binderRatio');
  const [rescale, setRescale] = useState(false);
  const binder = observable === 'binderRatio';
  const scaled = binder && rescale;
  const horizontalCoordinate = (kx: number, length: number) => scaled ? (sweep.critical / kx - 1) * length : kx;
  const inView = (x: number) => Number.isFinite(x) && (!scaled || Math.abs(x) <= 3);
  const horizontal = sweep.curves[0].samples[index].kx;
  const title = binder ? (ko ? '크기별 Binder 비율' : 'Binder ratio by size') : (ko ? '격자점당 감수율' : 'Susceptibility per site');
  return <>
    <label>{ko ? '관측량' : 'Observable'}
      <select className="ml-2 rounded border border-border bg-background p-1" value={observable}
        onChange={e => setObservable(e.target.value === 'binderRatio' ? 'binderRatio' : 'susceptibilityPerSite')}>
        <option value="binderRatio">{ko ? 'Binder 비율' : 'Binder ratio'}</option>
        <option value="susceptibilityPerSite">{ko ? '격자점당 감수율' : 'Susceptibility per site'}</option>
      </select>
    </label>
    <label>{ko ? '가로 결합' : 'Horizontal coupling'} Kₓ = {horizontal.toFixed(2)}
      <input className="block w-full" type="range" min={0} max={40} value={index} aria-valuetext={horizontal.toFixed(2)}
        onChange={e => setIndex(Number(e.target.value))}/>
    </label>
    {binder && <label className="flex items-center gap-2">
      <input type="checkbox" checked={rescale} onChange={e => setRescale(e.target.checked)}/>
      {ko ? '축을 tL로 변환하기 (ν = 1)' : 'Rescale the horizontal axis to tL (ν = 1)'}
    </label>}
    <WavePlot marker="ising-criticality" title={title} xLabel={scaled ? 'tL' : 'Kₓ'} yLabel={binder ? 'Uₗ' : 'χ/N'}
      xDomain={scaled ? [-3, 3] : undefined} vertical={[scaled ? 0 : sweep.critical]}
      series={sweep.curves.map((curve, i) => ({ name: `L = ${curve.length}`, color: colors[i],
        segments: [curve.samples.map((sample): [number, number] => [horizontalCoordinate(sample.kx, curve.length), sample[observable]])
          .filter(([x]) => inView(x))] }))}
      dots={sweep.curves.flatMap((curve, i) => {
        const x = horizontalCoordinate(horizontal, curve.length);
        return inView(x) ? [{ color: colors[i], point: [x, curve.samples[index][observable]] as [number, number] }] : [];
      })}
      caption={scaled
        ? (ko ? 't = Kₓ,c/Kₓ − 1입니다. |tL| ≤ 3인 표본을 비교해 유한 크기 보정이 남는지 살펴보세요.' : 'Here t = Kₓ,c/Kₓ − 1. Compare samples with |tL| ≤ 3 and look for finite-size corrections.')
        : (ko ? '세로 점선은 무한 격자의 임계점이고, 점은 선택한 유한 격자의 관측값입니다.' : 'The vertical dashed line marks the infinite-lattice critical point. Dots show the selected finite-lattice values.')}/>
    <dl className="grid grid-cols-2 gap-2 tabular-nums">
      <div className="contents"><dt>{ko ? '무한 격자 임계점' : 'Infinite-lattice critical point'} Kₓ</dt><dd>{sweep.critical.toFixed(6)}</dd></div>
      {sweep.crossings.map(({ lengths, kx }) => <div key={lengths.join(',')} className="contents">
        <dt>{ko ? 'Binder 교점' : 'Binder crossing'} L={lengths.join(', ')}</dt>
        <dd>{kx === null ? (ko ? '범위 안에서 교점 미검출' : 'No crossing detected in range') : kx.toFixed(6)}</dd>
      </div>)}
      {sweep.curves.map(curve => <div key={curve.length} className="contents">
        <dt>{binder ? 'Uₗ' : 'χ/N'} · L={curve.length}</dt><dd>{curve.samples[index][observable].toFixed(5)}</dd>
      </div>)}
    </dl>
  </>;
}

export default function IsingCriticalityLab() {
  const { language } = useLanguage();
  const ko = language === 'ko';
  const [ratio, setRatio] = useState(1);
  const [result, setResult] = useState<{ ratio: number; reply: IsingSweepReply }>();
  const reply = result?.ratio === ratio ? result.reply : undefined;

  useEffect(() => {
    let worker: Worker | undefined;
    let active = true;
    const finish = (reply: IsingSweepReply) => {
      worker?.terminate();
      if (active) setResult({ ratio, reply });
    };
    try {
      worker = new Worker(new URL('../../lib/learning-ising-cylinder.worker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = ({ data }: MessageEvent<IsingSweepReply>) => finish(data);
      worker.onerror = () => finish({ error: true });
      worker.onmessageerror = () => finish({ error: true });
      worker.postMessage(ratio);
    } catch {
      finish({ error: true });
    }
    return () => { active = false; worker?.terminate(); };
  }, [ratio]);

  return <section id="experiment" className={layout} aria-labelledby="ising-experiment-title">
    <h2 id="ising-experiment-title">{ko ? '유한 크기 교점 비교' : 'Compare finite-size crossings'}</h2>
    <label>{ko ? '세로 대 가로 결합비' : 'Vertical-to-horizontal coupling ratio'} Kᵧ/Kₓ
      <select className="ml-2 rounded border border-border bg-background p-1" value={ratio} onChange={e => setRatio(Number(e.target.value))}>
        {[.5, 1, 1.5].map(value => <option key={value}>{value}</option>)}
      </select>
    </label>
    {!reply && <p role="status">{ko ? '원통의 모멘트를 계산 중입니다.' : 'Computing the cylinder moments.'}</p>}
    {reply && ('sweep' in reply ? <IsingPlot sweep={reply.sweep} ko={ko}/> : <p role="alert">{ko ? '계산에 실패했습니다. 결합비를 바꿔 다시 계산하세요.' : 'Calculation failed. Change the coupling ratio to try again.'}</p>)}
    {isingCriticalityGuide.experiments.map(experiment => <details key={experiment.title.en}>
      <summary>{experiment.title[language]}</summary>
      <p>{experiment.predict[language]}</p><p>{experiment.action[language]}</p><p>{experiment.explain[language]}</p>
    </details>)}
  </section>;
}

export function IsingCriticalityConcept({ language }: { language: 'ko' | 'en' }) {
  const ko = language === 'ko';
  return <section className="ai-intuition" id="visual-intuition" aria-labelledby="ising-concept-title">
    <h2 id="ising-concept-title">{ko ? '이징 원통의 경계' : 'Ising cylinder boundaries'}</h2>
    <p>{isingCriticalityGuide.intuition[language]}</p>
    <figure className="my-4 rounded border border-border p-4">
      <div className="text-center font-mono text-xl leading-relaxed" aria-hidden="true">
        ↻ ● ─ ● ─ ● ↺<br/>│　　│　　│<br/>↻ ● ─ ● ─ ● ↺<br/>│　　│　　│<br/>↻ ● ─ ● ─ ● ↺
      </div>
      <figcaption>{isingCriticalityGuide.steps[0].explanation[language]}</figcaption>
    </figure>
    <a href="#experiment">{ko ? '크기별 곡선 비교하기' : 'Compare curves by size'}</a>
  </section>;
}
