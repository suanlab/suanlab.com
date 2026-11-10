'use client';
import {useLanguage} from '@/components/language-provider';
import type {WavesResult} from '@/lib/learning-waves';
import type {Vec2} from '@/lib/learning-simulations';

import {WavePlot,waveNumber as fmt,type Series,type Mark} from './waves-plot';
import CoupledModesFigure from './coupled-modes-figure';
import WaveEquationFigure from './wave-equation-figure';
import InterferenceFigure from './interference-figure';
import DispersionFigure from './dispersion-figure';
import SamplingSensorsFigure from './sampling-sensors-figure';
import FourierOpticsFigure from './fourier-optics-figure';
import RayWaveOpticsFigure from './ray-wave-optics-figure';
import PolarizationFigure from './polarization-figure';
export default function WavesFigure({result}: {result: WavesResult}) {
  const {language} = useLanguage(), t = (ko: string, en: string) => language === 'ko' ? ko : en;
  if('samplingSensors' in result)return <SamplingSensorsFigure result={result}/>;
  if('fourierOptics' in result)return <FourierOpticsFigure result={result}/>;
  if('rayWaveOptics' in result)return <RayWaveOpticsFigure result={result}/>;
  if('polarizationOptics' in result)return <PolarizationFigure result={result}/>;
  if('dispersivePackets' in result)return <DispersionFigure result={result}/>;
  if('interferenceOptics' in result)return <InterferenceFigure result={result}/>;
  if('waveEquation' in result)return <WaveEquationFigure result={result}/>;
  if(!('response' in result))return <CoupledModesFigure result={result}/>;
  const r = result.response, system = result.system;
  const name = (ko: string, en: string) => t(ko,en);
  const series = (title: string, points: Vec2[], color = 'blue'): Series => ({name: title, segments: [points], color});
  const plot = (marker: string, title: string, s: Series[], x: string, y: string, caption: string, dots: Mark[] = [], vertical: number[] = []) => <WavePlot key={marker} marker={marker} title={title} series={s} xLabel={x} yLabel={y} caption={caption} dots={dots} vertical={vertical}/>;
  const table = (marker: string, title: string, rows: (string | number | null | undefined)[][]) => <figure key={marker} data-waves-table={marker}><div className="ai-discrete-table" tabIndex={0} role="region" aria-label={title}><table><caption>{title}</caption><thead><tr><th scope="col">{t('항목','Quantity')}</th><th scope="col">{t('값','Value')}</th><th scope="col">{t('단위·의미','Unit or meaning')}</th></tr></thead><tbody>{rows.map((row,i) => <tr key={i}>{row.map((v,j) => j === 0 ? <th key={j} scope="row">{v}</th> : <td key={j}>{typeof v === 'number' || v == null ? fmt(v) : v}</td>)}</tr>)}</tbody></table></div></figure>;
  const status = ({'attracting-particular': t('초기 기억이 사라지는 특수해','Attracting particular solution'), 'nonattracting-particular': t('초기 자유 운동이 남는 특수해','Particular solution with persistent free motion'), 'no-bounded-periodic-response': t('유계 주기 응답 없음','No bounded periodic response'), 'singular-transfer-zero-input': t('영입력·특이 전달함수','Zero input, singular transfer')} as Record<string,string>)[r.status];
  const regime = ({underdamped:t('부족감쇠','Underdamped'),critical:t('임계감쇠','Critical damping'),overdamped:t('과감쇠','Overdamped')} as Record<string,string>)[r.dampingRegime];
  const panels: React.ReactNode[] = [];
  if (result.transient && result.query) {
    const a = result.transient, q = result.query;
    const curve = (key: 'position'|'velocity'|'force'|'energy'|'inputPower'|'heatPower') => a.states.map(p => [p.time,p[key]] as Vec2);
    const mark = (key: 'position'|'velocity'|'force'|'energy'|'inputPower'|'heatPower', color = 'orange') => [{point:[q.time,q[key]] as Vec2,color}];
    const sampled = t('선형 흐름의 표본 257개입니다. 연결선은 적분 갱신이 아닙니다. 점은 선택한 실제 시각입니다.','These are 257 samples of the linear flow, not integration updates. The dot marks the selected actual time.');
    panels.push(plot('transient-position',t('초기조건을 포함한 실제 변위','Actual displacement with initial data'),[series('x(t)',curve('position'))],'t · s','x · m',sampled,mark('position')));
    panels.push(plot('transient-force',t('정현파 외력의 시계','The sinusoidal force clock'),[series('F(t)',curve('force'))],'t · s','F · N',t('외력은 t=0에서 양의 최댓값으로 시작합니다.','The force starts at its positive maximum at t=0.'),mark('force')));
    panels.push(plot('transient-velocity',t('속도와 구동 타이밍','Velocity and drive timing'),[series('v(t)',curve('velocity'))],'t · s','v · m/s',sampled,mark('velocity')));
    panels.push(plot('transient-phase',t('실제 위치–속도 궤적','Actual position–velocity trajectory'),[series(t('시간 궤적','Time trajectory'),a.states.map(p => [p.position,p.velocity] as Vec2))],'x · m','v · m/s',t('단위가 다른 두 축입니다. 선분은 시간순으로 표본을 연결합니다.','The axes have different units. Segments join samples in time order.'),[{point:[q.position,q.velocity],color:'orange'}]));
    panels.push(plot('transient-energy',t('저장된 역학적 에너지','Stored mechanical energy'),[series('E(t)',curve('energy'))],'t · s','E · J',t('저장 에너지가 일정해도 입력과 열의 흐름은 0이 아닐 수 있습니다.','Constant stored energy can coexist with nonzero input and heat flows.'),mark('energy')));
    panels.push(plot('transient-power',t('순간 입력 전력과 열 발생률','Instantaneous input and dissipation'),[series(t('입력 Fv · 파랑','Input Fv · blue'),curve('inputPower')),series(t('열 cv² · 주황','Heat cv² · orange'),curve('heatPower'),'orange')],'t · s','P · W',t('입력은 음수일 수 있고 열 발생률은 음수가 아닙니다.','Input power can be negative; dissipation is nonnegative.'),[...mark('inputPower','blue'),...mark('heatPower','orange')]));
    if (a.particularInitial && a.homogeneousInitial) {
      const particular = a.states.map(p => ({time:p.time, position:a.particularInitial!.position*Math.cos(system.frequency*p.time)+(system.frequency===0?0:a.particularInitial!.velocity*Math.sin(system.frequency*p.time)/system.frequency)}));
      panels.push(plot('transient-decomposition',t('특수해와 초기 기억의 합','Particular solution plus initial memory'),[
        series(t('실제 · 파랑','Actual · blue'),curve('position')),
        series(t('특수해 · 주황','Particular · orange'),particular.map(p => [p.time,p.position] as Vec2),'orange'),
        series(t('동차 성분 · 초록','Homogeneous · green'),particular.map((p,i) => [p.time,a.states[i].position-p.position] as Vec2),'green')], 't · s','x · m',t('무감쇠에서는 동차 성분이 사라지지 않습니다. 큰 항의 상쇄는 실제 변위와 구분하세요.','Without damping, the homogeneous component persists. Large cancelling terms differ from the actual displacement.')));
    } else panels.push(<p key="no-split" role="status">{t('이 설정에는 유계 주기 특수해가 없어 분해 그림을 생략합니다. 실제 유한 시간 해는 위에 표시합니다.','No bounded periodic particular solution exists here, so its decomposition is omitted. The actual finite-time solution is shown above.')}</p>);
    panels.push(table('transient-energy-balance',t('종료 시간 전체의 독립 에너지 수지','Independent energy balance over the full duration'),[
      ['T',a.duration,'s'],['W',a.work,'J'],['Q',a.heat,'J'],['ΔE',a.energyChange,'J'],['ΔE − W + Q',a.balanceResidual,'J'],
      [t('적분 완료','Quadrature completed'),a.quadrature.completed?t('예','Yes'):t('아니요','No'),t('적응형 Simpson','Adaptive Simpson')],
      [t('함수 평가','Function evaluations'),a.quadrature.evaluations,t('그림 표본과 별도','Separate from drawing samples')],
      [t('일 오차 추정','Work error estimate'),a.quadrature.workErrorEstimate,'J'],[t('열 오차 추정','Heat error estimate'),a.quadrature.heatErrorEstimate,'J'],
      ['W/T',a.finiteWindowMeanInputPower,'W'],['⟨P⟩',r.meanPower,t('주기 특수해 · W','Periodic particular · W')]]));
  }
  if (result.sweep) {
    const segments = (key: 'gain'|'phaseLag'|'meanPower') => {
      const out: Vec2[][] = []; let current: Vec2[] = [];
      for (const row of result.sweep!) {const value = row.response[key]; if (value === null) {if (current.length) out.push(current); current=[];} else current.push([row.frequency,value]);}
      if (current.length) out.push(current);
      return out;
    };
    const selected = (value: number | null): Mark[] => value === null ? [] : [{point:[system.frequency,value],color:'orange'}];
    const marker = [r.naturalFrequency,...(r.displacementPeakFrequency===null?[]:[r.displacementPeakFrequency]),...(r.halfPower?[r.halfPower.lower,r.halfPower.upper]:[])];
    const family = t('각 가로 좌표는 서로 다른 구동 실험입니다. 시간 궤적이 아닙니다. 수직선은 아래 표의 특성 주파수입니다.','Each horizontal coordinate is a different drive experiment, not a time trajectory. Vertical lines mark characteristic frequencies listed below.');
    panels.push(plot('frequency-gain',t('변위 이득과 조건부 최대점','Displacement gain and conditional peak'),[{name:'G · m/N',segments:segments('gain'),color:'blue'}],'Ω · rad/s','G · m/N',family,selected(r.gain),marker));
    panels.push(plot('frequency-phase',t('측정 응답의 위상 지연','Measured response phase lag'),[{name:'φ · rad',segments:segments('phaseLag'),color:'blue'}],'Ω · rad/s','φ · rad',t('영입력·DC·유계 주기해 부재에서는 측정 위상점을 그리지 않습니다. 내재적 전달 위상과 구분합니다.','No measured phase point is drawn for zero input, DC or a missing bounded periodic response. Distinguish intrinsic transfer phase.'),selected(r.phaseLag),[r.naturalFrequency]));
    panels.push(plot('frequency-power',t('평균 흡수 전력과 반전력 폭','Mean absorbed power and half-power width'),[{name:'⟨P⟩ · W',segments:segments('meanPower'),color:'blue'}],'Ω · rad/s','⟨P⟩ · W',family,selected(r.meanPower),marker));
    panels.push(table('frequency-landmarks',t('서로 다른 주파수와 품질계수','Distinct frequencies and quality factor'),[
      ['ω₀',r.naturalFrequency,'rad/s'],['ωd',r.dampedFreeFrequency,'rad/s'],['Ωx',r.displacementPeakFrequency,'rad/s'],['ΩP',r.powerPeakFrequency,'rad/s'],
      ['Ω−',r.halfPower?.lower,'rad/s'],['Ω+',r.halfPower?.upper,'rad/s'],['ΔΩ',r.halfPower?.width,'rad/s'],['Q',r.qualityFactor,'1'],
      [t('변위 이득 최대점','Displacement gain maximum'),r.displacementPeakStatus==='maximum-at-dc'?'DC':r.displacementPeakStatus==='unbounded-resonance'?t('유계 최대 없음','No bounded maximum'):t('양의 내부 최대','Positive interior maximum'),''] ]));
    if (result.periodic) {
      const rows = result.periodic, xy = (key:'position'|'velocity'|'force'|'power'|'heat'|'energy') => rows.map(p=>[p.time,p[key]] as Vec2);
      const note = t('선택한 주파수의 주기 특수해 한 주기입니다. 초기값 문제의 시작 과정이 아닙니다.','One cycle of the selected periodic particular solution. This is not an initial-value startup.');
      panels.push(plot('periodic-position',t('선택 실험의 주기 변위','Periodic displacement of the selected experiment'),[series('xₚ',xy('position'))],'t · s','xₚ · m',note));
      panels.push(plot('periodic-velocity',t('주기 속도와 힘의 위상','Periodic velocity and force timing'),[series('vₚ',xy('velocity'))],'t · s','vₚ · m/s',note));
      panels.push(plot('periodic-force',t('선택 주파수의 외력','Force at the selected frequency'),[series('F',xy('force'))],'t · s','F · N',note));
      panels.push(plot('periodic-power',t('한 주기의 입력과 열','Input and heat during one cycle'),[series(t('입력 · 파랑','Input · blue'),xy('power')),series(t('열 · 주황','Heat · orange'),xy('heat'),'orange')],'t · s','P · W',note));
      panels.push(plot('periodic-energy',t('한 주기의 저장 에너지','Stored energy during one cycle'),[series('E',xy('energy'))],'t · s','E · J',note));
    } else panels.push(<p key="no-cycle" role="status">{t('DC에는 유한한 주기가 없고, 무감쇠 공진에는 유계 주기해가 없습니다. 이 설정의 주기 그림은 제공하지 않습니다.','DC has no finite cycle; undamped resonance has no bounded periodic response. Cycle plots are unavailable at this setting.')}</p>);
    panels.push(table('frequency-cycle',t('선택 응답의 전달함수와 주기 에너지','Selected susceptibility and cycle energy'),[
      ['Re χ',r.susceptibility?.real,'m/N'],['Im χ',r.susceptibility?.imaginary,'m/N'],['φ intrinsic',r.intrinsicLag,'rad'],['φ measured',r.phaseLag,'rad'],
      ['T',r.period,'s'],['⟨E⟩',r.meanStoredEnergy,'J'],['Q cycle',r.heatPerCycle,'J'],
      [t('DC 정적 에너지','DC static energy'),system.frequency===0?system.forceAmplitude**2/(2*system.stiffness):null,'J']]));
  }
  return <div className="ai-waves-figures" data-waves-kind={result.kind}><p role="status">{regime} · {status}</p>{panels}<p className="ai-small">{name('—는 0이 아닌 미정 값입니다. 오차 추정은 보증된 전체 오차 상한이 아닙니다.','A dash means unavailable, not zero. Error estimates are not certified global bounds.')}</p></div>;
}
