'use client';
import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { motion } from 'motion/react';
import { firstStepLessons } from '@/data/learning-first-steps';
import { projectionPreparation, descentPreparation, bayesPreparation, interferencePreparation, firstStepEquation, firstStepDisplayEquation, type FirstStepId } from '@/lib/learning-first-steps';
import { LearningMath, LearningText } from './learning-math';
function useGentleMotion() {
  const [reduced, setReduced] = useState(true);
  useEffect(() => { const media = matchMedia('(prefers-reduced-motion: reduce)'), update = () => setReduced(media.matches); update(); media.addEventListener('change', update); return () => media.removeEventListener('change', update); }, []);
  return reduced;
}
function Scene({ kind, example, stage, reduced, language, children }: { kind: FirstStepId; example: number; stage: number; reduced: boolean; children?: ReactNode; language: 'ko' | 'en' }) {
  const id = useId(), ko = language === 'ko', duration = reduced ? 0 : .5;
  let geometry;
  if (kind === 'inner-products') {
    const r = projectionPreparation(example), map = (p: number[]) => [220 + p[0] * 45, 235 - p[1] * 45];
    const u = map(r.u), p = map(r.projected), norm = Math.hypot(...r.v), v = r.v.map(x => x / norm);
    geometry = <>
      {Array.from({ length: 11 }, (_, i) => <g key={i} className="ai-studio-grid"><line x1={40 + i * 45} y1={30} x2={40 + i * 45} y2={300} />{i < 6 && <line x1={40} y1={55 + i * 45} x2={535} y2={55 + i * 45} />}</g>)}
      <path d="M40 235 H535 M220 30 V300" className="ai-studio-axis" />
      {stage >= 1 && <motion.line x1={220 - v[0] * 195} y1={235 + v[1] * 195} x2={220 + v[0] * 195} y2={235 - v[1] * 195} stroke="var(--learning-green)" strokeWidth={3} initial={false} animate={{ opacity: 1 }} />}
      <line x1={220} y1={235} x2={u[0]} y2={u[1]} stroke="var(--learning-blue)" strokeWidth={4} markerEnd={`url(#${id}-arrow)`} />
      {stage >= 2 && <><motion.line x1={p[0]} y1={p[1]} x2={u[0]} y2={u[1]} stroke="var(--learning-neutral)" strokeWidth={3} strokeDasharray="6 5" initial={false} animate={{ x1: p[0], y1: p[1] }} transition={{ duration }} /><motion.circle r={7} fill="var(--learning-orange)" initial={{ cx: reduced ? p[0] : u[0], cy: reduced ? p[1] : u[1] }} animate={{ cx: p[0], cy: p[1] }} transition={{ duration }} /></>}
      <circle cx={u[0]} cy={u[1]} r={7} fill="var(--learning-blue)" />
    </>;
  } else if (kind === 'gradients') {
    const r = descentPreparation(example, stage), x = (x: number) => 280 + 18 * x, y = (y: number) => 170 - 18 * y;
    geometry = <><path d="M40 170 H520 M280 30 V310" className="ai-studio-axis" />{[1, 4, 16, 32].map(level => <ellipse key={level} cx={280} cy={170} rx={18 * Math.sqrt(level)} ry={18 * Math.sqrt(2 * level)} fill="hsl(var(--primary)/.025)" stroke="hsl(var(--primary)/.3)" />)}<path d={r.points.map((p, i) => `${i ? 'L' : 'M'}${x(p.x)},${y(p.y)}`).join(' ')} fill="none" stroke="var(--learning-orange)" strokeWidth={3} />{r.points.map((p, i) => <circle key={i} cx={x(p.x)} cy={y(p.y)} r={4} fill="var(--learning-orange)" />)}<motion.circle r={8} fill="var(--learning-blue)" initial={false} animate={{ cx: x(r.current.x), cy: y(r.current.y) }} transition={{ duration }} /></>;
  } else {
    const bayes = kind === 'bayesian-inference', r = bayes ? bayesPreparation(example, stage) : interferencePreparation(example, stage);
    const values = bayes ? [(r as ReturnType<typeof bayesPreparation>).fair, (r as ReturnType<typeof bayesPreparation>).biased] : [(r as ReturnType<typeof interferencePreparation>).p0, (r as ReturnType<typeof interferencePreparation>).p1];
    const barX = bayes ? 80 : 285, barWidth = bayes ? 390 : 210;
    geometry = <>{values.map((v, i) => <g key={i}><rect x={barX} y={90 + i * 120} width={barWidth} height={35} rx={6} fill="hsl(var(--muted))" /><motion.rect data-studio-bar={i} x={barX} y={90 + i * 120} height={35} rx={6} fill={i ? 'var(--learning-orange)' : 'var(--learning-blue)'} initial={false} animate={{ width: v * barWidth }} transition={{ duration }} /></g>)}{!bayes && values.map((_, i) => {
      const amplitude = stage === 0 ? i === 0 ? 1 : 0 : stage < 3 ? (i === 1 && example === 0 && stage === 2 ? -1 : 1) / Math.sqrt(2) : Math.sqrt(values[i]);
      return <g key={i}><line x1={65} x2={235} y1={108 + i * 120} y2={108 + i * 120} className="ai-studio-axis" /><motion.line x1={150} y1={108 + i * 120} y2={108 + i * 120} stroke={i ? 'var(--learning-orange)' : 'var(--learning-blue)'} strokeWidth={5} markerEnd={amplitude ? `url(#${id}-arrow)` : undefined} initial={false} animate={{ x2: 150 + 65 * amplitude }} transition={{ duration }} /></g>;
    })}</>;
  }
  const description = kind === 'inner-products' ? (ko ? '파란 벡터와 초록 기준선, 주황 투영점 및 점선 잔차를 단계별로 비교합니다.' : 'Compare a blue vector, green reference line, orange projection and dashed residual in stages.') : kind === 'gradients' ? (ko ? '등고선 위의 파란 매개변수 점과 주황 이동 경로를 비교합니다. 실제 좌표와 손실은 다음 수식에서 확인할 수 있습니다.' : 'Compare the blue parameter point and orange update path on contours. Exact coordinates and loss follow in the equation.') : kind === 'bayesian-inference' ? (ko ? '위 파란 막대는 공정 동전, 아래 주황 막대는 편향 동전 가설의 확률입니다. 두 확률의 합은 하나입니다.' : 'The upper blue bar is the fair-coin hypothesis probability; the lower orange bar is the biased-coin probability. They sum to one.') : (ko ? '각 행에서 왼쪽 화살표는 진폭의 크기와 부호, 오른쪽 막대는 해당 상태의 확률입니다. 위는 영 상태, 아래는 일 상태입니다.' : 'In each row the left arrow shows amplitude magnitude and sign; the right bar shows probability. The upper row is state zero and the lower row is state one.');
  return <figure className="ai-studio-scene"><svg viewBox="0 0 560 340" role="img" aria-labelledby={`${id}-description`}><title id={`${id}-description`}>{description}</title><defs><marker id={`${id}-arrow`} viewBox="0 0 10 10" refX={8} refY={5} markerWidth={5} markerHeight={5} orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="context-stroke" /></marker></defs>{geometry}</svg>{children}<figcaption>{description}</figcaption></figure>;
}
export default function LearningFirstSteps({ kind, language }: { kind: FirstStepId; language: 'ko' | 'en' }) {
  const lesson = firstStepLessons[kind], ko = language === 'ko', id = useId(), frame = useRef<HTMLElement>(null);
  const [example, setExample] = useState(0), [stage, setStage] = useState(0), [playing, setPlaying] = useState(false), [revealed, setRevealed] = useState(false);
  const reduced = useGentleMotion();
  useEffect(() => { if (!playing || reduced) return; if (stage === 3) { setPlaying(false); return; } const timer = setTimeout(() => setStage(stage + 1), 1600); return () => clearTimeout(timer); }, [playing, stage, reduced]);
  useEffect(() => { if (reduced) setPlaying(false); }, [reduced]);
  useEffect(() => { const pause = () => { if (document.hidden) setPlaying(false); }; document.addEventListener('visibilitychange', pause); const observer = new IntersectionObserver(entries => { if (!entries[0].isIntersecting) setPlaying(false); }); if (frame.current) observer.observe(frame.current); return () => { document.removeEventListener('visibilitychange', pause); observer.disconnect(); }; }, []);
  const chooseStage = (n: number) => { setStage(n); setPlaying(false); };
  const playback = <div className="ai-studio-playback"><button type="button" disabled={stage === 0} onClick={() => chooseStage(stage - 1)}>{ko ? '이전 단계' : 'Previous stage'}</button><button type="button" disabled={reduced} onClick={() => { if (playing) setPlaying(false); else { if (stage === 3) setStage(0); setPlaying(true); } }}>{playing ? (ko ? '재생 멈춤' : 'Pause stages') : (ko ? '단계 재생' : 'Play stages')}</button><button type="button" disabled={stage === 3} onClick={() => chooseStage(stage + 1)}>{ko ? '다음 단계' : 'Next stage'}</button><button type="button" onClick={() => { chooseStage(0); setRevealed(false); }}>{ko ? '처음으로' : 'Restart'}</button></div>;
  return <section id="first-understanding" className="ai-learning-studio" ref={frame} data-learning-studio={kind} data-studio-stage={stage} data-studio-example={example} data-studio-playing={String(playing)} aria-labelledby={`${id}-title`}>
    <span className="ai-eyebrow">START WITH A PICTURE</span><h2 id={`${id}-title`}>{lesson.question[language]}</h2><p><LearningText text={lesson.purpose[language]} /></p>
    <div className="ai-studio-prediction"><strong>{ko ? '먼저 예상해보세요' : 'Make a prediction first'}</strong><p><LearningText text={lesson.prediction[language]} /></p></div>
    <label className="ai-studio-example-label" htmlFor={`${id}-example`}>{ko ? '비교할 예제' : 'Example to explore'}</label><select id={`${id}-example`} value={example} onChange={e => { setExample(Number(e.target.value)); setStage(0); setPlaying(false); setRevealed(false); }}>{lesson.examples.map((label, i) => <option key={i} value={i}>{label[language]}</option>)}</select>
    <div className="ai-studio-stages" role="group" aria-label={ko ? '그림 설명 단계' : 'Visual explanation stages'}>{lesson.stages.map((s, i) => <button key={i} type="button" aria-pressed={stage === i} onClick={() => chooseStage(i)}>{s.title[language]}</button>)}</div>
    <div className="ai-studio-comparison"><Scene kind={kind} example={example} stage={stage} reduced={reduced} language={language}>{playback}</Scene><div className="ai-studio-stage-text"><h3>{lesson.stages[stage].title[language]}</h3><p><LearningText text={lesson.stages[stage].body[language]} /></p><LearningMath tex={firstStepDisplayEquation(kind, example, stage)} display label={ko ? '현재 예제의 수식' : 'Equation for the current example'} />{kind === 'gradients' && <><p className="ai-small"><LearningText text={ko ? '요약 값은 소수 셋째 자리까지 표시합니다. $\\approx$는 반올림한 값입니다.' : 'Summary values use up to three decimal places. $\\approx$ marks a rounded value.'} /></p><details className="ai-studio-exact"><summary>{ko ? '정확한 분수 계산 보기' : 'Show the exact fractional calculation'}</summary><div data-studio-exact><LearningMath tex={firstStepEquation(kind, example, stage)} display label={ko ? '현재 예제의 정확한 수식' : 'Exact equation for the current example'} /></div></details></>}<p className="ai-studio-analogy"><LearningText text={lesson.analogy[language]} /></p><p>{ko ? '그림에서 바뀐 부분과 수식의 같은 부분을 찾아보세요.' : 'Find the change in the picture and its matching term in the equation.'}</p></div></div>

    {reduced && <p className="ai-small">{ko ? '동작 줄이기 설정에 따라 버튼으로 한 단계씩 살펴볼 수 있습니다.' : 'With reduced motion, use the buttons to explore one stage at a time.'}</p>}
    <button className="ai-studio-reveal" type="button" aria-expanded={revealed} aria-controls={`${id}-explanation`} onClick={() => setRevealed(!revealed)}>{ko ? '예측과 설명 비교하기' : 'Compare your prediction with the explanation'}</button><div id={`${id}-explanation`} hidden={!revealed}><p><LearningText text={lesson.explanation[language]} /></p></div>
    <div className="ai-studio-worked">{lesson.worked.map((work, i) => <article key={i}><h3>{work.title[language]}</h3><p><LearningText text={work.body[language]} /></p><LearningMath tex={work.tex} display label={ko ? `풀이 예제 ${i + 1}` : `Worked example ${i + 1}`} /></article>)}</div>
    <p className="ai-studio-bridge"><LearningText text={lesson.bridge[language]} /></p><a href="#theory">{ko ? '본문의 유도와 심화 설명 이어 읽기' : 'Continue to the derivation and deeper reading'}</a>
  </section>;
}
