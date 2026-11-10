'use client';
import React, { useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { thermalFieldSteps } from '@/data/thermal-field-teaching';

function Graphic({ step, language }: { step: number; language: 'ko' | 'en' }) {
  const t = (ko: string, en: string) => language === 'ko' ? ko : en;
  const blue = 'var(--learning-blue)', orange = 'var(--learning-orange)';
  const text = (x: number, y: number, label: string, fill = 'currentColor') => <text x={x} y={y} fill={fill} fontSize={16}>{label}</text>;
  return <svg viewBox="0 0 640 250" width="640" height="250" role="img" aria-label={thermalFieldSteps[step].caption[language]} className="thermal-concept-svg">
    {step === 0 && <><path d="M50 205H600M320 30V205" fill="none" stroke="currentColor" />{[1, 4].map((k, i) => <path key={k} d={Array.from({ length: 121 }, (_, j) => { const x = -3 + j / 20; return `${j ? 'L' : 'M'}${320 + x * 85},${205 - 155 * Math.exp(-k * x * x / 2)}`; }).join(' ')} fill="none" stroke={i ? orange : blue} strokeWidth={3} />)}{text(70, 35, 'K=1', blue)}{text(480, 35, 'K=4', orange)}{text(570, 235, 'φ')}{text(60, 235, t('가중치 exp(−Kφ²/2)', 'Weight exp(−Kφ²/2)'))}</>}
    {step === 1 && <><path d="M60 125H590M320 20V230" stroke="currentColor" /><path d="M230 35L410 215" stroke={blue} strokeWidth={4}/><path d="M320 125H380V185" stroke={orange} strokeDasharray="6 4" fill="none" />{text(390, 140, 'Δφ₁=1', orange)}{text(390, 190, 'Δφ₂=−1', orange)}{text(75, 50, 'dvol = √2 dz', blue)}{text(555, 115, 'φ₁')}{text(335, 30, 'φ₂')}{text(60, 225, 'φ₁+φ₂=0')}</>}
    {step === 2 && <><path d="M50 130H605" stroke="currentColor" />{[2, 0, -2].map((u, i) => <g key={i}><rect x={110 + i * 170} y={50} width={35} height={80} fill={blue}/><rect x={150 + i * 170} y={u < 0 ? 130 : 130 - u * 40} width={35} height={Math.abs(u) * 40 || 2} fill={orange}/>{text(110 + i * 170, 240, ['K=I', 'K=T I', 'K=T² I'][i])}{text(150 + i * 170, u < 0 ? 225 : 120 - u * 40, String(u), orange)}</g>)}{text(65, 25, '⟨E⟩=2', blue)}{text(460, 25, 'U=2,0,−2', orange)}</>}
    {step === 3 && <><rect x={50} y={70} width={235} height={95} rx={12} fill="none" stroke={blue} strokeWidth={2}/><rect x={360} y={70} width={230} height={95} rx={12} fill="none" stroke={orange} strokeWidth={2}/>{text(75, 110, 'S_G = log(2π)+1', blue)}{text(385, 110, 'S = log(2π)', orange)}{text(300, 125, '−1')}{text(75, 210, 'q=2, K=T I, φ*=1')}</>}
    {step === 4 && <>{text(50, 60, 'φ* → 2φ*', blue)}{text(290, 60, 'Δlog Z = −2 log 2', blue)}<path d="M50 100H590" stroke="currentColor" opacity={.3}/>{text(50, 145, 'φ=2z; Kz=4K; Jz=2J', orange)}{text(50, 180, 'φ*z=φ*/2', orange)}{text(330, 210, 'Δlog Z = 0', orange)}</>}
    {step === 5 && <>{[0, 1, 2, 3].map((k, i) => <g key={k}><rect x={60 + i * 145} y={65} width={105} height={75} rx={8} stroke={k % 2 ? orange : blue} fill="none" strokeWidth={2}/>{text(80 + i * 145, 110, `k=${k}`)}</g>)}<path d="M255 150V185H545V150" stroke={orange} fill="none" strokeWidth={2}/>{text(75, 35, 'N=4')}{text(220, 220, t('공액 쌍 → 실수 두 좌표', 'Conjugate pair → two real coordinates'))}{text(70, 165, 'real', blue)}{text(365, 165, 'real', blue)}</>}
  </svg>;
}
export default function ThermalFieldConcept({ language }: { language: 'ko' | 'en' }) {
  const [step, setStep] = useState(0), [motion, setMotion] = useState(true), id = useId(), current = thermalFieldSteps[step];
  const t = (ko: string, en: string) => language === 'ko' ? ko : en;
  return <section className="ai-intuition thermal-concept" aria-labelledby={id} data-thermal-concept={step}>
    <h2 id={id}>{t('그림으로 측도와 열 자유에너지 이해하기', 'Understand measure and thermal free energy visually')}</h2>
    <p>{t('각 그림은 고정된 설명 기준입니다. 아래에서 직접 적용한 입력의 계산 결과와 구별해 읽으세요.', 'Each diagram is a fixed teaching reference. Distinguish it from the computed result of your applied input below.')}</p>
    <div className="ai-concept-steps" role="group" aria-label={t('개념 단계', 'Concept steps')}>{thermalFieldSteps.map((s, i) => <Button key={s.id} variant="outline" aria-pressed={step === i} onClick={() => setStep(i)}>{i + 1}. {s.title[language]}</Button>)}</div>
    <label className="scalar-check"><input type="checkbox" checked={motion} onChange={e => setMotion(e.target.checked)}/>{t('개념 전환 효과', 'Animate concept transitions')}</label>
    <article className="information-step" key={step} style={!motion ? { animation: 'none' } : undefined} aria-live="polite"><h3>{current.title[language]}</h3><p>{current.body[language]}</p><p className="thermal-formula">{current.formula}</p>
      <figure><div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('개념 그림 · 좌우 이동 가능', 'Concept diagram; scroll horizontally')}><Graphic step={step} language={language}/></div><figcaption>{current.caption[language]}</figcaption></figure>
    </article>
  </section>;
}
