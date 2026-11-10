'use client';
import React, { useId } from 'react';
import { Button } from '@/components/ui/button';
import { blockOrderSteps } from '@/data/block-order-teaching';

function Graphic({ step,language }: { step: number; language: 'ko' | 'en' }) {
  const t = (ko: string,en: string) => language === 'ko' ? ko : en, blue = 'var(--learning-blue)', orange = 'var(--learning-orange)';
  const text = (x: number,y: number,label: string,color = 'currentColor') => <text x={x} y={y} fill={color} fontSize={16}>{label}</text>;
  const pair = (x: number,y: number,label: string) => <g><rect x={x} y={y} width={110} height={44} rx={8} stroke={blue} fill="none" strokeWidth={2}/>{text(x+16,y+28,label)}</g>;
  return <svg className="block-order-concept-svg" viewBox="0 0 640 280" width="640" height="280" role="img" aria-label={blockOrderSteps[step].caption[language]}>
    {step === 0 && <>{[['+','−','+','−'],['+','−','−','+']].map((spins,row) => <g key={row}>
      {[0,1].map(b => <rect key={b} x={35+b*122} y={55+row*120} width={114} height={62} rx={8} fill="none" stroke={blue} strokeWidth={2}/>)}
      {spins.map((s,i) => text(53+i*60,94+row*120,s,s === '+' ? blue : orange))}
      <path d={`M300 ${85+row*120}H350`} stroke="currentColor" strokeWidth={2}/>{text(380,91+row*120,'m=(0,0)',blue)}</g>)}{text(40,28,t('내부 배치','Internal arrangements'))}{text(380,28,t('같은 프로필','Same profile'))}</>}
    {step === 1 && <>{['+−|+−','+−|−+','−+|+−','−+|−+'].map((s,i) => <g key={s}>{text(45+i*150,32,s)}<rect x={55+i*150} y={210-[50,100,100,50][i]} width={65} height={[50,100,100,50][i]} fill={i===0||i===3 ? orange : blue}/>{text(65+i*150,245,i===0||i===3 ? '1/6' : '1/3')}</g>)}<path d="M40 210H610" stroke="currentColor"/>{text(60,80,'x=1/2; k=(1,1)')}</>}
    {step === 2 && <>{text(50,35,'M total=0; x=1/2')}{['(+2,−2)','(0,0)','(−2,+2)'].map((s,i) => <g key={s}><rect x={45+i*195} y={80} width={160} height={90} rx={10} fill="none" stroke={blue} strokeWidth={2}/>{text(70+i*195,115,s)}{text(90+i*195,150,['2/7','3/7','2/7'][i],orange)}</g>)}{text(70,220,'P(M total=0)=7/27')}{text(70,255,'P(k=(1,1))=1/9')}</>}
    {step === 3 && <><path d="M45 220H600M45 130H600" stroke={orange} strokeDasharray="6 4"/>{[1/3,2/3,2/3,1/3].map((m,i) => <g key={i}><rect x={65+i*145} y={220-180*m} width={58} height={180*m} fill={blue}/>{text(70+i*145,210-180*m,i===0||i===3 ? '1/3' : '2/3')}{text(80+i*145,250,String(i))}</g>)}{text(50,30,t('조건부 위치 평균','Conditional site means'))}{text(440,115,'m=1/2',orange)}</>}
    {step === 4 && <>{text(45,45,'Q=3/4',blue)}{text(300,45,'Qref=4x^(5/2)',orange)}{text(45,90,'log(w/wref)=log(9/8)/2')}{pair(50,160,'m_A')}{pair(260,160,'m_B')}{pair(470,160,'m_C')}<path d="M160 182H260M370 182H470" stroke={orange} strokeWidth={3}/>{text(80,250,t('차이 제곱 항 + 국소 이차항 이동','Difference penalties + local quadratic shifts'))}</>}
    {step === 5 && <>{pair(40,45,'A')}{pair(260,45,'B: M=0')}{pair(480,45,'C')}<path d="M150 67H260M370 67H480" stroke={blue} strokeWidth={3}/>{text(65,125,t('열린 경계: 끝 쌍 의존; OR=16/25','open: A,C dependent | B; OR=16/25'),orange)}{pair(40,205,'A')}{pair(260,205,'B')}{pair(480,205,'C')}<path d="M150 227H260M370 227H480M95 205V165H535V205" stroke={blue} strokeWidth={3} fill="none"/>{text(180,190,t('주기 연결: A와C는 이웃','Periodic: A and C are neighbors'))}</>}
  </svg>;
}
export default function BlockOrderConcept({ language,step,onStep,motion,onMotion }: { language: 'ko' | 'en'; step: number; onStep: (step: number) => void; motion: boolean; onMotion: (motion: boolean) => void }) {
  const id = useId(), current = blockOrderSteps[step], t = (ko: string,en: string) => language === 'ko' ? ko : en;
  return <section className="ai-intuition block-order-concept" id="visual-intuition" aria-labelledby={id} data-block-concept={step}>
    <h2 id={id}>{t('그림으로 국소 질서변수 이해하기','Understand local order parameters visually')}</h2>
    <p>{t('아래 개념 그림은 고정된 설명 기준입니다. 직접 적용한 입력의 결과는 실험 결과 영역에서 확인하세요.','These diagrams are fixed teaching references. Find results for your applied input in the experiment result below.')}</p>
    <div className="ai-concept-steps" role="group" aria-label={t('개념 단계','Concept steps')}>{blockOrderSteps.map((s,i) => <Button key={s.id} variant="outline" aria-pressed={step===i} onClick={() => onStep(i)}>{i+1}. {s.title[language]}</Button>)}</div>
    <label className="scalar-check"><input type="checkbox" checked={motion} onChange={e => onMotion(e.target.checked)}/>{t('개념 전환 효과','Animate concept transitions')}</label>
    <article className="information-step" key={step} style={!motion ? { animation: 'none' } : undefined} aria-live="polite"><h3>{current.title[language]}</h3><p>{current.body[language]}</p><p className="block-order-formula">{current.formula}</p>
      <figure><div className="qubit-scroll" role="region" tabIndex={0} aria-label={t('개념 그림 · 좌우 이동 가능','Concept diagram; scroll horizontally')}><Graphic step={step} language={language}/></div><figcaption>{current.caption[language]}</figcaption></figure>
    </article>
  </section>;
}
