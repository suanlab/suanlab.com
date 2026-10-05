'use client';

import { useEffect, useRef, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { ArrowRight, Pause, Play } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';
import { labIdentities, labIdentityName } from '@/data/lab-identity';

const READING_MS = 14_000;
const uWords = Array.from(new Set(labIdentities.map(identity => identity.unifying)));
const aWords = Array.from(new Set(labIdentities.map(identity => identity.approach)));
type TypingPhase = 'typing' | 'reading' | 'deleting';
const connections = [
  { title: 'Superintelligence Research', ko: '초지능을 향한 모델과 학습 방법론 연구', en: 'Exploring models and learning methods for superintelligence', href: '/research/' },
  { title: 'Applied Intelligence', ko: '데이터와 실제 문제를 연결하는 프로젝트', en: 'Connecting data with real-world problems', href: '/project/' },
  { title: 'Open Knowledge', ko: '논문에서 코드와 강의로 이어지는 지식 공유', en: 'Sharing knowledge through papers, code, and teaching', href: '/lecture/' },
];

function TypedIdentityWord({ word, reserve, length, slot }: { word: string; reserve: string[]; length: number; slot: 'u' | 'a' }) {
  return (
    <span className="identity-variable-word relative inline-grid whitespace-nowrap" data-slot={slot}>
      {reserve.map(term => <span key={term} className="invisible col-start-1 row-start-1">{term}</span>)}
      <span className="absolute left-0 top-0">{Array.from(word).map((letter, characterIndex) => (
        <span key={characterIndex} className={`identity-letter ${characterIndex === 0 ? 'text-cyan-200' : ''}`} style={{ opacity: characterIndex < length ? 1 : 0 }}>
          {letter}{characterIndex === (length === 0 ? 0 : length - 1) && <span className={`identity-cursor ${length === 0 ? 'identity-cursor-start' : ''}`} />}
        </span>
      ))}</span>
    </span>
  );
}

export function HomeIdentityHero({ stats }: { stats: { label: string; value: string }[] }) {
  const { language, t } = useLanguage();
  const [typing, setTyping] = useState<{ index: number; uLength: number; aLength: number; phase: TypingPhase }>({ index: 0, uLength: 0, aLength: 0, phase: 'typing' });
  const [paused, setPaused] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(true);
  const [visible, setVisible] = useState(true);
  const [inView, setInView] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);
  const ko = language === 'ko';
  const running = !paused && !reducedMotion && visible && inView;
  const { index, uLength, aLength, phase } = typing;
  const identity = labIdentities[index];

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const updateMotion = () => setReducedMotion(preference.matches);
    const updateVisibility = () => setVisible(!document.hidden);
    updateMotion(); updateVisibility();
    preference.addEventListener('change', updateMotion);
    document.addEventListener('visibilitychange', updateVisibility);
    const observer = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting));
    if (sectionRef.current) observer.observe(sectionRef.current);
    return () => {
      preference.removeEventListener('change', updateMotion);
      document.removeEventListener('visibilitychange', updateVisibility);
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    if (!running) return;
    const current = labIdentities[index];
    const nextIndex = (index + 1) % labIdentities.length;
    const next = labIdentities[nextIndex];
    const changeU = current.unifying !== next.unifying;
    const changeA = current.approach !== next.approach;
    const cleared = (!changeU || uLength === 0) && (!changeA || aLength === 0);
    const delay = phase === 'reading' ? READING_MS : phase === 'deleting' ? (cleared ? 350 : 18) : 65;
    const timer = window.setTimeout(() => {
      if (phase === 'typing') {
        const nextU = Math.min(uLength + 1, current.unifying.length);
        const nextA = Math.min(aLength + 1, current.approach.length);
        setTyping({ index, uLength: nextU, aLength: nextA, phase: nextU === current.unifying.length && nextA === current.approach.length ? 'reading' : 'typing' });
      } else if (phase === 'reading') {
        setTyping({ index, uLength, aLength, phase: 'deleting' });
      } else if (!cleared) {
        setTyping({ index, uLength: changeU ? Math.max(0, uLength - 1) : uLength, aLength: changeA ? Math.max(0, aLength - 1) : aLength, phase: 'deleting' });
      } else {
        setTyping({ index: nextIndex, uLength: changeU ? 0 : uLength, aLength: changeA ? 0 : aLength, phase: 'typing' });
      }
    }, delay);
    return () => window.clearTimeout(timer);
  }, [index, uLength, aLength, phase, running]);

  function togglePause() {
    // Pausing finishes the current sentence so the entire explanation is readable.
    if (!paused) setTyping({ index, uLength: identity.unifying.length, aLength: identity.approach.length, phase: 'reading' });
    setPaused(current => !current);
  }

  return (
    <section ref={sectionRef} className="identity-hero relative isolate overflow-hidden bg-slate-950 text-white" aria-labelledby="lab-identity-heading" data-motion={running ? 'running' : 'paused'}>
      <noscript><style>{'.identity-letter, .identity-copy { opacity: 1 !important; } .identity-cursor { display: none; }'}</style></noscript>
      <Image src="/assets/images/slider/2.jpg" alt="" fill priority sizes="100vw" className="-z-20 object-cover object-center" />
      <div aria-hidden="true" className="identity-photo-shade absolute inset-0 -z-10" />
      <div className="container grid items-center gap-10 py-14 md:py-20 lg:grid-cols-[1.5fr_1fr] lg:gap-16 lg:py-24">
        <div className="min-w-0">
          <p className="mb-5 flex items-center gap-3 text-xs font-medium uppercase tracking-[0.2em] text-cyan-200">
            <span aria-hidden="true" className="h-px w-8 shrink-0 bg-cyan-300" /> Superintelligence Research
          </p>
          <h1 id="lab-identity-heading" className="identity-wordmark text-[clamp(3.2rem,11vw,6.5rem)] font-semibold leading-none tracking-[-0.055em]" aria-label="SuanLab">
            <span className="text-white">SUAN</span><span className="text-cyan-200">LAB</span>
          </h1>
          <p className="mt-5 text-sm text-slate-200">{ko ? '초지능의 가능성을 연구하고, 실제 가치로 연결합니다.' : 'Exploring superintelligence. Connecting research to real-world value.'}</p>

          <div id="lab-identity-description" className="identity-panel mt-8 grid border-l-2 border-cyan-300/70 pl-5 md:pl-6">
            <p lang="en" className="identity-accessible-name sr-only">{labIdentityName(identity)}</p>
            <p lang="en" aria-hidden="true" className="identity-typed-name flex flex-wrap gap-x-2 gap-y-1 text-base font-medium leading-relaxed md:text-xl">
              <span className="identity-fixed-word whitespace-nowrap" data-slot="s"><span className="text-cyan-200">S</span>uperintelligence</span>
              <TypedIdentityWord slot="u" word={identity.unifying} reserve={uWords} length={uLength} />
              <TypedIdentityWord slot="a" word={identity.approach} reserve={aWords} length={aLength} />
              <span className="identity-fixed-word whitespace-nowrap" data-slot="n"><span className="text-cyan-200">N</span>eural-networks</span>
              <span className="identity-fixed-word whitespace-nowrap text-cyan-200" data-slot="lab">LAB</span>
            </p>
            <div className="mt-4 grid">
              {labIdentities.map((entry, position) => {
                const active = position === index;
                return <div key={labIdentityName(entry)} className={`identity-slide ${active ? 'identity-slide-active' : ''}`} aria-hidden={!active}>
                  <div className="identity-copy" data-readable={active && phase !== 'deleting'} style={{ opacity: active && phase !== 'deleting' ? 1 : 0 }}>
                    <h2 lang="ko" className="text-xl font-semibold leading-relaxed [word-break:keep-all] md:text-2xl">{entry.titleKo}</h2>
                    <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-200 [word-break:keep-all] md:text-base">{ko ? entry.descriptionKo : entry.descriptionEn}</p>
                  </div>
                </div>;
              })}
            </div>
          </div>

          <div className="mt-4 h-11">
            {!reducedMotion && <button type="button" className="identity-pause inline-flex min-h-11 items-center gap-2 rounded-md px-1 text-xs text-cyan-200 hover:text-white" aria-controls="lab-identity-description" onClick={togglePause}>
              {paused ? <Play aria-hidden="true" className="h-3 w-3" /> : <Pause aria-hidden="true" className="h-3 w-3" />}
              {ko ? (paused ? '애니메이션 이어보기' : '잠시 멈추고 읽기') : (paused ? 'Resume animation' : 'Pause to read')}
            </button>}
          </div>
          <div className="mt-7 flex flex-col gap-3 sm:flex-row">
            <Button size="lg" className="bg-blue-600 text-white hover:bg-blue-700" asChild><Link href="/publication/">{t('cta.btn.publications') as string}<ArrowRight aria-hidden="true" className="ml-2 h-4 w-4" /></Link></Button>
            <Button size="lg" variant="outline" className="border-white/30 bg-slate-950/40 text-white hover:bg-white/10 hover:text-white" asChild><Link href="/suan/">{t('hero.btn.profile') as string}</Link></Button>
          </div>
        </div>

        <div className="min-w-0 rounded-2xl border border-white/25 bg-slate-950/65 p-6 backdrop-blur-sm md:p-8">
          <div aria-hidden="true" className="identity-network relative mx-auto mb-6 max-w-xs">
            <svg viewBox="0 0 320 180" className="w-full overflow-visible" fill="none">
              <path d="M52 40L160 90L268 40M52 140L160 90L268 140M52 40L52 140M268 40L268 140" stroke="#67e8f9" strokeOpacity=".3" />
              <path className="identity-signal" d="M52 40L160 90L268 140M52 140L160 90L268 40" stroke="#a5f3fc" strokeWidth="2" strokeDasharray="8 100" />
              {[{ x: 52, y: 40, label: 'LLM' }, { x: 268, y: 40, label: 'RAG' }, { x: 52, y: 140, label: 'Agents' }, { x: 268, y: 140, label: 'Multimodal' }].map(({ x, y, label }, n) => (
                <g key={label}><circle className="identity-node" style={{ animationDelay: `${n * .6}s` }} cx={x} cy={y} r="20" stroke="#67e8f9" strokeOpacity=".45" /><circle cx={x} cy={y} r="4" fill="#a5f3fc" /><text x={x} y={y + 35} textAnchor="middle" fill="#e2e8f0" fontSize="11">{label}</text></g>
              ))}
              <circle cx="160" cy="90" r="37" fill="#0f172a" stroke="#67e8f9" strokeOpacity=".7" />
              <text x="160" y="94" textAnchor="middle" fill="#cffafe" fontSize="13" fontWeight="600" letterSpacing="1">SUANLAB</text>
            </svg>
          </div>
          <p className="mb-2 border-b border-white/20 pb-4 text-xs font-medium uppercase tracking-[0.16em] text-slate-200">Research / Practice / Education</p>
          {connections.map((item, position) => (
            <Link key={item.href} href={item.href} className="group flex items-start gap-4 rounded-lg py-4 transition-colors hover:bg-white/5">
              <span className="pt-1 font-mono text-xs text-cyan-200">0{position + 1}</span>
              <div className="min-w-0 flex-1"><h2 className="text-base font-semibold">{item.title}</h2><p className="mt-1 text-sm leading-relaxed text-slate-200">{ko ? item.ko : item.en}</p></div>
              <ArrowRight aria-hidden="true" className="mt-1 h-4 w-4 shrink-0 text-cyan-200 transition-transform group-hover:translate-x-1 motion-reduce:transform-none" />
            </Link>
          ))}
        </div>
      </div>

      <div className="relative border-t border-white/20 bg-slate-950/75 backdrop-blur-sm">
        <div className="container grid grid-cols-2 gap-8 py-7 md:grid-cols-4">{stats.map(stat => <div key={stat.label} className="text-center"><div className="text-3xl font-bold text-white md:text-4xl">{stat.value}</div><div className="mt-1 text-sm text-slate-200">{stat.label}</div></div>)}</div>
      </div>
    </section>
  );
}
