'use client';

import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { usePathname } from 'next/navigation';
import { Pause, Play } from 'lucide-react';
import { useLanguage } from '@/components/language-provider';

const STORAGE_KEY = 'suanlab-motion-paused';
const MotionContext = createContext({ paused: false, reduced: true, canAnimate: false, togglePaused: () => {} });

export function useSiteMotion() { return useContext(MotionContext); }

function ViewportReveals({ enabled }: { enabled: boolean }) {
  const pathname = usePathname();
  const revealed = useRef(new WeakSet<Element>());
  useEffect(() => {
    const root = document.getElementById('main-content');
    if (!enabled || !root || !('IntersectionObserver' in window) || !('animate' in Element.prototype)) return;
    const animations = new Set<Animation>();
    const watched = new WeakSet<Element>();
    const play = (node: Element, frames: Keyframe[], options: KeyframeAnimationOptions) => {
      const animation = node.animate(frames, options);
      animations.add(animation);
      animation.finished.then(() => animations.delete(animation), () => animations.delete(animation));
    };
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const node = entry.target;
        observer.unobserve(node);
        if (revealed.current.has(node)) continue;
        revealed.current.add(node);
        const siblings = node.parentElement ? [...node.parentElement.children] : [];
        const delay = Math.min(4, Math.max(0, siblings.indexOf(node))) * 35;
        // Keep text contrast intact throughout the entrance, including delayed cards.
        if (node.getAttribute('data-motion-reveal') !== 'route') play(node,
          [{ transform: `translateY(${innerWidth < 640 ? 8 : 16}px)` }, { transform: 'translateY(0)' }],
          { duration: 480, delay, fill: 'backwards', easing: 'cubic-bezier(.2,.7,.2,1)' });
        node.querySelectorAll('[data-motion-bar]').forEach((bar, index) => play(bar,
          [{ transform: 'scaleY(0)' }, { transform: 'scaleY(1)' }], { duration: 600, delay: Math.min(index * 20, 240), easing: 'cubic-bezier(.2,.7,.2,1)' }));
        node.querySelectorAll('[data-motion-path]').forEach((path, index) => play(path,
          [{ strokeDashoffset: '1' }, { strokeDashoffset: '0' }], { duration: 900, delay: index * 100, easing: 'ease-out' }));
      }
    }, { threshold: .08, rootMargin: '0px 0px -24px 0px' });
    const watch = (node: Element) => {
      const add = (element: Element) => {
        if (watched.has(element) || revealed.current.has(element)) return;
        watched.add(element); observer.observe(element);
      };
      if (node.matches('[data-motion-reveal]')) add(node);
      node.querySelectorAll('[data-motion-reveal]').forEach(add);
    };
    watch(root);
    // Inspect inserted elements only; typing and count-up text never rescan the page.
    const mutations = new MutationObserver(records => records.forEach(record => record.addedNodes.forEach(node => {
      if (node instanceof Element) watch(node);
    })));
    mutations.observe(root, { childList: true, subtree: true });
    return () => { observer.disconnect(); mutations.disconnect(); animations.forEach(animation => animation.cancel()); };
  }, [enabled, pathname]);
  return null;
}

export function SiteMotionProvider({ children }: { children: ReactNode }) {
  const [paused, setPaused] = useState(false);
  const [reduced, setReduced] = useState(true);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const preference = matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(preference.matches);
    update(); preference.addEventListener('change', update);
    try { setPaused(localStorage.getItem(STORAGE_KEY) === 'true'); } catch { /* Motion preferences also work without storage. */ }
    setReady(true);
    return () => preference.removeEventListener('change', update);
  }, []);
  const canAnimate = ready && !paused && !reduced;
  useEffect(() => { document.documentElement.dataset.siteMotion = canAnimate ? 'running' : 'paused'; }, [canAnimate]);
  function togglePaused() {
    const next = !paused; setPaused(next);
    try { localStorage.setItem(STORAGE_KEY, String(next)); } catch { /* Keep the in-session selection. */ }
  }
  return <MotionContext.Provider value={{ paused, reduced, canAnimate, togglePaused }}>
    <ViewportReveals enabled={canAnimate} />{children}
  </MotionContext.Provider>;
}

export function MotionControls({ compact = false }: { compact?: boolean }) {
  const { paused, reduced, togglePaused } = useSiteMotion();
  const { language } = useLanguage();
  const ko = language === 'ko';
  const label = reduced ? (ko ? '시스템 설정: 모션 감소' : 'System setting: reduced motion')
    : paused ? (ko ? '사이트 모션 재생' : 'Play site motion') : (ko ? '사이트 모션 일시정지' : 'Pause site motion');
  return <button type="button" disabled={reduced} onClick={togglePaused} aria-pressed={paused || reduced} aria-label={compact ? label : undefined} title={compact ? label : undefined}
    className={compact ? 'inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/20 bg-slate-950/65 text-cyan-100 hover:bg-slate-800 disabled:cursor-default'
      : 'inline-flex max-w-full items-center gap-2 rounded-lg border px-3 py-2 text-left text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-default'}>
    {paused || reduced ? <Play aria-hidden="true" className="h-3.5 w-3.5 shrink-0" /> : <Pause aria-hidden="true" className="h-3.5 w-3.5 shrink-0" />}
    {!compact && label}
  </button>;
}
