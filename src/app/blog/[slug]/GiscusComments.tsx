'use client';

import { useEffect, useRef, useState } from 'react';
import { useTheme } from 'next-themes';
import { useLanguage } from '@/components/language-provider';

const GISCUS_ORIGIN = 'https://giscus.app';
function themeName(theme: 'dark' | 'light') {
  if (theme !== 'dark') return 'light';
  // Public iframes cannot fetch styles from a visitor's private HTTP loopback host.
  if (window.location.protocol !== 'https:') return 'dark';
  return new URL('/assets/styles/giscus-dark.css', window.location.href).href;
}

function GiscusEmbed({ theme, language }: { theme: 'dark' | 'light'; language: 'ko' | 'en' }) {
  const ref = useRef<HTMLDivElement>(null);
  const initial = useRef({ theme, language });

  useEffect(() => {
    const container = ref.current;
    if (!container) return;
    const script = document.createElement('script');
    script.src = `${GISCUS_ORIGIN}/client.js`;
    const attributes = {
      repo: 'suanlab/suanlab.com', 'repo-id': 'R_kgDOQwIYFg',
      category: 'General', 'category-id': 'DIC_kwDOQwIYFs4C4WZ0',
      mapping: 'pathname', strict: '0', 'reactions-enabled': '1',
      'emit-metadata': '0', 'input-position': 'top',
      theme: themeName(initial.current.theme), lang: initial.current.language, loading: 'lazy',
    };
    for (const [key, value] of Object.entries(attributes)) script.setAttribute(`data-${key}`, value);
    script.crossOrigin = 'anonymous';
    script.async = true;
    container.appendChild(script);
    return () => { container.replaceChildren(); };
  }, []);

  useEffect(() => {
    const container = ref.current;
    if (!container) return;
    const frames = new Set<HTMLIFrameElement>();
    const update = () => {
      for (const frame of container.querySelectorAll('iframe')) {
        if (!frames.has(frame)) { frames.add(frame); frame.addEventListener('load', update); }
        frame.style.colorScheme = theme;
        frame.style.backgroundColor = theme === 'dark' ? '#0d1117' : '#ffffff';
        frame.contentWindow?.postMessage({ giscus: { setConfig: { theme: themeName(theme), lang: language } } }, GISCUS_ORIGIN);
      }
    };
    const observer = new MutationObserver(update);
    observer.observe(container, { childList: true, subtree: true });
    update();
    return () => { observer.disconnect(); for (const frame of frames) frame.removeEventListener('load', update); };
  }, [theme, language]);

  return <div ref={ref} style={{ colorScheme: theme }} />;
}

export default function GiscusComments() {
  const { resolvedTheme } = useTheme();
  const { language } = useLanguage();
  const [mounted, setMounted] = useState(false);
  // A stored theme can resolve before this streamed boundary hydrates.
  useEffect(() => { setMounted(true); }, []);
  return <div className="mt-12 border-t pt-8">
    <h3 className="mb-6 text-lg font-semibold">{language === 'ko' ? '댓글' : 'Comments'}</h3>
    {mounted && resolvedTheme && <GiscusEmbed theme={resolvedTheme === 'dark' ? 'dark' : 'light'} language={language} />}
  </div>;
}
