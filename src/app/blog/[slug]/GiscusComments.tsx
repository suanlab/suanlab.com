'use client';

import { useEffect, useRef } from 'react';
import { useTheme } from 'next-themes';

export default function GiscusComments() {
  const ref = useRef<HTMLDivElement>(null);
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    if (!ref.current) return;

    // Clear existing comments
    ref.current.innerHTML = '';

    const script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.setAttribute('data-repo', 'suanlab/suanlab.com');
    script.setAttribute('data-repo-id', 'R_kgDOQwIYFg');
    script.setAttribute('data-category', 'General');
    script.setAttribute('data-category-id', 'DIC_kwDOQwIYFs4C4WZ0');
    script.setAttribute('data-mapping', 'pathname');
    script.setAttribute('data-strict', '0');
    script.setAttribute('data-reactions-enabled', '1');
    script.setAttribute('data-emit-metadata', '0');
    script.setAttribute('data-input-position', 'top');
    script.setAttribute('data-theme', resolvedTheme === 'dark' ? 'dark' : 'light');
    script.setAttribute('data-lang', 'ko');
    script.crossOrigin = 'anonymous';
    script.async = true;

    ref.current.appendChild(script);
  }, [resolvedTheme]);

  return (
    <div className="mt-12 pt-8 border-t">
      <h3 className="text-lg font-semibold mb-6">댓글</h3>
      <div ref={ref} />
    </div>
  );
}
