'use client';
import { memo, useMemo } from 'react';
import { renderToString } from 'katex';
export const LearningMath = memo(function LearningMath({ tex, display = false, label }: { tex: string; display?: boolean; label?: string }) {
  const html = useMemo(() => renderToString(tex, { displayMode: display, throwOnError: true, strict: 'error', trust: false, output: 'htmlAndMathml' }), [tex, display]);
  return display ? <div className="ai-learning-math ai-learning-math-block" tabIndex={0} role="region" aria-label={label ?? 'Equation'} dangerouslySetInnerHTML={{ __html: html }} /> : <span className="ai-learning-math" dangerouslySetInnerHTML={{ __html: html }} />;
});
/** Authored prose only: mathematical spans are explicitly delimited, never guessed. */
export function LearningText({ text }: { text: string }) {
  return <>{text.split(/(\$[^$\n]+\$)/g).map((part, i) => part.startsWith('$') && part.endsWith('$') ? <LearningMath key={i} tex={part.slice(1, -1)} /> : part)}</>;
}
