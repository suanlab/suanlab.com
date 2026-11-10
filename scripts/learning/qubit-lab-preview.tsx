import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { LanguageProvider, useLanguage } from '../../src/components/language-provider';
import QubitLab from '../../src/components/learning/qubit-lab';
import { qubitTeaching } from '../../src/data/qubit-teaching';
import { isQubitTeachingId, qubitTeachingIds, type QubitTeachingId } from '../../src/lib/learning-qubit-teaching-input';
import '../../src/styles/qubit-lab.css';
const params = new URL(location.href).searchParams, requested = params.get('unit') ?? '';
const initial = isQubitTeachingId(requested) ? requested : qubitTeachingIds[0];
try { localStorage.setItem('suanlab-language', params.get('lang') === 'en' ? 'en' : 'ko'); } catch { /* Session language fallback. */ }
document.documentElement.classList.toggle('dark', params.get('theme') === 'dark');
function Preview() {
  const { language, setLanguage } = useLanguage(), [kind, setKind] = useState<QubitTeachingId>(initial);
  const unit = qubitTeaching.find(item => item.unit.id === kind)!.unit;
  return <main className="ai-book" data-qubit-preview={kind} data-language={language} style={{ maxWidth: 1100, padding: 16, margin: 'auto', minWidth: 0 }}>
    <button type="button" data-preview-language aria-label={language === 'ko' ? '언어 전환' : 'Change language'} onClick={() => setLanguage(language === 'ko' ? 'en' : 'ko')}>{language === 'ko' ? 'EN' : 'KO'}</button>
    <label htmlFor="unit-select">{language === 'ko' ? '검토할 단원' : 'Unit to review'}</label>
    <select id="unit-select" value={kind} style={{ width: '100%', minHeight: 40, background: 'hsl(var(--card))', color: 'hsl(var(--foreground))', border: '1px solid hsl(var(--border))', margin: '8px 0 24px' }} onChange={e => { if (isQubitTeachingId(e.target.value)) setKind(e.target.value); }}>{qubitTeaching.map(item => <option key={item.unit.id} value={item.unit.id}>{item.unit.title[language]}</option>)}</select>
    <h1 style={{ fontSize: 28, fontWeight: 700, lineHeight: 1.4, wordBreak: 'keep-all' }}>{unit.title[language]}</h1>
    <QubitLab kind={kind} language={language} />
  </main>;
}
createRoot(document.getElementById('root')!).render(<LanguageProvider><Preview /></LanguageProvider>);
