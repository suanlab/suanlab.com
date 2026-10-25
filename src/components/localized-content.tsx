'use client';

import { cloneElement, createContext, isValidElement, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactElement, type ReactNode } from 'react';
import { usePathname } from 'next/navigation';
import { useLanguage } from '@/components/language-provider';
import { englishText, type EnglishDictionary } from '@/data/i18n/translate';

const EMPTY: EnglishDictionary = {};
const DictionaryContext = createContext<EnglishDictionary>(EMPTY);

export function useDisplayText() {
  const { language } = useLanguage();
  const dictionary = useContext(DictionaryContext);
  return (text: string) => language === 'en' ? englishText(text, dictionary) : text;
}

type RegisterTitleScope = (dictionary: EnglishDictionary) => () => void;
const TitleScopeContext = createContext<RegisterTitleScope | null>(null);

// One owner updates document.title; nested page dictionaries only register their scope.
// Independent observers can otherwise overwrite each other during client navigation.
function TitleManager({ children, dictionary }: { children: ReactNode; dictionary: EnglishDictionary }) {
  const [scopes, setScopes] = useState<Map<symbol, EnglishDictionary>>(() => new Map());
  const register = useCallback<RegisterTitleScope>(scope => {
    const key = Symbol();
    setScopes(previous => new Map(previous).set(key, scope));
    return () => setScopes(previous => { const next = new Map(previous); next.delete(key); return next; });
  }, []);
  const combined = useMemo(() => Object.assign({}, dictionary, ...scopes.values()), [dictionary, scopes]);
  return <TitleScopeContext.Provider value={register}><LocalizedDocumentTitle dictionary={combined} />{children}</TitleScopeContext.Provider>;
}

function TitleScope({ dictionary }: { dictionary: EnglishDictionary }) {
  const register = useContext(TitleScopeContext);
  useEffect(() => register?.(dictionary), [register, dictionary]);
  return null;
}

interface TitleSource { path: string; original: string; translated: string; }

export function titleSourceForPath(previous: TitleSource, path: string, current: string): TitleSource {
  // Next can update metadata before usePathname changes. Keep the raw source when
  // that earlier update has already been translated by this same title owner.
  return current === previous.original || current === previous.translated
    ? { ...previous, path }
    : { path, original: current, translated: '' };
}

function LocalizedDocumentTitle({ dictionary }: { dictionary: EnglishDictionary }) {
  const { language } = useLanguage();
  const pathname = usePathname();
  const source = useRef({ path: '', original: '', translated: '' });
  useEffect(() => {
    const update = () => {
      const current = document.title;
      source.current = titleSourceForPath(source.current, pathname, current);
      const translated = source.current.original.split(' | ').map(part => englishText(part, dictionary)).join(' | ');
      source.current.translated = translated;
      const next = language === 'en' ? translated : source.current.original;
      if (document.title !== next) document.title = next;
    };
    update();
    const observer = new MutationObserver(update);
    observer.observe(document.head, { childList: true, subtree: true, characterData: true });
    return () => observer.disconnect();
  }, [language, dictionary, pathname]);
  return null;
}

/** Render one locale through React, preserving element keys, handlers and source URLs. */
export function localizeNodes(children: ReactNode, dictionary: EnglishDictionary): ReactNode {
  if (Array.isArray(children)) return children.map(node => localizeNodes(node, dictionary));
  if (typeof children === 'string') return englishText(children, dictionary);
  if (!isValidElement(children)) return children;
  const element = children as ReactElement<Record<string, unknown>>;
  const props = element.props;
  if (props.lang === 'ko' || props['data-original-language'] || (typeof element.type === 'string' && ['script', 'style', 'pre', 'code'].includes(element.type))) return children;
  const localized: Record<string, unknown> = {};
  for (const attribute of ['title', 'alt', 'placeholder', 'aria-label']) {
    if (typeof props[attribute] === 'string') localized[attribute] = englishText(props[attribute], dictionary);
  }
  if (props.children !== undefined) localized.children = localizeNodes(props.children as ReactNode, dictionary);
  return cloneElement(element, localized);
}

export default function LocalizedContent({ children, translations = EMPTY, translateTitle = false }: { children: ReactNode; translations?: EnglishDictionary; translateTitle?: boolean }) {
  const { language } = useLanguage();
  const parent = useContext(DictionaryContext);
  const dictionary = useMemo(() => ({ ...parent, ...translations }), [parent, translations]);
  const content = <DictionaryContext.Provider value={dictionary}>{!translateTitle && translations !== EMPTY && <TitleScope dictionary={dictionary} />}{language === 'en' ? localizeNodes(children, dictionary) : children}</DictionaryContext.Provider>;
  return translateTitle ? <TitleManager dictionary={dictionary}>{content}</TitleManager> : content;
}

export function OriginalLanguageNotice() {
  const { language } = useLanguage();
  if (language !== 'en') return null;
  return <p className="mb-6 rounded-lg border bg-muted/30 px-4 py-3 text-sm text-muted-foreground">This original content is available in Korean. Navigation and controls follow your selected language.</p>;
}
