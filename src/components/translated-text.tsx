'use client';

import { useLanguage } from '@/components/language-provider';

interface TranslatedTextProps {
  textKey: string;
  fallback?: string;
}

export function TranslatedText({ textKey, fallback }: TranslatedTextProps) {
  const { t } = useLanguage();
  const translated = t(textKey);
  
  if (typeof translated === 'string') {
    return <>{translated || fallback || textKey}</>;
  }
  
  return <>{fallback || textKey}</>;
}

export function TranslatedBadge({ textKey }: { textKey: string }) {
  const { t } = useLanguage();
  const translated = t(textKey);
  
  return <span>{typeof translated === 'string' ? translated : textKey}</span>;
}
