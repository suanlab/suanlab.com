'use client';

import { useLanguage } from '@/components/language-provider';

export function LanguageSwitcher() {
  const { language, setLanguage } = useLanguage();

  return (
    <div className="flex items-center rounded-md border bg-muted/50 p-0.5">
      <button
        onClick={() => setLanguage('ko')}
        className={`flex items-center gap-1 rounded-sm px-2 py-1 text-xs font-medium transition-colors ${
          language === 'ko'
            ? 'bg-background text-foreground shadow-sm'
            : 'text-muted-foreground hover:text-foreground'
        }`}
      >
        KO
      </button>
      <button
        onClick={() => setLanguage('en')}
        className={`flex items-center gap-1 rounded-sm px-2 py-1 text-xs font-medium transition-colors ${
          language === 'en'
            ? 'bg-background text-foreground shadow-sm'
            : 'text-muted-foreground hover:text-foreground'
        }`}
      >
        EN
      </button>
    </div>
  );
}
