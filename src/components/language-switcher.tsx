'use client';

import { Button } from '@/components/ui/button';
import { useLanguage } from '@/components/language-provider';

export function LanguageSwitcher() {
  const { language, setLanguage } = useLanguage();

  const toggleLanguage = () => {
    setLanguage(language === 'ko' ? 'en' : 'ko');
  };

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={toggleLanguage}
      className="h-9 px-3 font-medium"
    >
      {language === 'ko' ? 'EN' : 'KO'}
    </Button>
  );
}
