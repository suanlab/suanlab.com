'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

type Language = 'ko' | 'en';

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string | string[];
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

const translations = {
  ko: {
    'hero.badge': '데이터 사이언스 & 인공지능 연구실',
    'hero.title': '에 오신 것을 환영합니다',
    'hero.description': '데이터 사이언스, 딥러닝, 머신러닝, 빅데이터 연구실입니다. 논문, 강의, YouTube 콘텐츠를 통해 지식을 공유합니다.',
    'hero.btn.profile': '이수안 교수 소개',
    'hero.btn.research': '연구 분야 보기',
    
    'stats.publications': '논문',
    'stats.videos': 'YouTube 영상',
    'stats.projects': '프로젝트',
    'stats.lectures': '강의',
    
    'blog.badge': 'Blog',
    'blog.title': '최근 블로그 포스트',
    'blog.description': '최신 논문 리뷰와 AI 관련 글을 확인하세요',
    'blog.btn.more': '더 많은 글 보기',
    
    'youtube.badge': 'YouTube Channel',
    'youtube.title': '영상으로 배우기',
    'youtube.description': '150개 이상의 교육용 비디오를 통해 데이터 과학, 머신러닝, 딥러닝, 파이썬 프로그래밍을 배워보세요.',
    'youtube.btn.watch': '영상 보기',
    'youtube.topics': ['Python Programming', 'Data Science', 'Machine Learning', 'Deep Learning', 'Computer Vision', 'NLP'],
    
    'media.badge': '미디어 복도',
    'media.title': '미디어 복도',
    'media.description': 'SuanLab과 이수안 교수의 연구 및 활동에 관한 미디어 기사',
    
    'quicklinks.title': 'SuanLab 둘러보기',
    'quicklinks.description': '논문, 프로젝트, 강의, 영상 튜토리얼을 살펴 보세요',
    'quicklinks.research': '연구 분야 탐색',
    'quicklinks.youtube': '교육 영상 보기',
    'quicklinks.publications': '논문 목록',
    'quicklinks.projects': '프로젝트 목록',
    
    'research.title': '연구 분야',
    'research.description': '데이터 사이언스와 인공지능 분야의 최신 기술을 연구합니다',
    'research.areas': [
      { title: 'Data Science & Big Data', titleKo: '데이터과학 및 빅데이터' },
      { title: 'Deep Learning & ML', titleKo: '딥러닝 및 머신러닝' },
      { title: 'Natural Language Processing', titleKo: '자연어처리' },
      { title: 'Computer Vision', titleKo: '컴퓨터 비전' },
      { title: 'Graphs and Tensors', titleKo: '그래프 및 텐서' },
      { title: 'Spatio-Temporal', titleKo: '시공간 데이터' },
      { title: 'Audio & Speech Processing', titleKo: '오디오 음성 처리' },
    ],
    
    'cta.title': '협업에 관심이 있으신가요?',
    'cta.description': '데이터 사이언스와 인공지능 연구에 참여하거나 협업하고 싶으시면 연락 주세요.',
    'cta.btn.contact': '이수안 교수에게 연락하기',
    'cta.btn.publications': '논문 보기',
    
    'lang.ko': '한국어',
    'lang.en': 'English',
    'lang.switch': '언어 변경',
  },
  en: {
    'hero.badge': 'Data Science & AI Research Lab',
    'hero.title': 'Welcome to SuanLab',
    'hero.description': 'A research laboratory focused on Data Science, Deep Learning, Machine Learning, and Big Data. We share knowledge through publications, lectures, and YouTube content.',
    'hero.btn.profile': 'About Prof. Suan Lee',
    'hero.btn.research': 'View Research Areas',
    
    'stats.publications': 'Publications',
    'stats.videos': 'YouTube Videos',
    'stats.projects': 'Projects',
    'stats.lectures': 'Lectures',
    
    'blog.badge': 'Blog',
    'blog.title': 'Recent Blog Posts',
    'blog.description': 'Check out the latest paper reviews and AI-related articles',
    'blog.btn.more': 'View More Posts',
    
    'youtube.badge': 'YouTube Channel',
    'youtube.title': 'Learn Through Videos',
    'youtube.description': 'Learn Data Science, Machine Learning, Deep Learning, and Python programming through 150+ educational videos.',
    'youtube.btn.watch': 'Watch Videos',
    'youtube.topics': ['Python Programming', 'Data Science', 'Machine Learning', 'Deep Learning', 'Computer Vision', 'NLP'],
    
    'media.badge': 'In the News',
    'media.title': 'Media Coverage',
    'media.description': 'Media articles about SuanLab and Prof. Suan Lee\'s research and activities',
    
    'quicklinks.title': 'Explore SuanLab',
    'quicklinks.description': 'Browse publications, projects, lectures, and video tutorials',
    'quicklinks.research': 'Explore Research Areas',
    'quicklinks.youtube': 'Watch Educational Videos',
    'quicklinks.publications': 'View Publications',
    'quicklinks.projects': 'View Projects',
    
    'research.title': 'Research Areas',
    'research.description': 'We research cutting-edge technologies in Data Science and Artificial Intelligence',
    'research.areas': [
      { title: 'Data Science & Big Data', titleKo: 'Data Science & Big Data' },
      { title: 'Deep Learning & ML', titleKo: 'Deep Learning & Machine Learning' },
      { title: 'Natural Language Processing', titleKo: 'Natural Language Processing' },
      { title: 'Computer Vision', titleKo: 'Computer Vision' },
      { title: 'Graphs and Tensors', titleKo: 'Graphs and Tensors' },
      { title: 'Spatio-Temporal', titleKo: 'Spatio-Temporal Data' },
      { title: 'Audio & Speech Processing', titleKo: 'Audio & Speech Processing' },
    ],
    
    'cta.title': 'Interested in Collaboration?',
    'cta.description': 'Contact us if you are interested in participating in or collaborating on Data Science and AI research.',
    'cta.btn.contact': 'Contact Prof. Suan Lee',
    'cta.btn.publications': 'View Publications',
    
    'lang.ko': '한국어',
    'lang.en': 'English',
    'lang.switch': 'Change Language',
  },
};

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguageState] = useState<Language>('ko');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const saved = localStorage.getItem('suanlab-language') as Language;
    if (saved && (saved === 'ko' || saved === 'en')) {
      setLanguageState(saved);
    }
  }, []);

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
    localStorage.setItem('suanlab-language', lang);
  };

  const t = (key: string): string | string[] => {
    const keys = key.split('.');
    let value: unknown = translations[language];
    
    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = (value as Record<string, unknown>)[k];
      } else {
        return key;
      }
    }
    
    return value as string | string[];
  };

  if (!mounted) {
    return (
      <LanguageContext.Provider value={{ language: 'ko', setLanguage: () => {}, t: () => '' }}>
        {children}
      </LanguageContext.Provider>
    );
  }

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}
