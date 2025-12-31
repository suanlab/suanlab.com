import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { ThemeProvider } from '@/components/theme-provider';
import ModernHeader from '@/components/layout/Header/ModernHeader';
import ModernFooter from '@/components/layout/Footer/ModernFooter';
import {
  OrganizationJsonLd,
  PersonJsonLd,
  WebSiteJsonLd,
} from '@/components/seo/JsonLd';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

const BASE_URL = 'https://suanlab.com';

export const metadata: Metadata = {
  metadataBase: new URL(BASE_URL),
  title: {
    default: 'SuanLab | Data Science & AI Research',
    template: '%s | SuanLab',
  },
  description:
    '이수안 교수의 데이터 사이언스 연구실 - 딥러닝, 머신러닝, 빅데이터, 자연어처리, 컴퓨터 비전 연구 및 교육 콘텐츠 제공',
  keywords: [
    '이수안',
    'SuanLab',
    '데이터 사이언스',
    'Data Science',
    '딥러닝',
    'Deep Learning',
    '머신러닝',
    'Machine Learning',
    '빅데이터',
    'Big Data',
    '자연어처리',
    'NLP',
    '컴퓨터 비전',
    'Computer Vision',
    'AI',
    '인공지능',
    'PyTorch',
    'TensorFlow',
    '파이썬',
    'Python',
  ],
  authors: [{ name: '이수안 (Suan Lee)', url: `${BASE_URL}/suan` }],
  creator: '이수안',
  publisher: 'SuanLab',
  icons: {
    icon: '/favicon.ico',
    apple: '/apple-touch-icon.png',
  },
  manifest: '/site.webmanifest',
  openGraph: {
    type: 'website',
    locale: 'ko_KR',
    url: BASE_URL,
    siteName: 'SuanLab',
    title: 'SuanLab | Data Science & AI Research',
    description:
      '이수안 교수의 데이터 사이언스 연구실 - 딥러닝, 머신러닝, 빅데이터, 자연어처리, 컴퓨터 비전 연구 및 교육 콘텐츠 제공',
    images: [
      {
        url: '/assets/images/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'SuanLab - Data Science & AI Research',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'SuanLab | Data Science & AI Research',
    description:
      '이수안 교수의 데이터 사이언스 연구실 - 딥러닝, 머신러닝, 빅데이터, 자연어처리, 컴퓨터 비전',
    images: ['/assets/images/og-image.jpg'],
    creator: '@suanlab',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  alternates: {
    canonical: BASE_URL,
    types: {
      'application/rss+xml': `${BASE_URL}/blog/feed.xml`,
    },
  },
  verification: {
    google: 'your-google-verification-code',
  },
  category: 'technology',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" suppressHydrationWarning>
      <head>
        <OrganizationJsonLd />
        <PersonJsonLd />
        <WebSiteJsonLd />
      </head>
      <body className={`${inter.variable} font-sans antialiased`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="relative flex min-h-screen flex-col">
            <ModernHeader />
            <main className="flex-1">{children}</main>
            <ModernFooter />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
