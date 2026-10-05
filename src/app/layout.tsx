import { site } from '@/config/site';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Script from 'next/script';
import { ThemeProvider } from '@/components/theme-provider';
import { LanguageProvider } from '@/components/language-provider';
import ModernHeader from '@/components/layout/Header/ModernHeader';
import ModernFooter from '@/components/layout/Footer/ModernFooter';
import {
  OrganizationJsonLd,
  PersonJsonLd,
  WebSiteJsonLd,
} from '@/components/seo/JsonLd';
import './globals.css';

const GA_MEASUREMENT_ID = 'G-PYEC6PCW0P';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

const BASE_URL = site.url;

export const metadata: Metadata = {
  metadataBase: new URL(BASE_URL),
  title: {
    default: site.title,
    template: '%s | SuanLab',
  },
  description:
    site.description,
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
    'Superintelligence',
    'AI',
    '초지능',
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
    title: site.title,
    description:
      site.description,
    images: [
      {
        url: '/assets/images/og-image.jpg',
        width: 1200,
        height: 630,
        alt: site.title,
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: site.title,
    description:
      site.description,
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
      'application/rss+xml': `${BASE_URL}/feed.xml`,
    },
  },
  verification: {
    google: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION || undefined,
    other: {
      ...(process.env.NEXT_PUBLIC_NAVER_SITE_VERIFICATION
        ? { 'naver-site-verification': process.env.NEXT_PUBLIC_NAVER_SITE_VERIFICATION }
        : {}),
    },
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
        {/* Google Analytics */}
        <Script
          src={`https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`}
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', '${GA_MEASUREMENT_ID}');
          `}
        </Script>
        {/* JSON-LD Structured Data */}
        <OrganizationJsonLd />
        <PersonJsonLd />
        <WebSiteJsonLd />
      </head>
      <body className={`${inter.variable} antialiased`}>
        <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-[100] focus:px-4 focus:py-2 focus:bg-primary focus:text-primary-foreground focus:rounded-md">본문으로 건너뛰기</a>
        <LanguageProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            disableTransitionOnChange
          >
            <div className="relative flex min-h-screen flex-col">
              <ModernHeader />
              <main id="main-content" className="flex-1">{children}</main>
              <ModernFooter />
            </div>
          </ThemeProvider>
        </LanguageProvider>
      </body>
    </html>
  );
}
