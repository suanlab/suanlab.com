import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import { ThemeProvider } from '@/components/theme-provider';
import ModernHeader from '@/components/layout/Header/ModernHeader';
import ModernFooter from '@/components/layout/Footer/ModernFooter';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: 'SuanLab | Data Science & AI Research',
  description: 'Professor Suan Lee\'s Research Lab - Data Science, Deep Learning, Machine Learning, Big Data, NLP, Computer Vision',
  keywords: 'Suan Lee, SuanLab, Data Science, Deep Learning, Machine Learning, Big Data, NLP, Computer Vision, AI Research',
  authors: [{ name: 'Suan Lee' }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" suppressHydrationWarning>
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
