import type { Metadata } from 'next';
import { site } from '@/config/site';
import { InteractiveBookLibrary } from '@/components/learning/interactive-book';
import '@/styles/interactive-book.css';
export const metadata: Metadata = {
  title: { absolute: 'Interactive Book | SuanLab' },
  description: 'Interactive textbooks for learning mathematics and physics through visualizations, experiments and explained examples.',
  alternates: { canonical: `${site.url}/book/interactive/` },
  openGraph: { title: 'Interactive Book | SuanLab', description: 'Explore interactive textbooks from SuanLab.', url: `${site.url}/book/interactive/`, type: 'website' },
};
export default function InteractiveBookPage() { return <InteractiveBookLibrary />; }
