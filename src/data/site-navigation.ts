import { lectures } from './lectures';
import { playlists } from './youtube';
import { researchAreas } from './research';
import { User, Search, Youtube, Newspaper, FolderKanban, GraduationCap, Presentation, BookMarked, PenLine } from 'lucide-react';

export const mainNavigation = [
  { nameKey: 'nav.suan', href: '/suan', icon: User },
  {
    nameKey: 'nav.research',
    href: '/research',
    icon: Search,
    children: researchAreas.map(item => ({ name: item.titleEn, href: `/research/${item.slug}` })),
  },
  { nameKey: 'nav.project', href: '/project', icon: FolderKanban },
  { nameKey: 'nav.publication', href: '/publication', icon: Newspaper },
  { nameKey: 'nav.blog', href: '/blog', icon: PenLine },
  {
    nameKey: 'nav.book',
    href: '/book',
    icon: BookMarked,
    children: [
      { name: 'Online Book', href: '/book/online' },
      { name: 'Published Book', href: '/book/published' },
    ],
  },
  {
    nameKey: 'nav.lecture',
    href: '/lecture',
    icon: GraduationCap,
    children: lectures.map(item => ({ name: item.titleEn, href: `/lecture/${item.slug}` })),
  },
  { nameKey: 'nav.course', href: '/course', icon: Presentation },
  {
    nameKey: 'nav.youtube',
    href: '/youtube',
    icon: Youtube,
    children: playlists.map(item => ({ name: item.titleEn, href: `/youtube/${item.slug}` })),
  },
];
