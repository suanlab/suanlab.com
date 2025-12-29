'use client';

import { useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { Menu, ChevronDown, User, Search, Youtube, Newspaper, FolderKanban, GraduationCap, Presentation, BookMarked, PenLine } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger, SheetTitle } from '@/components/ui/sheet';
import { ThemeToggle } from '@/components/theme-toggle';
import { cn } from '@/lib/utils';

const navigation = [
  { name: 'SUAN', href: '/suan', icon: User },
  {
    name: 'RESEARCH',
    href: '/research',
    icon: Search,
    children: [
      { name: 'Data Science & Big Data', href: '/research/ds' },
      { name: 'Deep Learning & ML', href: '/research/dl' },
      { name: 'Natural Language Processing', href: '/research/nlp' },
      { name: 'Computer Vision', href: '/research/cv' },
      { name: 'Graphs and Tensors', href: '/research/graphs' },
      { name: 'Spatio-Temporal', href: '/research/st' },
      { name: 'Audio & Speech Processing', href: '/research/asp' },
    ],
  },
  { name: 'PROJECT', href: '/project', icon: FolderKanban },
  { name: 'PUBLICATION', href: '/publication', icon: Newspaper },
  {
    name: 'BOOK',
    href: '/book',
    icon: BookMarked,
    children: [
      { name: 'Online Book', href: '/book/online' },
      { name: 'Published Book', href: '/book/published' },
    ],
  },
  { name: 'BLOG', href: '/blog', icon: PenLine },
  {
    name: 'LECTURE',
    href: '/lecture',
    icon: GraduationCap,
    children: [
      { name: 'Artificial Intelligence', href: '/lecture/ai' },
      { name: 'Deep Learning', href: '/lecture/dl' },
      { name: 'Machine Learning', href: '/lecture/ml' },
      { name: 'Natural Language Processing', href: '/lecture/nlp' },
      { name: 'Computer Vision', href: '/lecture/cv' },
      { name: 'Audio Signal Processing', href: '/lecture/asp' },
      { name: 'Big Data Analysis', href: '/lecture/bd' },
      { name: 'Database', href: '/lecture/db' },
    ],
  },
  { name: 'COURSE', href: '/course', icon: Presentation },
  {
    name: 'YOUTUBE',
    href: '/youtube',
    icon: Youtube,
    children: [
      { name: 'Python Programming', href: '/youtube/pp' },
      { name: 'Data Science', href: '/youtube/ds' },
      { name: 'Machine Learning', href: '/youtube/ml' },
      { name: 'Deep Learning', href: '/youtube/dl' },
      { name: 'Computer Vision', href: '/youtube/cv' },
      { name: 'NLP', href: '/youtube/nlp' },
    ],
  },
];

export default function ModernHeader() {
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center space-x-2">
          <Image
            src="/assets/images/logo.png"
            alt="SuanLab"
            width={305}
            height={80}
            className="h-10 w-auto"
            priority
          />
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden lg:flex items-center space-x-1">
          {navigation.map((item) => (
            <div
              key={item.name}
              className="relative group"
              onMouseEnter={() => item.children && setOpenDropdown(item.name)}
              onMouseLeave={() => setOpenDropdown(null)}
            >
              <Link
                href={item.href}
                className={cn(
                  "flex items-center gap-1 px-3 py-2 text-sm font-medium rounded-md transition-colors",
                  "hover:bg-accent hover:text-accent-foreground"
                )}
              >
                <item.icon className="h-4 w-4" />
                {item.name}
                {item.children && <ChevronDown className="h-3 w-3" />}
              </Link>

              {/* Dropdown Menu */}
              {item.children && openDropdown === item.name && (
                <div className="absolute top-full left-0 pt-2 w-56">
                  <div className="rounded-md border bg-popover p-1 shadow-lg">
                    {item.children.map((child) => (
                      <Link
                        key={child.href}
                        href={child.href}
                        className="block px-3 py-2 text-sm rounded-md hover:bg-accent hover:text-accent-foreground transition-colors"
                      >
                        {child.name}
                      </Link>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </nav>

        {/* Dark Mode Toggle & Mobile Menu */}
        <div className="flex items-center gap-2">
          {/* Theme Toggle */}
          <ThemeToggle />

          {/* Mobile Menu */}
          <Sheet>
            <SheetTrigger asChild className="lg:hidden">
              <Button variant="ghost" size="icon">
                <Menu className="h-5 w-5" />
                <span className="sr-only">Toggle menu</span>
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-80 overflow-y-auto">
              <SheetTitle className="sr-only">Navigation Menu</SheetTitle>
              <div className="flex flex-col space-y-4 mt-8 pb-8">
                {navigation.map((item) => (
                  <div key={item.name}>
                    <Link
                      href={item.href}
                      className="flex items-center gap-2 px-2 py-2 text-lg font-medium hover:text-primary transition-colors"
                    >
                      <item.icon className="h-5 w-5" />
                      {item.name}
                    </Link>
                    {item.children && (
                      <div className="ml-7 mt-2 space-y-2">
                        {item.children.map((child) => (
                          <Link
                            key={child.href}
                            href={child.href}
                            className="block text-sm text-muted-foreground hover:text-primary transition-colors"
                          >
                            {child.name}
                          </Link>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  );
}
