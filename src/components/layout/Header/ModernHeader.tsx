'use client';

import { mainNavigation as navigation } from '@/data/site-navigation';
import { useState, useCallback, useEffect, useRef } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import Image from 'next/image';
import { Menu, ChevronDown, Search } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger, SheetTitle } from '@/components/ui/sheet';
import { ThemeToggle } from '@/components/theme-toggle';
import { LanguageSwitcher } from '@/components/language-switcher';
import { useLanguage } from '@/components/language-provider';
import { cn } from '@/lib/utils';



export default function ModernHeader() {
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);
  const pathname = usePathname();
  const dropdownRef = useRef<HTMLDivElement>(null);
  const { t, language } = useLanguage();
  const [fitsDesktop, setFitsDesktop] = useState(false);
  const rowRef = useRef<HTMLDivElement>(null);
  const logoRef = useRef<HTMLAnchorElement>(null);
  const navigationRef = useRef<HTMLElement>(null);
  const utilityRef = useRef<HTMLDivElement>(null);

  // Measure the translated labels instead of assuming one breakpoint fits both languages.
  useEffect(() => {
    const row = rowRef.current;
    const logo = logoRef.current;
    const nav = navigationRef.current;
    const utilities = utilityRef.current;
    if (!row || !logo || !nav || !utilities) return;
    const update = () => {
      const style = getComputedStyle(row);
      const available = row.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      setFitsDesktop(available >= logo.offsetWidth + nav.offsetWidth + utilities.offsetWidth + 32);
    };
    const observer = new ResizeObserver(update);
    [row, logo, nav, utilities].forEach(element => observer.observe(element));
    update();
    document.fonts.ready.then(update);
    return () => observer.disconnect();
  }, [language]);

  const isActive = useCallback((href: string) => {
    return pathname === href || (href !== '/' && pathname.startsWith(href));
  }, [pathname]);

  // Close dropdown on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpenDropdown(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div ref={rowRef} className="container relative flex h-16 items-center justify-between gap-2 px-4 sm:px-8">
        {/* Logo */}
        <Link ref={logoRef} href="/" className="flex items-center space-x-2 shrink-0">
          <Image
            src="/assets/images/logo.svg"
            alt="SuanLab"
            width={305}
            height={80}
            className="h-7 w-auto dark:brightness-0 dark:invert sm:h-8"
            priority
          />
        </Link>

        {/* Desktop Navigation */}
        <div className={fitsDesktop ? 'min-w-0' : 'invisible absolute h-0 w-0 overflow-hidden'}>
        <nav ref={navigationRef} aria-hidden={!fitsDesktop} className="flex w-max shrink-0 items-center space-x-0.5">
          {navigation.map((item) => (
            <div
              key={item.nameKey}
              className="relative group"
              ref={item.children ? dropdownRef : undefined}
              onMouseEnter={() => item.children && setOpenDropdown(item.nameKey)}
              onMouseLeave={() => setOpenDropdown(null)}
            >
              <Link
                href={item.href}
                className={cn(
                  "flex items-center gap-1 whitespace-nowrap px-2 py-2 text-sm font-medium rounded-md transition-colors",
                  "hover:bg-accent hover:text-accent-foreground",
                  isActive(item.href) && "text-primary font-semibold"
                )}
                aria-expanded={item.children ? openDropdown === item.nameKey : undefined}
                aria-haspopup={item.children ? "true" : undefined}
                onKeyDown={(e) => {
                  if (!item.children) return;
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setOpenDropdown(openDropdown === item.nameKey ? null : item.nameKey);
                  } else if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    setOpenDropdown(item.nameKey);
                    // Focus first dropdown item after render
                    setTimeout(() => {
                      const firstItem = e.currentTarget.parentElement?.querySelector('[role="menuitem"]');
                      if (firstItem instanceof HTMLElement) firstItem.focus();
                    }, 0);
                  }
                }}
              >
                <item.icon className="h-4 w-4" />
                {(t(item.nameKey) as string).toUpperCase()}
                {item.children && <ChevronDown className="h-3 w-3" />}
              </Link>

              {/* Dropdown Menu */}
              {item.children && openDropdown === item.nameKey && (
                <div className="absolute top-full left-0 pt-2 w-56" role="menu">
                  <div className="rounded-md border bg-popover p-1 shadow-lg">
                    {item.children.map((child) => (
                      <Link
                        key={child.href}
                        href={child.href}
                        role="menuitem"
                        tabIndex={0}
                        className={cn(
                          "block px-3 py-2 text-sm rounded-md hover:bg-accent hover:text-accent-foreground transition-colors",
                          isActive(child.href) && "text-primary font-semibold"
                        )}
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
        </div>

        {/* Dark Mode Toggle & Mobile Menu */}
        <div className="flex shrink-0 items-center gap-0 sm:gap-2">
          <div ref={utilityRef} className="flex shrink-0 items-center gap-0 sm:gap-2">
          {/* Search */}
          <Button variant="ghost" size="icon" asChild>
            <Link href="/search" aria-label={language === 'ko' ? '검색' : 'Search'}>
              <Search className="h-5 w-5" />
            </Link>
          </Button>
          <ThemeToggle />
          <LanguageSwitcher />
          </div>
          <Sheet>
            <SheetTrigger asChild className={fitsDesktop ? 'hidden' : ''}>
              <Button variant="ghost" size="icon">
                <Menu className="h-5 w-5" />
                <span className="sr-only">Toggle menu</span>
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-80 overflow-y-auto">
              <SheetTitle className="sr-only">Navigation Menu</SheetTitle>
              <div className="flex flex-col space-y-4 mt-8 pb-8">
                {navigation.map((item) => (
                  <div key={item.nameKey}>
                    <Link
                      href={item.href}
                      className={cn(
                        "flex items-center gap-2 px-2 py-2 text-lg font-medium hover:text-primary transition-colors",
                        isActive(item.href) && "text-primary font-semibold"
                      )}
                    >
                      <item.icon className="h-5 w-5" />
                      {(t(item.nameKey) as string).toUpperCase()}
                    </Link>
                    {item.children && (
                      <div className="ml-7 mt-2 space-y-2">
                        {item.children.map((child) => (
                          <Link
                            key={child.href}
                            href={child.href}
                            className={cn(
                              "block text-sm text-muted-foreground hover:text-primary transition-colors",
                              isActive(child.href) && "text-primary font-semibold"
                            )}
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
