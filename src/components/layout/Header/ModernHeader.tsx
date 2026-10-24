'use client';

import { mainNavigation as navigation } from '@/data/site-navigation';
import { useState, useCallback, useEffect, useRef, type KeyboardEvent } from 'react';
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
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = usePathname();
  const { t, language } = useLanguage();
  const [fitsDesktop, setFitsDesktop] = useState(false);
  const rowRef = useRef<HTMLDivElement>(null);
  const logoRef = useRef<HTMLAnchorElement>(null);
  const navigationRef = useRef<HTMLElement>(null);
  const utilityRef = useRef<HTMLDivElement>(null);

  useEffect(() => { setMobileOpen(false); setOpenDropdown(null); }, [pathname]);
  useEffect(() => { if (fitsDesktop) setMobileOpen(false); }, [fitsDesktop]);

  useEffect(() => {
    const row = rowRef.current, logo = logoRef.current, nav = navigationRef.current, utilities = utilityRef.current;
    if (!row || !logo || !nav || !utilities) return;
    const update = () => {
      const style = getComputedStyle(row);
      const available = row.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      setFitsDesktop(available >= logo.offsetWidth + nav.offsetWidth + utilities.offsetWidth + 24);
    };
    const observer = new ResizeObserver(update);
    [row, logo, nav, utilities].forEach(element => observer.observe(element));
    update();
    void document.fonts.ready.then(update);
    return () => observer.disconnect();
  }, [language]);

  const isActive = useCallback((href: string) => pathname.replace(/\/$/, '') === href
    || pathname.startsWith(`${href}/`), [pathname]);

  function openWithKeyboard(event: KeyboardEvent<HTMLElement>, key: string) {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    event.preventDefault();
    const group = event.currentTarget.closest('[data-nav-group]');
    const last = event.key === 'ArrowUp';
    setOpenDropdown(key);
    requestAnimationFrame(() => {
      const items = group?.querySelectorAll<HTMLElement>('[role="menuitem"]');
      if (items?.length) items[last ? items.length - 1 : 0].focus();
    });
  }

  function navigateMenu(event: KeyboardEvent<HTMLDivElement>) {
    const group = event.currentTarget;
    if (event.key === 'Escape') {
      event.preventDefault(); event.stopPropagation(); setOpenDropdown(null);
      group.querySelector<HTMLElement>('[data-menu-trigger]')?.focus();
      return;
    }
    if (!(event.target instanceof HTMLElement) || event.target.getAttribute('role') !== 'menuitem') return;
    const items = [...group.querySelectorAll<HTMLElement>('[role="menuitem"]')];
    const index = items.indexOf(event.target);
    const next = event.key === 'ArrowDown' ? (index + 1) % items.length
      : event.key === 'ArrowUp' ? (index - 1 + items.length) % items.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : -1;
    if (next >= 0) { event.preventDefault(); items[next].focus(); }
  }

  return <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
    <div ref={rowRef} className="container relative flex h-16 items-center justify-between gap-2 px-4 sm:px-8">
      <Link ref={logoRef} href="/" className="flex shrink-0 items-center" onClick={() => setMobileOpen(false)}>
        <Image src="/assets/images/logo.svg" alt="SuanLab" width={305} height={80} className="h-7 w-auto dark:brightness-0 dark:invert sm:h-8" priority />
      </Link>
      <div className={fitsDesktop ? 'min-w-0' : 'invisible absolute h-0 w-0 overflow-hidden'}>
        <nav ref={navigationRef} aria-label={language === 'ko' ? '주 메뉴' : 'Main navigation'} aria-hidden={!fitsDesktop} className="flex w-max shrink-0 items-center space-x-0.5">
          {navigation.map(item => {
            const label = (t(item.nameKey) as string).toUpperCase();
            const menuId = `menu-${item.nameKey.replace(/\./g, '-')}`;
            return <div key={item.nameKey} data-nav-group className="relative flex items-center"
              onMouseEnter={() => item.children && setOpenDropdown(item.nameKey)}
              onMouseLeave={event => { if (!event.currentTarget.contains(document.activeElement)) setOpenDropdown(null); }}
              onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setOpenDropdown(null); }}
              onKeyDown={navigateMenu}>
              <Link href={item.href} aria-current={isActive(item.href) ? 'page' : undefined}
                onKeyDown={event => { if (item.children) openWithKeyboard(event, item.nameKey); }}
                onClick={() => setOpenDropdown(null)}
                className={cn('flex items-center gap-1 whitespace-nowrap rounded-md px-2 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground', isActive(item.href) && 'font-semibold text-primary')}>
                <item.icon className="h-4 w-4" />{label}
              </Link>
              {item.children && <button type="button" data-menu-trigger aria-label={`${label} ${language === 'ko' ? '하위 메뉴' : 'submenu'}`}
                aria-expanded={openDropdown === item.nameKey} aria-haspopup="menu" aria-controls={menuId}
                className="rounded p-1 hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                onClick={() => setOpenDropdown(current => current === item.nameKey ? null : item.nameKey)}
                onKeyDown={event => openWithKeyboard(event, item.nameKey)}><ChevronDown className="h-3 w-3" /></button>}
              {item.children && openDropdown === item.nameKey && <div id={menuId} role="menu" aria-label={label} className="absolute left-0 top-full w-56 pt-2">
                <div className="rounded-md border bg-popover p-1 shadow-lg">{item.children.map(child => <Link key={child.href} href={child.href} role="menuitem"
                  onClick={() => setOpenDropdown(null)} aria-current={isActive(child.href) ? 'page' : undefined}
                  className={cn('block rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent hover:text-accent-foreground', isActive(child.href) && 'font-semibold text-primary')}>{child.name}</Link>)}</div>
              </div>}
            </div>;
          })}
        </nav>
      </div>
      <div className="flex shrink-0 items-center gap-0 sm:gap-2">
        <div ref={utilityRef} className="flex shrink-0 items-center gap-0 sm:gap-2">
          <Button variant="ghost" size="icon" className="h-9 w-9 sm:h-10 sm:w-10" asChild><Link href="/search" aria-label={language === 'ko' ? '검색' : 'Search'}><Search className="h-5 w-5" /></Link></Button>
          <ThemeToggle className="h-9 w-9 sm:h-10 sm:w-10" /><LanguageSwitcher />
        </div>
        <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
          <SheetTrigger asChild className={fitsDesktop ? 'hidden' : ''}><Button variant="ghost" size="icon" className="h-9 w-9 sm:h-10 sm:w-10"><Menu className="h-5 w-5" /><span className="sr-only">{language === 'ko' ? '메뉴 열기' : 'Open menu'}</span></Button></SheetTrigger>
          <SheetContent side="right" closeLabel={language === 'ko' ? '닫기' : 'Close'} className="w-80 max-w-full overflow-y-auto">
            <SheetTitle className="sr-only">{language === 'ko' ? '내비게이션 메뉴' : 'Navigation menu'}</SheetTitle>
            <div className="mt-8 flex flex-col space-y-4 pb-8">{navigation.map(item => <div key={item.nameKey}>
              <Link href={item.href} onClick={() => setMobileOpen(false)} aria-current={isActive(item.href) ? 'page' : undefined} className={cn('flex items-center gap-2 px-2 py-2 text-lg font-medium transition-colors hover:text-primary', isActive(item.href) && 'font-semibold text-primary')}><item.icon className="h-5 w-5" />{(t(item.nameKey) as string).toUpperCase()}</Link>
              {item.children && <div className="ml-7 mt-2 space-y-2">{item.children.map(child => <Link key={child.href} href={child.href} onClick={() => setMobileOpen(false)} aria-current={isActive(child.href) ? 'page' : undefined} className={cn('block text-sm text-muted-foreground transition-colors hover:text-primary', isActive(child.href) && 'font-semibold text-primary')}>{child.name}</Link>)}</div>}
            </div>)}</div>
          </SheetContent>
        </Sheet>
      </div>
    </div>
  </header>;
}
