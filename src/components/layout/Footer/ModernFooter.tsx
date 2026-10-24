'use client';

import { useLanguage } from '@/components/language-provider';
import { researchAreas } from '@/data/research';
import { site } from '@/config/site';
import Link from 'next/link';
import { Youtube, Facebook, Instagram, Linkedin, Mail, Phone, MapPin } from 'lucide-react';

// X (Twitter) icon component
const XIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
  </svg>
);

const socialLinks = [
  { name: 'YouTube', href: site.social.youtube, icon: Youtube },
  { name: 'Facebook', href: site.social.facebook, icon: Facebook },
  { name: 'Instagram', href: site.social.instagram, icon: Instagram },
  { name: 'LinkedIn', href: site.social.linkedin, icon: Linkedin },
  { name: 'X', href: site.social.x, icon: XIcon },
];

const quickLinks = [
  { nameKey: 'nav.research', href: '/research' },
  { nameKey: 'nav.publication', href: '/publication' },
  { nameKey: 'nav.lecture', href: '/lecture' },
  { nameKey: 'nav.youtube', href: '/youtube' },
  { nameKey: 'nav.book', href: '/book' },
  { nameKey: 'nav.project', href: '/project' },
  { nameKey: 'nav.prompts', href: '/prompts' },
  { nameKey: 'nav.deadlines', href: '/deadlines' },
  { nameKey: 'nav.qt', href: '/qt' },
  { nameKey: 'nav.contact', href: '/contact' },
];

export default function ModernFooter() {
  const { language, t } = useLanguage();
  return (
    <footer className="border-t bg-muted/50">
      <div className="container py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* About */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">SuanLab</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {t('footer.about') as string}
            </p>
            <div className="flex space-x-3">
              {socialLinks.map((social) => (
                <a
                  key={social.name}
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground hover:text-primary transition-colors"
                  aria-label={social.name}
                >
                  <social.icon className="h-5 w-5" />
                </a>
              ))}
            </div>
          </div>

          {/* Quick Links */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">{t('footer.links') as string}</h3>
            <ul className="space-y-2">
              {quickLinks.map((link) => (
                <li key={link.nameKey}>
                  <Link
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-primary transition-colors"
                  >
                    {t(link.nameKey) as string}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">{t('footer.contact') as string}</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-3 text-sm text-muted-foreground">
                <MapPin className="h-4 w-4 mt-0.5 shrink-0" />
                <span>{site.contact.address}<br /></span>
              </li>
              <li className="flex items-center gap-3 text-sm text-muted-foreground">
                <Phone className="h-4 w-4 shrink-0" />
                <span>{site.contact.phone}</span>
              </li>
              <li className="flex items-center gap-3 text-sm text-muted-foreground">
                <Mail className="h-4 w-4 shrink-0" />
                <a href={`mailto:${site.contact.emails[0]}`} className="hover:text-primary transition-colors">
                  {site.contact.emails[0]}
                </a>
              </li>
            </ul>
          </div>

          {/* Research Areas */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">{t('footer.research') as string}</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              {researchAreas.map(area => <li key={area.slug}><Link href={`/research/${area.slug}`} className="hover:text-primary transition-colors">{language === 'ko' ? area.titleKo : area.titleEn}</Link></li>)}
            </ul>
          </div>
        </div>
      </div>

      {/* Copyright */}
      <div className="border-t">
        <div className="container py-6 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-muted-foreground">
            © {new Date().getFullYear()} SuanLab. All rights reserved.
          </p>
          <div className="flex gap-4 text-sm text-muted-foreground">
            <Link href="/privacy" className="hover:text-primary transition-colors">{t('footer.privacy') as string}</Link>
            <Link href="/terms" className="hover:text-primary transition-colors">{t('footer.terms') as string}</Link>
            <Link href="/sitemap.xml" className="hover:text-primary transition-colors">{t('footer.sitemap') as string}</Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
