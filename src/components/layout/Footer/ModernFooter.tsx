import Link from 'next/link';
import { Youtube, Facebook, Instagram, Linkedin, Mail, Phone, MapPin } from 'lucide-react';

// X (Twitter) icon component
const XIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
  </svg>
);

const socialLinks = [
  { name: 'YouTube', href: 'https://www.youtube.com/channel/UCFfALXX0DOx7zv6VeR5U_Bg', icon: Youtube },
  { name: 'Facebook', href: 'https://www.facebook.com/suanlab', icon: Facebook },
  { name: 'Instagram', href: 'https://www.instagram.com/suanlab', icon: Instagram },
  { name: 'LinkedIn', href: 'https://www.linkedin.com/in/suan-lee-46aaa15b/', icon: Linkedin },
  { name: 'X', href: 'https://x.com/leesuanlab', icon: XIcon },
];

const quickLinks = [
  { name: 'Research', href: '/research' },
  { name: 'Publication', href: '/publication' },
  { name: 'Lecture', href: '/lecture' },
  { name: 'YouTube', href: '/youtube' },
  { name: 'Book', href: '/book' },
  { name: 'Project', href: '/project' },
];

export default function ModernFooter() {
  return (
    <footer className="border-t bg-muted/50">
      <div className="container py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* About */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">SuanLab</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              이수안 교수의 데이터 사이언스 &amp; 인공지능 연구실입니다.
              <br />
              강의, 논문, YouTube 콘텐츠를 통해 지식을 공유합니다.
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
            <h3 className="text-lg font-semibold">Quick Links</h3>
            <ul className="space-y-2">
              {quickLinks.map((link) => (
                <li key={link.name}>
                  <Link
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-primary transition-colors"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Contact</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-3 text-sm text-muted-foreground">
                <MapPin className="h-4 w-4 mt-0.5 shrink-0" />
                <span>65 Semyung-ro, Jecheon-si,<br />Chungcheongbuk-do, 27136 Korea</span>
              </li>
              <li className="flex items-center gap-3 text-sm text-muted-foreground">
                <Phone className="h-4 w-4 shrink-0" />
                <span>+82-43-649-1273</span>
              </li>
              <li className="flex items-center gap-3 text-sm text-muted-foreground">
                <Mail className="h-4 w-4 shrink-0" />
                <a href="mailto:suanlab@gmail.com" className="hover:text-primary transition-colors">
                  suanlab@gmail.com
                </a>
              </li>
            </ul>
          </div>

          {/* Research Areas */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Research Areas</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><Link href="/research/ds" className="hover:text-primary transition-colors">Data Science &amp; Big Data</Link></li>
              <li><Link href="/research/dl" className="hover:text-primary transition-colors">Deep Learning &amp; Machine Learning</Link></li>
              <li><Link href="/research/nlp" className="hover:text-primary transition-colors">Natural Language Processing</Link></li>
              <li><Link href="/research/cv" className="hover:text-primary transition-colors">Computer Vision</Link></li>
              <li><Link href="/research/graphs" className="hover:text-primary transition-colors">Graphs and Tensors</Link></li>
              <li><Link href="/research/st" className="hover:text-primary transition-colors">Spatio-Temporal</Link></li>
              <li><Link href="/research/asp" className="hover:text-primary transition-colors">Audio &amp; Speech Processing</Link></li>
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
            <Link href="/privacy" className="hover:text-primary transition-colors">Privacy Policy</Link>
            <Link href="/terms" className="hover:text-primary transition-colors">Terms of Service</Link>
            <Link href="/sitemap.xml" className="hover:text-primary transition-colors">Sitemap</Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
