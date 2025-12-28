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
              Professor Suan&apos;s research lab focused on Data Science and Artificial Intelligence.
              Sharing knowledge through lectures, publications, and YouTube content.
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
              <li>Data Science & Big Data</li>
              <li>Deep Learning & Machine Learning</li>
              <li>Natural Language Processing</li>
              <li>Computer Vision</li>
              <li>Graphs and Tensors</li>
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
            <Link href="#" className="hover:text-primary transition-colors">Privacy Policy</Link>
            <Link href="#" className="hover:text-primary transition-colors">Terms of Service</Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
