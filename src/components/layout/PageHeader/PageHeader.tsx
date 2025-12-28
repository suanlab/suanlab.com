import Link from 'next/link';
import { ChevronRight, Home } from 'lucide-react';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  backgroundImage?: string;
  breadcrumbs?: { label: string; href?: string }[];
}

export default function PageHeader({
  title,
  subtitle,
  backgroundImage = '/assets/images/slider/blue.jpg',
  breadcrumbs = [],
}: PageHeaderProps) {
  return (
    <section
      className="relative py-16 md:py-20 lg:py-24 text-white overflow-hidden"
      style={{
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
      }}
    >
      <div className="absolute inset-0 bg-slate-900/70" />
      <div className="container relative z-10">
        <div className="max-w-3xl">
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl md:text-5xl">
            {title}
          </h1>
          {subtitle && (
            <p className="mt-4 text-lg text-slate-300">{subtitle}</p>
          )}
          {breadcrumbs.length > 0 && (
            <nav className="mt-6 flex items-center space-x-2 text-sm">
              <Link
                href="/"
                className="flex items-center text-slate-300 hover:text-white transition-colors"
              >
                <Home className="h-4 w-4" />
                <span className="ml-1">Home</span>
              </Link>
              {breadcrumbs.map((crumb, index) => (
                <span key={index} className="flex items-center">
                  <ChevronRight className="h-4 w-4 text-slate-500" />
                  {crumb.href ? (
                    <Link
                      href={crumb.href}
                      className="ml-2 text-slate-300 hover:text-white transition-colors"
                    >
                      {crumb.label}
                    </Link>
                  ) : (
                    <span className="ml-2 text-white font-medium">{crumb.label}</span>
                  )}
                </span>
              ))}
            </nav>
          )}
        </div>
      </div>
    </section>
  );
}
