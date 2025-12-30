import Link from 'next/link';
import { Home, PenLine } from 'lucide-react';

export const metadata = {
  title: 'Admin | SuanLab',
  description: 'SuanLab Blog Admin Dashboard',
};

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-muted/30">
      {/* Admin Header */}
      <header className="sticky top-0 z-50 border-b bg-background">
        <div className="container flex h-14 items-center justify-between">
          <div className="flex items-center gap-6">
            <Link href="/admin" className="font-bold text-lg">
              SuanLab Admin
            </Link>
            <nav className="flex items-center gap-4 text-sm">
              <Link
                href="/admin"
                className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors"
              >
                <Home className="h-4 w-4" />
                Dashboard
              </Link>
              <Link
                href="/admin/blog/new"
                className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors"
              >
                <PenLine className="h-4 w-4" />
                New Post
              </Link>
            </nav>
          </div>
          <Link
            href="/"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            View Site
          </Link>
        </div>
      </header>

      {/* Main Content */}
      <main>{children}</main>
    </div>
  );
}
