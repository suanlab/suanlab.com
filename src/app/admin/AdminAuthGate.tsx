'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

const ADMIN_PASSWORD = 'suan1004';

export default function AdminAuthGate({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  // Check localStorage on mount
  useEffect(() => {
    const authenticated = localStorage.getItem('admin_authenticated') === 'true';
    setIsAuthenticated(authenticated);
    setIsLoading(false);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (password === ADMIN_PASSWORD) {
      localStorage.setItem('admin_authenticated', 'true');
      setIsAuthenticated(true);
      setPassword('');
    } else {
      setError('비밀번호가 올바르지 않습니다.');
      setPassword('');
    }
  };

  if (isLoading) {
    return null;
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-muted/30 flex items-center justify-center p-4">
        <Card className="w-full max-w-sm">
          <CardHeader>
            <CardTitle className="text-center">관리자 인증</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <Input
                  type="password"
                  placeholder="비밀번호를 입력하세요"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoFocus
                />
              </div>
              {error && <p className="text-sm text-red-500">{error}</p>}
              <Button type="submit" className="w-full">
                인증
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    );
  }

  return <>{children}</>;
}
