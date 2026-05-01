export default function ContactLoading() {
  return (
    <div className="container py-16 md:py-20">
      <div className="animate-pulse grid gap-12 lg:grid-cols-3">
        <aside className="lg:col-span-1 space-y-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="rounded-lg border p-6">
              <div className="flex gap-4">
                <div className="h-12 w-12 bg-muted rounded-lg shrink-0" />
                <div className="space-y-2 flex-1">
                  <div className="h-4 bg-muted rounded w-16" />
                  <div className="h-3 bg-muted rounded w-full" />
                </div>
              </div>
            </div>
          ))}
        </aside>
        <div className="lg:col-span-2 space-y-6">
          <div className="rounded-lg border p-6 space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="space-y-2">
                <div className="h-4 bg-muted rounded w-24" />
                <div className="h-10 bg-muted rounded" />
              </div>
            ))}
            <div className="space-y-2">
              <div className="h-4 bg-muted rounded w-16" />
              <div className="h-32 bg-muted rounded" />
            </div>
            <div className="h-10 bg-muted rounded w-32" />
          </div>
        </div>
      </div>
    </div>
  );
}
