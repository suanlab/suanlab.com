export default function PublicationLoading() {
  return (
    <div className="container py-16 md:py-20">
      <div className="animate-pulse grid gap-8 lg:grid-cols-4">
        <aside className="lg:col-span-1 space-y-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-10 bg-muted rounded" />
          ))}
          <div className="mt-6 space-y-2">
            <div className="h-6 bg-muted rounded w-1/2" />
            <div className="rounded-lg border p-4 space-y-3">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="flex justify-between">
                  <div className="h-4 bg-muted rounded w-24" />
                  <div className="h-5 bg-muted rounded-full w-8" />
                </div>
              ))}
            </div>
          </div>
        </aside>
        <main className="lg:col-span-3 space-y-4">
          <div className="flex gap-2 mb-6">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-9 bg-muted rounded-full w-20" />
            ))}
          </div>
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="rounded-lg border p-4 space-y-2">
              <div className="flex gap-2">
                <div className="h-5 bg-muted rounded-full w-12" />
                <div className="h-5 bg-muted rounded w-3/4" />
              </div>
              <div className="h-3 bg-muted rounded w-1/2" />
              <div className="h-3 bg-muted rounded w-1/3" />
            </div>
          ))}
        </main>
      </div>
    </div>
  );
}
