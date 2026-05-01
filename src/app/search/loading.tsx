export default function SearchLoading() {
  return (
    <div className="container py-16 md:py-20">
      <div className="animate-pulse space-y-8">
        <div className="h-12 bg-muted rounded-lg" />
        <div className="flex gap-2">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-9 bg-muted rounded-full w-20" />
          ))}
        </div>
        <div className="space-y-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="rounded-lg border p-4 space-y-2">
              <div className="flex items-center gap-2">
                <div className="h-5 bg-muted rounded-full w-16" />
                <div className="h-5 bg-muted rounded w-3/4" />
              </div>
              <div className="h-3 bg-muted rounded w-full" />
              <div className="h-3 bg-muted rounded w-5/6" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
