export default function BookLoading() {
  return (
    <div className="container py-16 md:py-20">
      <div className="animate-pulse">
        <div className="mx-auto max-w-2xl text-center mb-12">
          <div className="h-9 bg-muted rounded w-48 mx-auto" />
          <div className="h-4 bg-muted rounded w-64 mx-auto mt-4" />
        </div>
        <div className="grid gap-8 md:grid-cols-2 max-w-4xl mx-auto">
          {[1, 2].map((i) => (
            <div key={i} className="rounded-lg border p-8 space-y-4">
              <div className="h-16 w-16 bg-muted rounded-xl" />
              <div className="h-7 bg-muted rounded w-36" />
              <div className="h-4 bg-muted rounded w-full" />
              <div className="h-10 bg-muted rounded w-28" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
