export default function DownloadCard({ name, version, size, notes, href, checksum }: {
  name: string;
  version: string;
  size: string;
  notes: string;
  href: string;
  checksum?: string;
}) {
  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="mb-1">{name}</h2>
          <p className="text-sm text-gray-500">{version} • {size}</p>
        </div>
        {/* Was: <a href={href} className="btn btn-primary" download>Download</a> */}
        <button className="btn btn-primary" disabled aria-disabled="true">
          Download
        </button>
      </div>
      <p className="text-gray-600 mt-4">{notes}</p>
      {checksum && (
        <p className="mt-3 text-xs text-gray-400">Checksum: {checksum}</p>
      )}
    </div>
  )
}