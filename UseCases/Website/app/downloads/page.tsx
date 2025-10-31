import DownloadCard from '@/components/DownloadCard'

const models = [
  {
    name: 'GESTURE Lite (TGCN)',
    version: 'September 2025',
    size: '10 MB',
    notes: 'Lightweight ASL → natural language translation model. Open Source, Open Weight, Efficient.',
    href: '#'
  },
  {
    name: 'GESTURE Heavy (UniSign)',
    version: 'September 2025',
    size: '1.1 GB',
    notes: 'Our heavier ASL → natural language translation model. For more accurate translations at an inference cost.',
    href: '#',
  },
]

export default function DownloadsPage() {
  return (
    <div className="space-y-6">
      <h1>Model Downloads</h1>
      <p className="text-gray-600">Grab our latest models releases below.</p>
      <div className="grid md:grid-cols-2 gap-6">
        {models.map((m) => (
          <DownloadCard key={m.name} {...m} />
        ))}
      </div>
    </div>
  )
}
