import Image from 'next/image'
import Link from 'next/link'

export default function HomePage() {
  return (
    <div className="space-y-10">
      <section className="card flex flex-col md:flex-row items-center gap-8">
        <Image src="/logo.svg" alt="GESTURE " width={80} height={80} />
        <div className="space-y-3">
          <h1>GESTURE</h1>
          <p className="text-gray-600 max-w-2xl">
            A central hub for ASL-powered AI. Download our models and use them in real use cases like
            <strong> HAND</strong> (Human–AI Natural Dialogue) and <strong>SIGNIT</strong> (Sign-based Interactive Quiz).
          </p>
          <div className="flex gap-3">
            <Link href="/downloads" className="btn btn-primary">Download Models</Link>
            <Link href="/use-cases" className="btn btn-outline">Explore Use Cases</Link>
          </div>
        </div>
      </section>

      <section className="grid md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="mb-2">HAND</h2>
          <p className="text-gray-600">Human–AI Natural Dialogue: two-way ASL ↔ natural language interface with an ASL avatar.</p>
          <div className="mt-4">
            <Link href="/use-cases/hand" className="btn btn-outline">View</Link>
          </div>
        </div>
        <div className="card">
          <h2 className="mb-2">SIGNIT</h2>
          <p className="text-gray-600">Sign-based Interactive Quiz: sign answers, get instant ASL feedback from an avatar.</p>
          <div className="mt-4">
            <Link href="/use-cases/signit" className="btn btn-outline">View</Link>
          </div>
        </div>
      </section>
    </div>
  )
}
