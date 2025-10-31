import Link from 'next/link'

export default function UseCasesIndex() {
  return (
    <div className="space-y-6">
      <h1>Use Cases</h1>
      <p className="text-gray-600">At GESTURE, we're dedicated to not only creating AI models for accessibility, but also producing real-world use case scenarios to bring them to users in the hopes of benefiting them in their daily life. </p>
      <div className="grid md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="mb-2">HAND</h2>
          <p className="text-gray-600">Two-way ASL ↔ natural language interface with an ASL avatar.</p>
          <div className="mt-4"><Link className="btn btn-outline" href="/use-cases/hand">Open</Link></div>
        </div>
        <div className="card">
          <h2 className="mb-2">SIGNIT</h2>
          <p className="text-gray-600">Sign-based Interactive Quiz: sign answers, get feedback live.</p>
          <div className="mt-4"><Link className="btn btn-outline" href="/use-cases/signit">Open</Link></div>
        </div>
      </div>
    </div>
  )
}
