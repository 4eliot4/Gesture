import Link from 'next/link'

export default function HANDPage() {
  return (
    <div className="space-y-6">
      <h1>HAND – Human–AI Natural Dialogue</h1>
      <p className="text-gray-600 max-w-3xl">
        HAND enables a complete ASL ↔ natural language loop: talk to the LLM in ASL, receive a response in ASL.
      </p>

      <div className="card">
        <h2 className="mb-2">Download</h2>
        <p className="text-gray-600">Download the code for the HAND use case below.</p>
        <div className="mt-4 flex gap-3">
          <button className="btn btn-primary">Download</button>
          <button className="btn btn-outline">View Docs</button>
        </div>
      </div>

      <div className="card">
        <h2 className="mb-2">Getting Started</h2>
        <ol className="list-decimal ml-6 text-gray-700 space-y-2">
          <li>Download the Gesture Lite model or the Gesture Heavy model from the <Link href="/downloads" className="underline">Downloads</Link> page.</li>
          <li>Download the HAND codebase from the link above.</li>
          <li>Wire up webcam input. Try a conversation!</li>
        </ol>
      </div>
    </div>
  )
}
