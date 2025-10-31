import Link from 'next/link'

export default function SIGNITPage() {
  return (
    <div className="space-y-6">
      <h1>SIGNIT – Sign-based Interactive Quiz</h1>
      <p className="text-gray-600 max-w-3xl">
        SIGNIT is a quiz bot that accepts answers in ASL.
      </p>

      <div className="card">
        <h2 className="mb-2">Download</h2>
        <p className="text-gray-600">Download the code for the SIGNIT use case.</p>
        <div className="mt-4 flex gap-3">
          <button className="btn btn-primary">Download</button>
          <button className="btn btn-outline">View Docs</button>
        </div>
      </div>

      <div className="card">
        <h2 className="mb-2">Getting Started</h2>
        <ol className="list-decimal ml-6 text-gray-700 space-y-2">
          <li>Download the Gesture Lite model or the Gesture Heavy model from the <Link href="/downloads" className="underline">Downloads</Link> page.</li>
          <li>Download the SIGNIT codebase from the link above.</li>
          <li>Wire up webcam input. When prompted, suggest a subject you'd like to be quizzed about. Enjoy your quiz!</li>
        </ol>
      </div>
    </div>
  )
}
