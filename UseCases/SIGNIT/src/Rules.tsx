import React from 'react';
import { Link } from 'react-router-dom';

export default function Rules() {
  return (
    <div className="mx-auto max-w-3xl p-6 space-y-6">
      <header className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">SIGNIT Rules</h1>
        <Link to="/" className="rounded-xl border px-3 py-1 text-sm">Back to quiz</Link>
      </header>

      <section className="rounded-2xl border p-4 space-y-3">
        <h2 className="text-xl font-semibold">How answers work</h2>
        <ul className="list-disc pl-6 space-y-2 text-sm">
          <li><strong>Signing only.</strong> You answer every question using ASL via your camera.</li>
          <li><strong>Multiple Choice Questions (MCQ):</strong> Options are conceptually labeled <code>A</code>, <code>B</code>, <code>C</code>, <code>D</code>. Sign the option (e.g., the word/phrase that best matches), and we map your signed phrase to the closest option automatically.</li>
          <li><strong>Full Answer Questions:</strong> Sign a full sentence. The model transcribes your signing and we auto-submit.</li>
          <li><strong>Recording:</strong> Your signed attempt is recorded and attached to the question for review.</li>
        </ul>
      </section>

      <section className="rounded-2xl border p-4 space-y-3">
        <h2 className="text-xl font-semibold">Privacy & tips</h2>
        <ul className="list-disc pl-6 space-y-2 text-sm">
          <li>Ensure good lighting and keep your hands in frame.</li>
          <li>Hold poses briefly for clearer recognition.</li>
          <li>You can move to the next question after your answer is submitted.</li>
        </ul>
      </section>
    </div>
  );
}
