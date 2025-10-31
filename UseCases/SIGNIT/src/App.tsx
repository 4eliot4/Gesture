import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Check, CircleHelp, Keyboard, ListChecks, RotateCcw } from 'lucide-react';
import ASLSession from './ASLSession';
import { Link } from 'react-router-dom';

// =========================
// SIGNIT with ASL model plug-in
// Model outputs expected via ws://localhost:5174/asl (streamed by ASLSession).
// =========================

type Mode = 'MCQ' | 'FULL';

interface BaseQuestion {
  id: string;
  prompt: string;
  explanation?: string;
}

interface MCQQuestion extends BaseQuestion {
  type: 'MCQ';
  options: Array<{ key: 'a' | 'b' | 'c' | 'd'; label: string }>;
  answer: 'a' | 'b' | 'c' | 'd';
}

interface FullAnswerQuestion extends BaseQuestion {
  type: 'FULL';
  expected?: string | RegExp;
}

type Question = MCQQuestion | FullAnswerQuestion;

const SAMPLE_QUESTIONS: Question[] = [
  {
    id: 'q1',
    type: 'MCQ',
    prompt: 'Which letter comes after B?',
    options: [
      { key: 'a', label: 'A' },
      { key: 'b', label: 'B' },
      { key: 'c', label: 'C' },
      { key: 'd', label: 'D' },
    ],
    answer: 'c',
    explanation: 'The sequence is A, B, C, D — so C comes after B.',
  },
  {
    id: 'q2',
    type: 'FULL',
    prompt: 'In one short sentence, describe what photosynthesis does.',
    expected: /converts\s+light|sunlight\s+into\s+(chemical\s+)?energy|glucose/i,
    explanation: 'At a high level, plants convert sunlight into chemical energy (glucose).',
  },
  {
    id: 'q3',
    type: 'MCQ',
    prompt: '2 + 2 = ?',
    options: [
      { key: 'a', label: '3' },
      { key: 'b', label: '4' },
      { key: 'c', label: '22' },
      { key: 'd', label: '5' },
    ],
    answer: 'b',
  },
];

function scoreMCQ(q: MCQQuestion, response?: 'a' | 'b' | 'c' | 'd') {
  return Number(response === q.answer);
}

function scoreFull(q: FullAnswerQuestion, response?: string) {
  if (!response || !q.expected) return 0;
  if (typeof q.expected === 'string') {
    return Number(response.trim().toLowerCase().includes(q.expected.trim().toLowerCase()));
  }
  return Number(q.expected.test(response));
}

// helpers for MCQ auto-mapping
function normalize(s:string){ return s.toLowerCase().replace(/[^a-z0-9]+/g,'').trim(); }
function sim(a:string,b:string){
  a = normalize(a); b = normalize(b);
  if (!a || !b) return 0;
  const L = Math.max(a.length,b.length);
  let same = 0;
  for (let i=0;i<Math.min(a.length,b.length);i++) if (a[i]===b[i]) same++;
  return same / L;
}

// =========================
// UI components
// =========================
function Chip({ label }: { label: string }) {
  return <span className="inline-flex items-center rounded-full border px-2 py-0.5 text-xs">{label}</span>;
}

function Section({ title, children }: React.PropsWithChildren<{ title: string }>) {
  return (
    <div className="space-y-3">
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      <div className="rounded-2xl border p-4 shadow-sm">{children}</div>
    </div>
  );
}

function MCQCard({
  q,
  value,
  locked,
}: {
  q: MCQQuestion;
  value?: 'a' | 'b' | 'c' | 'd';
  locked?: boolean;
}) {
  const answered = Boolean(value);
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <CircleHelp className="h-5 w-5" />
        <p className="text-base font-medium">{q.prompt}</p>
        <Chip label="MCQ" />
      </div>
      <div className="text-sm opacity-70">
        Options exist conceptually (A / B / C / D). Sign your answer—our model maps it to the closest option automatically.
      </div>
      {answered && (
        <div className="text-xs opacity-70">
          Selected (by model): <b>{value?.toUpperCase()}</b>
        </div>
      )}
    </div>
  );
}

function FullAnswerCard({
  q,
  value,
  locked,
}: {
  q: FullAnswerQuestion;
  value?: string;
  locked?: boolean;
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Keyboard className="h-5 w-5" />
        <p className="text-base font-medium">{q.prompt}</p>
        <Chip label="Full Answer" />
      </div>
      <div className="min-h-[96px] w-full rounded-2xl border p-3 text-sm opacity-70">
        Signing only — your sentence will appear here after the model emits a final result.
      </div>
      {value && (
        <div className="text-xs opacity-70">
          Answer (by model): <b>{value}</b>
        </div>
      )}
    </div>
  );
}

function Result({ correct, explanation }: { correct: boolean; explanation?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm">
      <Check className="h-4 w-4" />
      <span className={correct ? 'font-medium' : 'font-medium'}>
        {correct ? 'Marked correct' : 'Marked incorrect'}
      </span>
      {explanation && <span className="opacity-80">· {explanation}</span>}
    </div>
  );
}

// =========================
// Main App
// =========================
export default function App() {
  const [mode, setMode] = useState<Mode>('MCQ');
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string | undefined>>({});
  const [submitted, setSubmitted] = useState<Record<string, boolean>>({});
  const [aslOpen, setAslOpen] = useState(false);
  const [answerVideos, setAnswerVideos] = useState<Record<string, string>>({});

  const questions = useMemo(() => SAMPLE_QUESTIONS.filter((q) => q.type === mode), [mode]);
  const current = questions[index];

  const totalScore = useMemo(() => {
    return SAMPLE_QUESTIONS.reduce((acc, q) => {
      const val = answers[q.id];
      if (q.type === 'MCQ') return acc + scoreMCQ(q, val as any);
      return acc + scoreFull(q, val as string);
    }, 0);
  }, [answers]);

  const isLast = index === Math.max(0, questions.length - 1);
  const locked = submitted[current?.id ?? ''] === true;

  const next = () => {
    if (isLast) return;
    setIndex((i) => i + 1);
  };

  const prev = () => {
    setIndex((i) => Math.max(0, i - 1));
  };

  const resetAll = () => {
    setAnswers({});
    setSubmitted({});
    setIndex(0);
  };

  const correctness = (() => {
    if (!current) return undefined;
    const val = answers[current.id];
    if (val == null || !submitted[current.id]) return undefined;
    if (current.type === 'MCQ') return scoreMCQ(current, val as any) === 1;
    return scoreFull(current, val as string) === 1;
  })();

  // Auto-open signing modal on question load/change if not already submitted
  useEffect(() => {
    if (!current) return;
    if (!submitted[current.id]) {
      setAslOpen(true);
    }
  }, [current?.id]);

  // ASL mapping logic for MCQ and FULL
  function onLiveToken(live: string) {
    if (!current || current.type !== 'MCQ' || locked) return;
    const token = normalize(live);

    // direct letter choice (e.g., signer spells 'A', 'B', 'C', 'D')
    if (['a', 'b', 'c', 'd'].includes(token)) {
      setAnswers((a) => ({ ...a, [current.id]: token as any }));
      return;
    }

    // fuzzy match token against option labels
    let bestKey: 'a' | 'b' | 'c' | 'd' | undefined;
    let bestScore = 0;
    for (const opt of current.options) {
      const s = sim(token, opt.label);
      if (s > bestScore) { bestScore = s; bestKey = opt.key; }
    }
    if (bestKey && bestScore >= 0.8) {
      setAnswers((a) => ({ ...a, [current.id]: bestKey! }));
    }
  }

  function onFinalSentence(sentence: string) {
    if (!current) return;
    if (current.type === 'FULL') {
      if (locked) return;
      setAnswers((a) => ({ ...a, [current.id]: sentence }));
      setSubmitted((s) => ({ ...s, [current.id]: true }));
      setAslOpen(false);
      return;
    }
    // MCQ: map final sentence to option and submit
    const token = normalize(sentence);
    let bestKey: 'a' | 'b' | 'c' | 'd' | undefined;
    let bestScore = 0;
    for (const opt of current.options) {
      const s = sim(token, opt.label);
      if (s > bestScore) { bestScore = s; bestKey = opt.key; }
    }
    if (bestKey) {
      setAnswers((a) => ({ ...a, [current.id]: bestKey! }));
      setSubmitted((s) => ({ ...s, [current.id]: true }));
      setAslOpen(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl border text-lg font-bold">S</div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">SIGNIT</h1>
            <p className="text-sm opacity-70">ASL-enabled quiz bot</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link to="/rules" className="rounded-2xl border px-3 py-2 text-sm">Rules</Link>
          <button
            onClick={() => setMode('MCQ')}
            className={[
              'inline-flex items-center gap-2 rounded-2xl border px-3 py-2',
              mode === 'MCQ' ? 'ring-2 ring-offset-2' : 'hover:shadow',
            ].join(' ')}
          >
            <ListChecks className="h-4 w-4" /> MCQ
          </button>
          <button
            onClick={() => setMode('FULL')}
            className={[
              'inline-flex items-center gap-2 rounded-2xl border px-3 py-2',
              mode === 'FULL' ? 'ring-2 ring-offset-2' : 'hover:shadow',
            ].join(' ')}
          >
            <Keyboard className="h-4 w-4" /> Full answers
          </button>
        </div>
      </header>

      <Section title="Question">
        {!current ? (
          <div className="text-sm opacity-80">No questions available for this mode.</div>
        ) : current.type === 'MCQ' ? (
          <MCQCard
            q={current}
            value={answers[current.id] as any}
            locked={locked}
          />
        ) : (
          <FullAnswerCard
            q={current}
            value={(answers[current.id] as string) ?? ''}
            locked={locked}
          />
        )}

        {correctness !== undefined && (
          <div className="mt-3">
            <Result correct={correctness} explanation={current.explanation} />
          </div>
        )}

        {/* Recorded video preview */}
        {answerVideos[current?.id ?? ''] && (
          <div className="mt-4">
            <div className="text-sm opacity-60 mb-1">Your signed answer (recorded):</div>
            <video controls className="w-full max-w-md rounded-xl border" src={answerVideos[current!.id]} />
          </div>
        )}

        {/* Auto-opened ASL capture modal */}
        <ASLSession
          open={aslOpen}
          onClose={() => setAslOpen(false)}
          onLiveToken={onLiveToken}
          onFinalSentence={onFinalSentence}
          onVideoReady={(url) => {
            if (!current) return;
            setAnswerVideos(v => ({ ...v, [current.id]: url }));
          }}
        />
      </Section>

      <Section title="Progress">
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <Chip label={`Mode: ${mode}`} />
          <Chip label={`Question ${questions.length ? index + 1 : 0}/${questions.length}`} />
          <Chip label={`Total Score: ${totalScore}/${SAMPLE_QUESTIONS.length}`} />
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4">
          {questions.map((q, i) => {
            const isSubmitted = submitted[q.id];
            const val = answers[q.id];
            const correct =
              q.type === 'MCQ' ? scoreMCQ(q, val as any) === 1 : scoreFull(q as FullAnswerQuestion, val as string) === 1;
            return (
              <button
                key={q.id}
                onClick={() => setIndex(i)}
                className={[
                  'rounded-2xl border px-3 py-2 text-left text-sm transition',
                  i === index ? 'ring-2 ring-offset-2' : 'hover:shadow',
                ].join(' ')}
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs opacity-70">#{i + 1}</span>
                  <span className="truncate">{q.prompt}</span>
                </div>
                <div className="mt-1 text-xs opacity-70">
                  {isSubmitted ? (correct ? 'Correct' : 'Incorrect') : 'Unanswered'}
                </div>
              </button>
            );
          })}
        </div>
        <div className="mt-3 flex items-center gap-2">
          <button onClick={prev} disabled={index === 0} className="rounded-2xl border px-3 py-2 disabled:opacity-50">
            Prev
          </button>
          <button onClick={next} disabled={isLast} className="rounded-2xl border px-3 py-2 disabled:opacity-50">
            Next
          </button>
          <button onClick={resetAll} className="ml-auto inline-flex items-center gap-2 rounded-2xl border px-3 py-2">
            <RotateCcw className="h-4 w-4" /> Reset
          </button>
        </div>
      </Section>

      <footer className="pt-2 text-center text-xs opacity-60">
        <p>Signing-only. The camera opens automatically for each question. The model picks the MCQ option or records your full-sentence answer.</p>
      </footer>
    </div>
  );
}
