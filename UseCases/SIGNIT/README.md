# SIGNIT

A very basic quiz-bot framework with two modes:

- **MCQ**: single-choice multiple choice
- **Full answers**: free-text answers (basic regex/string check)

Built with React + TypeScript + Vite + Tailwind.

## Quickstart

```bash
# 1) Install deps
npm i

# 2) Run locally
npm run dev

# 3) Production build
npm run build

# 4) Preview local prod build
npm run preview
```

Deploy the contents of `dist/` to any static host (Vercel, Netlify, S3, Firebase, GitHub Pages, etc.).

## Editing Questions

Open `src/App.tsx` and update `SAMPLE_QUESTIONS`. Replace with your own loader/API when ready.

## Notes

- This is intentionally minimal. No state management lib, no router.
- Tailwind is set up out of the box.
- Icons via `lucide-react`.


## ASL workflow (hardcoded endpoints)

- Click **Answer by signing** → opens a modal with webcam preview.
- The app records `video/webm` with `MediaRecorder` and streams 500ms chunks to `ws://localhost:5174/upload`.
- Simultaneously it listens to `ws://localhost:5174/asl` for JSON outputs from your model:
  - `{"type":"letter","t":<ms>,"char":"A","conf":0.9}`
  - `{"type":"word","t":<ms>,"word":"GLUCOSE","conf":0.85}`
  - `{"type":"final","t":<ms>,"text":"...","conf":0.9}`
- Live tokens map to MCQ options automatically; final sentences auto-submit in Full mode.
- When you stop, the recorded video is attached to the current answer and previewed in the UI.

> Swap these endpoints with your actual backend when ready (see `src/ASLSession.tsx`).

## Rules page
- Visit `/rules` to see the answering rules.
- MCQ options are conceptually A/B/C/D; you will **not** click them—answers are provided by signing only.
- The camera/signing modal opens automatically when each question appears.
