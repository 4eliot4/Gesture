import React, { useEffect, useRef, useState } from 'react';

type Props = {
  open: boolean;
  onClose: () => void;
  onLiveToken?: (s: string) => void;
  onFinalSentence?: (s: string) => void;
  // returns a blob URL of the recorded video when session ends
  onVideoReady?: (url: string, blob: Blob) => void;
};

/**
 * Hardcoded endpoints:
 * - Video upload (binary chunks): ws://localhost:5174/upload
 * - Model outputs (JSON):         ws://localhost:5174/asl
 *
 * Upload WS receives binary video/webm chunks from MediaRecorder every 500ms.
 * Model WS sends JSON messages like:
 *  { "type":"letter", "t": <ms>, "char":"A", "conf":0.9 }
 *  { "type":"word",   "t": <ms>, "word":"GLUCOSE", "conf":0.85 }
 *  { "type":"final",  "t": <ms>, "text":"plants convert sunlight into glucose", "conf":0.9 }
 */
export default function ASLSession({ open, onClose, onLiveToken, onFinalSentence, onVideoReady }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const uploadWSRef = useRef<WebSocket | null>(null);
  const modelWSRef = useRef<WebSocket | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let stopped = false;
    (async () => {
      try {
        // get camera
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (stopped) return;
        mediaStreamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }

        // connect websockets
        const uploadWS = new WebSocket('ws://localhost:5174/upload');
        const modelWS  = new WebSocket('ws://localhost:5174/asl');
        uploadWS.binaryType = 'arraybuffer';

        uploadWS.onopen = () => {
          // start recording once upload socket is open
          const rec = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });
          mediaRecorderRef.current = rec;
          chunksRef.current = [];
          rec.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
              chunksRef.current.push(e.data);
              try {
                // stream chunk to backend
                if (uploadWS.readyState === WebSocket.OPEN) {
                  e.data.arrayBuffer().then((buf) => uploadWS.send(buf));
                }
              } catch {}
            }
          };
          rec.start(500); // send a chunk every 500ms
          setRecording(true);
          setReady(true);
        };

        modelWS.onmessage = (e) => {
          try {
            const msg = JSON.parse(e.data);
            if (msg.type === 'letter' && msg.char) {
              onLiveToken?.(String(msg.char));
            } else if (msg.type === 'word' && msg.word) {
              onLiveToken?.(String(msg.word));
            } else if (msg.type === 'final' && msg.text) {
              onFinalSentence?.(String(msg.text));
            }
          } catch {}
        };

        modelWSRef.current = modelWS;
        uploadWSRef.current = uploadWS;
      } catch (err: any) {
        setError(err?.message ?? 'Failed to open camera or sockets');
      }
    })();

    return () => {
      stopped = true
    }
  }, [open]);

  const stopAll = async () => {
    try {
      mediaRecorderRef.current?.stop();
    } catch {}
    try {
      mediaStreamRef.current?.getTracks().forEach(t => t.stop());
    } catch {}
    try { uploadWSRef.current?.close(); } catch {}
    try { modelWSRef.current?.close(); } catch {}

    setRecording(false);

    // Build a single Blob for local playback
    const blob = new Blob(chunksRef.current, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    onVideoReady?.(url, blob);
  };

  const closeSession = async () => {
    await stopAll();
    onClose();
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="w-full max-w-2xl rounded-2xl bg-white p-4 shadow-xl">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Record your ASL answer</h3>
          <button onClick={closeSession} className="rounded-xl border px-3 py-1">Close</button>
        </div>

        {error && <div className="mt-2 rounded border border-red-300 bg-red-50 p-2 text-sm text-red-700">{error}</div>}

        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <div className="space-y-2">
            <div className="text-sm opacity-60">Camera</div>
            <video ref={videoRef} className="aspect-video w-full rounded-xl border bg-black" muted playsInline />
            <div className="flex items-center gap-2">
              <button
                className="rounded-xl border px-3 py-1 disabled:opacity-50"
                onClick={recording ? stopAll : undefined}
                disabled={!ready || !recording}
              >
                Stop
              </button>
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-sm opacity-60">Model output</div>
            <div className="rounded-xl border p-2 text-sm">
              <p className="opacity-60">Your backend should stream letters/words/final sentences here via WS.</p>
              <p className="opacity-60">This UI forwards live tokens and final sentences to the quiz logic.</p>
            </div>
          </div>
        </div>

        <p className="mt-2 text-xs opacity-60">
          Video chunks are streamed to <code>ws://localhost:5174/upload</code>. Model results are read from <code>ws://localhost:5174/asl</code>.
        </p>
      </div>
    </div>
  );
}
