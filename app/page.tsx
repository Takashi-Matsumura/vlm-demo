'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { extractFramesFromVideo, type ExtractedFrame } from '@/lib/video';

type Status = 'idle' | 'extracting' | 'streaming' | 'error' | 'done';
type Mode = 'image' | 'video';

const DEFAULT_IMAGE_PROMPT = 'この画像を詳細に日本語で説明してください。';
const DEFAULT_VIDEO_PROMPT = 'この動画で何が起きているかを時系列に沿って日本語で説明してください。';

const MIN_FRAMES = 2;
const MAX_FRAMES = 12;
const DEFAULT_FRAME_COUNT = 6;

function detectMode(file: File): Mode | null {
  if (file.type.startsWith('image/')) return 'image';
  if (file.type.startsWith('video/')) return 'video';
  return null;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<Mode | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState(DEFAULT_IMAGE_PROMPT);
  const [promptTouched, setPromptTouched] = useState(false);
  const [frameCount, setFrameCount] = useState(DEFAULT_FRAME_COUNT);
  const [frames, setFrames] = useState<ExtractedFrame[]>([]);
  const [extractProgress, setExtractProgress] = useState<{ done: number; total: number } | null>(null);
  const [output, setOutput] = useState('');
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const frameUrls = useMemo(() => frames.map((f) => URL.createObjectURL(f.blob)), [frames]);
  useEffect(() => {
    return () => frameUrls.forEach((u) => URL.revokeObjectURL(u));
  }, [frameUrls]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const onPickFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    if (!f) {
      setFile(null);
      setMode(null);
      setFrames([]);
      return;
    }
    const m = detectMode(f);
    if (!m) {
      setError(`サポートしていないファイル形式: ${f.type || '不明'}`);
      setStatus('error');
      return;
    }
    setFile(f);
    setMode(m);
    setFrames([]);
    setOutput('');
    setError(null);
    setStatus('idle');
    if (!promptTouched) {
      setPrompt(m === 'video' ? DEFAULT_VIDEO_PROMPT : DEFAULT_IMAGE_PROMPT);
    }
  };

  const onExtract = async () => {
    if (!file || mode !== 'video') return;
    setStatus('extracting');
    setError(null);
    setExtractProgress({ done: 0, total: frameCount });
    try {
      const result = await extractFramesFromVideo(file, frameCount, (done, total) =>
        setExtractProgress({ done, total }),
      );
      setFrames(result);
      setStatus('idle');
    } catch (err) {
      setError((err as Error).message);
      setStatus('error');
    } finally {
      setExtractProgress(null);
    }
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || status === 'streaming' || status === 'extracting') return;
    if (mode === 'video' && frames.length === 0) {
      setError('先に「フレーム抽出」を実行してください。');
      setStatus('error');
      return;
    }

    const ac = new AbortController();
    abortRef.current = ac;
    setOutput('');
    setError(null);
    setStatus('streaming');

    const fd = new FormData();
    fd.append('prompt', prompt);
    if (mode === 'video') {
      frames.forEach((f, i) => {
        fd.append('frames', new File([f.blob], `frame-${i}.jpg`, { type: 'image/jpeg' }));
      });
    } else {
      fd.append('image', file);
    }

    try {
      const res = await fetch('/api/vlm', { method: 'POST', body: fd, signal: ac.signal });
      if (!res.ok || !res.body) {
        let detail = `HTTP ${res.status}`;
        try {
          const body = await res.json();
          detail = body.error ? `${body.error}${body.url ? ` (${body.url})` : ''}` : detail;
        } catch {
          // body wasn't JSON
        }
        setError(detail);
        setStatus('error');
        return;
      }
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) setOutput((o) => o + value);
      }
      setStatus('done');
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        setStatus('idle');
        return;
      }
      setError((err as Error).message);
      setStatus('error');
    } finally {
      abortRef.current = null;
    }
  };

  const onCancel = () => abortRef.current?.abort();

  const isBusy = status === 'streaming' || status === 'extracting';
  const showWarmupHint = status === 'streaming' && output === '';
  const canSubmit =
    !!file && !isBusy && (mode === 'image' || (mode === 'video' && frames.length > 0));

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold">VLM Demo — Qwen2.5-VL × llama.cpp</h1>
        <p className="mt-1 text-sm opacity-70">
          画像 1 枚または動画 1 本を入力すると、Qwen2.5-VL が日本語で言語化します。
          動画はブラウザ内で {MIN_FRAMES}〜{MAX_FRAMES} 枚のフレームに分解して送信されます。
        </p>
      </header>

      <form onSubmit={onSubmit} className="grid grid-cols-1 gap-8 md:grid-cols-2">
        <section className="space-y-3">
          <label className="block text-sm font-medium">
            画像 / 動画 {mode && <span className="ml-2 text-xs opacity-60">({mode})</span>}
          </label>
          <input
            type="file"
            accept="image/*,video/*"
            onChange={onPickFile}
            className="block w-full text-sm file:mr-3 file:rounded file:border file:border-current/20 file:bg-transparent file:px-3 file:py-1.5 file:text-sm"
          />

          <div className="aspect-video w-full overflow-hidden rounded border border-current/15 bg-black/[.03] dark:bg-white/[.04]">
            {previewUrl && mode === 'image' && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={previewUrl} alt="プレビュー" className="h-full w-full object-contain" />
            )}
            {previewUrl && mode === 'video' && (
              <video
                src={previewUrl}
                controls
                playsInline
                muted
                className="h-full w-full object-contain"
              />
            )}
            {!previewUrl && (
              <div className="flex h-full w-full items-center justify-center text-sm opacity-50">
                画像または動画を選択してください
              </div>
            )}
          </div>

          {mode === 'video' && (
            <div className="space-y-2 rounded border border-current/15 p-3">
              <div className="flex items-center justify-between gap-3">
                <label htmlFor="frameCount" className="text-sm font-medium">
                  抽出フレーム数: <span className="tabular-nums">{frameCount}</span>
                </label>
                <input
                  id="frameCount"
                  type="range"
                  min={MIN_FRAMES}
                  max={MAX_FRAMES}
                  step={1}
                  value={frameCount}
                  onChange={(e) => setFrameCount(Number(e.target.value))}
                  disabled={isBusy}
                  className="flex-1"
                />
              </div>
              <button
                type="button"
                onClick={onExtract}
                disabled={isBusy}
                className="rounded border border-current/30 px-3 py-1.5 text-sm font-medium disabled:opacity-40"
              >
                {status === 'extracting'
                  ? `フレーム抽出中… (${extractProgress?.done}/${extractProgress?.total})`
                  : frames.length > 0
                    ? 'フレームを再抽出'
                    : 'フレーム抽出'}
              </button>

              {frames.length > 0 && (
                <div className="grid grid-cols-3 gap-1.5 pt-2 sm:grid-cols-4">
                  {frameUrls.map((u, i) => (
                    <figure key={i} className="overflow-hidden rounded border border-current/15">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={u} alt={`frame ${i}`} className="aspect-video w-full object-cover" />
                      <figcaption className="px-1 py-0.5 text-[10px] tabular-nums opacity-60">
                        {frames[i].timestampSec.toFixed(1)}s
                      </figcaption>
                    </figure>
                  ))}
                </div>
              )}
            </div>
          )}
        </section>

        <section className="space-y-3">
          <label htmlFor="prompt" className="block text-sm font-medium">
            プロンプト
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => {
              setPrompt(e.target.value);
              setPromptTouched(true);
            }}
            rows={3}
            className="w-full resize-y rounded border border-current/20 bg-transparent px-3 py-2 text-sm outline-none focus:border-current/40"
          />

          <div className="flex gap-2">
            <button
              type="submit"
              disabled={!canSubmit}
              className="rounded bg-foreground px-4 py-2 text-sm font-medium text-background disabled:opacity-40"
            >
              {status === 'streaming' ? '生成中…' : '送信'}
            </button>
            {status === 'streaming' && (
              <button
                type="button"
                onClick={onCancel}
                className="rounded border border-current/30 px-4 py-2 text-sm font-medium"
              >
                キャンセル
              </button>
            )}
          </div>

          {showWarmupHint && (
            <p className="text-xs opacity-70">
              初回はモデルロードで 30〜60 秒かかることがあります。そのままお待ちください。
            </p>
          )}
          {error && (
            <div className="rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-700 dark:text-red-300">
              {error}
            </div>
          )}

          <div className="min-h-[12rem] rounded border border-current/15 bg-black/[.02] dark:bg-white/[.03] p-3">
            {output ? (
              <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
                {output}
                {status === 'streaming' && <span className="opacity-50">▍</span>}
              </pre>
            ) : (
              <div className="text-sm opacity-50">
                {status === 'idle' && '応答はここに表示されます。'}
                {status === 'extracting' && 'フレーム抽出中…'}
                {status === 'streaming' && '応答待ち…'}
                {status === 'done' && '（出力なし）'}
                {status === 'error' && 'エラーが発生しました。'}
              </div>
            )}
          </div>
        </section>
      </form>
    </main>
  );
}
