'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { extractFramesFromVideo, type ExtractedFrame } from '@/lib/video';

type Status = 'idle' | 'extracting' | 'streaming' | 'error' | 'done';
type Mode = 'image' | 'video';
type BackendMode = 'llama_cpp' | 'llama_cpp_72b' | 'mlx_native' | 'parallel';

interface Metrics {
  ttft: number | null;
  duration: number | null;
}

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

function formatMs(ms: number | null): string {
  if (ms === null) return '—';
  return ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(1)} s`;
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
  const [backendMode, setBackendMode] = useState<BackendMode>('llama_cpp');
  const [mlxHealth, setMlxHealth] = useState<{ loaded: boolean; error?: string } | null>(null);

  // llama_cpp / primary output
  const [output, setOutput] = useState('');
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const [llamaMetrics, setLlamaMetrics] = useState<Metrics>({ ttft: null, duration: null });
  const abortRef = useRef<AbortController | null>(null);

  // mlx_native / secondary output (parallel mode)
  const [mlxOutput, setMlxOutput] = useState('');
  const [mlxStatus, setMlxStatus] = useState<Status>('idle');
  const [mlxError, setMlxError] = useState<string | null>(null);
  const [mlxMetrics, setMlxMetrics] = useState<Metrics>({ ttft: null, duration: null });
  const mlxAbortRef = useRef<AbortController | null>(null);

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
    return () => {
      abortRef.current?.abort();
      mlxAbortRef.current?.abort();
    };
  }, []);

  const [llama72bHealth, setLlama72bHealth] = useState<{ loaded: boolean; error?: string } | null>(null);

  // Poll backend health
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('/api/vlm');
        if (res.ok) {
          const data = await res.json();
          setMlxHealth(data.mlx ?? null);
          setLlama72bHealth(data.llama_72b ?? null);
        }
      } catch {
        setMlxHealth({ loaded: false, error: 'unreachable' });
        setLlama72bHealth({ loaded: false, error: 'unreachable' });
      }
    };
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
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
    setMlxOutput('');
    setError(null);
    setMlxError(null);
    setStatus('idle');
    setMlxStatus('idle');
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

  const streamFromBackend = async (
    fd: FormData,
    setOut: React.Dispatch<React.SetStateAction<string>>,
    setSt: React.Dispatch<React.SetStateAction<Status>>,
    setErr: React.Dispatch<React.SetStateAction<string | null>>,
    setMetrics: React.Dispatch<React.SetStateAction<Metrics>>,
    signal: AbortSignal,
  ) => {
    const startTime = performance.now();
    let firstChunk = true;
    setOut('');
    setErr(null);
    setSt('streaming');
    try {
      const res = await fetch('/api/vlm', { method: 'POST', body: fd, signal });
      if (!res.ok || !res.body) {
        let detail = `HTTP ${res.status}`;
        try {
          const body = await res.json();
          detail = body.error ? `${body.error}${body.url ? ` (${body.url})` : ''}` : detail;
        } catch {
          // body wasn't JSON
        }
        setErr(detail);
        setSt('error');
        return;
      }
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) {
          if (firstChunk) {
            setMetrics((m) => ({ ...m, ttft: performance.now() - startTime }));
            firstChunk = false;
          }
          setOut((o) => o + value);
        }
      }
      setMetrics((m) => ({ ...m, duration: performance.now() - startTime }));
      setSt('done');
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        setSt('idle');
        return;
      }
      setErr((err as Error).message);
      setSt('error');
    }
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || isBusy) return;

    setOutput('');
    setMlxOutput('');
    setError(null);
    setMlxError(null);
    setLlamaMetrics({ ttft: null, duration: null });
    setMlxMetrics({ ttft: null, duration: null });

    if (backendMode === 'parallel') {
      if (mode === 'video' && frames.length === 0) {
        setError('並列比較モードでは先に「フレーム抽出」を実行してください。');
        setStatus('error');
        return;
      }

      const llamaAc = new AbortController();
      const mlxAc = new AbortController();
      abortRef.current = llamaAc;
      mlxAbortRef.current = mlxAc;

      const llamaFd = new FormData();
      llamaFd.append('prompt', prompt);
      llamaFd.append('mode', 'llama_cpp');
      if (mode === 'video') {
        frames.forEach((f, i) =>
          llamaFd.append('frames', new File([f.blob], `frame-${i}.jpg`, { type: 'image/jpeg' })),
        );
      } else {
        llamaFd.append('image', file);
      }

      const mlxFd = new FormData();
      mlxFd.append('prompt', prompt);
      mlxFd.append('mode', 'mlx_native');
      if (mode === 'video') {
        mlxFd.append('video', file);
      } else {
        mlxFd.append('image', file);
      }

      Promise.all([
        streamFromBackend(llamaFd, setOutput, setStatus, setError, setLlamaMetrics, llamaAc.signal),
        streamFromBackend(mlxFd, setMlxOutput, setMlxStatus, setMlxError, setMlxMetrics, mlxAc.signal),
      ]).finally(() => {
        abortRef.current = null;
        mlxAbortRef.current = null;
      });
      return;
    }

    // Single backend
    if ((backendMode === 'llama_cpp' || backendMode === 'llama_cpp_72b') && mode === 'video' && frames.length === 0) {
      setError('先に「フレーム抽出」を実行してください。');
      setStatus('error');
      return;
    }

    const ac = new AbortController();
    abortRef.current = ac;

    const fd = new FormData();
    fd.append('prompt', prompt);
    fd.append('mode', backendMode);

    if (backendMode === 'mlx_native') {
      if (mode === 'video') {
        fd.append('video', file);
      } else {
        fd.append('image', file);
      }
    } else {
      if (mode === 'video') {
        frames.forEach((f, i) =>
          fd.append('frames', new File([f.blob], `frame-${i}.jpg`, { type: 'image/jpeg' })),
        );
      } else {
        fd.append('image', file);
      }
    }

    const [setOut, setSt, setErr, setMet] =
      backendMode === 'mlx_native'
        ? [setMlxOutput, setMlxStatus, setMlxError, setMlxMetrics]
        : [setOutput, setStatus, setError, setLlamaMetrics];

    await streamFromBackend(fd, setOut, setSt, setErr, setMet, ac.signal);
    abortRef.current = null;
  };

  const onCancel = () => {
    abortRef.current?.abort();
    mlxAbortRef.current?.abort();
  };

  const isBusy =
    status === 'streaming' || mlxStatus === 'streaming' || status === 'extracting';

  const showWarmupHint =
    (status === 'streaming' && output === '') ||
    (mlxStatus === 'streaming' && mlxOutput === '');

  const canSubmit =
    !!file &&
    !isBusy &&
    (mode === 'image' ||
      (mode === 'video' && backendMode === 'mlx_native') ||
      (mode === 'video' && backendMode !== 'mlx_native' && frames.length > 0));

  const needsFrames =
    mode === 'video' && (backendMode === 'llama_cpp' || backendMode === 'llama_cpp_72b' || backendMode === 'parallel');

  const showFrameExtractor = needsFrames;
  const showNativeVideoHint = mode === 'video' && backendMode === 'mlx_native';

  const parallelDone =
    backendMode === 'parallel' && (status === 'done' || mlxStatus === 'done');
  const isParallel = backendMode === 'parallel';

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-6">
        <div className="flex items-center gap-3 flex-wrap">
          <h1 className="text-2xl font-semibold">VLM Demo — Qwen2.5-VL</h1>
          {llama72bHealth !== null && (
            <span
              className={`rounded px-2 py-0.5 text-xs font-medium ${
                llama72bHealth.loaded
                  ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                  : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
              }`}
            >
              llama 72B {llama72bHealth.loaded ? '✓ ready' : '✗ offline'}
            </span>
          )}
          {mlxHealth !== null && (
            <span
              className={`rounded px-2 py-0.5 text-xs font-medium ${
                mlxHealth.loaded
                  ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                  : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
              }`}
            >
              mlx-vlm {mlxHealth.loaded ? '✓ ready' : '✗ offline'}
            </span>
          )}
        </div>
        <p className="mt-1 text-sm opacity-70">
          画像 1 枚または動画 1 本を入力すると Qwen2.5-VL が日本語で言語化します。
          バックエンドを切り替えて精度・速度の違いを比較できます。
        </p>
      </header>

      {/* Backend selector */}
      <div className="mb-6 rounded border border-current/15 p-3">
        <fieldset>
          <legend className="mb-2 text-xs font-medium opacity-60">バックエンド</legend>
          <div className="flex flex-wrap gap-4">
            {(
              [
                ['llama_cpp',     'llama.cpp 7B',  'フレーム多枚画像方式'],
                ['llama_cpp_72b', 'llama.cpp 72B', 'フレーム多枚画像方式 (高品質)'],
                ['mlx_native',    'mlx-vlm 72B',   'ネイティブ動画入力 (M-RoPE)'],
                ['parallel',      '並列比較',       '同じ入力で両方同時実行'],
              ] as const
            ).map(([val, label, desc]) => (
              <label key={val} className="flex cursor-pointer items-start gap-2">
                <input
                  type="radio"
                  name="backendMode"
                  value={val}
                  checked={backendMode === val}
                  onChange={() => setBackendMode(val)}
                  disabled={isBusy}
                  className="mt-0.5"
                />
                <span>
                  <span className="text-sm font-medium">{label}</span>
                  <span className="ml-1.5 text-xs opacity-50">{desc}</span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>
      </div>

      <form onSubmit={onSubmit} className="grid grid-cols-1 gap-8 md:grid-cols-2">
        {/* Input column */}
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

          {/* Frame extractor — shown only when llama_cpp or parallel mode with video */}
          {showFrameExtractor && (
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

          {/* Native video hint for mlx_native mode */}
          {showNativeVideoHint && (
            <p className="rounded border border-blue-500/20 bg-blue-500/5 px-3 py-2 text-sm">
              動画をそのまま mlx-vlm に送信します — フレーム抽出は不要です。
            </p>
          )}
        </section>

        {/* Output column */}
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
              {isBusy ? '生成中…' : '送信'}
            </button>
            {isBusy && (
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
              初回はモデルロードで時間がかかることがあります (llama.cpp: 30〜60 秒、mlx-vlm 72B:
              数分)。そのままお待ちください。
            </p>
          )}

          {/* Output panels */}
          <div className={isParallel ? 'grid grid-cols-2 gap-3' : ''}>
            {/* Primary output (llama_cpp or single mlx_native) */}
            {(backendMode !== 'mlx_native') && (
              <div>
                {isParallel && (
                  <p className="mb-1 text-xs font-medium opacity-60">llama.cpp 7B</p>
                )}
                {error && (
                  <div className="mb-2 rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-700 dark:text-red-300">
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
                {isParallel && llamaMetrics.ttft !== null && (
                  <p className="mt-1 text-[10px] tabular-nums opacity-50">
                    TTFT {formatMs(llamaMetrics.ttft)} / 総時間 {formatMs(llamaMetrics.duration)}
                  </p>
                )}
              </div>
            )}

            {/* MLX output (mlx_native or parallel) */}
            {(backendMode === 'mlx_native' || isParallel) && (
              <div>
                {isParallel && (
                  <p className="mb-1 text-xs font-medium opacity-60">mlx-vlm 72B</p>
                )}
                {mlxError && (
                  <div className="mb-2 rounded border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-700 dark:text-red-300">
                    {mlxError}
                  </div>
                )}
                <div className="min-h-[12rem] rounded border border-current/15 bg-black/[.02] dark:bg-white/[.03] p-3">
                  {mlxOutput ? (
                    <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
                      {mlxOutput}
                      {mlxStatus === 'streaming' && <span className="opacity-50">▍</span>}
                    </pre>
                  ) : (
                    <div className="text-sm opacity-50">
                      {mlxStatus === 'idle' && '応答はここに表示されます。'}
                      {mlxStatus === 'streaming' && '応答待ち…'}
                      {mlxStatus === 'done' && '（出力なし）'}
                      {mlxStatus === 'error' && 'エラーが発生しました。'}
                    </div>
                  )}
                </div>
                {mlxMetrics.ttft !== null && (
                  <p className="mt-1 text-[10px] tabular-nums opacity-50">
                    TTFT {formatMs(mlxMetrics.ttft)} / 総時間 {formatMs(mlxMetrics.duration)}
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Metrics comparison table (parallel mode, after at least one side finishes) */}
          {parallelDone && (
            <table className="w-full border-collapse text-xs">
              <thead>
                <tr className="border-b border-current/10">
                  <th className="py-1 pr-4 text-left font-medium opacity-70">指標</th>
                  <th className="py-1 pr-4 text-right font-medium opacity-70">llama.cpp 7B</th>
                  <th className="py-1 text-right font-medium opacity-70">mlx-vlm 72B</th>
                </tr>
              </thead>
              <tbody className="opacity-80">
                <tr>
                  <td className="py-0.5 pr-4">初回出力まで (TTFT)</td>
                  <td className="py-0.5 pr-4 text-right tabular-nums">
                    {formatMs(llamaMetrics.ttft)}
                  </td>
                  <td className="py-0.5 text-right tabular-nums">{formatMs(mlxMetrics.ttft)}</td>
                </tr>
                <tr>
                  <td className="py-0.5 pr-4">完了までの総時間</td>
                  <td className="py-0.5 pr-4 text-right tabular-nums">
                    {formatMs(llamaMetrics.duration)}
                  </td>
                  <td className="py-0.5 text-right tabular-nums">
                    {formatMs(mlxMetrics.duration)}
                  </td>
                </tr>
                <tr>
                  <td className="py-0.5 pr-4">出力文字数</td>
                  <td className="py-0.5 pr-4 text-right tabular-nums">{output.length}</td>
                  <td className="py-0.5 text-right tabular-nums">{mlxOutput.length}</td>
                </tr>
              </tbody>
            </table>
          )}

          {/* Educational info — shown in parallel mode */}
          {isParallel && (
            <details className="rounded border border-current/10 text-xs">
              <summary className="cursor-pointer px-3 py-2 font-medium opacity-70">
                各バックエンドの技術解説
              </summary>
              <div className="grid grid-cols-1 gap-3 px-3 pb-3 pt-2 sm:grid-cols-2">
                <div>
                  <p className="mb-1 font-medium">llama.cpp 7B — フレーム多枚方式</p>
                  <ul className="space-y-1 opacity-70">
                    <li>
                      ブラウザで {frameCount} 枚の JPEG を等間隔に抽出し、独立した画像トークンとして送信
                    </li>
                    <li>フレーム間の時間情報は「並び順」のみ — 本物の時間軸エンコーディングはない</li>
                    <li>抽出間隔より短い動きやイベントは欠落する</li>
                    <li>7B モデルのため推論は速いが精度は限定的</li>
                  </ul>
                </div>
                <div>
                  <p className="mb-1 font-medium">mlx-vlm 72B — ネイティブ動画入力</p>
                  <ul className="space-y-1 opacity-70">
                    <li>動画ファイルを直接入力 — フレーム抽出は不要</li>
                    <li>
                      <strong>M-RoPE</strong> (時間・縦・横 3 軸の位置エンコーディング) で時間情報を正確に保持
                    </li>
                    <li>隣接フレーム間の空間トークン圧縮で短時間の動きも捉えやすい</li>
                    <li>72B モデルのため推論は遅いが理解精度が高い</li>
                  </ul>
                </div>
              </div>
            </details>
          )}
        </section>
      </form>
    </main>
  );
}
