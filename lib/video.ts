export type ExtractedFrame = {
  blob: Blob;
  timestampSec: number;
};

const MAX_FRAME_DIMENSION = 768;

export async function extractFramesFromVideo(
  file: File,
  count: number,
  onProgress?: (done: number, total: number) => void,
): Promise<ExtractedFrame[]> {
  if (count < 1) throw new Error('count must be >= 1');

  const url = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.preload = 'auto';
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = 'anonymous';
  video.src = url;

  try {
    await new Promise<void>((resolve, reject) => {
      const onMeta = () => {
        cleanup();
        if (!isFinite(video.duration) || video.duration <= 0) {
          reject(new Error('動画のメタデータを読み取れませんでした'));
        } else {
          resolve();
        }
      };
      const onErr = () => {
        cleanup();
        reject(new Error('動画ファイルをデコードできませんでした'));
      };
      const cleanup = () => {
        video.removeEventListener('loadedmetadata', onMeta);
        video.removeEventListener('error', onErr);
      };
      video.addEventListener('loadedmetadata', onMeta, { once: true });
      video.addEventListener('error', onErr, { once: true });
    });

    const srcW = video.videoWidth;
    const srcH = video.videoHeight;
    if (!srcW || !srcH) throw new Error('動画の解像度を取得できませんでした');

    const scale = Math.min(1, MAX_FRAME_DIMENSION / Math.max(srcW, srcH));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(srcW * scale);
    canvas.height = Math.round(srcH * scale);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context を取得できませんでした');

    const duration = video.duration;
    const frames: ExtractedFrame[] = [];

    for (let i = 0; i < count; i++) {
      const t = (duration * (i + 0.5)) / count;
      await seekTo(video, t);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob(
          (b) => (b ? resolve(b) : reject(new Error('フレーム JPEG 化に失敗'))),
          'image/jpeg',
          0.85,
        );
      });
      frames.push({ blob, timestampSec: t });
      onProgress?.(i + 1, count);
    }

    return frames;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function seekTo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const onSeeked = () => {
      cleanup();
      resolve();
    };
    const onErr = () => {
      cleanup();
      reject(new Error(`seek 失敗: ${time}s`));
    };
    const cleanup = () => {
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onErr);
    };
    video.addEventListener('seeked', onSeeked, { once: true });
    video.addEventListener('error', onErr, { once: true });
    video.currentTime = Math.max(0, Math.min(time, video.duration - 0.001));
  });
}
