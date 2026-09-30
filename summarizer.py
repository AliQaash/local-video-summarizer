import time
import glob
import json
import os
import re
import sys
import textwrap

import ollama
from faster_whisper import WhisperModel

# Force UTF-8 output regardless of whether stdout is an interactive console or
# a redirected pipe/file — Windows otherwise falls back to a legacy codepage
# (cp1252) on redirect, which crashes on the emoji used throughout this script.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────
# CONFIG  ← edit these if needed
# ─────────────────────────────────────────────
# large-v3 via faster-whisper (CTranslate2): same weights and accuracy as plain
# Whisper's large-v3, but quantized inference brings it to ~4.7GB at float16,
# where plain openai-whisper's large-v3 (~10GB) will not fit a 6GB card at all.
WHISPER_MODEL  = "large-v3"
COMPUTE_TYPE   = "float16"    # auto-falls back to int8_float16 on OOM, and to CPU int8 with no CUDA
# gemma2:9b, not llama3 — on UrduMMLU (a native-Urdu academic benchmark), it's
# the strongest open model that actually fits a 6GB card, ahead of Qwen3-8B.
# Urdu-specific fine-tuned models score WORSE on broad Urdu comprehension than
# strong general models, so a niche "Urdu model" would be a downgrade, not a fix.
OLLAMA_MODEL   = "gemma2:9b"
VIDEO_LANGUAGE = "ur"         # "ur" = Urdu  |  None = auto-detect

# Three-pass chaptering, so chapters land on real topic changes instead of a
# fixed clock interval:
#   Pass 1 — scan the transcript in large windows, ask the LLM to mark real
#            topic-change boundaries (not every sentence).
#   Pass 2 — for each resulting chapter, summarize using only that chapter's
#            actual transcript text.
#   Pass 3 — one final pass over the whole chapter list to merge adjacent
#            near-duplicate chapters, tighten titles, and write the overview.
BOUNDARY_WINDOW_MINUTES = 15  # window size for pass 1 (bigger = more context per boundary judgment)

# Any stretch longer than this with no transcribed speech gets flagged in the
# console and marked inline in the transcript file, so silent drops can't hide.
GAP_REPORT_SECONDS = 3.0

# Where Whisper model weights are cached. Defaults to the standard Hugging Face
# cache location, which works on any machine straight after a clone. Set the
# WHISPER_CACHE_DIR environment variable to keep models somewhere else:
#   PowerShell:  $env:WHISPER_CACHE_DIR = "D:\models\whisper"
#   bash:        export WHISPER_CACHE_DIR=/mnt/models/whisper
WHISPER_CACHE_DIR = os.environ.get("WHISPER_CACHE_DIR") or None
# ─────────────────────────────────────────────


def find_local_video():
    # Audio formats included so voice notes work too. WhatsApp exports .opus on
    # Android and .m4a on iOS; ffmpeg decodes all of these, so Whisper handles
    # them exactly like a video's audio track.
    patterns = ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.flv",
                "*.opus", "*.ogg", "*.m4a", "*.mp3", "*.wav", "*.aac", "*.wma"]
    for ext in patterns:
        hits = glob.glob(ext)
        if hits:
            return hits[0]
    return None


def format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_ts_to_seconds(ts: str):
    parts = [int(p) for p in ts.strip().split(":")]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    return None


def detect_device():
    """Return (device, compute_type) for whatever this machine can actually run.

    Loads the tiny model as a throwaway probe. faster-whisper runs on
    CTranslate2 rather than torch, so a torch.cuda.is_available() check can
    report True and still fail here when cuDNN/cuBLAS is missing. Actually
    constructing a model is the only honest test, and tiny costs ~75MB once.
    """
    try:
        probe = WhisperModel("tiny", device="cuda", compute_type=COMPUTE_TYPE,
                             download_root=WHISPER_CACHE_DIR)
        del probe
        return "cuda", COMPUTE_TYPE
    except Exception:
        print("ℹ️   No usable CUDA device. Falling back to CPU with int8.")
        print("    Expect this to be several times slower; large-v3 on CPU is slow but accurate.")
        return "cpu", "int8"


def load_whisper_model(model_size, device, compute_type, download_root):
    """Load faster-whisper, auto-downgrading precision once on VRAM failure
    instead of crashing the whole run."""
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=download_root)
        return model, compute_type
    except Exception as e:
        if device == "cuda" and compute_type != "int8_float16":
            print(f"⚠️  Load failed with compute_type={compute_type} ({e})")
            print("    Retrying with int8_float16 (lower VRAM footprint)...")
            model = WhisperModel(model_size, device=device, compute_type="int8_float16", download_root=download_root)
            return model, "int8_float16"
        raise


def snap_to_nearest_segment(ts_seconds, segments):
    """LLMs sometimes slightly mangle a copied timestamp. Snap whatever it
    returns to the nearest real segment start so chapter slicing stays exact."""
    return min(segments, key=lambda s: abs(s.start - ts_seconds)).start


def ollama_chat_with_retry(model, prompt, retries=2, delay_seconds=3):
    """One retry on transient failures (e.g. a CUDA shared-object crash in
    Ollama's llama-server right after a GPU handoff) instead of silently
    losing that chunk's content, as happened on the previous run."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            return response["message"]["content"]
        except Exception as e:
            last_error = e
            if attempt < retries:
                print(f"      ⚠️  Ollama call failed ({e}); retrying in {delay_seconds}s ({attempt+1}/{retries})…")
                time.sleep(delay_seconds)
    raise last_error


def detect_chapter_boundaries(window_text, model, prior_title):
    """Pass 1: ask the LLM to mark genuine topic changes within one window,
    not every sentence."""
    continuity_note = (
        f'This window may continue the previous topic, "{prior_title}". '
        "Don't mark a boundary right at the start unless the topic has actually changed."
        if prior_title else
        "This is the start of the video."
    )
    prompt = textwrap.dedent(f"""
        You are finding TOPIC boundaries in a video transcript — points where the
        speaker moves to a genuinely different subject. Not every sentence, only
        real topic shifts. Aim for natural chapter lengths (roughly 1-4 minutes),
        not one boundary per sentence.

        {continuity_note}

        RULES:
        1. Use ONLY timestamps that appear verbatim in the transcript below.
        2. One boundary per line: [MM:SS] Short Topic Title
        3. No other text, no numbering, no headers.

        TRANSCRIPT WINDOW:
        {window_text}

        BOUNDARIES:
    """).strip()

    text = ollama_chat_with_retry(model, prompt)

    boundaries = []
    for line in text.splitlines():
        match = re.match(r"\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+)", line)
        if match:
            secs = parse_ts_to_seconds(match.group(1))
            if secs is not None:
                boundaries.append((secs, match.group(2).strip()))
    return boundaries


def summarize_chapter(chapter_text, title_hint, model):
    """Pass 2: polished title + one-sentence description from the chapter's
    actual transcript content (not just the provisional title from pass 1)."""
    prompt = textwrap.dedent(f"""
        Summarize this section of a video transcript as one chapter.
        Provisional working title (you may keep, improve, or replace it): "{title_hint}"

        Respond in EXACTLY this format, nothing else:
        TITLE: <short chapter title, a few words>
        DESCRIPTION: <one sentence describing what's covered>

        Respond in English regardless of the transcript's language. Do not translate
        the transcript itself — just describe the topic.

        TRANSCRIPT:
        {chapter_text}
    """).strip()

    text = ollama_chat_with_retry(model, prompt)

    title_match = re.search(r"TITLE:\s*(.+)", text)
    desc_match = re.search(r"DESCRIPTION:\s*(.+)", text)
    title = title_match.group(1).strip() if title_match else title_hint
    desc = desc_match.group(1).strip() if desc_match else text.strip()
    return title, desc


def consolidate_and_overview(chapters, model):
    """Pass 3: merge adjacent near-duplicate chapters, tighten titles, and
    write a short overview paragraph — one combined call over the whole list."""
    chapter_block = "\n".join(
        f"[{format_ts(c['start'])}] {c['title']} — {c['desc']}" for c in chapters
    )
    prompt = textwrap.dedent(f"""
        Below is a draft chapter list for a video, generated section by section
        (so adjacent chapters may sometimes be the same topic split in two).

        TASKS:
        1. Merge any adjacent chapters that are really the same topic — keep the
           earlier timestamp, write one clean title and description.
        2. Tighten titles so they're consistent and concise.
        3. Write a 2-4 sentence OVERVIEW of the whole video.

        Respond in EXACTLY this format:
        OVERVIEW:
        <paragraph>

        CHAPTERS:
        [MM:SS] Title — Description
        [MM:SS] Title — Description
        ...

        DRAFT CHAPTERS:
        {chapter_block}
    """).strip()

    text = ollama_chat_with_retry(model, prompt)

    if "CHAPTERS:" in text:
        overview_part, chapters_part = text.split("CHAPTERS:", 1)
        overview = overview_part.replace("OVERVIEW:", "").strip()
        chapters_text = chapters_part.strip()
    else:
        # Model didn't follow the format — fall back to the draft list untouched.
        overview = ""
        chapters_text = chapter_block

    return overview, chapters_text


def pick_available_model(preferred: str) -> str:
    try:
        available = [m["name"] for m in ollama.list()["models"]]
        if any(preferred in m for m in available):
            return preferred
        for fallback in ["gemma2:9b", "llama3.1:8b", "llama3:latest", "llama3.2:3b", "mistral", "llama3.2:1b"]:
            if any(fallback in m for m in available):
                print(f"⚠️  '{preferred}' not found — using '{fallback}' instead.")
                return fallback
        if available:
            chosen = available[0]
            print(f"⚠️  Using first available model: {chosen}")
            return chosen
    except Exception:
        pass
    return preferred


class CachedSegment:
    """Stand-in for a faster-whisper Segment, rebuilt from the JSON cache.

    Downstream code only ever reads .start, .end and .text, so this is all the
    surface a cached run needs.
    """
    __slots__ = ("start", "end", "text")

    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


def transcript_cache_path(video_path):
    return os.path.splitext(video_path)[0] + "_transcript_cache.json"


def cache_fingerprint(video_path):
    """Identifies the exact inputs a cached transcript was produced from.

    Any change to the media file or the transcription settings invalidates the
    cache, so a stale transcript can never be silently reused.
    """
    stat = os.stat(video_path)
    return {
        "file": os.path.basename(video_path),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "model": WHISPER_MODEL,
        "language": VIDEO_LANGUAGE,
    }


def load_cached_transcript(video_path):
    """Return (segments, duration) from cache, or None when unusable."""
    path = transcript_cache_path(video_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️   Transcript cache unreadable ({e}); re-transcribing.")
        return None

    if data.get("fingerprint") != cache_fingerprint(video_path):
        print("ℹ️   Transcript cache is stale (file or settings changed); re-transcribing.")
        return None

    segments = [CachedSegment(s["start"], s["end"], s["text"]) for s in data["segments"]]
    return segments, data["duration"]


def save_cached_transcript(video_path, segments, duration):
    path = transcript_cache_path(video_path)
    payload = {
        "fingerprint": cache_fingerprint(video_path),
        "duration": duration,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️   Could not write transcript cache: {e}")


def transcribe_or_load(video_path):
    """Transcription is by far the most expensive stage, and the LLM stage that
    follows it fails for unrelated reasons (Ollama not running, a model not
    pulled). Caching means those failures cost seconds to retry instead of
    re-running the whole transcription."""
    cached = load_cached_transcript(video_path)
    if cached:
        segments, duration = cached
        print(f"📂  Reusing cached transcript ({len(segments)} segments). "
              f"Delete {os.path.basename(transcript_cache_path(video_path))} to force a fresh run.")
        return segments, duration

    # CUDA when it's actually usable, CPU otherwise. Probing with a throwaway
    # tiny model is more honest than trusting torch.cuda.is_available(), since
    # faster-whisper runs on CTranslate2, not torch: CUDA can look present to
    # torch and still fail here on a missing cuDNN/cuBLAS.
    device, compute_type = detect_device()
    print(f"⏳  Loading faster-whisper ({WHISPER_MODEL}, {compute_type}) on {device.upper()} …")
    try:
        model, used_compute_type = load_whisper_model(WHISPER_MODEL, device, compute_type, WHISPER_CACHE_DIR)
    except Exception as e:
        print(f"❌  Whisper load failed even after fallback: {e}")
        return None
    if used_compute_type != compute_type:
        print(f"    (running at {used_compute_type} instead of {compute_type})")

    print("🎧  Transcribing …")
    t0 = time.time()
    try:
        segments_gen, info = model.transcribe(
            video_path,
            language=VIDEO_LANGUAGE,
            beam_size=5,
            # VAD OFF. Silero VAD misclassifies reverberant, quiet, or
            # music-backed speech as non-speech, and anything it marks is never
            # transcribed at all. Off means Whisper walks the entire audio in
            # 30s windows, so no region can be skipped before it is even seen.
            vad_filter=False,
            # None (not just a loose number) fully DISABLES both segment-drop
            # mechanisms. Whisper normally discards a segment when it looks like
            # non-speech or when model confidence is low; on Urdu religious
            # speech, confidence is legitimately low and real content gets
            # thrown away. With both set to None the checks never run, so every
            # decoded segment is kept.
            no_speech_threshold=None,
            log_prob_threshold=None,
            # Each window stands alone, so one bad patch cannot derail the rest
            # through a repetition loop.
            condition_on_previous_text=False,
            # Word-level timing. Tightens segment boundaries and lets the
            # coverage check below detect real gaps precisely.
            word_timestamps=True,
        )
        segments = list(segments_gen)
    except Exception as e:
        print(f"❌  Transcription error: {e}")
        return None

    print(f"✅  Done in {round(time.time()-t0, 1)}s  |  {len(segments)} segments")

    del model  # release faster-whisper's VRAM before Ollama needs the GPU

    duration = getattr(info, "duration", None) or (segments[-1].end if segments else 0)
    save_cached_transcript(video_path, segments, duration)
    return segments, duration


def process_video():
    video_path = find_local_video()
    if not video_path:
        print("❌ No video file found in this folder.")
        return
    print(f"🎬  {video_path}")

    result = transcribe_or_load(video_path)
    if not result:
        return
    segments, audio_duration = result
    if not segments:
        print("❌  No speech segments produced.")
        return

    # ---- Coverage check ----
    # The decoded audio duration is the ground truth to measure against.
    # Anything not covered by a segment is either real silence or dropped
    # audio; both are shown so you can verify rather than trust.
    covered = sum(s.end - s.start for s in segments)
    coverage_pct = (covered / audio_duration * 100) if audio_duration else 0

    gaps = []
    cursor = 0.0
    for s in segments:
        if s.start - cursor > GAP_REPORT_SECONDS:
            gaps.append((cursor, s.start))
        cursor = max(cursor, s.end)
    if audio_duration - cursor > GAP_REPORT_SECONDS:
        gaps.append((cursor, audio_duration))

    print(f"📊  Coverage: {coverage_pct:.1f}% of {format_ts(audio_duration)} "
          f"({format_ts(covered)} transcribed)")
    if gaps:
        print(f"    {len(gaps)} gap(s) over {GAP_REPORT_SECONDS}s — listen to these to confirm they're silence:")
        for g_start, g_end in gaps:
            print(f"      {format_ts(g_start)} → {format_ts(g_end)}  ({round(g_end - g_start, 1)}s)")
    else:
        print(f"    No gaps over {GAP_REPORT_SECONDS}s. Transcript is continuous.")

    # Gap markers go into the transcript file too, so nothing is invisible.
    raw_lines = []
    cursor = 0.0
    for s in segments:
        if s.start - cursor > GAP_REPORT_SECONDS:
            raw_lines.append(f"[{format_ts(cursor)}] <<< NO SPEECH DETECTED for {round(s.start - cursor, 1)}s >>>")
        raw_lines.append(f"[{format_ts(s.start)}] {s.text.strip()}")
        cursor = max(cursor, s.end)
    if audio_duration - cursor > GAP_REPORT_SECONDS:
        raw_lines.append(f"[{format_ts(cursor)}] <<< NO SPEECH DETECTED for {round(audio_duration - cursor, 1)}s (to end) >>>")

    raw_text = "\n".join(raw_lines)
    transcript_path = os.path.splitext(video_path)[0] + "_transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"📄  Raw transcript → {transcript_path}")

    llm = pick_available_model(OLLAMA_MODEL)

    # ---- Pass 1: boundary detection over large windows ----
    print(f"🔎  Pass 1/3 — detecting chapter boundaries with {llm} …")
    all_boundaries = []
    prior_title = None
    window_start = 0.0
    window = []
    windows = []
    for s in segments:
        if s.start - window_start >= BOUNDARY_WINDOW_MINUTES * 60 and window:
            windows.append(window)
            window = []
            window_start = s.start
        window.append(s)
    if window:
        windows.append(window)

    for wi, win in enumerate(windows):
        win_text = "\n".join(f"[{format_ts(s.start)}] {s.text.strip()}" for s in win)
        print(f"   window {wi+1}/{len(windows)}")
        try:
            boundaries = detect_chapter_boundaries(win_text, llm, prior_title)
        except Exception as e:
            print(f"   ⚠️  window {wi+1} boundary detection failed: {e}")
            boundaries = []
        if boundaries:
            all_boundaries.extend(boundaries)
            prior_title = boundaries[-1][1]

    if not all_boundaries or all_boundaries[0][0] > 5:
        all_boundaries.insert(0, (0.0, "Introduction"))

    all_boundaries.sort(key=lambda b: b[0])
    deduped = []
    for secs, title in all_boundaries:
        snapped = snap_to_nearest_segment(secs, segments)
        if deduped and snapped - deduped[-1][0] < 15:
            continue  # too close to the previous boundary — treat as noise
        deduped.append((snapped, title))

    # ---- Pass 2: summarize each chapter from its actual transcript text ----
    print(f"✍️   Pass 2/3 — summarizing {len(deduped)} chapters …")
    chapters = []
    for i, (start, title_hint) in enumerate(deduped):
        end = deduped[i + 1][0] if i + 1 < len(deduped) else segments[-1].end
        chapter_segments = [s for s in segments if start <= s.start < end]
        chapter_text = "\n".join(f"[{format_ts(s.start)}] {s.text.strip()}" for s in chapter_segments)
        if not chapter_text.strip():
            continue
        try:
            title, desc = summarize_chapter(chapter_text, title_hint, llm)
        except Exception as e:
            print(f"   ⚠️  chapter {i+1} summarization failed: {e}")
            title, desc = title_hint, "[summary failed]"
        chapters.append({"start": start, "title": title, "desc": desc})
        print(f"   chapter {i+1}/{len(deduped)}: [{format_ts(start)}] {title}")

    # ---- Pass 3: consolidate + overview ----
    print("🧩  Pass 3/3 — consolidating chapters and writing overview …")
    try:
        overview, chapters_text = consolidate_and_overview(chapters, llm)
    except Exception as e:
        print(f"⚠️  Consolidation failed: {e}")
        overview = ""
        chapters_text = "\n".join(f"[{format_ts(c['start'])}] {c['title']} — {c['desc']}" for c in chapters)

    video_title = os.path.splitext(os.path.basename(video_path))[0]
    total_duration = format_ts(segments[-1].end) if segments else "?"

    parts = [f"📹  {video_title}", f"⏱   Duration: {total_duration}", "─" * 60]
    if overview:
        parts += ["📝  OVERVIEW", overview, "─" * 60]
    parts += ["📌  KEY TOPICS & TIMESTAMPS", "─" * 60, chapters_text]
    full_summary = "\n\n".join(parts)

    print("\n" + "═" * 60)
    print(full_summary)
    print("═" * 60)

    summary_path = os.path.splitext(video_path)[0] + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(full_summary)
    print(f"\n💾  Summary saved → {summary_path}")


if __name__ == "__main__":
    process_video()
