import json
import os
import time
from faster_whisper import WhisperModel
import ollama

# --- Transcription settings ---
# MX450 only has 2GB VRAM, which is why "base"/"small" were the ceiling on GPU.
# faster-whisper's CPU + int8 path lets us run a much larger, more accurate
# model instead — trading some speed for real accuracy on hard audio.
WHISPER_MODEL_SIZE = "medium"   # try "large-v3" if quality still isn't enough
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"   # keeps CPU memory/time reasonable
LANGUAGE = "ur"

SUMMARY_MODEL = "llama3.1:8b"   # big step up from 1b — run: ollama pull llama3.1:8b
                                  # (will be noticeably slower per chunk on CPU — expected)
CHUNK_SECONDS = 180
TRANSCRIPT_CACHE_FILE = "transcript_cache.json"


def transcribe(video_path):
    # Reuse a cached transcript if one exists — transcription is the slow,
    # expensive step (minutes on CPU), and re-running it every time we're
    # just debugging the summarization step wastes real time.
    if os.path.exists(TRANSCRIPT_CACHE_FILE):
        print(f"📂 Found cached transcript at {TRANSCRIPT_CACHE_FILE} — reusing it instead of re-transcribing.")
        print("   (Delete this file if you want to force a fresh transcription.)")
        with open(TRANSCRIPT_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"⏳ Loading faster-whisper '{WHISPER_MODEL_SIZE}' on CPU (int8)...")
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )

    print(f"🎧 Transcribing audio from: {video_path} (this will be slower than 'base' — that's expected)")
    start_time = time.time()

    segments_iter, info = model.transcribe(
        video_path,
        language=LANGUAGE,        # tells Whisper the source audio is Urdu
        task="translate",         # KEY CHANGE: output English text, not Urdu
        condition_on_previous_text=False,
    )

    segments = []
    for seg in segments_iter:
        segments.append({"start": seg.start, "text": seg.text})

    print(f"✅ Transcription complete in {round(time.time() - start_time, 2)} seconds.")
    print(f"   Detected/forced language: {info.language} (probability {round(info.language_probability, 2)})")

    with open(TRANSCRIPT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"💾 Cached transcript to {TRANSCRIPT_CACHE_FILE} for reuse.")

    return segments


def group_into_chunks(segments, chunk_seconds=CHUNK_SECONDS):
    chunks = []
    current = {"start": segments[0]["start"], "text": ""}

    for seg in segments:
        if seg["start"] - current["start"] > chunk_seconds and current["text"]:
            chunks.append(current)
            current = {"start": seg["start"], "text": ""}
        current["text"] += seg["text"]

    if current["text"]:
        chunks.append(current)

    return chunks


def summarize_chunk(chunk_text, start_label):
    prompt = f"""Summarize ONLY what is explicitly said in this transcript segment.
Do not add, infer, or guess any information that is not directly stated.
Do not add parenthetical clarifications, identifications, or interpretations
of names or terms that are not explicitly explained in the text itself
(e.g. do not guess who a name or title refers to).
Do not include any meta-commentary about your own response, accuracy, or
process — output ONLY the summary text itself.
If the segment is unclear, garbled, or too short to summarize meaningfully,
say so explicitly instead of inventing content.

Transcript segment (starting at {start_label}):
{chunk_text}

Summary (2-4 sentences max):"""

    response = ollama.chat(model=SUMMARY_MODEL, messages=[
        {"role": "user", "content": prompt}
    ])
    return response["message"]["content"].strip()


def synthesize_final_summary(partial_summaries, video_title=""):
    combined = "\n".join(partial_summaries)
    prompt = f"""You are given a set of timestamped partial summaries from
different segments of the same video, generated independently. Combine them
into ONE coherent, well-structured summary of the video as a whole.

Rules:
- Use ONLY information present in the partial summaries below — do not add
  outside knowledge, even if you recognize the topic or speaker.
- Organize it as a short intro sentence followed by 3-6 bullet points
  covering the main themes, each keeping its approximate timestamp.
- Do not include meta-commentary about your own process — output only the
  final summary.

Partial summaries:
{combined}

Final structured summary:"""

    response = ollama.chat(model=SUMMARY_MODEL, messages=[
        {"role": "user", "content": prompt}
    ])
    return response["message"]["content"].strip()


def process_video(video_path):
    segments = transcribe(video_path)

    # Print the raw transcript first — check THIS looks coherent before
    # trusting the summary. If this is still garbled, the fix is a bigger
    # model or better audio, not the summarization step.
    print("\n📝 ENGLISH TRANSLATION (sanity check before summarizing):\n")
    full_text = " ".join(seg["text"] for seg in segments)
    print(full_text[:1000] + ("..." if len(full_text) > 1000 else ""))
    print()

    print("✂️  Splitting transcript into chunks before summarization...")
    chunks = group_into_chunks(segments)

    print(f"🤖 Summarizing {len(chunks)} chunk(s) with {SUMMARY_MODEL}...")
    partial_summaries = []
    for chunk in chunks:
        start_label = time.strftime("%M:%S", time.gmtime(chunk["start"]))
        summary = summarize_chunk(chunk["text"], start_label)
        partial_summaries.append(f"[{start_label}] {summary}")

    print("\n" + "=" * 50)
    print("📊 PER-SEGMENT SUMMARIES (raw, before synthesis)")
    print("=" * 50 + "\n")
    for line in partial_summaries:
        print(line + "\n")

    print("🧵 Synthesizing one coherent final summary from the segments above...")
    final_summary = synthesize_final_summary(partial_summaries)

    print("\n" + "=" * 50)
    print("📊 FINAL VIDEO SUMMARY")
    print("=" * 50 + "\n")
    print(final_summary)


if __name__ == "__main__":
    local_video_file = "my_video.mp4"
    process_video(local_video_file)
