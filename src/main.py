"""Trail Companion Phase 1: local microphone-to-speaker pipeline.

Microphone -> Whisper (base.en) -> local places -> Qwen3.5-4B -> Piper -> speaker.

All inference is local and offline (CPU-only). One interaction per run:

1. Press Enter, then ask your question (fixed recording duration).
2. Transcription -> nearest places -> grounded LLM answer -> spoken reply.
3. Stage timings are printed; then the program exits.

`--text "..."` skips the microphone (deterministic runs, headless use).
`--no-playback` synthesizes the reply WAV but does not play it.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.audio.play import PlaybackError, play_wav  # noqa: E402
from src.audio.record import AudioError, record_question  # noqa: E402
from src.llm.answer import LLMError, build_prompt, ensure_complete_answer, run_llm  # noqa: E402
from src.location.providers import (  # noqa: E402
    LocationError,
    check_fresh,
    gpsd_location,
    manual_location,
)
from src.places.geo import GeoError, nearest_in_pack  # noqa: E402
from src.places.nearby import load_places, nearest_places  # noqa: E402
from src.places.pack import load_pack  # noqa: E402
from src.stt.transcribe import STTError, is_empty_transcription, transcribe  # noqa: E402
from src.tts.speak import TTSError, synthesize  # noqa: E402

TARGET_SECONDS = 10.0  # AGENTS.md latency target (upper bound)
NO_PLACES_EXIT = 4  # coordinates unusable or no nearby places: honest message, no LLM call


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traily Phase 2 local voice pipeline.")
    parser.add_argument("--text", default=None, help="question text (skips microphone)")
    parser.add_argument("--record-seconds", type=float, default=config.RECORD_SECONDS)
    parser.add_argument("--no-playback", action="store_true", help="do not play reply audio")
    parser.add_argument("--workdir", default=None, help="artifact dir (default: temp dir)")
    parser.add_argument("--pack", default=config.PACK_FILE, help="offline trail pack (data/packs/*.json); default: Phase 1 fixture")
    parser.add_argument("--lat", type=float, default=float(config.DEF_LAT) if config.DEF_LAT else None, help="manual TEST latitude (labelled, never live)")
    parser.add_argument("--lon", type=float, default=float(config.DEF_LON) if config.DEF_LON else None, help="manual TEST longitude (labelled, never live)")
    parser.add_argument(
        "--location-source",
        choices=("manual", "gpsd"),
        default=config.LOCATION_SOURCE,
        help="location source for --pack mode (default: manual test coords)",
    )
    parser.add_argument("--max-age", type=float, default=config.MAX_LOCATION_AGE_S, help="max location age in seconds")
    return parser.parse_args(argv)


def resolve_places(args: argparse.Namespace) -> tuple[list[dict], str]:
    """Return (places, origin_label) for the LLM prompt.

    Pack mode: coordinates -> local pack -> haversine -> nearest 10.
    Default: Phase 1 fictional fixture (backward compatible).
    """
    if args.pack is None:
        return nearest_places(load_places(), config.MAX_PLACES), "local fixture data"
    try:
        pack = load_pack(args.pack)
    except ValueError as exc:
        print(f"Place data unavailable: {exc}")
        raise SystemExit(2) from exc
    try:
        if args.location_source == "gpsd":
            loc = check_fresh(gpsd_location(), args.max_age)
            origin = "live GPSD fix"
        else:
            if args.lat is None or args.lon is None:
                print("Pack mode needs coordinates: --lat/--lon (test) or --location-source gpsd.")
                raise SystemExit(2)
            loc = manual_location(args.lat, args.lon)
            origin = f"MANUAL test coordinates ({loc.lat}, {loc.lon})"
    except LocationError as exc:
        print(f"Location unavailable: {exc}")
        raise SystemExit(2) from exc
    try:
        places = nearest_in_pack(pack, loc.lat, loc.lon, config.MAX_PLACES)
    except GeoError as exc:
        print(f"No nearby places: {exc}")
        raise SystemExit(NO_PLACES_EXIT) from exc
    if not places:
        print(f"No suitable places near ({loc.lat}, {loc.lon}) in pack '{pack['area_name']}'.")
        raise SystemExit(NO_PLACES_EXIT)
    acc = f", accuracy ~{loc.accuracy_m:.0f} m" if loc.accuracy_m is not None else ""
    print(f"Location: ({loc.lat}, {loc.lon}) via {loc.source}{acc}.")
    return places, f"offline pack '{pack['area_name']}'"


def run_pipeline(
    question: str | None,
    record_seconds: float,
    playback: bool,
    workdir: Path,
    args: argparse.Namespace | None = None,
) -> int:
    # Distinct time.perf_counter() timestamps for every stage boundary.
    # Post-recording latency is measured from RECORDING FINISH (not true
    # end-of-speech: fixed-duration recording has no VAD) to playback start.
    t_pipeline_start = time.perf_counter()
    ts: dict[str, float] = {"pipeline_start": t_pipeline_start}
    workdir.mkdir(parents=True, exist_ok=True)
    question_wav = workdir / "question.wav"
    reply_wav = workdir / "reply.wav"

    # 1-2. Record (or use --text).
    if question is None:
        input("Press Enter, then ask your question after the beep... (recording %.0fs)" % record_seconds)
        print(f"Recording {record_seconds:.0f}s...", flush=True)
        ts["record_start"] = time.perf_counter()
        try:
            record_question(question_wav, seconds=record_seconds)
        except AudioError as exc:
            print(f"Recording failed: {exc}")
            return 2
        ts["record_end"] = time.perf_counter()
    else:
        ts["record_start"] = ts["record_end"] = time.perf_counter()
        print(f"Text mode (mic skipped): {question}")

    # 3. Transcribe (skipped in --text mode).
    if question is None:
        ts["stt_start"] = time.perf_counter()
        try:
            transcript, _ = transcribe(question_wav)
        except STTError as exc:
            print(f"Speech-to-text failed: {exc}")
            return 2
        ts["stt_end"] = time.perf_counter()
        print(f'Heard: "{transcript}"')
        if is_empty_transcription(transcript):
            print("I didn't catch that — no speech recognized. Please try again.")
            return 3
        question = transcript
    else:
        ts["stt_start"] = ts["stt_end"] = time.perf_counter()

    # 4. Places (validated, sorted, limited — never computed by the LLM).
    if args is None:
        args = parse_args([])
    try:
        places, origin = resolve_places(args)
    except SystemExit as exc:
        return int(exc.code or 0)
    except ValueError as exc:
        print(f"Place data unavailable: {exc}")
        return 2
    print(f"Using {len(places)} nearby place(s) from {origin}.")

    # 5. LLM (reasoning always off inside run_llm; see src/llm/answer.py).
    prompt = build_prompt(question, places)
    ts["llm_start"] = time.perf_counter()
    try:
        answer, _ = run_llm(prompt)
    except LLMError as exc:
        print(f"Answer generation failed: {exc}")
        return 2
    ts["llm_end"] = time.perf_counter()
    answer, used_fallback = ensure_complete_answer(answer, places)
    if used_fallback:
        print("Note: LLM omitted a place; using deterministic complete summary.")
    print(f'Traily: "{answer}"')

    # 6. TTS synthesis, then playback (timed separately).
    ts["tts_start"] = time.perf_counter()
    try:
        synthesize(answer, reply_wav)
    except TTSError as exc:
        print(f"Speech synthesis failed: {exc}")
        return 2
    ts["tts_end"] = time.perf_counter()
    ts["playback_start"] = time.perf_counter()
    if playback:
        try:
            play_wav(reply_wav)
            played = True
        except PlaybackError as exc:
            print(f"Warning: playback failed ({exc}); reply saved at {reply_wav}")
            played = False
    else:
        print(f"Playback skipped; reply saved at {reply_wav}")
        played = False
    ts["playback_end"] = time.perf_counter()

    # 7. Timing summary (AGENTS.md: separate stages, honest about limits).
    record_s = ts["record_end"] - ts["record_start"]
    stt_s = ts["stt_end"] - ts["stt_start"]
    llm_s = ts["llm_end"] - ts["llm_start"]
    tts_s = ts["tts_end"] - ts["tts_start"]
    post_recording_s = ts["playback_start"] - ts["record_end"]
    total_s = ts["playback_start"] - ts["pipeline_start"]
    print("\n--- timings (seconds, time.perf_counter) ---")
    print(f"record (mic):                        {record_s:8.2f}")
    print(f"STT (whisper base.en):               {stt_s:8.2f}")
    print("LLM load:                            (included below; llama-cli cold-starts per run)")
    print(f"LLM elapsed, incl. CLI startup:      {llm_s:8.2f} (Qwen3.5-4B, reasoning off)")
    print("LLM first-token:                     N/A (not exposed by llama-cli; not estimated)")
    print(f"TTS synthesis (piper):               {tts_s:8.2f}")
    print(f"playback started:                    {'yes' if played else 'no (skipped/failed)'}")
    print(f"post-recording latency (record-finish -> playback-start): {post_recording_s:8.2f}")
    print("  (fixed-duration recording without VAD: this is NOT true end-of-speech latency)")
    print(f"total interaction (pipeline-start -> playback-start):     {total_s:8.2f} (target <= ~{TARGET_SECONDS:.0f}s post-recording)")
    if post_recording_s <= TARGET_SECONDS:
        print("Latency target met for this run.")
    else:
        print("Latency target NOT met for this run (see notes/failures.md).")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workdir:
        workdir = Path(args.workdir)
        return run_pipeline(args.text, args.record_seconds, not args.no_playback, workdir, args)
    with tempfile.TemporaryDirectory(prefix="traily-") as tmp:
        return run_pipeline(args.text, args.record_seconds, not args.no_playback, Path(tmp), args)


if __name__ == "__main__":
    raise SystemExit(main())
