"""Speak the command instead of typing it: Speechmatics -> planner.

This is the Speechmatics bonus track, and it is not bolted on for the sake of it
-- it is the natural interface for this task. Nobody sitting at a table types a
JSON task graph; they say "set the table and pour me some water". The transcript
goes straight into the same planner the typed path uses, so voice adds an input
modality without forking the reasoning.

Real-time transcription is used rather than batch so partial hypotheses can be
shown while the person is still speaking, which makes the demo video legible:
the viewer sees the words appear, then the plan, then the arms move.

Set SPEECHMATICS_API_KEY before use.
"""
from __future__ import annotations

import asyncio
import json
import os
import wave
from dataclasses import dataclass, field

RT_URL = os.environ.get("SPEECHMATICS_RT_URL", "wss://eu2.rt.speechmatics.com/v2")
CHUNK = 4096


class VoiceError(RuntimeError):
    pass


@dataclass
class Transcript:
    text: str = ""
    partials: list[str] = field(default_factory=list)
    language: str = "en"
    audio_seconds: float = 0.0
    latency_s: float = 0.0
    # Informational frames the server sent (quota, warnings). Kept rather than
    # dropped: they are what a session looks like from the outside, and the one
    # that arrives before RecognitionStarted is the one that broke this client.
    info: list = field(default_factory=list)


class SpeechmaticsClient:
    """Minimal real-time client over the v2 websocket protocol."""

    def __init__(self, api_key: str | None = None, language: str = "en",
                 sample_rate: int = 16000, max_delay: float = 1.0,
                 on_partial=None):
        self.api_key = api_key or os.environ.get("SPEECHMATICS_API_KEY")
        if not self.api_key:
            raise VoiceError(
                "SPEECHMATICS_API_KEY is not set. Get a key at portal.speechmatics.com "
                "and export it, or run with --instruction to type the command instead."
            )
        self.language = language
        self.sample_rate = sample_rate
        self.max_delay = max_delay
        self.on_partial = on_partial

    def _start_message(self) -> str:
        return json.dumps({
            "message": "StartRecognition",
            "audio_format": {"type": "raw", "encoding": "pcm_s16le",
                             "sample_rate": self.sample_rate},
            "transcription_config": {
                "language": self.language,
                "enable_partials": True,
                "max_delay": self.max_delay,
                "operating_point": "enhanced",
            },
        })

    async def _run(self, audio_chunks) -> Transcript:
        try:
            import websockets                                     # noqa: PLC0415
        except ImportError as exc:
            raise VoiceError("pip install websockets to use the voice interface") from exc

        import time
        tr = Transcript(language=self.language)
        t0 = time.perf_counter()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with websockets.connect(RT_URL, additional_headers=headers,
                                      max_size=None) as ws:
            await ws.send(self._start_message())
            # Wait for RecognitionStarted, skipping anything informational that
            # arrives ahead of it. The server sends an `Info` frame first --
            # `concurrent_session_usage`, reporting "1 of quota 2" -- and reading
            # exactly one message treated that as a refusal and killed the
            # session. It cost nothing to find here and would have cost the whole
            # voice demo live on stage, which is the kind of failure a first
            # message is guaranteed to produce and a unit test never will.
            while True:
                ack = json.loads(await ws.recv())
                kind = ack.get("message")
                if kind == "RecognitionStarted":
                    break
                if kind == "Info":
                    tr.info.append(ack)
                    continue
                if kind == "Warning":
                    tr.info.append(ack)
                    continue
                if kind == "Error":
                    raise VoiceError(f"Speechmatics refused the session: {ack}")
                raise VoiceError(
                    f"unexpected message while starting recognition: {ack}")

            async def send_audio():
                n = 0
                for chunk in audio_chunks:
                    await ws.send(chunk)
                    n += len(chunk)
                    await asyncio.sleep(0)
                tr.audio_seconds = n / (2 * self.sample_rate)
                await ws.send(json.dumps({"message": "EndOfStream",
                                          "last_seq_no": 0}))

            sender = asyncio.create_task(send_audio())
            try:
                while True:
                    msg = json.loads(await ws.recv())
                    kind = msg.get("message")
                    if kind == "AddPartialTranscript":
                        txt = msg["metadata"]["transcript"].strip()
                        if txt:
                            tr.partials.append(txt)
                            if self.on_partial:
                                self.on_partial(txt)
                    elif kind == "AddTranscript":
                        tr.text += msg["metadata"]["transcript"]
                    elif kind == "EndOfTranscript":
                        break
                    elif kind == "Error":
                        raise VoiceError(f"Speechmatics error: {msg}")
            finally:
                sender.cancel()
        tr.text = " ".join(tr.text.split())
        tr.latency_s = time.perf_counter() - t0
        return tr


def _wav_chunks(path: str, expect_rate: int):
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise VoiceError(f"{path} must be 16-bit mono PCM WAV")
        if wf.getframerate() != expect_rate:
            raise VoiceError(
                f"{path} is {wf.getframerate()} Hz; expected {expect_rate}. "
                f"Convert it: ffmpeg -i in.wav -ar {expect_rate} -ac 1 out.wav"
            )
        while True:
            data = wf.readframes(CHUNK)
            if not data:
                return
            yield data


def transcribe_file(path: str, *, language: str = "en", api_key: str | None = None,
                    on_partial=None) -> Transcript:
    """Transcribe a WAV file. Deterministic, so it is what the demo video uses."""
    client = SpeechmaticsClient(api_key=api_key, language=language, on_partial=on_partial)
    return asyncio.run(client._run(_wav_chunks(path, client.sample_rate)))


def transcribe_microphone(seconds: float = 6.0, *, language: str = "en",
                          api_key: str | None = None, on_partial=None) -> Transcript:
    """Record from the default input device and transcribe it live."""
    try:
        import sounddevice as sd                                   # noqa: PLC0415
    except ImportError as exc:
        raise VoiceError("pip install sounddevice to use the microphone") from exc

    client = SpeechmaticsClient(api_key=api_key, language=language, on_partial=on_partial)
    rate = client.sample_rate
    frames = int(seconds * rate)
    print(f"Listening for {seconds:.0f}s...", flush=True)
    audio = sd.rec(frames, samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    raw = audio.tobytes()
    chunks = [raw[i:i + CHUNK * 2] for i in range(0, len(raw), CHUNK * 2)]
    return asyncio.run(client._run(chunks))
