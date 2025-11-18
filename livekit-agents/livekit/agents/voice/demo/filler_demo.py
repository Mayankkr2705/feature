"""
Demo script showing filler-word filtering during agent speech.

This script creates a minimal AgentSession and AgentActivity, sets up
a FakeSTT that returns predetermined transcripts, and simulates VAD events
to demonstrate that filler words are ignored while real interruptions
trigger agent speech interruption.

Run from repo root:
    python -m livekit.agents.voice.demo.filler_demo
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add livekit-agents to sys.path for easier testing without editable install
repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
livekit_agents_path = repo_root / "livekit-agents"
if livekit_agents_path.exists() and str(livekit_agents_path) not in sys.path:
    sys.path.insert(0, str(livekit_agents_path))

import asyncio
from dataclasses import dataclass
from livekit import rtc
from livekit.agents import stt, vad
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.speech_handle import SpeechHandle


@dataclass
class FakeSTTResponse:
    alternatives: list[stt.SpeechData]


class FakeSTT(stt.STT):
    """Minimal STT that returns a predetermined transcript."""

    def __init__(self, transcript: str):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.transcript = transcript

    async def recognize(self, buffer, *, language: str | None = None):
        """Return a fake SpeechEvent with the predetermined transcript."""
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=self.transcript, language="en-US")],
        )


async def main():
    print("=== Filler Word Demo ===\n")

    # Create a minimal agent and session
    agent = Agent(instructions="You are a helpful assistant.")
    session = AgentSession(llm="gpt-4o-mini")
    
    # Set runtime filler words
    filler_words = ["um", "uh", "like", "you know"]
    session.update_filler_words(filler_words)
    print(f"Configured filler words: {filler_words}\n")

    # Create a minimal activity (normally done by session.start)
    activity = session._activity  # type: ignore
    if activity is None:
        from livekit.agents.voice.agent_activity import AgentActivity
        activity = AgentActivity(agent, session)
        session._activity = activity  # type: ignore

    # Simulate agent is currently speaking
    activity._current_speech = SpeechHandle.create(allow_interruptions=True)
    print("Agent is now speaking (simulated)...\n")

    # Test 1: Filler-only transcript should be ignored
    print("Test 1: Filler-only transcript ('um uh like')")
    activity.stt = FakeSTT("um uh like")
    vad_event = vad.VADEvent(
        type=vad.VADEventType.INFERENCE_DONE,
        samples=rtc.AudioFrame(data=b"", sample_rate=16000, num_channels=1, samples_per_channel=0),
        frames=[],
        duration=0.6,
        speech_duration=0.6,
        silence_duration=0.0,
        probability=0.9,
        inference_duration=0.0,
        speaking=True,
    )
    await activity._classify_vad_frames_and_decide(vad_event, filler_words)
    
    if activity._current_speech.interrupted:
        print("  ❌ FAILED: Agent was interrupted (should have been ignored)\n")
    else:
        print("  ✅ PASSED: Filler ignored, agent continues speaking\n")

    # Reset for test 2
    activity._current_speech = SpeechHandle.create(allow_interruptions=True)

    # Test 2: Interrupt keyword should trigger interruption
    print("Test 2: Interrupt keyword ('wait stop')")
    activity.stt = FakeSTT("wait stop")
    await activity._classify_vad_frames_and_decide(vad_event, filler_words)
    
    if activity._current_speech.interrupted:
        print("  ✅ PASSED: Agent was interrupted by keyword\n")
    else:
        print("  ❌ FAILED: Agent not interrupted (should have been)\n")

    # Reset for test 3
    activity._current_speech = SpeechHandle.create(allow_interruptions=True)

    # Test 3: Mixed content (filler + real words) should interrupt
    print("Test 3: Mixed content ('um I have a question')")
    activity.stt = FakeSTT("um I have a question")
    await activity._classify_vad_frames_and_decide(vad_event, filler_words)
    
    if activity._current_speech.interrupted:
        print("  ✅ PASSED: Agent was interrupted by real content\n")
    else:
        print("  ❌ FAILED: Agent not interrupted (should have been)\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
