r"""Demo script showing filler-word filtering during agent speech.

This script directly tests the _classify_vad_frames_and_decide method
with mocked STT responses to demonstrate that filler words are ignored
while real interruptions are detected.

Run from repo root:
    python livekit-agents\livekit\agents\voice\demo\filler_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add the parent of 'livekit' package to sys.path for easier testing without editable install
# The structure is: livekit-agents/livekit/agents/... so we need livekit-agents in the path
demo_file = Path(__file__).resolve()
livekit_package_parent = demo_file.parent.parent.parent.parent  # Goes up to livekit-agents/
if livekit_package_parent.exists() and str(livekit_package_parent) not in sys.path:
    sys.path.insert(0, str(livekit_package_parent))

import asyncio
from unittest.mock import MagicMock
from livekit import rtc
from livekit.agents import stt, vad
from livekit.agents.voice.speech_handle import SpeechHandle


class FakeSTT(stt.STT):
    """Minimal STT that returns a predetermined transcript."""

    def __init__(self, transcript: str):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.transcript = transcript

    async def _recognize_impl(self, buffer, *, language: str | None = None, conn_options=None):
        """Return a fake SpeechEvent with the predetermined transcript."""
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=self.transcript, language="en-US")],
        )
    
    def stream(self, *, language: str | None = None):
        """Not needed for this demo."""
        raise NotImplementedError("Streaming not supported in demo")


class MockActivity:
    """Minimal mock of AgentActivity for testing the classifier."""
    
    def __init__(self):
        self.stt = None
        self._current_speech = None
        self._interrupt_by_audio_activity = MagicMock()
    
    # Import the actual method to test
    from livekit.agents.voice.agent_activity import AgentActivity
    _classify_vad_frames_and_decide = AgentActivity._classify_vad_frames_and_decide


async def main():
    print("=== Filler Word Demo ===\n")

    filler_words = ["um", "uh", "like", "you know"]
    print(f"Configured filler words: {filler_words}\n")

    # Create a mock activity
    activity = MockActivity()
    activity._current_speech = SpeechHandle.create(allow_interruptions=True)
    print("Agent is now speaking (simulated)...\n")

    # Test 1: Filler-only transcript should be ignored
    print("Test 1: Filler-only transcript ('um uh like')")
    activity.stt = FakeSTT("um uh like")
    vad_event = vad.VADEvent(
        type=vad.VADEventType.INFERENCE_DONE,
        samples_index=0,
        timestamp=0.0,
        frames=[],
        speech_duration=0.6,
        silence_duration=0.0,
        probability=0.9,
        inference_duration=0.0,
        speaking=True,
    )
    await activity._classify_vad_frames_and_decide(vad_event, filler_words)
    
    if activity._interrupt_by_audio_activity.called:
        print("  ❌ FAILED: Agent was interrupted (should have been ignored)\n")
    else:
        print("  ✅ PASSED: Filler ignored, agent continues speaking\n")

    # Reset for test 2
    activity._current_speech = SpeechHandle.create(allow_interruptions=True)
    activity._interrupt_by_audio_activity.reset_mock()

    # Test 2: Interrupt keyword should trigger interruption
    print("Test 2: Interrupt keyword ('wait stop')")
    activity.stt = FakeSTT("wait stop")
    await activity._classify_vad_frames_and_decide(vad_event, filler_words)
    
    if activity._interrupt_by_audio_activity.called:
        print("  ✅ PASSED: Agent was interrupted by keyword\n")
    else:
        print("  ❌ FAILED: Agent not interrupted (should have been)\n")

    # Reset for test 3
    activity._current_speech = SpeechHandle.create(allow_interruptions=True)
    activity._interrupt_by_audio_activity.reset_mock()

    # Test 3: Mixed content (filler + real words) should interrupt
    print("Test 3: Mixed content ('um I have a question')")
    activity.stt = FakeSTT("um I have a question")
    await activity._classify_vad_frames_and_decide(vad_event, filler_words)
    
    if activity._interrupt_by_audio_activity.called:
        print("  ✅ PASSED: Agent was interrupted by real content\n")
    else:
        print("  ❌ FAILED: Agent not interrupted (should have been)\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
