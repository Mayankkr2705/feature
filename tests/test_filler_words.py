"""
Unit tests for filler-word classification during agent speech.

These tests verify that the _classify_vad_frames_and_decide method
in AgentActivity correctly distinguishes between filler words and
real interruptions using mocked STT responses.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit import rtc
from livekit.agents import stt, vad
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.speech_handle import SpeechHandle


@dataclass
class MockSTTResponse:
    """Mock STT response with alternatives."""

    alternatives: list[stt.SpeechData]


class MockSTT(stt.STT):
    """Mock STT that returns configurable transcripts."""

    def __init__(self, transcript: str = ""):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.transcript = transcript

    async def recognize(self, buffer, *, language: str | None = None):
        """Return a mock SpeechEvent with the configured transcript."""
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=self.transcript, language="en-US")],
        )


@pytest.fixture
def agent():
    """Create a minimal test agent."""
    return Agent(instructions="Test agent")


@pytest.fixture
def session():
    """Create a minimal test session."""
    return AgentSession()


@pytest.fixture
def activity(agent, session):
    """Create a test activity with a current speech."""
    act = AgentActivity(agent, session)
    # Set up minimal state for testing
    act._current_speech = SpeechHandle.create(allow_interruptions=True)
    return act


@pytest.fixture
def vad_event():
    """Create a test VAD event with sufficient duration."""
    return vad.VADEvent(
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


@pytest.mark.asyncio
async def test_filler_only_transcript_ignored(activity, vad_event):
    """Test that transcripts containing only filler words are ignored."""
    filler_words = ["um", "uh", "like"]
    activity.stt = MockSTT("um uh like")

    # Mock the interrupt method to track calls
    activity._interrupt_by_audio_activity = MagicMock()

    await activity._classify_vad_frames_and_decide(vad_event, filler_words)

    # Should NOT have called interrupt
    activity._interrupt_by_audio_activity.assert_not_called()
    # Speech should not be marked as interrupted
    assert not activity._current_speech.interrupted


@pytest.mark.asyncio
async def test_interrupt_keyword_triggers_interruption(activity, vad_event):
    """Test that interrupt keywords trigger immediate interruption."""
    filler_words = ["um", "uh"]
    activity.stt = MockSTT("wait stop")

    # Mock the interrupt method to track calls
    activity._interrupt_by_audio_activity = MagicMock()

    await activity._classify_vad_frames_and_decide(vad_event, filler_words)

    # Should have called interrupt
    activity._interrupt_by_audio_activity.assert_called_once()


@pytest.mark.asyncio
async def test_mixed_content_triggers_interruption(activity, vad_event):
    """Test that mixed filler + real words trigger interruption."""
    filler_words = ["um", "uh"]
    activity.stt = MockSTT("um I have a question")

    activity._interrupt_by_audio_activity = MagicMock()

    await activity._classify_vad_frames_and_decide(vad_event, filler_words)

    # Should have called interrupt (contains non-filler words)
    activity._interrupt_by_audio_activity.assert_called_once()


@pytest.mark.asyncio
async def test_empty_transcript_triggers_interruption(activity, vad_event):
    """Test that empty transcript is treated as interruption (safe default)."""
    filler_words = ["um", "uh"]
    activity.stt = MockSTT("")

    activity._interrupt_by_audio_activity = MagicMock()

    await activity._classify_vad_frames_and_decide(vad_event, filler_words)

    # Should have called interrupt (empty transcript -> safe default)
    activity._interrupt_by_audio_activity.assert_called_once()


@pytest.mark.asyncio
async def test_stt_error_triggers_interruption(activity, vad_event):
    """Test that STT errors fall back to interruption (safe default)."""
    filler_words = ["um", "uh"]

    # Create an STT that raises an error
    class FailingSTT(stt.STT):
        def __init__(self):
            super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))

        async def recognize(self, buffer, *, language: str | None = None):
            raise RuntimeError("STT failed")

    activity.stt = FailingSTT()
    activity._interrupt_by_audio_activity = MagicMock()

    await activity._classify_vad_frames_and_decide(vad_event, filler_words)

    # Should have called interrupt (error -> safe default)
    activity._interrupt_by_audio_activity.assert_called_once()


@pytest.mark.asyncio
async def test_no_current_speech_is_noop(activity, vad_event):
    """Test that classifier is a no-op when there's no current speech."""
    filler_words = ["um", "uh"]
    activity.stt = MockSTT("um uh")
    activity._current_speech = None  # No active speech

    activity._interrupt_by_audio_activity = MagicMock()

    await activity._classify_vad_frames_and_decide(vad_event, filler_words)

    # Should NOT have called interrupt (no active speech)
    activity._interrupt_by_audio_activity.assert_not_called()


@pytest.mark.asyncio
async def test_already_interrupted_is_noop(activity, vad_event):
    """Test that classifier is a no-op if speech is already interrupted."""
    filler_words = ["um", "uh"]
    activity.stt = MockSTT("hello there")
    activity._current_speech._interrupted = True  # Already interrupted

    activity._interrupt_by_audio_activity = MagicMock()

    await activity._classify_vad_frames_and_decide(vad_event, filler_words)

    # Should NOT have called interrupt (already interrupted)
    activity._interrupt_by_audio_activity.assert_not_called()


@pytest.mark.asyncio
async def test_case_insensitive_filler_matching(activity, vad_event):
    """Test that filler matching is case-insensitive."""
    filler_words = ["UM", "Uh", "LiKe"]
    activity.stt = MockSTT("um UH like")

    activity._interrupt_by_audio_activity = MagicMock()

    await activity._classify_vad_frames_and_decide(vad_event, filler_words)

    # Should NOT have called interrupt (all tokens are fillers, case-insensitive)
    activity._interrupt_by_audio_activity.assert_not_called()


@pytest.mark.asyncio
async def test_interrupt_keywords_case_insensitive(activity, vad_event):
    """Test that interrupt keywords work case-insensitively."""
    filler_words = ["um"]
    activity.stt = MockSTT("WAIT STOP")

    activity._interrupt_by_audio_activity = MagicMock()

    await activity._classify_vad_frames_and_decide(vad_event, filler_words)

    # Should have called interrupt (interrupt keywords present)
    activity._interrupt_by_audio_activity.assert_called_once()


@pytest.mark.asyncio
async def test_punctuation_stripped_from_tokens(activity, vad_event):
    """Test that punctuation is properly stripped during tokenization."""
    filler_words = ["um", "uh"]
    activity.stt = MockSTT("um, uh! like?")

    activity._interrupt_by_audio_activity = MagicMock()

    await activity._classify_vad_frames_and_decide(vad_event, filler_words)

    # Should NOT have called interrupt (all are fillers after punctuation removal)
    activity._interrupt_by_audio_activity.assert_not_called()


@pytest.mark.asyncio
async def test_all_interrupt_keywords(activity, vad_event):
    """Test all defined interrupt keywords trigger interruption."""
    filler_words = ["um"]
    interrupt_keywords = ["stop", "wait", "pause", "hold", "cut", "no"]

    for keyword in interrupt_keywords:
        # Reset state
        activity._current_speech = SpeechHandle.create(allow_interruptions=True)
        activity.stt = MockSTT(keyword)
        activity._interrupt_by_audio_activity = MagicMock()

        await activity._classify_vad_frames_and_decide(vad_event, filler_words)

        # Should have called interrupt for this keyword
        activity._interrupt_by_audio_activity.assert_called_once(), f"Keyword '{keyword}' should trigger interruption"
