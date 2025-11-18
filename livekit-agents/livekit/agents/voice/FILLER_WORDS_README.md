
Filler-word handling for LiveKit Agents

Summary

This small extension helps the agent ignore harmless "fillers" (things like "uh", "umm", "hmm", or similar) while it is speaking, so brief user noises don't interrupt speech unnecessarily. At the same time, it preserves legitimate interruptions — words like "stop" or "wait" will always stop the agent.

How it works :

- We add a configurable `filler_words` list to `AgentSessionOptions` and a runtime helper `AgentSession.update_filler_words(...)` so you can change the list while the session is running.
- When the VAD reports a short burst of user speech while the agent is speaking, the code schedules a quick STT recognition on the VAD frames.
- If the resulting transcript consists only of words in your `filler_words` list (case-insensitive), the event is treated as a filler and ignored — the agent keeps talking.
- If the transcript contains interruption keywords (e.g. `stop`, `wait`) or any other non-filler words, it is treated as a real interruption and the agent stops as usual.

Why this approach

- Non-invasive: this is implemented as an extension on top of the existing VAD and event loop — no changes to the VAD internals or LiveKit core are needed.
- Safe fallback: if there is no STT available or you haven't configured `filler_words`, the system falls back to the original behavior (i.e., treat detected speech as a potential interruption).
- Language-agnostic: the filler list is just a list of strings that you control; matching is case-insensitive and token-based, which makes it flexible for mixed-language scenarios.

Configuration

You can set the filler words at session creation time or update them at runtime.

Examples

At startup:

```
session = AgentSession(...)
session._opts.filler_words = ["uh", "umm", "hmm", "haan"]
```

At runtime:

```
session.update_filler_words(["uh", "umm", "hmm", "haan"]) 
```

Tips: keep the list short and focused on common speech disfluencies for best results.

Testing checklist

1. Run an `AgentSession` with an STT backend configured and with some filler words set.
2. While the agent is speaking, trigger a brief filler utterance (e.g., "uh" or "umm") — the agent should continue speaking and you should see a `DEBUG` log stating the filler was ignored.
3. Trigger a clear interruption (e.g., "stop" or "wait one second") — the agent should stop immediately and you should see an `INFO` log indicating a valid interruption.
4. Try mixed phrases like "umm okay stop" — these should still interrupt because they contain an explicit command.

Logging behavior

- Ignored filler events: `DEBUG` level with a message like `Ignored filler interruption while agent speaking` plus the recognized transcript.
- Valid interruptions: `INFO` level with a message like `Valid interruption detected during agent speech` and the transcript.

Notes and caveats

- This feature relies on having an STT available during agent playback. If STT is missing or the filler list is empty, the code conservatively falls back to the previous behavior.
- If STT fails while trying to classify a short segment, the implementation errs on the side of caution and treats the audio as a real interruption (so the agent will stop).

Future ideas

- Make the interruption keywords configurable, like `filler_words`, and expose simple heuristics (confidence thresholds, fuzzy matching) to reduce false positives across languages.
- Optionally use language detection or token-confidence from the STT to make decisions more robust in noisy conditions.

If you'd like, I can:
- Add unit tests that mock STT to verify ignored vs. valid interruption behavior.
- Expose interruption keywords as a runtime option as well.
- Produce a short demo script that shows the behaviour with mocked audio input.


Further improvements (future work)

- Allow configuring interruption keywords at runtime (similar to `filler_words`).
- More advanced NLP heuristics (language detection, fuzzy matching) for better multi-language support.
- Optional confidence thresholding from STT results to avoid false positives from low-confidence transcripts.

Configuration

- When creating an `AgentSession`, pass `filler_words` via `AgentSessionOptions` (new parameter) or call `session.update_filler_words([...])` at runtime.

Example:

```
session = AgentSession(...)
# at startup
session._opts.filler_words = ["uh", "umm", "hmm", "haan"]
# or at runtime
session.update_filler_words(["uh", "umm", "hmm", "haan"]) 
```

Notes and caveats

- The system requires an STT backend available (`session.stt`) to perform the filler classification during agent speech. If STT is not available or the filler list is empty, behavior falls back to the previous interruption behavior.
- This implementation uses a conservative approach: if STT raises an error while classifying, we interrupt to avoid missing genuine interruptions.
- The interruption keyword list is intentionally small and can be extended if you want richer command detection.

Testing

1. Start a session with an STT configured and set `filler_words` to include the filler tokens you want to ignore.
2. While the agent is speaking, say a filler like "uh" or "umm" — the agent should continue speaking and logs should show an ignored filler interruption.
3. Say a real interruption like "stop" or "wait one second" — the agent must stop immediately and logs show a valid interruption.
4. Mixed utterances such as "umm okay stop" should trigger a valid interruption because of the presence of `stop`.

Logging

- Ignored fillers are logged with level `DEBUG` and the message: `Ignored filler interruption while agent speaking` plus the recognized transcript.
- Valid interruptions are logged with level `INFO` and the message: `Valid interruption detected during agent speech`.

Further improvements (future work)

- Allow configuring interruption keywords at runtime (similar to `filler_words`).
- More advanced NLP heuristics (language detection, fuzzy matching) for better multi-language support.
- Optional confidence thresholding from STT results to avoid false positives from low-confidence transcripts.


