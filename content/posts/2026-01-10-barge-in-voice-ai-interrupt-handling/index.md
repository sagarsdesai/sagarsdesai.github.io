---
title: "Barge-In Support - Interrupt Handling in Real-Time Voice AI Pipelines"
date: 2026-01-10
categories: [voice-ai, architecture, real-time]
featured: true
---

Real-time voice AI systems face a fundamental interaction problem: **users expect to interrupt**. When the assistant speaks for too long or provides unwanted information, users naturally try to interrupt—but most voice pipelines ignore this input entirely.

This post explores the system design for **barge-in support**: the ability to detect user speech during assistant playback, cancel ongoing generation, and transition back to listening mode with minimal latency.

The recommended architecture—**Hot-Cut with Rolling Buffer**—is the industry-standard approach used by Siri, Google Assistant, and Alexa. It achieves **zero-latency mute** and **zero audio loss** by combining client-side VAD with speculative audio buffering.

**Target Environment:** Headphones or phone interaction (acoustic isolation eliminates echo cancellation complexity).

---

#### Problem Statement

Consider a production voice pipeline using NVIDIA NIMs:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      CURRENT STATE: Half-Duplex Pipeline                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   User       ASR NIM        Gateway         LLM NIM       TTS NIM      User   │
│  ┌─────┐   ┌─────────┐   ┌───────────┐   ┌─────────┐   ┌─────────┐   ┌─────┐ │
│  │ Mic │──▶│Parakeet │──▶│ Producer- │──▶│ Qwen 2.5│──▶│ Magpie  │──▶│Spkr │ │
│  │     │   │  1.1B   │   │ Consumer  │   │   7B    │   │   TTS   │   │     │ │
│  └─────┘   └─────────┘   └───────────┘   └─────────┘   └─────────┘   └─────┘ │
│     │                                                                    │    │
│     │  ◀───────────────────────────────────────────────────────────────▶│    │
│     │                                                                    │    │
│     │             PROBLEM: Mic disabled during TTS playback              │    │
│     │                      User cannot interrupt                         │    │
│     │                                                                    │    │
└──────────────────────────────────────────────────────────────────────────────┘
```

The pipeline operates in **half-duplex mode**: the system either listens OR speaks, never both. State transitions follow a strict sequence:

```
State Flow:  LISTENING ──▶ PROCESSING ──▶ SPEAKING ──▶ LISTENING
                            │
                            └── (filler audio at 600ms if slow)
```

**User Experience Without Barge-In:**

```
User: "What's the capital of France?"
AI:   "The capital of France is Paris. It's a beautiful city known for the 
       Eiffel Tower, the Louvre Museum, the Seine River, Notre-Dame Cathedral,
       and countless cafés. The city has a rich history dating back to..."
       
User: [WAITS 15+ SECONDS]
User: "Actually, what about Germany?"
AI:   "The capital of Germany is Berlin."
```

**User Experience With Barge-In:**

```
User: "What's the capital of France?"
AI:   "The capital of France is Par—"
User: "Actually, never mind, what about Germany?"
AI:   "The capital of Germany is Berlin."
```

---

#### Assumptions and Constraints

The design operates within these constraints:

| Constraint | Value | Rationale |
|:---|:---|:---|
| **Audio Direction** | Half-duplex | Mic disabled during TTS playback |
| **Producer-Consumer** | Background LLM→TTS→Queue | Queue contains pending audio that must be cancelled |
| **State Machine** | 4 states | `LISTENING`, `PROCESSING`, `SPEAKING`, `ERROR` |
| **Client Architecture** | Single gRPC bidirectional stream | Can send messages anytime |
| **Session Memory** | Redis per-turn write | Interrupted response must be saved (partial) |
| **Latency Budget** | <200ms interrupt | From user speech detected → TTS stops |
| **Turn Transition** | <300ms | From TTS stops → ASR begins processing |

---

#### Data Flow Analysis

The current gateway uses an `asyncio.Queue` to decouple LLM generation from TTS streaming:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        PRODUCER-CONSUMER ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   PRODUCER TASK (Background)                  CONSUMER LOOP (Main)            │
│   ══════════════════════════                  ════════════════════            │
│                                                                               │
│   ┌─────────────┐                            ┌─────────────┐                 │
│   │  LLM NIM    │                            │  Consumer   │                 │
│   │  (Streaming)│                            │  Loop       │                 │
│   └──────┬──────┘                            └──────┬──────┘                 │
│          │                                          │                         │
│          │ tokens                                   │                         │
│          ▼                                          │                         │
│   ┌─────────────┐       ┌─────────────────┐        │                         │
│   │  TTS NIM    │──────▶│  asyncio.Queue  │◀───────┘                         │
│   │  (Chunked)  │       │  (audio_chunk)  │                                  │
│   └─────────────┘       └────────┬────────┘                                  │
│                                  │                                            │
│                                  ▼                                            │
│                          ┌─────────────┐                                      │
│                          │  gRPC Yield │                                      │
│                          │  to Client  │                                      │
│                          └─────────────┘                                      │
│                                                                               │
│   PROBLEM: When barge-in occurs, the queue may contain 2-5 seconds of        │
│            pre-generated audio that must be discarded.                        │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Cancellation Requirements:**

1. **Cancel producer task:** Stop LLM generation and TTS synthesis
2. **Drain queue:** Discard all pending audio chunks
3. **Save partial response:** Write interrupted text to session store
4. **Notify client:** Send `INTERRUPTED` event
5. **Transition state:** Move to `LISTENING`

---

#### Implementation Options

Four architectural approaches exist, each with distinct trade-offs.

##### Option A - Client-Side VAD with Server Cancel

The client detects speech using Voice Activity Detection (VAD) and sends a cancel signal.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         OPTION A: Client-Side VAD                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│    CLIENT                                         SERVER                      │
│   ┌──────────────────────────────┐              ┌──────────────────────────┐ │
│   │                              │              │                          │ │
│   │  ┌─────────┐                 │              │    TTS Stream Active     │ │
│   │  │   VAD   │◀── Mic Input    │              │          │               │ │
│   │  └────┬────┘                 │              │          ▼               │ │
│   │       │                      │ cancel_tts   │   ┌────────────────┐     │ │
│   │       ▼                      │─────────────▶│   │ 1. Cancel      │     │ │
│   │  Speech detected?            │   (gRPC)     │   │    producer    │     │ │
│   │       │                      │              │   │ 2. Drain queue │     │ │
│   │       │ YES                  │              │   │ 3. Save partial│     │ │
│   │       ▼                      │              │   │ 4. Send        │     │ │
│   │  1. Stop local playback      │              │   │    INTERRUPTED │     │ │
│   │  2. Send cancel_tts=true     │              │   └────────────────┘     │ │
│   │                              │              │          │               │ │
│   │                              │◀─────────────│──────────┘               │ │
│   │                              │  INTERRUPTED │                          │ │
│   │  3. Resume mic streaming     │   + LISTENING│                          │ │
│   │                              │              │                          │ │
│   └──────────────────────────────┘              └──────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Technical Characteristics:**

| Aspect | Behavior |
|:---|:---|
| VAD Method | Energy-based RMS threshold |
| Detection Latency | ~50-100ms (local) |
| Network Latency | ~10-50ms (cancel signal) |
| Total Interrupt Latency | ~60-150ms |
| Implementation Effort | 2-3 days |

**Echo Problem:**

```
HEADPHONES (Works):                    SPEAKERS (Broken):
┌────────────────────────┐            ┌────────────────────────┐
│                        │            │                        │
│ TTS ──▶ Headphones     │            │ TTS ──▶ Speakers ──┐   │
│                        │            │                    │   │
│ Mic ◀── User voice     │            │         ┌─────────┘   │
│         │              │            │         │             │
│         ▼              │            │         ▼             │
│   VAD detects:         │            │       Mic ◀───────────│
│   User speech only ✓   │            │         │             │
│                        │            │         ▼             │
│                        │            │   VAD detects:        │
│                        │            │   TTS audio ✗         │
│                        │            │   (False positive)    │
│                        │            │                       │
└────────────────────────┘            └────────────────────────┘
```

**Mitigation:** Require headphones for v1, or implement dynamic threshold adjustment during TTS playback.

---

##### Option B - Server-Side Full-Duplex

Keep the mic stream active on the server even during TTS playback.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       OPTION B: Server-Side Full-Duplex                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│    CLIENT                                         SERVER                      │
│   ┌──────────────────────────────┐              ┌──────────────────────────┐ │
│   │                              │              │                          │ │
│   │  Mic ALWAYS streaming ───────┼─audio_chunk─▶│  ┌──────────────────┐   │ │
│   │  (even during TTS playback)  │              │  │  Server-side VAD │   │ │
│   │                              │              │  │       OR         │   │ │
│   │  Speaker playing TTS ◀───────┼─audio_chunk──│  │  ASR partial →   │   │ │
│   │                              │              │  │  speech detected │   │ │
│   │                              │              │  └────────┬─────────┘   │ │
│   │                              │              │           │             │ │
│   │                              │              │   If speech during      │ │
│   │                              │              │   SPEAKING:             │ │
│   │                              │              │   1. Cancel producer    │ │
│   │                              │              │   2. Drain queue        │ │
│   │                              │              │   3. Start ASR on       │ │
│   │                              │              │      interrupting audio │ │
│   │                              │              │                         │ │
│   └──────────────────────────────┘              └──────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Technical Characteristics:**

| Aspect | Behavior |
|:---|:---|
| Bandwidth | 2× (send while receiving) |
| VAD Location | Server (Silero VAD or ASR partials) |
| Detection Latency | ~100-200ms (network + processing) |
| Implementation Effort | 1-2 weeks |
| Echo Handling | Still required if speakers used |

---

##### Option C - Hybrid with Handshake

Enhanced version of Option A with explicit server acknowledgment.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       OPTION C: Hybrid with Handshake                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│    CLIENT                                         SERVER                      │
│   ┌──────────────────────────────┐              ┌──────────────────────────┐ │
│   │                              │              │                          │ │
│   │  1. Detect speech (VAD)      │              │                          │ │
│   │  2. Stop local audio         │              │                          │ │
│   │  3. Buffer mic input         │ cancel_tts   │                          │ │
│   │  4. Send cancel_tts ─────────┼─────────────▶│  5. Cancel producer      │ │
│   │                              │              │  6. Drain queue          │ │
│   │                              │              │  7. Save partial         │ │
│   │                              │◀─────────────┼──8. Send INTERRUPTED     │ │
│   │  9. Receive INTERRUPTED      │  INTERRUPTED │                          │ │
│   │                              │              │                          │ │
│   │  10. Send buffered audio ────┼─audio_chunk─▶│  11. Start ASR stream    │ │
│   │  11. Continue mic stream     │              │                          │ │
│   │                              │◀─────────────┼──12. Send LISTENING      │ │
│   │  12. Confirm state           │   LISTENING  │                          │ │
│   │                              │              │                          │ │
│   └──────────────────────────────┘              └──────────────────────────┘ │
│                                                                               │
│   Benefits:                                                                   │
│   • Clean state transition (no race conditions)                              │
│   • Server controls state machine                                            │
│   • Partial response saved before transition                                 │
│   • Can log interruption metrics                                             │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

##### Option D - ASR-Based Detection

Use ASR partial transcripts to detect interrupting speech (no VAD needed).

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       OPTION D: ASR-Based Detection                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   During SPEAKING state, keep ASR stream open:                                │
│                                                                               │
│   audio_chunk ──▶ ASR NIM ──▶ partial transcript                             │
│                                     │                                         │
│                                     ▼                                         │
│                             ┌───────────────┐                                │
│                             │ Transcript    │                                │
│                             │ detected?     │                                │
│                             └───────┬───────┘                                │
│                                     │                                         │
│                      ┌──────────────┼──────────────┐                         │
│                      │ YES          │              │ NO                       │
│                      ▼              │              ▼                          │
│               ┌────────────┐        │       ┌────────────┐                   │
│               │ User is    │        │       │ Continue   │                   │
│               │ speaking → │        │       │ TTS        │                   │
│               │ Interrupt  │        │       │ playback   │                   │
│               └────────────┘        │       └────────────┘                   │
│                                     │                                         │
│   Advanced: Energy threshold first, then ASR confirmation                    │
│   (reduces GPU cost by filtering silence before ASR)                          │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Technical Characteristics:**

| Aspect | Behavior |
|:---|:---|
| Detection Latency | 300-500ms (ASR streaming latency) |
| GPU Cost | Additional ASR inference during TTS |
| Echo Handling | Built-in (ASR trained on speech) |
| Keyword Commands | Can detect "stop", "wait", "hold on" |

---

##### Option E - Hot-Cut Architecture (Recommended)

Options A-D each have trade-offs that compromise either latency or completeness. The industry-standard approach used by high-performance voice assistants (Siri, Google Assistant) combines client-side detection with **speculative audio buffering**.

**Key Insight:** When the user says "**Sto**-p that," the first syllable occurs *while* the VAD is still deciding if speech is present. Without buffering, this audio is lost.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              OPTION E: Hot-Cut Architecture (Client-Gated Full-Duplex)        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   CONCEPT: Client behaves like a walkie-talkie that can override incoming    │
│            signal. Audio is buffered speculatively, sent only on interrupt.   │
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────┐│
│   │                         CLIENT STATE MACHINE                             ││
│   │                                                                          ││
│   │   PLAYING TTS                           BARGE-IN TRIGGERED               ││
│   │   ───────────                           ──────────────────               ││
│   │                                                                          ││
│   │   ┌─────────────────┐                  ┌─────────────────┐               ││
│   │   │ Speaker: ON     │                  │ Speaker: OFF    │  ◀─ INSTANT   ││
│   │   │ Mic: Recording  │   VAD Trigger    │ (Hardware mute) │               ││
│   │   │ Buffer: Rolling │ ───────────────▶ │                 │               ││
│   │   │ Send: Nothing   │                  │ Send: Signal +  │               ││
│   │   │                 │                  │       Buffer +  │               ││
│   │   │                 │                  │       New Audio │               ││
│   │   └─────────────────┘                  └─────────────────┘               ││
│   │                                                                          ││
│   │   Rolling Buffer (500ms):  [chunk][chunk][chunk][chunk][chunk]           ││
│   │                             ─────────────────────────────────            ││
│   │                             Captures "Sto-" before VAD fires             ││
│   │                                                                          ││
│   └─────────────────────────────────────────────────────────────────────────┘│
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Why Hot-Cut is Superior:**

| Aspect | Options A-D | Option E (Hot-Cut) |
|:---|:---|:---|
| Mute Latency | Network RTT (~50ms) | 0ms (local hardware) |
| Audio Clipping | First syllable lost | Captured in pre-roll buffer |
| Bandwidth | Continuous OR burst | Efficient (burst on interrupt only) |
| User Perception | "Slightly delayed" | "Instant response" |

**Sequence Diagram:**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        HOT-CUT SEQUENCE (Timeline)                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   TIME     CLIENT                           SERVER                            │
│   ════     ══════                           ══════                            │
│                                                                               │
│   T+0ms    ┌─────────────────────────────┐                                   │
│            │ TTS Playing: "The capital   │  TTS Stream Active                │
│            │ of France is Pa—"           │        │                          │
│            │                             │        ▼                          │
│            │ Mic → Rolling Buffer        │  Audio Chunk n ──────▶ Client    │
│            │ [silence][silence][sil...]  │                                   │
│            └─────────────────────────────┘                                   │
│                                                                               │
│   T+100ms  User speaks: "Wait, stop."                                        │
│            Buffer: [sil][sil]["Wa"][...]                                     │
│                                                                               │
│   T+150ms  ┌─────────────────────────────┐                                   │
│            │ VAD: Energy > Threshold     │                                   │
│            │                             │                                   │
│            │ ⚡ HOT CUT SEQUENCE:        │                                   │
│            │                             │                                   │
│            │ 1. Stop speaker (0ms)       │                                   │
│            │ 2. Send BargeInMessage ─────┼───────────────────▶ Server       │
│            │    - cancel_tts: true       │                                   │
│            │    - pre_roll: ["Wa"]["it"] │                                   │
│            │ 3. Continue streaming ──────┼───────────────────▶ ASR          │
│            │                             │                                   │
│            └─────────────────────────────┘                                   │
│                                                                               │
│   T+160ms                                 ┌─────────────────────────────┐    │
│                                           │ Server receives BargeIn     │    │
│                                           │                             │    │
│                                           │ 1. Cancel TTS producer      │    │
│                                           │ 2. Drain audio queue        │    │
│                                           │ 3. Inject pre_roll → ASR    │    │
│                                           │ 4. Save partial response    │    │
│                                           │ 5. Yield LISTENING state    │    │
│                                           │                             │    │
│                                           └─────────────────────────────┘    │
│                                                                               │
│   T+200ms  Client streaming new audio ────────────────────────▶ ASR          │
│                                                                               │
│   T+800ms  ASR Final: "Wait stop"                                            │
│            Server processes new turn                                          │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Technical Characteristics:**

| Aspect | Behavior |
|:---|:---|
| Mute Latency | 0ms (local hardware control) |
| Audio Loss | None (pre-roll buffer captures onset) |
| Pre-roll Buffer | 200-500ms rolling window |
| Network Efficiency | No streaming during TTS (buffer only) |
| Implementation Effort | 3-4 days |

**Why This Works for Headphones/Phone:**

1. **Acoustic Isolation:** Headphones physically separate mic from speaker—no echo cancellation needed
2. **Proximity:** Phone mics are close to mouth—high signal-to-noise ratio makes energy detection reliable
3. **Controlled Environment:** Known audio hardware enables tuned VAD thresholds

---

#### Decision Matrix

| Criteria | Weight | A (Client VAD) | B (Server Duplex) | C (Hybrid) | D (ASR) | E (Hot-Cut) |
|:---|:---|:---|:---|:---|:---|:---|
| Mute Latency | Critical | ⚠️ 60-150ms | ❌ 150-200ms | ⚠️ 100-150ms | ❌ 300-500ms | ✅ **0ms** |
| Audio Clipping | High | ❌ First syllable lost | ❌ Lost | ❌ Lost | ❌ Lost | ✅ **Pre-roll captured** |
| Implementation | Medium | ✅ 2-3 days | ❌ 1-2 weeks | ⚠️ 3-4 days | ⚠️ 4-5 days | ✅ 3-4 days |
| Bandwidth | Low | ✅ Low | ❌ 2× | ✅ Low | ⚠️ 1.5× | ✅ **Optimal** |
| Client Complexity | Medium | ⚠️ VAD only | ✅ None | ⚠️ VAD + buffer | ✅ None | ⚠️ VAD + buffer |
| Echo Handling | N/A | N/A (headphones) | N/A | N/A | ✅ Built-in | N/A (headphones) |

**Recommendation: Option E (Hot-Cut Architecture)**

For headphone/phone interactions where acoustic echo cancellation is not required, Option E provides the best user experience:

1. **Zero-latency mute:** Speaker cuts locally the instant speech is detected—feels instant to user
2. **No audio clipping:** Rolling buffer captures the start of the interrupting sentence
3. **Network efficiency:** No continuous streaming during TTS playback
4. **Industry standard:** Architecture used by Siri, Google Assistant, Alexa

The trade-off is increased client complexity (VAD + rolling buffer), which is acceptable for dedicated applications.

---

#### Updated State Machine

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         STATE MACHINE WITH BARGE-IN                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│                                    cancel_tts                                 │
│                     ┌─────────────────────────────────────────┐               │
│                     │                                         │               │
│                     ▼                                         │               │
│     ┌───────────────────────┐                          ┌──────┴───────┐       │
│     │                       │   final ASR transcript   │              │       │
│     │       LISTENING       │─────────────────────────▶│  PROCESSING  │       │
│     │                       │                          │              │       │
│     └───────────────────────┘                          └──────┬───────┘       │
│              ▲                                                │               │
│              │                                                │               │
│              │                                                │ first TTS     │
│              │                                                │ chunk         │
│              │                                                ▼               │
│              │                                         ┌──────────────┐       │
│              │          END_OF_TURN or                 │              │       │
│              └─────────────────────────────────────────│   SPEAKING   │       │
│              │                                         │              │       │
│              │                                         └──────┬───────┘       │
│              │                                                │               │
│              │                                                │ cancel_tts    │
│              │                                                ▼               │
│              │                                         ┌──────────────┐       │
│              │                                         │              │       │
│              └─────────────────────────────────────────│ INTERRUPTED  │       │
│                                                        │              │       │
│                                                        └──────────────┘       │
│                                                                               │
│   New State: INTERRUPTED (transient)                                          │
│   • Cleanup occurs in this state                                              │
│   • Session saved with partial response                                       │
│   • Metrics recorded                                                          │
│   • Transitions immediately to LISTENING                                      │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

#### Protocol Changes

The Hot-Cut architecture requires a specialized message that carries both the cancel signal AND the buffered audio in a single packet:

```protobuf
// voice_workflow.proto additions

message ClientMessage {
  oneof content {
    VoiceConfig config = 1;
    bytes audio_chunk = 2;
    string text_input = 3;
    BargeInMessage barge_in = 4;  // NEW: Interrupt with pre-roll audio
  }
}

// Specialized interrupt packet: signal + buffered audio in one message
message BargeInMessage {
  bool cancel_tts = 1;
  bytes pre_roll_audio = 2;  // 200-500ms of audio captured BEFORE VAD trigger
}

enum EventType {
  EVENT_UNSPECIFIED = 0;
  LISTENING = 1;
  PROCESSING = 2;
  SPEAKING = 3;
  ERROR = 4;
  END_OF_TURN = 5;
  INTERRUPTED = 6;  // NEW: Response was interrupted
}
```

**Why a combined message?**

Sending `cancel_tts` and `pre_roll_audio` separately creates a race condition: the server might start a new ASR stream before the buffered audio arrives. The combined `BargeInMessage` ensures atomicity.

---

#### Server Barge-In Handler

The server must handle `BargeInMessage` with high priority and inject the pre-roll audio into the ASR stream:

```python
class VoiceGatewayServicer:
    async def StreamAudio(self, request_iterator, context):
        barge_in_event = asyncio.Event()
        pre_roll_audio = None
        
        async def input_processor():
            """Handle incoming messages with priority interrupt."""
            nonlocal pre_roll_audio
            async for request in request_iterator:
                if request.HasField('barge_in'):
                    # Priority interrupt: cancel + inject audio
                    logger.info("Barge-in: Hot-cut signal received")
                    pre_roll_audio = request.barge_in.pre_roll_audio
                    barge_in_event.set()
                    METRICS.barge_in_count.inc()
                else:
                    await input_queue.put(request)
        
        # In consumer loop: check for barge-in
        while True:
            if barge_in_event.is_set():
                # 1. Immediate cleanup
                if not producer_task.done():
                    producer_task.cancel()
                    try:
                        await producer_task
                    except asyncio.CancelledError:
                        pass
                
                # 2. Wipe pending audio (fresh queue)
                audio_queue = asyncio.Queue()
                
                # 3. Save partial turn to history
                partial = full_response_container[0].strip()
                if partial:
                    session_state.add_turn(
                        transcript, 
                        partial + " [interrupted]"
                    )
                    await self.session_store.save(session_id, session_state)
                
                # 4. Inject pre-roll audio into ASR
                # Critical: Treat pre-roll as start of new turn
                if pre_roll_audio:
                    await asr_queue.put(pre_roll_audio)
                
                # 5. Notify client of state change
                yield ServerMessage(
                    event=ServerEvent(type=INTERRUPTED)
                )
                yield ServerMessage(
                    event=ServerEvent(type=LISTENING)
                )
                
                barge_in_event.clear()
                pre_roll_audio = None
                # Continue to process new turn
```

**Key Detail:** The pre-roll audio is injected into the ASR queue *before* the client's subsequent audio chunks arrive. This ensures the complete utterance ("**Wait**, stop") is transcribed, not just ("stop").

---

#### Client Hot-Cut Implementation

The client maintains a **rolling buffer** during TTS playback, capturing audio speculatively without sending it:

```python
import audioop
import collections
from threading import Thread

class VoiceClient:
    def __init__(self):
        self.is_playing = False
        self.stream_mode = False
        # Rolling buffer: 10 chunks × 20ms = 200ms pre-roll
        self.audio_buffer = collections.deque(maxlen=10)
        self.bargein_threshold = 1500  # RMS threshold
        
    def mic_loop(self):
        """Main mic processing loop with hot-cut capability."""
        while self.running:
            chunk = self.input_stream.read(1024, exception_on_overflow=False)
            
            if self.is_playing:
                # STATE: TTS playing, buffer silently
                rms = audioop.rms(chunk, 2)
                
                if rms > self.bargein_threshold:
                    # ⚡ HOT CUT SEQUENCE
                    
                    # 1. Stop speaker immediately (0ms latency)
                    self.stop_speaker()
                    self.is_playing = False
                    
                    # 2. Send barge-in message with buffered audio
                    pre_roll = b''.join(self.audio_buffer)
                    self.send_barge_in(
                        cancel_tts=True,
                        pre_roll_audio=pre_roll + chunk
                    )
                    
                    # 3. Switch to streaming mode
                    self.stream_mode = True
                    self.audio_buffer.clear()
                else:
                    # Keep buffering silently (no network traffic)
                    self.audio_buffer.append(chunk)
            else:
                # STATE: Normal streaming
                self.send_audio_chunk(chunk)
    
    def send_barge_in(self, cancel_tts, pre_roll_audio):
        """Send combined cancel + audio message."""
        msg = ClientMessage(
            barge_in=BargeInMessage(
                cancel_tts=cancel_tts,
                pre_roll_audio=pre_roll_audio
            )
        )
        self.stream.write(msg)
```

**Rolling Buffer Visualization:**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          ROLLING BUFFER OPERATION                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   TIME        BUFFER STATE                      ACTION                        │
│   ════        ════════════                      ══════                        │
│                                                                               │
│   T+0ms       [sil][sil][sil][sil][sil]        TTS playing, user silent      │
│                                                                               │
│   T+20ms      [sil][sil][sil][sil][sil]        New chunk, shift buffer       │
│                ───▶                                                           │
│                                                                               │
│   T+100ms     [sil][sil]["Wa"]["it"][","]      User starts speaking          │
│                                                Buffer captures onset          │
│                                                                               │
│   T+120ms     VAD triggers (RMS > threshold)                                 │
│                                                                               │
│               ┌──────────────────────────────────────────────────────────┐   │
│               │ HOT CUT:                                                  │   │
│               │  1. Stop speaker (hardware mute)         ← 0ms            │   │
│               │  2. Flush buffer → pre_roll_audio        ← instant        │   │
│               │  3. Send BargeInMessage to server        ← ~10ms network  │   │
│               │  4. Switch to stream mode                                 │   │
│               └──────────────────────────────────────────────────────────┘   │
│                                                                               │
│   T+130ms     Streaming ["st"]["op"][...]      Normal ASR streaming          │
│                                                                               │
│   RESULT: Server receives "Wait, stop" not just "op"                         │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

#### Edge Cases

| Edge Case | Handling |
|:---|:---|
| Cancel during filler audio | Treat same as regular TTS—stop and transition |
| Cancel before any TTS sent | Skip TTS, save "thinking..." as response |
| Rapid repeated cancels | Debounce: ignore cancels within 500ms of last |
| Cancel during LLM (pre-TTS) | Cancel LLM stream, don't start TTS |
| Network delay on cancel | Client stops playback immediately, server catches up |
| Echo triggers false positive | Require sustained speech (150ms+) and higher threshold during playback |

---

#### Metrics

```python
# Prometheus metrics for barge-in monitoring
METRICS.barge_in_count = Counter(
    'voice_gateway_barge_in_total',
    'Number of barge-in interruptions'
)

METRICS.barge_in_latency = Histogram(
    'voice_gateway_barge_in_latency_seconds',
    'Time from cancel signal to LISTENING state',
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
)
```

---

#### Known Limitations (v1)

| Limitation | Workaround | Future Fix |
|:---|:---|:---|
| Echo problem with speakers | Require headphones/earbuds | WebRTC AEC integration |
| Client must implement VAD + buffer | Provide reference implementation | Server-side full-duplex |
| Pre-roll buffer size fixed | 200-500ms covers most cases | Adaptive buffer sizing |

**Assumptions for Hot-Cut Architecture:**

1. **Headphone/phone interaction:** Physical acoustic isolation eliminates echo
2. **Controlled client:** Application can implement VAD and buffer management
3. **Low-latency network:** Pre-roll + signal fits in single packet (~10-50KB)

---

#### Summary

Barge-in support transforms voice AI from a turn-taking system to a conversational one. The key insight is that **client-side detection with speculative buffering provides zero-latency mute without audio loss**.

| Implementation | Mute Latency | Audio Loss | Effort | Best For |
|:---|:---|:---|:---|:---|
| **Hot-Cut (Option E)** | **0ms** | **None** | 3-4 days | Production (headphones) |
| Client VAD (Option A) | 60-150ms | First syllable | 2-3 days | Simple clients |
| Server Full-Duplex (B) | 150-200ms | First syllable | 1-2 weeks | Thin clients |
| ASR-Based (Option D) | 300-500ms | First syllable | 4-5 days | Keyword commands |

**The Hot-Cut architecture is the industry standard** used by Siri, Google Assistant, and Alexa. It provides:

1. **Instant mute:** Speaker cuts locally (0ms)—feels immediate to user
2. **Complete capture:** Rolling buffer preserves the start of the interrupting sentence
3. **Network efficiency:** No continuous streaming during TTS playback
4. **Clean handoff:** Combined `BargeInMessage` ensures atomicity

The trade-off is increased client complexity (VAD + rolling buffer). For controlled environments with headphones/phone mics, this is the optimal choice. For telephony/PSTN integrations where client modification is impossible, server-side approaches (Option B) become necessary.

Barge-in is a UX feature, not a performance optimization. The goal is natural conversation flow—the assistant should feel **alert**, not **laggy**.


