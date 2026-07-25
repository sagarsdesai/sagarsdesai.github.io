---
title: "Voice Clone Cache - Solving Filler Audio Latency in Voice AI Pipelines"
date: 2026-01-09
categories: [voice-ai, latency, architecture]
featured: true
---

Real-time voice AI systems face a fundamental UX problem: **silence during processing feels broken**. When a user finishes speaking, the system must transcribe (ASR), generate a response (LLM), and synthesize speech (TTS). Even at 200-300ms total latency, the silence can feel awkward. Under load, this extends to seconds.

This post explores architectural approaches to filler audio—acknowledgment sounds like "Hmm..." or "Let me think..."—with a focus on the **Voice Clone Cache** pattern that achieves zero-latency playback while maintaining voice consistency.

---

#### The Latency Problem in Voice Pipelines

A typical voice AI pipeline executes sequentially:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        VOICE AI PIPELINE (Sequential)                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐  │
│   │  User   │    │   ASR   │    │   LLM   │    │   TTS   │    │  Audio   │  │
│   │ Speech  │───▶│ (STT)   │───▶│  (Gen)  │───▶│ Synth   │───▶│ Playback │  │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘    └──────────┘  │
│                                                                              │
│   ◀─────────────────────────────────────────────────────────────────────────▶│
│                                                                              │
│   T=0ms          T=50ms         T=100ms        T=200ms        T=350ms       │
│   User stops     ASR final      LLM starts     LLM done       Audio starts  │
│   speaking       transcript     generating     + TTS starts   playing       │
│                                                                              │
│                  ◀──────────────────────────────────────────▶                │
│                           "DEAD AIR" - User hears nothing                    │
│                                  (~250-350ms baseline)                       │
│                                  (~1-5 seconds under load)                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

Under normal conditions with warm GPU inference:

| Stage | Baseline Latency | Under Load |
|-------|------------------|------------|
| ASR final | ~50ms | ~100ms |
| LLM TTFT | ~40-80ms | 500ms+ |
| TTS synthesis | ~150-200ms | 500ms+ |
| **Total E2E** | **~250ms** | **1-5 seconds** |

The baseline 250ms is acceptable—barely perceptible. The problem emerges under load when LLM queues build up or GPU saturation occurs. A 2-5 second silence after speaking feels like a dropped connection.

---

#### Approaches to Filler Audio

There are four main architectural approaches:

| Approach | Latency | Voice Match | Complexity |
|----------|---------|-------------|------------|
| Real-time TTS | 150-300ms | ✅ Perfect | Low |
| Pre-recorded audio files | ~0ms | ❌ Mismatch | Low |
| **Voice Clone Cache** | ~0ms | ✅ Perfect | Medium |
| Client-side signaling | ~0ms | N/A | Medium |

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  FILLER AUDIO APPROACHES COMPARISON                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. REAL-TIME TTS                     2. PRE-RECORDED FILES                  │
│  ════════════════                     ═════════════════════                  │
│                                                                              │
│  Request ──▶ TTS GPU ──▶ Audio        Request ──▶ Disk ──▶ Audio             │
│              │                                    │                          │
│              ▼                                    ▼                          │
│         150-300ms                             ~5-10ms                        │
│         Voice: ✓ Match                        Voice: ✗ Mismatch              │
│         Cost: GPU cycles/req                  Cost: Disk I/O                 │
│                                                                              │
│  Problem: Filler takes as long        Problem: Different voice               │
│  as the response itself               breaks user immersion                  │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  3. VOICE CLONE CACHE (Recommended)   4. CLIENT-SIDE SIGNAL                  │
│  ══════════════════════════════════   ═════════════════════                  │
│                                                                              │
│  Startup: TTS GPU ──▶ RAM Cache       Server ──▶ {"status":"thinking"}       │
│                                                        │                     │
│  Runtime: Request ──▶ RAM ──▶ Audio            Client plays local audio      │
│                       │                                │                     │
│                       ▼                                ▼                     │
│                   ~0ms (memcpy)                    ~0ms network              │
│                   Voice: ✓ Match                  Voice: N/A                 │
│                   Cost: None                      Cost: None                 │
│                                                                              │
│  Best of both: Speed of files,        Best when you control the              │
│  quality of TTS                       client application                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

#### Real-Time TTS Generation

The naive approach: when latency is detected, send filler text to TTS.

```python
async def send_filler(self, tts_client):
    filler_text = random.choice(["Hmm...", "Let me think.", "One moment."])
    audio = await tts_client.synthesize(filler_text)  # 150-300ms
    return audio
```

**Problem:** The filler itself takes 150-300ms to generate. If your pipeline latency is 300ms, the filler arrives at the same time as the actual response. Under load when TTS is saturated, filler generation adds to the queue.

---

#### Pre-Recorded Audio Files

Store `.wav` files and play them directly:

```python
FILLER_FILES = {
    "hmm": load_wav("assets/fillers/hmm.wav"),
    "thinking": load_wav("assets/fillers/thinking.wav"),
}

def get_filler(self):
    return random.choice(list(FILLER_FILES.values()))
```

**Problem:** Voice mismatch. If your TTS engine produces a specific voice (cloned or selected), pre-recorded files from a different voice/speaker break immersion. Users notice the inconsistency.

---

#### Voice Clone Cache (Recommended)

The Voice Clone Cache combines the speed of pre-recorded files with the voice consistency of TTS:

1. **At startup:** Generate filler phrases using your production TTS engine
2. **Store in RAM:** Keep the raw PCM/WAV bytes in memory
3. **At runtime:** Serve cached bytes with zero inference cost

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    VOICE CLONE CACHE ARCHITECTURE                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STARTUP PHASE (Once per pod lifecycle)                                      │
│  ═══════════════════════════════════════                                     │
│                                                                              │
│      ┌─────────────┐         ┌─────────────┐         ┌─────────────────┐    │
│      │   Phrases   │         │  TTS Engine │         │   RAM Cache     │    │
│      │   Config    │────────▶│   (GPU)     │────────▶│   (dict)        │    │
│      │             │         │             │         │                 │    │
│      │ "Hmm..."    │  5 req  │  Synthesize │  Store  │ {"Hmm": bytes,  │    │
│      │ "Let me..." │ ──────▶ │  each phrase│ ──────▶ │  "Let me": ...} │    │
│      │ "One moment"│  ~1 sec │             │         │                 │    │
│      └─────────────┘         └─────────────┘         └─────────────────┘    │
│                                                                              │
│                              Cost: ~1 second                                 │
│                              Frequency: Once at startup                      │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  RUNTIME PHASE (Every request needing filler)                                │
│  ════════════════════════════════════════════                                │
│                                                                              │
│      ┌─────────────┐                             ┌─────────────────┐         │
│      │  Latency    │    Threshold exceeded?      │   RAM Cache     │         │
│      │  Estimator  │────────────────────────────▶│   (dict)        │         │
│      │             │         │                   │                 │         │
│      │ queue_depth │    Yes  │    O(1) lookup    │ {"Hmm": bytes}──┼──▶ Audio│
│      │ > threshold │ ───────▶│    No GPU call    │                 │   Stream│
│      └─────────────┘         │                   └─────────────────┘         │
│                              │                                               │
│                              Cost: ~0ms (memory read)                        │
│                              Frequency: Per-request (when needed)            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

```python
class FillerAudioCache:
    def __init__(self, tts_client):
        self.tts_client = tts_client
        self.cache: dict[str, bytes] = {}
        self.phrases = [
            "Hmm...",
            "Let me think.",
            "One moment.",
            "Just a second.",
            "Alright...",
        ]
    
    async def initialize(self):
        """Generate fillers at startup. ~1 second total."""
        for phrase in self.phrases:
            audio_bytes = await self.tts_client.synthesize(phrase)
            self.cache[phrase] = audio_bytes
    
    def get_random_filler(self) -> bytes:
        """O(1) RAM lookup. Zero inference."""
        phrase = random.choice(self.phrases)
        return self.cache[phrase]
```

**Key properties:**
- **Zero runtime latency:** Memory read, not GPU inference
- **Voice-matched:** Generated by the same TTS engine/voice as responses
- **One-time cost:** Pay for TTS generation once at startup, not per-request

---

#### When to Send Fillers

Sending fillers on every request degrades UX—a 200ms response prefaced with "Hmm..." feels artificial. Use adaptive triggering:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE FILLER TRIGGERING                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          ┌───────────────┐                                   │
│                          │  ASR Final    │                                   │
│                          │  Transcript   │                                   │
│                          └───────┬───────┘                                   │
│                                  │                                           │
│                                  ▼                                           │
│                    ┌─────────────────────────────┐                           │
│                    │   Estimate Expected Latency │                           │
│                    │   ─────────────────────────  │                           │
│                    │   • LLM queue depth          │                           │
│                    │   • GPU utilization %        │                           │
│                    │   • Rolling P95 latency      │                           │
│                    └─────────────┬───────────────┘                           │
│                                  │                                           │
│                                  ▼                                           │
│                    ┌─────────────────────────────┐                           │
│                    │  estimated_latency > 500ms? │                           │
│                    └─────────────┬───────────────┘                           │
│                                  │                                           │
│               ┌──────────────────┼──────────────────┐                        │
│               │ YES              │                  │ NO                     │
│               ▼                  │                  ▼                        │
│    ┌──────────────────┐          │       ┌──────────────────┐                │
│    │ Send Filler      │          │       │ Skip Filler      │                │
│    │ ────────────────  │          │       │ ────────────────  │                │
│    │ Stream cached    │          │       │ Proceed directly │                │
│    │ audio bytes      │          │       │ to LLM call      │                │
│    │ (~300ms audio)   │          │       │                  │                │
│    └────────┬─────────┘          │       └────────┬─────────┘                │
│             │                    │                │                          │
│             └────────────────────┴────────────────┘                          │
│                                  │                                           │
│                                  ▼                                           │
│                    ┌─────────────────────────────┐                           │
│                    │   Continue Pipeline         │                           │
│                    │   LLM → TTS → Audio         │                           │
│                    └─────────────────────────────┘                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

```python
FILLER_THRESHOLD_MS = 500

async def maybe_send_filler(self, estimated_latency_ms: int) -> Optional[bytes]:
    """Only send filler when latency is noticeable."""
    if estimated_latency_ms > FILLER_THRESHOLD_MS:
        return self.filler_cache.get_random_filler()
    return None
```

Latency estimation approaches:
- **Queue depth:** If LLM request queue > N, expect delays
- **Rolling average:** Track recent P95 latency
- **GPU utilization:** If > 80%, expect queueing

---

#### Scaling Behavior

A common concern: does filler generation at startup cause problems during scale-out?

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              SCALING: FILLER CACHE INITIALIZATION (1 → 5 Pods)               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BEFORE SCALE (Normal operation)                                             │
│  ═══════════════════════════════                                             │
│                                                                              │
│      ┌─────────┐                           ┌─────────────────┐               │
│      │  Pod 1  │                           │   TTS Service   │               │
│      │ ─────── │   Production TTS          │   (GPU)         │               │
│      │ Cache ✓ │──────────────────────────▶│                 │               │
│      │ Ready   │   ~11 req/s sustained     │   Handling      │               │
│      └─────────┘                           │   normal load   │               │
│                                            └─────────────────┘               │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  DURING SCALE-UP (HPA triggers 1 → 5)                                        │
│  ════════════════════════════════════                                        │
│                                                                              │
│      ┌─────────┐                                                             │
│      │  Pod 1  │────┐                      ┌─────────────────┐               │
│      │ Cache ✓ │    │   Production TTS     │   TTS Service   │               │
│      └─────────┘    ├─────────────────────▶│   (GPU)         │               │
│                     │                      │                 │               │
│      ┌─────────┐    │   5 phrases each     │   Receives:     │               │
│      │  Pod 2  │────┼─────────────────────▶│   • Prod load   │               │
│      │ Init... │    │   (25 total)         │   • 25 filler   │               │
│      └─────────┘    │                      │     requests    │               │
│                     │   ~2-3 sec burst     │                 │               │
│      ┌─────────┐    │                      │   Impact:       │               │
│      │  Pod 3  │────┼─────────────────────▶│   Equivalent to │               │
│      │ Init... │    │                      │   ~2 sec of     │               │
│      └─────────┘    │                      │   prod traffic  │               │
│                     │                      │                 │               │
│      ┌─────────┐    │                      │                 │               │
│      │  Pod 4  │────┼─────────────────────▶│                 │               │
│      │ Init... │    │                      │                 │               │
│      └─────────┘    │                      └─────────────────┘               │
│                     │                                                        │
│      ┌─────────┐    │                                                        │
│      │  Pod 5  │────┘                                                        │
│      │ Init... │                                                             │
│      └─────────┘                                                             │
│                                                                              │
│  ────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  AFTER SCALE (All pods ready)                                                │
│  ════════════════════════════                                                │
│                                                                              │
│      Pod 1 ─────┐                          ┌─────────────────┐               │
│      Cache ✓    │                          │   TTS Service   │               │
│                 │                          │   (GPU)         │               │
│      Pod 2 ─────┤   Production TTS only    │                 │               │
│      Cache ✓    │   (no filler gen)        │   Normal load   │               │
│                 ├─────────────────────────▶│   distributed   │               │
│      Pod 3 ─────┤                          │   across 5 pods │               │
│      Cache ✓    │                          │                 │               │
│                 │                          │                 │               │
│      Pod 4 ─────┤                          └─────────────────┘               │
│      Cache ✓    │                                                            │
│                 │                                                            │
│      Pod 5 ─────┘                                                            │
│      Cache ✓                                                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

Consider a Kubernetes deployment with HPA scaling from 1 → 5 pods:

```
T+0s:  Load spike detected
T+30s: HPA triggers scale-up
T+31s: 5 new pods starting simultaneously
       Each pod initializes filler cache:
       - 5 phrases × ~200ms TTS = ~1 second per pod
       - All 5 pods hit TTS concurrently = 25 requests
T+33s: All pods have cached fillers, ready to serve
```

**Impact analysis:**
- 25 short-phrase TTS requests = trivial workload
- Equivalent to ~2-3 seconds of normal production traffic
- New pods don't receive traffic until ready (K8s readiness probes)
- Existing pods continue serving unaffected

The one-time startup cost is negligible compared to ongoing production TTS load.

---

#### Client-Side Signaling Alternative

If you control the frontend application, consider not sending audio at all:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     CLIENT-SIDE SIGNALING ARCHITECTURE                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   SERVER                              NETWORK              CLIENT            │
│   ══════                              ═══════              ══════            │
│                                                                              │
│  ┌────────────┐                                        ┌────────────────┐   │
│  │  Voice     │     {"status": "thinking"}             │  Mobile App /  │   │
│  │  Gateway   │───────────────────────────────────────▶│  Web Client    │   │
│  │            │         ~50 bytes                      │                │   │
│  │  Detects   │         (JSON signal)                  │  Receives      │   │
│  │  latency   │                                        │  signal        │   │
│  │  threshold │                                        │       │        │   │
│  └────────────┘                                        │       ▼        │   │
│                                                        │  ┌──────────┐  │   │
│                                                        │  │ Local    │  │   │
│                 COMPARE: Audio stream                  │  │ Audio    │  │   │
│                 ─────────────────────                  │  │ Files    │  │   │
│                 ~50KB for 1 sec audio                  │  │ (bundled)│  │   │
│                 Network dependent                      │  └────┬─────┘  │   │
│                 Can drop packets                       │       │        │   │
│                                                        │       ▼        │   │
│                                                        │  Play locally  │   │
│                                                        │  (0ms latency) │   │
│                                                        └────────────────┘   │
│                                                                              │
│   Benefits:                                                                  │
│   • Zero bandwidth for filler audio                                         │
│   • Instant playback (no network RTT)                                       │
│   • Works with high packet loss                                             │
│   • Can trigger haptic feedback                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

```python
# Server sends signal, not audio
async def notify_thinking(self, websocket):
    await websocket.send(json.dumps({"status": "thinking"}))
```

```javascript
// Client plays local audio
socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.status === "thinking") {
        playLocalSound("thinking.mp3");  // Already on device
    }
};
```

**Advantages:**
- Zero bandwidth for filler audio
- Instant playback (no network latency)
- Works even with packet loss
- Can trigger haptic feedback on mobile

**Disadvantage:** Requires client modification; not applicable for telephony/PSTN integrations.

---

#### Comfort Noise as Alternative

Instead of linguistic fillers ("Hmm..."), non-linguistic audio avoids repetition fatigue:

- **Breathing sounds:** Subtle "intake of breath" signals "about to speak"
- **Room tone:** Low-volume ambient noise keeps the line "alive"
- **Typing sounds:** For text-based contexts, keyboard clicks indicate processing

These loops are tiny (KB-sized), can be extremely low bitrate, and don't require TTS generation.

---

#### Implementation Checklist

1. **Choose filler phrases:** 5-10 variations to avoid repetition
2. **Generate at startup:** Initialize cache before accepting traffic
3. **Match audio format:** Same sample rate (16kHz/24kHz), encoding (PCM/Opus) as TTS output
4. **Add latency estimation:** Only trigger when delay > threshold
5. **Monitor:** Track filler send rate to understand real-world latency distribution

---

#### Summary

| If your problem is... | Solution |
|----------------------|----------|
| Users don't know they were heard | Visual feedback / client signals |
| 1-2 second silence feels broken | Voice Clone Cache |
| Latency varies unpredictably | Threshold-based filler triggering |
| 5+ second latency is common | Scale GPU capacity (root cause) |

The Voice Clone Cache pattern provides the best balance: zero-latency playback with perfect voice consistency. The one-time startup cost (~1 second) is negligible, and the approach scales cleanly with horizontal pod scaling.

Filler audio is a UX optimization, not a fix for underlying throughput problems. If your P95 latency is consistently > 2 seconds, address GPU capacity first.

