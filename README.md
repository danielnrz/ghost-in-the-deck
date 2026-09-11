# Ghost in the Deck

A local, autonomous personal DJ with a full-body 3D avatar and a two-deck
workstation. Give it a folder of songs: it analyzes the collection, chooses
compatible next tracks, performs transitions, and keeps playing through the
library. The avatar follows musical context and actual audio decisions.
Everything runs on your machine, with no accounts, paid APIs, or cloud services.

## Setup

Linux is the validated platform. Use Python 3.12 or 3.13, an OpenGL-capable display,
an audio output device, and **ffmpeg with the `rubberband` filter**. The bundled
avatar is ready to use; Blender is needed only to rebuild it.

On Debian/Ubuntu, install system dependencies with:

```bash
sudo apt install python3-venv ffmpeg libgl1 libopenal1
ffmpeg -hide_banner -filters 2>/dev/null | grep rubberband
```

From this checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

`requirements.lock` pins the complete validated environment, including test
tools. `requirements.txt` pins the direct runtime dependencies if you prefer a
smaller installation; `requirements-dev.txt` adds pytest.

## Play a collection

```bash
./run.sh --music-dir "$HOME/Music"
```

Folders are searched recursively for WAV, FLAC, MP3, M4A, AAC, OGG, and Opus.
Symbolic/hardlink duplicates are omitted. Analysis runs before the window opens;
the terminal shows progress. Later launches reuse content-validated analysis.
Broken files are reported and skipped. With one usable song, that song plays;
with several, the set plays each once and exits at the end.

To try the complete system without supplying music:

```bash
./run.sh --demo
```

This synthesizes three short instrumental tracks locally in a temporary folder,
plays an automatic set, and cleans up the temporary files on exit.

The window shows the avatar and workstation, the active deck, source position,
measured tempo, and the next track while mixing. The avatar briefly reaches toward the incoming deck as a real transition
starts, monitors the blend, then acknowledges the handoff and settles. Deck A
is the avatar's left; deck B is its right. Labels on the workstation match the
status display.

Control gestures consume effects actually rendered into the committed audio
and the real transition ledger. They never schedule audio. Most playback is
restrained groove: measured short/broad energy and trend change body intensity,
with relaxed articulated hands. Real effects get a small knob turn; transition
starts get a button tap or a platter-side cue check that releases as playback starts.
A sustained measured energy rise can earn a rare cheer and small bounce, with
at least one song skipped afterward and a 90-second minimum gap. Effects can be
represented on the owning deck, with
12-second minimum spacing, 24 seconds before the same effect repeats, and six
seconds of settling after handoff. Disabling effects also removes their control
gestures. Brief contact follows a smooth approach and recovery; the body blends
across source handoffs without moving the audio boundary.

| Control or option | Meaning |
| --- | --- |
| Esc / close window / Ctrl-C | End playback and shut down |
| M | Mute/unmute |
| Up / Down | Output volume (starts at 80%) |
| `--track NAME` | Start with a filename containing NAME |
| `--transition-bars 4` | Blend length in assumed four-beat bars; default 4 |
| `--dwell 20` | Minimum solo seconds before another blend; default 20 |
| `--seconds 60` | End after 60 seconds of set playback |
| `--cache-dir PATH` | Analysis cache root; default `~/.cache/ghost-in-the-deck` |
| `--refresh` | Recompute analysis |
| `--no-fx` | Disable solo filter sweeps/gain risers; keep transitions |
| `--no-actions` | Groove only; disables solo effects and action gestures |
| `--headless` | Offscreen rendering; still needs a working graphics display |
| `--no-audio` | Silent runtime exercise using the same set engine |

Pause/seek, manual deck control, and endless repeat are not implemented.
Preparation may finish its current decode before shutdown returns.

## How the mixing works

Selection ranks feasible candidates by measured tempo difference, local energy
continuity, and stable cue positions. Ties are deterministic; already played
tracks are removed. It prefers an outgoing cue after 65% of the song and an
incoming cue in the first 35%. These are explicit policies, not detected song
sections. Bars assume four beats; periodic phrase position is an assumption.
There is no claimed key, chorus, verse, or drop detection.

For a blend, the incoming window is pitch-preservingly stretched by local
Rubber Band to the outgoing tempo. The direct playback-rate ratio must remain
within **0.80–1.25**. Detected beat timestamps anchor the cues, with the accepted
initial integer sample correction (at most one sample). This is initial phase
alignment, not continuous beat tracking or correction of variable-tempo songs.
After the blend the incoming track returns to native tempo, with a 20 ms boundary
smoothing ramp; this tempo change can be audible on widely separated BPMs.

The linear crossfade and sample ledger share exact boundaries. Incoming ownership
passes once at the end; the next transition starts from that source position.
A preparation thread supplies a bounded queue to Panda3D/OpenAL with up to 30
seconds buffered. Playback time drives the scene; rendering does not schedule
mixes. Slow or incompatible candidates are rejected before their audio is queued.
When no supported blend is possible, the current song finishes with a short fade
and the next starts with a short fade, explicitly without a beat-match claim.

Mono sources become stereo and sample rates are converted to 44.1 kHz. Peak gain
is reduced when necessary after conversion, effects, and stretching. The 0.85
sample ceiling and complementary linear gains avoid integer output clipping.
This is peak management, not loudness mastering, true-peak limiting, or perceptual
quality optimization. Existing distortion in a recording cannot be repaired.

The avatar is a simple untextured anatomical mannequin with a calibrated fixed
rig and workstation. Its gestures represent planned actions; they never cause
an audio decision. The design is functional rather than a finished character
art treatment. Final musical taste and animation appeal remain subjective.

## Checks and local smoke test

```bash
PYTHONPATH=src:tests .venv/bin/python -m pytest -q
PYTHONPATH=src:tests .venv/bin/python -m pytest tests/test_set_engine.py::test_synthetic_library_to_two_real_transitions -q
.venv/bin/python scripts/git_guard.py
```

Set `DISPLAY` to a working local X display to include rendered tests. Without a
display, pixel-dependent tests skip; a Linux CI runner can use `xvfb-run -a`.
The synthetic end-to-end test goes through discovery, actual analysis, selection,
planning, PCM execution, ownership transfer, and a second transition. It also
checks source bytes and exact repeatability after reloading cached analysis.

Private local music smoke test, from this checkout:

```bash
./run.sh --music-dir ./testMusic --seconds 90
```

This never writes to the songs. `testMusic` is ignored; automated tests do not
read it. Analysis and decode caches are disposable. Do not place private songs
or rendered audio in tracked directories.

For actual rendered transition review:

```bash
DISPLAY=:1 .venv/bin/python scripts/render_set.py
```

This synthesizes a library, consumes the production set engine, and renders both
transition directions, low/high solo passages, effects, and handoff/recovery
sequences into ignored `out/set_review`, with frame and behavior JSON manifests. Use your actual display
instead of `:1`. The live runtime can also save frames with
`--capture-at 10 40 --capture-dir out/set_review`.

See [live performance policies and review commands](docs/visual-behavior.md)
for the hand variants and rare musical accents.

The earlier single-track diagnostics remain available via `--single-track`;
`--show-action hand_to_deck` selects an isolated gesture review. Beat reports,
`--verbose`, and `--debug-motion` are legacy single-track diagnostic options.
The offline pair preview remains available with
`PYTHONPATH=src .venv/bin/python -m ghost_in_the_deck.transition_preview --help`.
See [historical engineering notes](docs/engineering-history.md) for the accepted
DSP, timing, rig calibration, and synthetic quality evaluator contracts.
