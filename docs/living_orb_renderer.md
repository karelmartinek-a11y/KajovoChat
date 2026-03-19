# Living Orb Renderer

## Instalace

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

`living orb` používá `PySide6` a `moderngl`. V test/offscreen prostředí přepne widget na bezpečný 2D fallback, ale produkční cesta je GPU přes `QOpenGLWidget`.
Při startu se provede Qt offscreen probe OpenGL contextu. Když neprojde, aplikace nepoužije GPU widget a rovnou přepne na fallback backend s diagnostickou hláškou.

## Demo

```bash
python -m kajovochat.orb_demo
```

Klávesy:

- `1` idle
- `2` listening
- `3` thinking
- `4` speaking
- `Space` zapne nebo vypne scripted cycle

## Integrace

Veřejné API drží třída `OrbEngine`:

```python
from kajovochat.orb import OrbEngine

engine = OrbEngine()
engine.set_state("listening")
engine.push_audio_frame(samples, sample_rate=24000)
engine.set_audio_features({"loudness": 0.4, "speaking_gate": 1.0})
engine.update(1.0 / 60.0)
engine.render()
```

Pravidlo priority:

- `set_audio_features(...)` přepíše nejbližší následující `update()`
- po tomto `update()` se engine vrátí k datům z `push_audio_frame(...)`

## Moduly

- `kajovochat/orb/config.py`: centrální parametry, barvy a state profily
- `kajovochat/orb/audio.py`: feature extraction, smoothing, hold a VAD-like gate
- `kajovochat/orb/state.py`: eased blend mezi `idle`, `listening`, `thinking`, `speaking`
- `kajovochat/orb/controller.py`: mapování stavů a audio feature do shader parametrů
- `kajovochat/orb/renderer.py`: `moderngl` renderer a shader management
- `kajovochat/orb/widget.py`: Qt host widget pro aplikaci a demo

## Klíčové konfigurační hodnoty

- `core_radius`, `aura_radius`
- `shell_deformation_strength`
- `domain_warp_primary`, `domain_warp_secondary`
- `attack_seconds`, `release_seconds`
- `speaking_hold_seconds`
- `state_transition_seconds`
- `low_band_weight`, `mid_band_weight`, `high_band_weight`
- barevné téma v `background_color`, `core_color`, `glow_color`, `aura_color`
