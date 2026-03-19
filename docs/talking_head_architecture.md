# Talking Head Architecture

## Přehled

Talking head je nově rozdělený do tří vrstev:

- `kajovochat/animation/`
  Zpracování audio-driven motion dat nad skutečně přehrávaným audiem.
- `kajovochat/widgets/talking_head_renderer.py`
  Samostatný renderer, který převádí `PerformanceFrame` a rig definici na kreslení do `QPainter`.
- `kajovochat/widgets/talking_head_widget.py`
  Qt widget kompatibilní s dosavadním `HeadWidget`, který skládá runtime stav, timer loop a volání rendereru.

## Datový tok

1. `AudioPlayer` průběžně odebírá playback PCM16 a předává ho do `LipSyncEngine`.
2. `LipSyncEngine` vrací kompatibilní snapshot pro stávající UI a současně umí bohatší viseme data.
3. `ConversationWorker.output_pose` dál posílá snapshot do hlavního okna.
4. `TalkingHeadWidget.set_lipsync_snapshot(...)` snapshot převede na `VisemeFrame`.
5. `PerformanceDriver` při každém ticku spojí:
   - stav aplikace
   - input/output level
   - lipsync frame
   - blink, gaze a head motion enginy
6. Renderer vykreslí hlavu podle `PerformanceFrame` a rig definice.

## Fallback vs production rig

Rig definice se načítá z `kajovochat/resources/talking_head_manifest.json`.

- `layers.fallback`
  Minimální bezpečný rig nad `head_photo.png`.
- `layers.production`
  Budoucí vrstvený rig s oddělenými assety.

Technická specifikace production assetů je v `docs/talking_head_asset_spec.md`.

Pokud production vrstvy nejsou dostupné nebo nejsou kompletní, runtime automaticky přejde do fallback režimu. Pokud selže samotná inicializace `TalkingHeadWidget`, `main.py` se vrátí na původní `HeadWidget`.

## Integrace v aplikaci

`kajovochat/main.py` nyní používá továrnu na head widget:

- preferovaně `TalkingHeadWidget`
- při chybě nebo při feature flagu `KAJOVOCHAT_HEAD_WIDGET=legacy` původní `HeadWidget`

Zachované API body:

- `set_state(...)`
- `set_running(...)`
- `set_input_level(...)`
- `set_output_level(...)`
- `set_lipsync_snapshot(...)`
- `set_error_text(...)`

## Runtime fallback při chybách

- chybějící production assety:
  talking head běží ve fallback rigu a aplikace pokračuje
- nevalidní manifest nebo chyba inicializace:
  chyba se zaloguje a `main.py` přepne vykreslování na původní `HeadWidget`
- startup notice:
  hlavní okno zobrazí textový kontext v captions a tooltip na head widgetu
