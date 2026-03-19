# Talking Head Tuning

## Kde se ladí rig

Primární místo pro ladění je `kajovochat/resources/talking_head_manifest.json`.

Požadavky na skutečný production layered rig jsou popsané v `docs/talking_head_asset_spec.md`.

Nejdůležitější sekce:

- `canvas`
  Základní rozměry a bezpečné okraje.
- `layers`
  Fallback a production vrstvy.
- `pivots`
  Referenční body pro hlavu, oči a ústa.
- `deformation_ranges`
  Mouth a eye oblasti pro fallback deformace.
- `state_presets`
  Základní vizuální akcenty stavů.

## Kde se ladí motion kanály

Audio-driven motion se ladí hlavně v:

- `kajovochat/animation/viseme_engine.py`
  Kanály rtů a čelisti, cluster mapping, hysterese, hold a smoothing.
- `kajovochat/animation/blink_engine.py`
  Rytmus mrkání a suppression při speaking attacku.
- `kajovochat/animation/gaze_engine.py`
  Idle drift, mikro-saccady a speaking stabilizace.
- `kajovochat/animation/head_motion_engine.py`
  Idle breathing, listening lean, thinking offset a speaking nod.
- `kajovochat/animation/performance_driver.py`
  Převod aplikačních stavů na výsledný `PerformanceFrame`.

## Doporučený postup ladění

1. Nejprve dolaď `VisemeFrame` a kanály v `viseme_engine.py`.
2. Potom uprav `deformation_ranges.mouth`, aby fallback oblast seděla na fotografii.
3. Nakonec dolaď `head_motion_engine.py` a `state_presets`, aby motion nepůsobil přehnaně.

Při ladění fallbacku drž tyto zásady:

- neotevírat čelist za limity v manifestu
- nepřehánět corner stretch a upper lip raise
- speaking asymetrii držet jen jako mikro akcent, ne jako viditelný tik
- fokus ve speaking stabilizovat spíš tlumením driftu než velkým pohybem očí

## Runtime fallback

Fallback funguje ve dvou úrovních:

- production rig není kompletní:
  `TalkingHeadWidget` zůstane aktivní a vykresluje fallback rig
- manifest nebo inicializace talking head selže:
  `main.py` použije původní `HeadWidget`

Rychlý návrat na starý widget během stabilizace:

```powershell
$env:KAJOVOCHAT_HEAD_WIDGET = "legacy"
python -m kajovochat
```

Návrat na nový widget:

```powershell
Remove-Item Env:KAJOVOCHAT_HEAD_WIDGET
python -m kajovochat
```
