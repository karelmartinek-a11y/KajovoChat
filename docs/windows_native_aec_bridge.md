# Windows native AEC bridge

Datum: 2026-03-23

## Cíl

Připravit první volbu pro AEC na Windows tak, aby aplikace mohla využít
nativní capture pipeline nebo helper postavený nad Windows audio stackem.

Současný stav je hybridní:

- `windows_native_preferred` je výchozí režim
- pokud není nativní helper dostupný, runtime padá na `webrtc_preferred`
- `custom_only` zůstává poslední nouzová cesta

## Co je připravené v Pythonu

V repozitáři už existuje tenká adapter vrstva:

- [`kajovochat/services/windows_native_aec.py`](../kajovochat/services/windows_native_aec.py)
- detekce helperu přes `KAJOVOCHAT_WINDOWS_AEC_DLL`
- přepínání backendů podle `audio_aec_mode`
- fallback na WebRTC, pokud helper není dostupný

Současně je v repozitáři připraven i zdrojový skeleton helperu:

- [`native/windows_aec_helper/`](../native/windows_aec_helper/)
- [`native/windows_apo_helper/`](../native/windows_apo_helper/)
- CMake projekt pro DLL
- VS Code tasky pro konfiguraci a build
- stub exporty se stejným C ABI, jaký očekává Python bridge

## Očekávaný nativní helper

Helper DLL má být samostatná Windows binárka. Python část očekává tyto symboly:

- `kajovochat_aec_create(int samplerate, int filter_length, int max_shift_samples) -> void*`
- `kajovochat_aec_destroy(void* handle)`
- `kajovochat_aec_process(void* handle, mic, mic_samples, reference, reference_samples, delay_ms, out_pcm, out_capacity, out_quality, out_improvement, out_residual, out_is_strong) -> int`

Helper v repozitáři už existuje jako první time-domain NLMS prototyp a vedle
něj je připraven i APO skeleton s totožným ABI. Není to hotový systémový APO,
ale je to skutečně funkční echo canceler s pevným C ABI, který se dá dále
vylepšovat bez změny Python bridge.

## Fázování

1. Přepnout default do `windows_native_preferred`.
2. Připravit nativní bridge a detekci helperu.
3. Používat helper jako první backend, fallback na WebRTC.
4. Měřit backend zvlášť v `aec_diag` a `aec_summary`.
5. Dlouhodobě vyhodnotit, zda je nativní helper stabilnější než WebRTC fallback.

## Praktický závěr

Tahle vrstva je nyní připravená na nativní Windows AEC a repozitář už obsahuje
první funkční helper prototyp i samostatný APO skeleton. Není to systémový APO,
ale už to není jen skelet: helper umí skutečné echo potlačení a zachovává
stabilní rozhraní pro další růst.

Pro build helperu je potřeba mit nainstalovane CMake, MSVC/Visual Studio Build Tools a Windows SDK.
