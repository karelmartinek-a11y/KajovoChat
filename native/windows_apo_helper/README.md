# Windows APO helper

Tahle slozka pripravi samostatny native projekt pro budoucni APO cestu ve
Windows. Zatim nejde o skutecny systemovy APO registracni balicek, ale o
oddeleny host/helper projekt, ktery:

- ma vlastni CMake build
- exportuje stejny C ABI kontrakt jako Windows AEC helper
- lze otevrit ve VS Code a sestavit pres tasky
- lze testovat samostatnym nativnim smoke testem

## Co je zatim pripraveno

- DLL `kajovochat_windows_apo`
- exporty `kajovochat_aec_*` pro kompatibilitu s existujicim bridge
- exporty `kajovochat_apo_capture_*` pro cistou capture-only APO cestu
- test pro synteticky echo scenar
- integrace do `probe_windows_native_aec()`

## Jak to otevrit ve VS Code

1. Otevri repozitar v `VS Code`.
2. Spust task `Native APO: Configure`.
3. Spust task `Native APO: Build`.
4. Spust task `Native APO: Test`.
5. Vystupni DLL hledej v `native/windows_apo_helper/build/bin/Release/`.

## Poznamka k funkcnosti

Tenhle helper je mezikrok k cistejsi APO ceste. Nezapojuje se jeste do plneho
systemoveho WASAPI/WDK capture kontraktu, ale uz exposeuje samostatny
capture-only ABI `kajovochat_apo_capture_*`, ktery nevyzaduje app-level
playback reference.
