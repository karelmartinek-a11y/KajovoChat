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
- test pro synteticky echo scenar
- integrace do `probe_windows_native_aec()`

## Jak to otevrit ve VS Code

1. Otevri repozitar v `VS Code`.
2. Spust task `Native APO: Configure`.
3. Spust task `Native APO: Build`.
4. Spust task `Native APO: Test`.
5. Vystupni DLL hledej v `native/windows_apo_helper/build/bin/Release/`.

## Poznamka k funkcnosti

 Tenhle helper je prvni mezikrok k APO ceste. Nezapojuje se jeste do skutecneho
systemoveho APO registracniho mechanismu. Slouzi jako samostatna host vrstva a
uz dnes umi echo cancellation pres stejny C ABI jako AEC helper.
