# Windows native AEC helper

Tahle slozka obsahuje zdrojovy skeleton pro budoucni nativni Windows helper,
který muze obslouzit echo cancellation mimo Python vrstvu.

## Co je tu zatim pripraveno

- CMake projekt pro DLL `kajovochat_windows_aec`
- C ABI kontrakt, ktery ceka Python bridge
- prvni funkční time-domain NLMS prototyp pro echo cancellation
- VS Code konfigurace pro sestaveni helperu z repozitare

## Jak to otevrit ve VS Code

1. Otevri repozitar v `VS Code`.
2. Spust task `Native AEC: Configure`.
3. Spust task `Native AEC: Build`.
4. Spust task `Native AEC: Test`.
5. Vystupni DLL hledej v `native/windows_aec_helper/build/bin/Release/` nebo
   podle generatoru v odpovidajicim config adresari.

Pokud chces test spustit rucne, pouzij:

```powershell
ctest --test-dir native/windows_aec_helper/build -C Release --output-on-failure
```

## Co je potreba mit nainstalovane

- CMake
- MSVC nebo Visual Studio Build Tools s C++ workloadem
- Windows SDK

## Poznamka k funkcnosti

Helper je zatim prototyp, ne plnohodnotny systemovy APO. Z pohledu Python bridge
ale uz poskytuje skutecne potlaceni echo a stejny C ABI kontrakt, takze jej lze
dale vylepsovat bez zmen v aplikaci.
