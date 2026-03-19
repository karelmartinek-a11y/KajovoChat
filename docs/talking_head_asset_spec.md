# Talking Head Asset Spec

## Účel

Tento dokument popisuje minimální technické požadavky na production layered rig, aby šel zapojit do současné architektury bez změny kódu.

Aktuální repo production vrstvy neobsahuje. Fallback rig je hotový a funkční, ale filmová produkční kvalita bude vyžadovat nové vrstvené assety podle této specifikace.

## Povinné vrstvy production rigu

Minimální sada vrstev:

- `head_base`
  Základ hlavy bez dynamických úst.
- `eyes`
  Oči a víčka jako samostatná vrstva pro jemné gaze posuny a blink zásahy.
- `mouth_upper`
  Horní ret a okolí.
- `mouth_lower`
  Dolní ret a spodní část úst.

Doporučené rozšiřující vrstvy:

- `mouth_interior`
- `teeth_upper`
- `teeth_lower`
- `tongue`
- `cheek_shadow`
- `specular_skin`
- `specular_lips`

## Naming convention

Soubory mají být v `kajovochat/resources/assets/talking_head/`.

Povinné názvy:

- `head_base.png`
- `eyes.png`
- `mouth_upper.png`
- `mouth_lower.png`

Doporučené názvy rozšiřujících vrstev:

- `mouth_interior.png`
- `teeth_upper.png`
- `teeth_lower.png`
- `tongue.png`
- `cheek_shadow.png`
- `specular_skin.png`
- `specular_lips.png`

Role v manifestu musí odpovídat významu vrstvy. Názvy je vhodné držet stabilní, aby nebylo nutné upravovat kód ani testy.

## Jednotné rozměry pláten

Všechny production vrstvy musí mít shodné rozměry plátna.

Současný runtime očekává:

- jednotné plátno `1024x1024`
- stejný ořez, stejný framing a stejný referenční head center

Pokud budou dodány jiné rozměry, musí být promítnuty do manifestu a vrstvy musí stále zůstat vzájemně přesně zarovnané.

## Pivoty

Manifest musí mít konzistentní referenční body:

- `head_center`
- `eye_left`
- `eye_right`
- `mouth_center`

Vrstvy musí být exportované tak, aby tyto pivoty seděly ve stejných normalizovaných souřadnicích napříč všemi PNG.

## Alpha pravidla

- export jako `RGBA` s čistou alfou
- bez barevných lemů v plně transparentních pixelech
- bez premultiplication artefaktů z exportu
- poloprůhledné hrany musí být čisté a stabilní

Není přijatelné:

- bílý nebo černý fringe kolem rtů a víček
- rozdílný matte color mezi vrstvami
- nečistý background uvnitř alfa kanálu

## Shadow a specular vrstvy

Pokud jsou použité:

- musí mít stejné plátno jako ostatní vrstvy
- nesmí obsahovat baked geometrický posun mimo definovaný pivot
- intenzita musí být jemná a vrstvy musí fungovat i při opacity pod `0.25`

Doporučení:

- `cheek_shadow` jen pro jemné modelování tváří
- `specular_skin` pro čelo, nos a líce
- `specular_lips` jen pro ret, ne pro zuby

## Alignment pravidla

- všechny vrstvy musí být exportované z jednoho master souboru bez změny framingu
- head center musí být identický
- mouth oblast musí přesně sedět mezi `mouth_upper` a `mouth_lower`
- oči nesmí měnit absolutní polohu mezi vrstvami

## Export pravidla

- `PNG`, `8-bit RGBA`
- bez barevného pozadí
- bez automatického trimu
- bez resize při exportu jednotlivých vrstev
- bez odlišného sharpeningu nebo odlišné gamma korekce mezi vrstvami

## Minimální technické požadavky pro zapojení bez změny kódu

Pro zapojení bez zásahu do Python kódu musí být splněno:

1. vrstvy jsou fyzicky dodané v `kajovochat/resources/assets/talking_head/`
2. názvy souborů odpovídají naming convention z tohoto dokumentu
3. všechny vrstvy mají jednotné plátno
4. pivoty a alignment sedí s hodnotami v `talking_head_manifest.json`
5. alpha kanál je čistý
6. production větev v manifestu ukazuje na reálné soubory

Pokud bude potřeba jiné názvosloví nebo jiné rozměry, je nutná úprava manifestu. Pokud nebudou sedět pivoty a alignment, samotný manifest nestačí a bude potřeba i další ladění renderu.
