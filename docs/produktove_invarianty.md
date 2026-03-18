# Produktové invarianty

Tento dokument popisuje produktový kontrakt, který má aplikace držet.

## Jazyk odpovědi

Existují jen dva režimy:

- `follow_input`
  - asistent odpovídá ve stejném jazyce, jakým mluví nebo píše uživatel
- `fixed`
  - asistent odpovídá vždy ve zvoleném jazyce bez ohledu na jazyk vstupu

Rozpoznávání vstupu zůstává na autodetekci.

## Styl odpovědi

Existují jen tři styly:

- `stručný`
- `vědecký_s_analýzou`
- `normální`

V produktu nesmí být natvrdo zadrátovaná cizí persona ani skryté sarkastické chování.

## Runtime audio profil

Následující chování je řízené interně, ne běžným uživatelským UI:

- realtime model
- hlas
- rychlost výstupu
- `server_vad`
- `far_field` noise reduction
- VAD threshold, prefix a silence
- výběr audio zařízení přes systémový default a interní heuristiku

## Hands-free invarianty

- barge-in musí zůstat povolený i během playbacku asistenta
- self-hearing musí být softwarově tlumený kombinací playback guardu a echo similarity
- lipsync musí vycházet ze skutečně přehrávaného audia, ne z textu ani síťových chunků
