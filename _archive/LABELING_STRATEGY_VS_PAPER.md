# Är vår etiketteringsstrategi vettig mot kmae250140.pdf (Sundin et al.)?

**Källa:** Pappret har lästs via extraherad text. PDF extraherades med `scripts/extract_pdf_text.py` (PyMuPDF) till **kmae250140.txt** (~9 400 ord). Denna bedömning bygger på den extraherade texten och er **faktiska strategi** i repot.

**Referens:** Sundin et al. (2026). *Insights from a nation-wide environmental relicensing of hydropower facilities in Sweden: a review of court verdicts from a biological perspective.* Knowl. Manag. Aquat. Ecosyst. 2026, 427, 6. https://doi.org/10.1051/kmae/2025034

---

## 1. Vad pappret beskriver (från kmae250140.txt)

- **33 NAP-domstolsfall** avslutade till slutet av 2024 (beslut med laga kraft): **22 permit withdrawal and dam removal**, **11 granted continued hydropower production** under villkor om moderna miljökrav.
- **Dataextraktion (Sektion 2.1):** Från domarna extraherades bl.a.: capacity (Q), effect (kW), MQ, MLQ, **upstream passage solutions**, **downstream passage solutions**, krav på **guidance** (typ, max vinkel alfa/beta rack), **max gap width** (rack), **discharge through downstream bypass**, **eel ramps** (ja/nej), **type of fishway**, **required slope of fishway**, **flow in fishway**, **hydropeaking restrictions**, **e-flow requirements**, **monitoring requirements**. För ål söktes "ål", "ålen", "ålyngel" och kontext noterades.
- **Tabell 2** (11 anläggningar som fick fortsatt drift): **Downstream** – Guidance, Angle (°), Gap width (mm), Bypass (L/s), Eel ramp. **Upstream** – Fishway (typ), Slope (%), Flow (L/s). **Flow** – Min. flow (L/s), No peaking, Reduced rate. Monitoring nämns i texten (begränsad i domarna).
- **Fokus:** Longitudinal connectivity och fiskpassage; miljöflöde får mindre uppmärksamhet; övervakningskrav nästan frånvarande. Rekommendationer: åtgärder med adaptiv design, funktionalitet och övervakning före detaljerade tekniska specifikationer.

Pappret ger alltså en **binär utfallstyp** (22 removal vs 11 permit) och en **tydlig variabellista** för vad som extraherats – inte en färdig "risk"-skala.

---

## 2. Er nuvarande etiketteringsstrategi (så som den är implementerad)

- **Tre klasser:** HIGH_RISK / MEDIUM_RISK / LOW_RISK.
- **Domstolsbeslut:** Etikett sätts utifrån **domslut** (utfall), kostnader, åtgärder, tidslinjer – inte enbart "removal vs permit".
- **Sundin:** Används som **feature-taxonomi** i `scripts/sundin_feature_extraction.py`. **Trösklar och vikter** lärs från era 40/44 märkta beslut (hybriden; RF feature importance, klustring som diagnos).
- **Ansökningar:** Weak labels i `weakly_labeled_applications.json`, tydligt separerade från domar; "proposal, not verdict".

---

## 3. Papprets variabler vs våra Sundin-features (korskontroll)

Korskontroll mot papprets Sektion 2.1 och Tabell 2 visar att vår feature-extraktion **täcker** papprets variabler:

| Pappret (Sektion 2.1 / Tabell 2) | Våra features (`sundin_feature_extraction.py`) | Kommentar |
|----------------------------------|------------------------------------------------|------------|
| Downstream: guidance (type), angle, gap width, bypass discharge | `downstream_has_screen`, `downstream_angle_degrees`, `downstream_gap_mm`, `downstream_bypass_ls` | Direkt mappning. |
| Eel ramps (yes/no) | `upstream_has_eel_ramp` | Täckt. |
| Upstream: type of fishway, slope, flow in fishway | `upstream_has_fishway`, `upstream_type_int` (nature-like / vertical-slot / eel-ramp), `upstream_slope_pct`, `upstream_discharge_ls` | Täckt. |
| Min flow, hydropeaking, e-flow | `flow_min_ls`, `flow_hydropeaking_banned`, `flow_percent_mq` | Täckt. |
| Monitoring requirements | `monitoring_required`, `monitoring_functional` | Täckt; pappret noterar att monitoring nästan saknas i domarna. |

**Tillägg i vår modell:** Vi extraherar dessutom **cost_msek** och **timeline_years** (utfall/tyngd). Det finns inte som extraherade variabler i pappret men är metodiskt vettiga för "risk" (kostnad och tidslinje för operatören).

**Slutsats korskontroll:** Er Sundin-feature-lista är **väl anpassad** till papprets ramverk. Ingen justering av feature-extraktion behövs utifrån pappret; etiketteringen (HIGH/MEDIUM/LOW) förblir er egen, baserad på domslut + påföljd.

---

## 4. Är strategin vettig mot pappret?

**Ja.**

| Papprets fokus | Er strategi | Vettighet |
|----------------|------------|-----------|
| 22 vs 11 = operatör förlorar vs vinner | Tre klasser (HIGH/MEDIUM/LOW) som fångar **grad av ogynnsamhet** (kostnad, åtgärder, tidslinje), inte bara binärt removal/permit. | **Vettig** – ni modellerar "hur ogynnsamt" för rådgivning. |
| Variabler i Sektion 2.1 / Tabell 2 | Samma dimensioner extraheras i `sundin_feature_extraction.py`; trösklar och vikter lärs från data. | **Vettig** – pappret ger *vad* som är relevant, era data ger *hur* det vägs. |
| 33 fall i pappret | Ni har 40/44 egna märkta beslut; ert urval kan skilja sig (andra tingsrätter, tidsperioder). | **OK** – pappret är ramverk, inte er träningsdata. |
| "Removals" vs "permits" | HIGH_RISK ≈ starkt ogynnsamt (nära removal/tunga krav); LOW_RISK ≈ gynnsamt (nära permit/låga krav). MEDIUM = mellanläge. | **Vettig** – treklass mappar naturligt på papprets kontinuum. |

**Rekommendation kvar:** Dokumentera er etikettdefinition (1–2 sidor): hur HIGH/MEDIUM/LOW mappar mot utfallstyper (t.ex. removal, tillstånd med stränga/milda villkor). Referera till Sundin et al. 2026 som underlag för **vilka faktorer** som är relevanta (passage, flöde, övervakning), inte för exakt samma binära indelning.

---

## 5. Verktyg för att läsa PDF:en igen

- **Script:** `scripts/extract_pdf_text.py`  
- **Användning:** `python scripts/extract_pdf_text.py [sökväg/till/fil.pdf]`  
- **Output:** Samma sökväg med `.txt`-ändring (t.ex. `kmae250140.txt`).  
- **Krav:** PyMuPDF (`pip install pymupdf`); används redan i `add_4_new_domar.py`.

---

## 6. Sammanfattning

- Er **etiketteringsstrategi** (Sundin som feature-taxonomi, etiketter från era märkta domar, tre klasser HIGH/MEDIUM/LOW) är **vettig och förenlig** med kmae250140 (Sundin et al. 2026).
- Er **Sundin-feature-lista** har **korskontrollerats** mot papprets Sektion 2.1 och Tabell 2 och täcker papprets variabler; tilläggen cost_msek och timeline_years är rimliga för risk.
- För fortsatt spårbarhet: behåll referensen till pappret (doi ovan) i dokumentationen och i ev. metodsektion.
