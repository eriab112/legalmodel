# Har vi förutsättningar att bygga ett bra beslutstöd för NAP-rådgivare?

**Kort svar:** **Ja, med tydliga villkor.** Ni har en **stabil grund**: domdata, pipeline, akademisk ram (Sundin), tolkbar feature-extraktion, och en fungerande prototyp. För att det ska bli ett **bra** beslutstöd krävs mer data, tydligare begränsningsuttalanden, förbättrad tolkbarhet och produktionsanpassning.

---

## 1. Vad som redan stödjer ett bra beslutstöd

| Förutsättning | Status |
|---------------|--------|
| **Domänspecifik data** | 46 unika NAP-relaterade domar, 40 märkta; lagstiftning och ansökningar tillgängliga. Rätt typ av källor för rådgivning. |
| **Reproducerbar pipeline** | Rå text → rensning → etiketter → träning → app. Tydliga script (02–06) och separata processed-filer. |
| **Akademisk förankring** | Sundin et al. 2026 ger en strukturerad modell för vad domstolar väger (passage, flöde, övervakning, kostnad). Ni använder den som taxonomi, inte som stela regler. |
| **Tolkbarhet** | Sundin-features (gapbredd, kostnad, tidslinje, fiskväg, övervakning) + RF-importance ger möjlighet att förklara *vilka* faktorer som driver risk. |
| **Separation ansökning vs dom** | Tydlig distinktion mellan vad operatörer *föreslår* och vad domstolar *beslutar*. Weak labels är tydligt märkta. Viktigt för rådgivning. |
| **Sökning + risk i samma gränssnitt** | Rådgivare kan både söka i precedens och få en riskindikation. Semantisk sökning (MiniLM) ger snabb, relevant träffbild. |
| **Utrymme att förbättra** | DAPT-korpus, 4 nya domar redo att inkluderas, ensemble av folds, kostnadsmodell i nap_model-main (R²=0,96) – allt finns i repot eller är påbörjat. |

Så: **ja**, ni har förutsättningar – data, metod, kod och en väg framåt.

---

## 2. Vad som idag begränsar “bra” beslutstöd

| Begränsning | Konsekvens | Vad som behövs |
|-------------|------------|----------------|
| **Liten träningsmängd (40/44)** | Modellen har ~65 % accuracy och stark MEDIUM_RISK-bias; osäker generalisering. | Fler märkta domar (mål: 80–100+), ev. extern validering. |
| **Ingen explicit osäkerhetskommunikation** | Rådgivare ser en risknivå utan tydlig “baserat på 40 domar”, “osäkerhet hög”. | UI som visar begränsning, ev. konfidensintervall eller kvalitetsetikett. |
| **Keyword-baserad chatt** | Svar styrs av nyckelord, inte verklig förståelse. Svårt att ställa följdfrågor. | LLM-integration eller tydligare “stöd för sökning, inte juridisk rådgivning”. |
| **Saknad “varför”-förklaring i appen** | RF/Sundin-features används inte i UI för att förklara *varför* HIGH/MEDIUM/LOW. | Visa vilka Sundin-faktorer som drev prediktionen (t.ex. kostnad, tidslinje, passage). |
| **Ingen spårbarhet** | Inga användare, loggning eller granskning. Olämpligt för verkligt beslutsstöd i organisation. | Auth, audit log, tydlig användningspolicy. |
| **SSL-workaround** | Certifikatverifiering avstängd. Oacceptabelt i produktion. | Korrekt proxy/CA eller begränsning till dev. |

Dessa är **hinder för att kalla det “bra” idag**, men de är adresserbara med mer data, UI- och modellförbättringar och produktionskrav.

---

## 3. Vad “bra beslutstöd för rådgivare” bör innefatta

För att systemet ska vara **bra** i rollen som stöd till rådgivare inom NAP:

1. **Tydlig roll** – “Stöd för att hitta och jämföra precedens och få en indikation på risk”, inte “juridiskt beslut” eller “ersättning för professionell bedömning”.
2. **Tolkbarhet** – Rådgivaren ska kunna se *vilka* faktorer (kostnad, åtgärder, tidslinje, passage) som driver riskklassen, t.ex. via Sundin-features eller SHAP.
3. **Källreferens** – Sökresultat och risk ska kopplas till konkreta domar (id, målnummer) så att rådgivaren kan läsa källan.
4. **Begränsningsuttalande** – Tydligt i UI: tränat på 40/44 domar, osäkerhet vid nya typer av mål, inte ersättning för juridisk expertis.
5. **Uppdaterbarhet** – Möjlighet att lägga in nya domar och omträna (pipelinen finns redan).
6. **Integration med befintligt arbete** – Länk till VISS/MCDA, kostnadsmodell eller NAP-dashboard (nap_model-main) ökar nyttan.

Ni har byggstenar för 1, 3, 5 och delvis 2 och 6; 4 är huvudsakligen formulering och UI.

---

## 4. Sammanfattande bedömning

- **Grunden är bra:** Domdata, pipeline, Sundin-ram, feature-extraktion, sökning och risk i samma app, och en genomtänkt strategi (Phase A, hybrid, multi-task) ger **förutsättningar** att bygga ett bra beslutstöd.
- **Idag är det en lovande prototyp**, inte ett färdigt “bra” beslutstöd: träningsunderlaget är litet, osäkerhet och tolkbarhet syns inte tillräckligt i UI, och produktionskrav (säkerhet, spårbarhet) är inte uppfyllda.
- **Väg till “bra”:**  
  - Få in de 4 nya domarna, DAPT och omträning (73–77 % som mål).  
  - Visa Sundin-faktorer eller SHAP i appen (“varför denna risk?”).  
  - Tydlig roll- och begränsningsuttalande i gränssnittet.  
  - När ni växer mot produktion: auth, logging, borttaget SSL-workaround, ev. integration med nap_model-main (kostnad, VISS).

**Slutsats:** Ja, ni har förutsättningar att bygga ett bra beslutstöd för rådgivare inom NAP, om ni bygger vidare på den grund ni har och explicit adresserar datamängd, tolkbarhet, begränsningsuttalande och driftkrav.
