# NAP Legal AI Advisor — Användarguide

## Översikt

NAP Legal AI Advisor är ett Streamlit-baserat verktyg för analys av miljödomstolsbeslut inom ramen for Nationella planen för moderna miljövillkor (NAP). Systemet erbjuder:

- **Chat-läge:** Ställ frågor om domstolsbeslut på svenska och få svar baserade på den analyserade datan.
- **Sök-läge:** Semantisk sökning bland alla domstolsbeslut med filtrering på risknivå, domstol och datum.
- **Riskindikation:** Automatisk klassificering av beslut som HIGH_RISK / MEDIUM_RISK / LOW_RISK med hjälp av fine-tunad KB-BERT (LegalBERT).

---

## Starta appen

```bash
# Från repo root
streamlit run nap-legal-ai-advisor/app.py
```

Öppna http://localhost:8501 i webbläsaren. Första starten tar ~45 sekunder (MiniLM-modellen laddas och sökindex byggs).

### Krav

- Python 3.10+
- Beroenden installerade: `pip install -r nap-legal-ai-advisor/requirements.txt`
- Tränade modeller i `models/nap_legalbert_cv/fold_4/best_model/`
- Processade data i `Data/processed/` (minst `cleaned_court_texts.json` och `labeled_dataset.json`)

---

## Chat-läge

Chatten använder ett RAG-system (Retrieval-Augmented Generation) baserat på keyword-routing. Inga externa LLM-anrop — allt sker lokalt.

### Exempel på frågor

| Fråga | Vad händer |
|-------|------------|
| "Visa hög risk beslut" | Listar alla beslut klassificerade som HIGH_RISK |
| "Vilka åtgärder är vanligast?" | Visar frekvens av extraherade åtgärder (fiskväg, minimitappning etc.) |
| "Visa statistik" | Övergripande statistik: antal beslut, domstolar, datumspann |
| "Visa riskfördelning" | Fördelning HIGH/MEDIUM/LOW med antal |
| "Visa senaste besluten" | De senaste besluten sorterade efter datum |
| "Vad kostar det?" | Sammanfattning av extraherade kostnader |
| "Jämför M 1234-22 med M 5678-23" | Sida-vid-sida-jämförelse av två specifika beslut |
| "Berätta om Ume älv" | Semantisk sökning efter relevanta beslut |

Snabbknappar ("quick actions") visas under chatfönstret för vanliga frågor.

---

## Sök-läge

Semantisk sökning med `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers). Sökfrågor matchas mot domstolsbeslutens text via cosine similarity.

### Filter

- **Risknivå:** HIGH_RISK, MEDIUM_RISK, LOW_RISK
- **Domstol:** Nacka TR, Växjö TR, etc.

### Resultat

Varje sökträff visar:
- Beslutsid och filnamn
- Risknivå och konfidensgrad
- Det mest relevanta textutdraget
- Likhetsvärde (0–1)

Klicka på ett beslut för detaljer: full text, metadata, extraherade åtgärder/kostnader och riskprediktion.

---

## Riskmodellen

- **Bas:** KB/bert-base-swedish-cased (110M parametrar)
- **Träning:** 5-fold korsvalidering med sliding window (512 tokens, stride 256)
- **Precision:** ~65 % genomsnittlig accuracy över folderna (hög varians 50–75 %)
- **Bias:** Överrepresentation av MEDIUM_RISK (95 % recall, medan HIGH/LOW har ~30 %)

Modellen används som **indikator**, inte som beslutsunderlag.

---

## Data

Appen läser från:

| Fil | Innehåll |
|-----|----------|
| `Data/processed/cleaned_court_texts.json` | 50 rensade domstolsbeslut med sektioner, metadata, extraherade åtgärder |
| `Data/processed/labeled_dataset.json` | Train/val/test-splits med etiketter (44 märkta) |
| `Data/processed/linkage_table.json` | Koppling till VISS-vattenförekomster (om tillgänglig) |

### Lägga till nya beslut

1. Placera rå TXT-filer i `Data/Domar/data/processed/court_decisions/`
2. Kör `python scripts/02_clean_court_texts.py`
3. Lägg etiketter i `Data/processed/label_overrides.json` (format: `{ "decision_id": "HIGH_RISK" }`)
4. Kör `python scripts/03_create_labeled_dataset.py`
5. (Valfritt) Omträna modellen: `python scripts/05_finetune_legalbert.py`
6. Starta om appen

---

## Felsökning

| Problem | Lösning |
|---------|---------|
| "Model not found" | Kontrollera att `models/nap_legalbert_cv/fold_4/best_model/` finns och innehåller `model.safetensors` |
| Långsam start | Normalt vid första start (~45 s). Efterföljande starter använder cache (`.cache/embeddings.pkl`) |
| SSL-fel | Företagsproxy: se `utils/ssl_fix.py`. Inte för produktion. |
| CUDA out of memory | Sänk BATCH_SIZE i `.env` (standard: 4, prova 2) |
| Inga sökresultat | Kontrollera att `cleaned_court_texts.json` finns och inte är tom |
