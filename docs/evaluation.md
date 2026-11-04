# Ewaluacja — automatyczna, sędzia LLM i ocena ręczna

Ten dokument opisuje warstwę ewaluacji z P0.4 oraz protokół pomiarowy z P0.5.
Jej zadaniem jest odpowiedzieć na trzy różne pytania:

1. **Czy odpowiedź jest poprawna?** — exact match, F1, sędzia LLM, ocena ręczna.
2. **Czy odpowiedź jest wierna kontekstowi?** — sędzia LLM i ocena ręczna.
3. **Czy retriever dostarczył właściwe dowody?** — gold title/span w kontekstach.

Wyniki są zapisywane równolegle do JSONL, CSV i SQLite, żeby surowe dane nie
zależały od pojedynczego formatu ani od późniejszych obliczeń.

## 1. Automatyczne metryki

### 1.1. Exact match i F1

Implementacja znajduje się w `src/evaluation/metrics.py`.

- Normalizacja jest wzorowana na SQuAD: małe litery, usunięcie interpunkcji,
  usunięcie angielskich rodzajników (`a`, `an`, `the`) i spacje.
- `exact_match` zwraca `1.0`, gdy znormalizowana odpowiedź jest równa
  któremukolwiek aliasowi z `answer_aliases`.
- `token_f1` liczy precision/recall na tokenach i zwraca najlepszy F1 po
  wszystkich referencjach.
- Dla pytań `answerable: false` EM i F1 są `null` — nie ma sensu liczyć
  podobieństwa tekstu do nieistniejącej odpowiedzi.

W benchmarkach SQuAD aliasy pochodzą z oficjalnych odpowiedzi. Dla TriviaQA i
wiki aliasy są zapisane w pliku benchmarku. Dzięki temu odpowiedź
`"Paris, France"` może zostać uznana za poprawną, jeśli taki alias istnieje.

### 1.2. Detekcja odmowy

Detektor obsługuje angielskie i polskie sformułowania, m.in.:

- `I don't have enough information...`
- `I cannot answer...`
- `Nie mam wystarczających informacji...`
- `Nie wiem.`

Pusta odpowiedź jest traktowana jako odmowa. Dla pytania answerable odmowa
jest błędem (`automatic_correct = False`); dla kontroli unanswerable poprawna
odmowa daje `refusal_correct = True` i `automatic_correct = True`.

### 1.3. Diagnostyka retrievera

Dla każdego pytania answerable runner sprawdza:

- `gold_title_rank` — pozycję dokumentu `gold_title` wśród pobranych chunków,
- `retrieval_hit` — czy gold dokument w ogóle trafił do kontekstu,
- `gold_span_in_context` — czy dokładny gold span występuje w którymś chunku.

To nie jest pełna metryka context precision/recall, ale tani sygnał, czy
odpowiedź mogła w ogóle powstać z dostarczonego kontekstu.

### 1.4. Sędzia LLM

Sędzia jest w `src/evaluation/judge.py`. Dla każdej odpowiedzi zwraca cztery
wartości 0/1 oraz krótkie uzasadnienie:

| Pole | Znaczenie |
|---|---|
| `correct` | odpowiedź semantycznie zgadza się z referencją; dla unanswerable — czy odmawia |
| `faithful` | każde twierdzenie faktu jest poparte pobranym kontekstem |
| `context_relevant` | co najmniej jeden kontekst jest istotny dla pytania |
| `refusal_appropriate` | model odmawia dokładnie wtedy, gdy pytanie jest unanswerable |

Prompt wymaga **tylko** obiektu JSON:

```json
{
  "correct": 1,
  "faithful": 1,
  "context_relevant": 1,
  "refusal_appropriate": 1,
  "reason": "one short sentence"
}
```

Parser akceptuje też JSON w bloku markdown i usuwa znaczniki `think`. Jeśli
odpowiedź nie jest poprawnym JSON-em, sędzia ponawia próbę raz; trwały błąd
jest zapisywany w rekordzie, ale nie przerywa całego runu.

Sędzia jest wstrzykiwalny (`LLMJudge(llm=...)`), więc testy nie potrzebują
modelu ani sieci. W CLI można wskazać osobny model sędziego przez
`--judge-model`.

> **Dlaczego nie bezpośrednio RAGAS?** W obecnym środowisku RAGAS 0.4.3 nie
> importuje się, bo oczekuje modułu
> `langchain_community.chat_models.vertexai`, którego nie ma w zainstalowanej
> wersji `langchain-community`. Zamiast obchodzić ten problem kruchym pinem,
> P0.4 używa bezpośredniego LLM-as-judge z jawnym kontraktem JSON. Metryki
> RAGAS można dodać później, gdy zależność zostanie naprawiona; nie zmieni to
> formatu zapisu wyników.

## 2. Uruchamianie automatycznej ewaluacji

### 2.1. Listy dostępnych modeli i benchmarków

```bash
uv run python -m src.evaluation.run_eval --list-benchmarks
uv run python -m src.evaluation.run_eval --list-models
```

`--list-models` pokazuje wszystkie skonfigurowane modele oraz te, które są
aktualnie dostępne (Ollama albo klucz API).

### 2.2. Pojedynczy run

```bash
uv run python -m src.evaluation.run_eval \
  --benchmark wikipedia-simple \
  --models "Gemma 3 4B" \
  --rag-config basic-512 \
  --judge-model "Gemma 3 4B" \
  --limit 5
```

`--limit 5` służy do smoke testu. Bez `--limit` uruchamiane są wszystkie
pytania benchmarku. Dla pierwszego uruchomienia warto użyć małego limitu, żeby
sprawdzić, czy Ollama, Qdrant i embeddingi działają.

Protokół pomiarowy można nadpisać jawnie:

```bash
uv run python -m src.evaluation.run_eval \
  --benchmark squad \
  --models "Gemma 3 4B" \
  --rag-config basic-512 \
  --judge-model "Gemma 3 4B" \
  --warmup 2 \
  --repetitions 3 \
  --max-tokens 1024 \
  --num-ctx 8192 \
  --confidence-level 0.95
```

Wartości domyślne są takie same jak powyższe. `--repetitions 1 --warmup 0`
przywraca pojedynczy przebieg wygodny do smoke testu, ale nie daje podstaw do
estymacji wariancji.

### 2.3. Wiele modeli i konfiguracji RAG

```bash
uv run python -m src.evaluation.run_eval \
  --benchmark squad \
  --models "Gemma 3 4B" \
  --models "Groq GPT-OSS 20B" \
  --rag-config basic-256 \
  --rag-config basic-512 \
  --rag-config basic-1024 \
  --rag-config reranked-512 \
  --judge-model "Gemma 3 4B"
```

Każda para model × RAG config dostaje osobny `run_id`, własny JSONL, CSV i
summary. `--all-models` uruchamia wszystkie modele dostępne w danej chwili.
`--models` można podać wielokrotnie albo rozdzielić przecinkami.

### 2.4. Run bez sędziego

```bash
uv run python -m src.evaluation.run_eval \
  --benchmark trivia \
  --models "Gemma 3 4B" \
  --rag-config basic-512 \
  --no-judge
```

Zostaną zapisane EM/F1, detekcja odmowy i diagnostyka retrievera, bez kosztu
wywołań sędziego.

### 2.5. Protokół pomiarowy P0.5

Protokół jest zdefiniowany w `src/evaluation/protocol.py` i zapisywany w
metadanych każdego runu. Domyślne wartości:

| Parametr | Domyślnie | Znaczenie |
|---|---:|---|
| `warmup_runs` | 2 | liczba odrzucanych żądań rozgrzewających |
| `repetitions` | 3 | liczba mierzonych, sekwencyjnych przebiegów |
| `max_tokens` | 1024 | wspólny limit wyjścia dla wszystkich modeli |
| `num_ctx` | 8192 | wspólny kontekst Ollamy (chmura: zarządzany przez dostawcę) |
| `confidence_level` | 0.95 | poziom ufności przedziałów |
| `execution_mode` | `sequential` | brak równoległości i współdzielenia GPU/API |

Przebieg jest zawsze sekwencyjny:

1. `warmup_runs` pierwszych pytań benchmarku jest wykonywanych i **odrzucanych**
   (nie trafiają do metryk jakości ani do plików wynikowych). Ich czasy trafiają
   tylko do podsumowania `warmup` w metadanych.
2. Następnie wykonywane jest `repetitions` pełnych przejść po benchmarku.
   Każdy rekord ma numer `repetition` (1..N), więc surowe dane pozwalają
   odtworzyć zarówno wynik zbiorczy, jak i stabilność między powtórzeniami.

`max_tokens` z protokołu nadpisuje per-modelowe wartości z
`src/config/models.py`. Dzięki temu Qwen 3 8B i Groq Qwen 3.8 nie dostają
większego limitu niż pozostałe modele. `num_ctx` jest przekazywany do
`ChatOllama`; dostawcy chmurowi nie mają wspólnego parametru kontekstu, więc
metadata rozróżnia `num_ctx` (żądany), `num_ctx_supported` i
`num_ctx_effective` (`null` dla chmury).

Jeśli włączony jest sędzia LLM, ocenia on **każde** powtórzenie. Dla dużego
kampanijnego runu oznacza to `repetitions × liczba pytań` wywołań sędziego;
`--no-judge` lub `--repetitions 1` zmniejszają ten koszt.

### 2.6. Czas generacji a retry

`generation_time_ms` obejmuje **tylko udaną próbę wywołania modelu**. Osobno
zapisywane są:

- `retry_count` — liczba ponowień przed sukcesem,
- `failed_attempt_time_ms` — czas prób zakończonych błędem,
- `retry_backoff_time_ms` — czas spania między próbami.

`total_time_ms` nadal jest czasem end-to-end i obejmuje retry. Dzięki temu
`tokens_per_second` i średni czas generacji nie są zaniżane przez oczekiwanie
na limity API. Surowe rekordy zachowują wszystkie cztery wartości.

## 3. Gdzie trafiają wyniki

Domyślnie wszystko ląduje w `data/evaluation/` (katalog jest ignorowany przez
git — surowe wyniki nie powinny trafiać do repozytorium):

```text
data/evaluation/
├── evaluation.sqlite
├── runs/
│   ├── <run_id>.jsonl
│   ├── <run_id>.csv
│   └── <run_id>_summary.json
└── manual/
    ├── <run_id>_manual_sheet.csv
    ├── <run_id>_human_ratings.jsonl
    ├── <run_id>_human_ratings.csv
    └── <run_id>_agreement.json
```

### 3.1. JSONL

Jedna linia = jedno pytanie w jednym powtórzeniu. To jest źródło prawdy:
zawiera odpowiedź, konteksty, metryki, werdykt sędziego i błąd (jeśli
wystąpił). Przykładowe pola:

```json
{
  "run_id": "20260501T120000Z_gemma-3-4b-basic-512_ab12",
  "question_id": "wiki-001",
  "repetition": 2,
  "answerable": true,
  "answer": "13 million",
  "retrieval_time_ms": 41.2,
  "generation_time_ms": 812.4,
  "retry_count": 0,
  "failed_attempt_time_ms": 0.0,
  "retry_backoff_time_ms": 0.0,
  "total_time_ms": 861.7,
  "max_tokens": 1024,
  "num_ctx": 8192,
  "exact_match": 1.0,
  "f1": 1.0,
  "is_refusal": false,
  "gold_title_rank": 1,
  "retrieval_hit": true,
  "gold_span_in_context": true,
  "judge": {
    "correct": 1,
    "faithful": 1,
    "context_relevant": 1,
    "refusal_appropriate": 1,
    "reason": "supported by context",
    "error": null
  },
  "error": null
}
```

### 3.2. CSV

Ten sam rekord w formie płaskiej. Konteksty są zapisane jako JSON w kolumnach
`contexts_json` i `context_titles_json`, żeby nie zgubić granic chunków.
Kolumny `judge_*` są spłaszczone, a `error` pozwala odfiltrować nieudane
pytania.

### 3.3. SQLite

Plik `evaluation.sqlite` ma trzy tabele:

- `runs` — metadane runu: benchmark, dataset, model, provider, RAG config,
  top-k, język, model sędziego i pełny `metadata_json` (wersje pakietów,
  commit git, GPU, platforma),
- `results` — jeden wiersz na pytanie × powtórzenie, z metrykami i `raw_json`;
  klucz to `(run_id, repetition, question_id)`,
- `human_ratings` — oceny ręczne: `(run_id, question_id, rater_id)` jest
  kluczem, więc ponowny import aktualizuje ocenę zamiast ją dublować.

Otwarcie starej bazy z P0.4 automatycznie migruje tabelę `results` do schematu
z kolumną `repetition` i polami retry (istniejące wiersze dostają
`repetition = 1`). Migracja nie usuwa danych.

Przykładowe zapytanie:

```bash
sqlite3 data/evaluation/evaluation.sqlite \
  "SELECT model_name, rag_config, COUNT(*), AVG(f1) FROM runs JOIN results USING(run_id) GROUP BY 1,2;"
```

### 3.4. Summary

`<run_id>_summary.json` zawiera zagregowane metryki jakości, liczby błędów oraz
statystyki wydajności. Dla każdego pola z `PERFORMANCE_FIELDS` (retrieval,
generation, total, tokens_out, tokens_per_second, retry_backoff) zapisany jest
blok `performance` z polami `n`, `mean`, `median`, `stdev`, `min`, `max`,
`ci_low`, `ci_high`. Obok niego są:

- `question_statistics` — statystyki per pytanie (po wszystkich powtórzeniach),
- `repetition_statistics` — jakość i wydajność per przebieg,
- `warmup` — liczność i czasy odrzuconych żądań rozgrzewających,
- starsze płaskie klucze `mean_*` — zachowane dla kompatybilności.

Przedziały ufności używają przybliżenia normalnego
`mean ± z * s / sqrt(n)`. Dla `n < 2` przedział nie jest raportowany. Summary
jest wygodne do szybkiego raportu, ale **nie zastępuje** JSONL — zawsze można
odtworzyć agregaty z surowych rekordów.

## 4. Ocena ręczna

### 4.1. Rubryka

```bash
uv run python -m src.evaluation.manual_eval rubric
```

Rubryka ma cztery wymiary 0/1:

- **correct** — zgodność z `expected_answer`; dla unanswerable tylko odmowa,
- **faithful** — każde twierdzenie faktu wynika z pobranego kontekstu,
- **context_relevant** — co najmniej jeden kontekst jest istotny,
- **refusal_appropriate** — odmowa dokładnie przy unanswerable.

### 4.2. Eksport próbki

```bash
uv run python -m src.evaluation.manual_eval \
  sheet --run-id RUN_ID --sample-size 20
```

Domyślnie powstaje
`data/evaluation/manual/RUN_ID_manual_sheet.csv`. Próbka jest deterministyczna
(`--seed 42`) i stratyfikowana: co najmniej jedno pytanie kontrolne (jeśli run
je zawiera), resztę uzupełniają pytania answerable. Pytania z błędem pipeline'u
są pomijane, bo nie mają odpowiedzi do oceny.

Sheet zawiera m.in.:

- `question`, `expected_answer`, `gold_context`, `gold_span`,
- `retrieved_contexts`, `answer` (materiał do oceny),
- `rater_1_*` i `rater_2_*` — osobne kolumny dla dwóch oceniających,
- `rater_1_notes`, `rater_2_notes`.

Wartości wpisuje się jako `0`, `1` albo puste. Puste oznacza brak oceny i nie
jest zamieniane na 0. Oceniający nie zmieniają `run_id` ani `question_id`.

### 4.3. Import i porównanie

Po wypełnieniu arkusza:

```bash
uv run python -m src.evaluation.manual_eval \
  import --sheet data/evaluation/manual/RUN_ID_manual_sheet.csv
```

Import zapisuje oceny do `human_ratings` i dodatkowo eksportuje je do
`manual/RUN_ID_human_ratings.jsonl` oraz `.csv`. Ponowny import tej samej pary
`(run_id, question_id, rater_id)` aktualizuje wpis.

Raport zgodności:

```bash
uv run python -m src.evaluation.manual_eval agreement --run-id RUN_ID
```

Raport zawiera dla każdego wymiaru liczbę par, procent zgodności i Cohen's
kappa. Kappa jest liczona tylko dla par, w których **obaj** oceniający
wypełnili dany wymiar. Interpretacja orientacyjna: `< 0.20` słaba, `0.21–0.40`
dostateczna, `0.41–0.60` umiarkowana, `0.61–0.80` dobra, `> 0.80` bardzo dobra.

### 4.4. Podgląd runów

```bash
uv run python -m src.evaluation.manual_eval list-runs
```

Jeśli używasz innego katalogu niż `data/evaluation`, opcję globalną podaj
**przed** nazwą podkomendy:

```bash
uv run python -m src.evaluation.manual_eval \
  --output-dir /tmp/eval sheet --run-id RUN_ID
```

## 5. Co jest zapisywane w metadanych

Runner zapisuje w `runs.metadata_json`:

- model, provider, `model_id`, parametry, `configured_max_tokens` i
  `model_quantization`,
- pełny blok `model`: kwantyzacja, źródło jej pochodzenia
  (`config` / `ollama` / `fallback`), rodzina, rozmiar parametrów i długość
  kontekstu zgłoszona przez Ollamę (jeśli dostępna),
- `protocol`: warm-up, powtórzenia, `max_tokens`, `num_ctx`, poziom ufności i
  tryb sekwencyjny,
- `max_tokens`, `num_ctx`, `num_ctx_supported`, `num_ctx_effective`,
- RAG config, chunk size, overlap, reranking, `top_k`, język,
- benchmark, dataset, liczbę pytań i limit,
- model sędziego,
- wersję Pythona, platformę, nazwę CPU, liczbę rdzeni, RAM, nazwę GPU, VRAM i
  wersję sterownika,
- commit git i informację, czy working tree był brudny,
- wersje kluczowych pakietów (`langchain`, `qdrant-client`,
  `sentence-transformers`, `torch`, `datasets`, `ragas` itd.).

Kwantyzacja modeli lokalnych jest najpierw odczytywana z działającej Ollamy
(`/api/show`, pole `quantization_level`), a `ModelConfig.quantization` jest
wartością zapasową. Dla modeli chmurowych zapisywane jest
`provider-managed`, bo dostawcy nie ujawniają wagi ani kwantyzacji.

## 5.1. Ograniczenia protokołu

- Warm-up odbywa się raz na początku runu, a nie przed każdym powtórzeniem.
  Kolejne przebiegi mogą więc podlegać dryfowi termicznemu lub zmianom
  obciążenia API; `repetition_statistics` pozwala ten dryf zobaczyć.
- `num_ctx` jest w pełni kontrolowany tylko dla Ollamy. Dla Groq, Google i
  Mistrala kontekst jest zarządzany przez dostawcę i nie jest stałym parametrem
  porównawczym.
- Kwantyzacja modeli chmurowych nie jest znana. Porównanie lokalne vs chmura
  dotyczy zatem całego stosu (wagi + kwantyzacja + infrastruktura), a nie
  wyłącznie rozmiaru modelu.
- Przedziały ufności są przybliżeniem normalnym. Przy trzech powtórzeniach
  należy je traktować jako orientacyjne, a w pracy podać `n` i odchylenie
  standardowe.
- Retry są nadal ponawiane wewnątrz pipeline'u. `generation_time_ms` jest
  wolny od backoffu, ale `total_time_ms` pokazuje rzeczywisty koszt
  end-to-end.

## 6. Ograniczenia

- Sędzia LLM może być stronniczy, zwłaszcza gdy ocenia model z tej samej
  rodziny. Dlatego obok niego są EM/F1, detekcja odmowy i ocena ręczna.
- Sędzia widzi `expected_answer` i gold context. To pomaga w ocenie
  poprawności, ale może wpływać na ocenę faithfulness. Ręczna rubryka jest
  niezależnym sprawdzianem.
- Próbka ręczna jest mała (domyślnie 20 pytań), więc kappa ma szeroki
  przedział niepewności. W pracy trzeba podać liczebność i sposób losowania.
- Benchmarki są angielskie; polski prompt jest obsługiwany, ale polskie
  benchmarki wymagają osobnego zbioru.
- Surowe wyniki są ignorowane przez git. Do repozytorium trafia kod i
  dokumentacja, nie wynik jednego runu.
- Powtórzenia tego samego pytania nie są w pełni niezależne: dzielą ten sam
  kontekst, model i sesję. Służą do estymacji wariancji pomiaru, a nie do
  traktowania odpowiedzi jak niezależnych próbek z populacji.

## 7. Szybka weryfikacja

```bash
# schemat benchmarków (bez Qdrant)
uv run python -m src.evaluation.benchmarks --validate

# testy jednostkowe warstwy ewaluacji
uv run pytest tests/test_metrics.py tests/test_judge.py \
  tests/test_evaluation_store.py tests/test_eval_runner.py \
  tests/test_manual_eval.py tests/test_measurement_protocol.py \
  tests/test_rag.py -q

# lista benchmarków i rubryka (bez modeli)
uv run python -m src.evaluation.run_eval --list-benchmarks
uv run python -m src.evaluation.manual_eval rubric
```
