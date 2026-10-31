# Benchmarki — schemat, pochodzenie i walidacja

Ten dokument opisuje, skąd pochodzą pytania benchmarkowe, jakie pola musi mieć
każde pytanie i jak sprawdzić, że gold evidence naprawdę znajduje się w bazie
wiedzy, którą przeszukuje RAG.

## 1. Dlaczego stare benchmarki były nieważne

Poprzednie pliki `data/benchmarks/*.json` miały tylko cztery pola:
`question`, `expected_answer`, `source`, `answerable`. To za mało, żeby
odróżnić prawdziwy benchmark od listy pytań przepisanych z pamięci:

- brakowało `question_id`, więc nie było stabilnego identyfikatora pytania,
- brakowało `gold_context` i `gold_span`, więc nie było czym zweryfikować, że
  odpowiedź jest w ogóle w bazie wiedzy,
- zbiór „Wikipedia + NQ” w rzeczywistości nie pochodził z oficjalnego splitu
  Natural Questions; baza wiedzy to pierwsze 500 artykułów Simple English
  Wikipedia, a nie korpus NQ,
- SQuAD i TriviaQA były ręcznie napisanymi pytaniami, nie próbką oficjalnych
  splitów, i nie miały odpowiadających im kolekcji Qdrant.

Wynik z takiego zestawu nie mówi nic o jakości RAG — mówi tylko, czy model
zna odpowiedź z treningu.

## 2. Schemat pytania (schema version 2.0)

Każdy plik benchmarku to lista obiektów. Każdy obiekt jest walidowany przez
`src.evaluation.benchmarks.BenchmarkQuestion`.

| Pole | Typ | Znaczenie |
|---|---|---|
| `question_id` | `str` | Stabilny identyfikator, unikalny w pliku |
| `question` | `str` | Pytanie zadawane pipeline’owi |
| `answerable` | `bool` | Czy odpowiedź istnieje w tej bazie wiedzy |
| `expected_answer` | `str \| null` | Oficjalna odpowiedź lub ręcznie zatwierdzona odpowiedź |
| `gold_context` | `str \| null` | Dokładny fragment źródła zawierający odpowiedź |
| `gold_span` | `str \| null` | Dokładny fragment `gold_context`, który jest odpowiedzią |
| `gold_title` | `str \| null` | Tytuł dokumentu w Qdrant (`metadata.document_title`) |
| `gold_url` | `str \| null` | URL źródła, jeśli dostępny |
| `dataset` | `wiki \| squad \| trivia` | Nazwa bazy wiedzy |
| `split` | `str` | `validation`, `custom` itp. |
| `set_type` | `official \| custom` | Pochodzenie zestawu |
| `source` | `str` | Czytelny opis pochodzenia pytania |
| `answer_aliases` | `list[str]` | Alternatywne poprawne odpowiedzi do EM/F1 |

Zasady egzekwowane przez walidator:

1. `question_id` jest unikalny w pliku.
2. Dla `answerable: true`: `expected_answer`, `gold_context`, `gold_span` i
   `gold_title` są niepuste, a `gold_span` jest **dokładnym** fragmentem
   `gold_context`.
3. Dla `answerable: false`: wszystkie pola gold są `null`, a
   `answer_aliases` jest puste.
4. `gold_context` ma co najmniej 20 znaków.
5. Nieznane pola są odrzucane (`extra="forbid"`), żeby literówka nie
   przechodziła niezauważona.

## 3. Zestawy

### 3.1. `wikipedia-simple` — custom

- Plik: `data/benchmarks/wikipedia_simple_benchmark.json`
- Rozmiar: 40 pytań — 30 answerable + 10 kontrolnych (unanswerable)
- Źródło korpusu: `wikimedia/wikipedia`, konfiguracja `20231101.simple`,
  `train[:500]`; po odfiltrowaniu stubów w Qdrant jest 486 artykułów
- Kolekcja: `rag_wiki_512` (sufiks zależy od chunk size)
- Pochodzenie pytań: **custom**, napisane ręcznie na podstawie artykułów,
  które faktycznie są w bazie wiedzy
- Gold evidence: `gold_context` i `gold_span` są wycinane z tekstu
  odtworzonego z Qdrant, więc nie mogą rozjechać się z bazą wiedzy
- Walidacja: `30/30 gold spans found` w `rag_wiki_512`

To **nie jest** oficjalny split Natural Questions. Nazwa `wikipedia-simple`
została wybrana celowo, żeby nie sugerować związku z NQ. Pytania kontrolne
pochodzą z ręcznie napisanej listy tematów, których w tym korpusie nie ma
(np. bieżące modele iPhone’a, fikcyjna Wakanda, przepis na tiramisu).

### 3.2. `squad` — official

- Plik: `data/benchmarks/squad_benchmark.json`
- Rozmiar: 50 pytań — 40 answerable + 10 unanswerable z SQuAD 2.0
- Split: oficjalny `validation` z `rajpurkar/squad_v2`
- Próbka: pierwsze 500 unikalnych kontekstów, potem deterministyczny wybór
  `random.Random(42).sample`
- `question_id`: `squad-<oficjalne id>`
- `gold_context`: fragment kontekstu wokół odpowiedzi
- `gold_span`: oficjalna odpowiedź z SQuAD (pierwsza z listy)
- `answer_aliases`: wszystkie oficjalne odpowiedzi dla pytania
- Status w API: **aktywny**, ale UI pokazuje go dopiero po ingestii kolekcji
  `rag_squad_*` (endpoint `/api/datasets` sprawdza Qdrant)

Ingestia używa tego samego splitu i tej samej próbki kontekstów:

```bash
uv run python -m src.data.ingest --dataset squad --all-configs
```

### 3.3. `trivia` — official

- Plik: `data/benchmarks/trivia_benchmark.json`
- Rozmiar: 50 pytań answerable (TriviaQA rc nie ma oficjalnych unanswerable)
- Split: oficjalny `validation` z `mandarjoshi/trivia_qa`,
  konfiguracja `rc.wikipedia`
- Próbka: pierwsze 500 unikalnych kontekstów entity pages, deterministyczny
  wybór `random.Random(42).sample`
- `question_id`: `trivia-<oficjalne question_id>`
- Gold evidence: pierwsza strona entity page, w której występuje oficjalna
  wartość odpowiedzi; pytania, w których odpowiedź nie występuje w pierwszej
  stronie, są pomijane
- Filtr relevancji: jeśli pytanie zawiera nazwę własną, musi ona występować
  w tytule lub tekście evidence; bez tego pytanie o flagę Szwajcarii mogłoby
  „pasować” do artykułu o Brazylii tylko dlatego, że zawiera słowo „white”
- Status w API: **aktywny**, ale UI pokazuje go dopiero po ingestii kolekcji
  `rag_trivia_*` (endpoint `/api/datasets` sprawdza Qdrant)

```bash
uv run python -m src.data.ingest --dataset trivia --all-configs
```

### 3.4. Wybór datasetu w UI

Dropdown „Dataset” jest zasilany przez `GET /api/datasets`. Endpoint zwraca
tylko te datasety, dla których istnieje co najmniej jedna kolekcja
`rag_<dataset>_<chunk_size>`, więc nie da się wybrać bazy, której nie ma w
Qdrant. Lista benchmarków w Auto Test jest filtrowana po wybranym datasecie —
dla `wiki` widać `wikipedia-simple`, dla `squad` — `squad`, dla `trivia` —
`trivia`. Dzięki temu pytania benchmarku zawsze trafiają do właściwego korpusu.

Po zmianie datasetu lista benchmarków przeładowuje się automatycznie.
Wszystkie trzy datasety mają kolekcje dla chunk size 256, 512 i 1024, więc
każdy RAG config (w tym `reranked-512`) działa z każdym z nich.

## 4. Walidacja

### 4.1. Sama struktura (działa w CI, bez Qdrant)

```bash
uv run python -m src.evaluation.benchmarks --validate
```

Oczekiwany wynik:

```text
Validated 140 questions in 3 file(s)
  squad_benchmark.json: 50 questions (40 answerable)
  trivia_benchmark.json: 50 questions (50 answerable)
  wikipedia_simple_benchmark.json: 40 questions (30 answerable)
```

### 4.2. Zgodność z bazą wiedzy

```bash
uv run python -m src.evaluation.benchmarks --validate --check-corpus
```

Dla każdej answerable question walidator:

1. znajduje kolekcję `rag_<dataset>_<suffix>`,
2. odtwarza dokumenty po `metadata.document_title`,
3. sprawdza, czy `gold_span` występuje w tekście dokumentu `gold_title`.

Brak kolekcji jest raportowany jako `skipped`, chyba że użyjesz
`--strict-corpus`. Wtedy walidacja kończy się błędem.

Przykład po ingestii wszystkich trzech korpusów:

```text
Corpus alignment:
  rag_squad_512: 40/40 gold spans found
  rag_trivia_512: 50/50 gold spans found
  rag_wiki_512: 30/30 gold spans found
```

### 4.3. Testy jednostkowe

```bash
uv run pytest tests/test_benchmarks.py -q
```

Testy sprawdzają m.in. unikalność ID, obecność gold evidence, dokładność
`gold_span` w `gold_context` oraz to, że `extract_gold_context` dopasowuje
całe słowa (żeby odpowiedź `O` nie została znaleziona w środku słowa
`blood`).

## 5. Odtwarzanie plików

```bash
# wiki — wymaga Qdrant z kolekcją rag_wiki_512
uv run python -m src.evaluation.build_benchmarks --dataset wiki

# SQuAD — wymaga pobrania oficjalnego validation split
uv run python -m src.evaluation.build_benchmarks --dataset squad

# TriviaQA — wymaga pobrania parquet rc.wikipedia validation
uv run python -m src.evaluation.build_benchmarks --dataset trivia

# wszystko
uv run python -m src.evaluation.build_benchmarks --dataset all
```

Wszystkie próbki są deterministyczne (`--seed 42`). Wiki korzysta z
`--collection rag_wiki_512`; dla innego chunk size trzeba podać odpowiednią
kolekcję.

Źródło ręcznych pytań wiki znajduje się w
`data/benchmarks/sources/wiki_custom_qa.json`. Builder sam wycina gold
context i span; jeśli artykuł albo odpowiedź zniknie z korpusu, build kończy
się błędem.

## 6. Ograniczenia

- Zestaw wiki jest custom, nie oficjalnym splitem NQ. W pracy trzeba to
  nazwać wprost i opisać sposób walidacji.
- SQuAD i TriviaQA są próbkami oficjalnych splitów, ale ograniczonymi do
  pierwszych 500 kontekstów. To nie jest pełny benchmark; to kontrolowany
  wycinek zgodny z ingestowanym korpusem.
- Gold contexts to fragmenty źródeł, a nie całe artykuły. Nadają się do
  sprawdzania retrievera i faithfulness, ale nie zastępują ręcznej oceny
  odpowiedzi.
- Zestawy są angielskie. Polski prompt jest obsługiwany przez pipeline, ale
  benchmarki PL trzeba zbudować osobno.
- UI pokazuje tylko te datasety, które mają co najmniej jedną kolekcję
  `rag_<dataset>_<chunk_size>`. Po ingestii wszystkich trzech korpusów
  dostępne są Wikipedia, SQuAD 2.0 i TriviaQA. Jeśli kolekcji brakuje,
  dataset jest ukryty, a bezpośrednie zapytanie API kończy się czytelnym 503
  zamiast wyników z pustej bazy.

## 7. Atrybucja

- SQuAD 2.0: Rajpurkar et al., licencja CC BY-SA 4.0.
- TriviaQA: Joshi et al., licencja Apache 2.0.
- Simple English Wikipedia: CC BY-SA 4.0; URL artykułu jest zapisany w
  `gold_url`.
