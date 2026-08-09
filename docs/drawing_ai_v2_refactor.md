# Drawing AI v2 — score-preserving refactor

## Maqsad

Etalon va ixtiyoriy OpenCV evaluatorlarining mavjud baholash mezonlari va
og‘irliklarini o‘zgartirmasdan, backend integratsiyasini xavfsiz, testlanuvchi
va versiyalanadigan holatga keltirish.

## Locked rubrikalar

### Etalon — 100 ball

`3 + 6 + 8 + 12 + 18 + 10 + 24 + 15 + 4 = 100`

### Ixtiyoriy — 75 raw ball → 100 ball

`15 + 10 + 15 + 15 + 10 + 10 = 75`

Rubrika nomi, tartibi va maksimal ballari `app/ai/drawing_ai_v2/rubrics.py`
modulida immutable contract sifatida saqlanadi. Engine har bir evaluationdan
keyin qaytgan jadvalni shu contract bilan tekshiradi.


## Modular yadro

Oldingi ikki monolit fayl compatibility facade sifatida saqlandi:

- `app/ai/etalon_mode_final_backend.py`;
- `app/ai/optional_mode_v1_backend.py`.

Asosiy kod quyidagi paketlarga ajratildi:

- `app/ai/etalon/` — `io`, `frame`, `placement`, `line_types`,
  `dimensions`, `projections`, `projection_sections`, `visible_view`,
  `visible_section`, `cleanliness`, `reporting`, `pipeline`;
- `app/ai/optional/` — `io`, `normalization`, `layout`, `roles`,
  `completeness`, `hatching`, `dimensions`, `line_semantics`, `cleanliness`,
  `task_requirements`, `reporting`, `pipeline`, `backend`.

Shu sabab har bir mezon mustaqil testlanadi va keyinchalik detector/OCR modeli
faqat kerakli modul ichida almashtiriladi. Eski importlar buzilmaydi.

## Yangi qatlamlar

- `contracts.py` — typed request/result va mode enum;
- `validation.py` — extension, magic bytes, fayl hajmi, image pixels va PDF
  page count validatsiyasi;
- `adapters.py` — legacy evaluatorlarni lazy import qiluvchi adapterlar;
- `normalization.py` — yagona result contract va rubric guard;
- `engine.py` — validation, execution timing, normalization va version metadata;
- `exceptions.py` — domain-level xatolar.

## Score-preserving optimizatsiyalar

- etalon modulidan takrorlangan importlar olib tashlandi;
- ishlatilmagan `matplotlib` dependency olib tashlandi;
- faqat score table uchun ishlatilgan `pandas` o‘rniga Python list/dict ishlatildi;
- PDF va PIL resurslari context manager orqali yopiladi;
- ixtiyoriy rejim artifact nomlari UUID bilan ajratildi, parallel submissionlarda
  fayl ustiga yozib yuborish bartaraf etildi;
- optional `task_text` teacher description’dan evaluatorga uzatiladi, ammo
  criterion maksimal balli va scoring formulasi o‘zgarmaydi;
- result ichiga `drawing_ai_v2` runtime metadata qo‘shiladi.

## Regression tekshiruvi

```bash
python scripts/run_drawing_ai_regression.py
```

Runner 5 ta nazorat chizmasini ikkala rejimda baholaydi va total score hamda
criterion table’ni `tests/fixtures/drawing_ai/baseline_results.json` bilan
solishtiradi. Runtime qiymati snapshot taqqoslashga kiritilmaydi.

Unit testlar:

```bash
python -m unittest tests.test_drawing_ai_v2 -v
```

## Muhim chegara

Ushbu bosqich mavjud heuristic aniqlik kamchiliklarini yashirmaydi va yangi
PyTorch detector qo‘shmaydi. U V1 scoringni reproducible baseline sifatida
muzlatadi. Keyingi detector/OCR/geometriya modullari shu contract ortida
bosqichma-bosqich ulanishi mumkin.

## Lokal benchmark

Bir xil sintetik `identical.png` fixture va fresh Python process’da uch martalik
median natija:

| Oqim | Oldingi median | Refaktor median | Peak RSS oldin | Peak RSS keyin |
|---|---:|---:|---:|---:|
| Etalon service | 1.88 s | 1.61 s | ~301 MB | ~266 MB |
| Optional service | 1.12 s | 1.09 s | ~228 MB | ~228 MB |

Etalon oqimida taxminan 14% vaqt va 11% peak xotira kamaydi. Bu raqamlar test
muhitiga bog‘liq; productionda alohida load-test bilan qayta o‘lchanadi.
