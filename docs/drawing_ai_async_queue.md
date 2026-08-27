# Drawing AI async queue

## Maqsad

`POST /api/v1/submissions` endi Drawing AI baholashni HTTP request ichida
kutmaydi. Chizma storage'ga yoziladi, `Submission(status=pending)` va
`DrawingEvaluationJob(status=pending)` bitta DB transactionda yaratiladi va
201 javob tez qaytadi.

Mavjud `app/ai/` scoring/detection algoritmi o'zgartirilmagan.

## Oqim

```text
Student upload
  -> R2 submissions/
  -> submissions row (pending)
  -> drawing_evaluation_jobs row (pending)
  -> HTTP 201

Drawing worker
  -> SELECT ... FOR UPDATE SKIP LOCKED
  -> R2 input -> temporary local file
  -> existing Drawing AI evaluator
  -> R2 results/
  -> submission evaluated
  -> job succeeded
```

`FOR UPDATE SKIP LOCKED` sabab bir nechta worker process bir xil job'ni
takroran olmaydi. Shu sabab queue keyinchalik horizontal worker scale uchun
ham tayyor.

## Statuslar

`Submission.status` backward-compatible qoladi:

- `pending`
- `evaluated`
- `failed`

API qo'shimcha ravishda:

- `evaluation_status=queued`
- `evaluation_status=evaluating`
- `evaluation_status=evaluated`
- `evaluation_status=failed`

hamda `evaluation_progress_percent` va `evaluation_attempts` qaytaradi.

## Retry va stale recovery

Transient xatoda job exponential delay bilan qayta `pending` bo'ladi.
`DRAWING_JOB_MAX_ATTEMPTS` tugagach submission `failed` bo'ladi.

Worker restart bo'lsa `DRAWING_JOB_STALE_MINUTES` dan eski `running` joblar
startup vaqtida qayta queue'ga qo'yiladi yoki retry limiti tugagan bo'lsa
`failed` qilinadi.

## Local / single web instance

```env
DRAWING_BACKGROUND_WORKER_ENABLED=true
DRAWING_WORKER_POLL_SECONDS=1
DRAWING_JOB_MAX_ATTEMPTS=3
DRAWING_JOB_RETRY_BASE_SECONDS=10
DRAWING_JOB_STALE_MINUTES=20
```

Bu rejimda bitta process ichida bir vaqtning o'zida bitta Drawing AI job
ishlaydi. Upload requestlar esa AI tugashini kutmaydi.

## Dedicated worker

Web service:

```env
DRAWING_BACKGROUND_WORKER_ENABLED=false
```

Worker command:

```bash
python -m scripts.run_drawing_worker
```

Bir nechta worker instance bir xil PostgreSQL queue'ni xavfsiz consume qila
oladi.

## Migration

Deploydan oldin/yoki yangi kod startidan oldin:

```bash
alembic upgrade head
```

Yangi migration `drawing_evaluation_jobs` jadvalini yaratadi. Oldindan
`pending` bo'lib qolgan submissionlar ham queue'ga seed qilinadi.

## 140+ bir vaqtdagi submission

Queue 140 ta Drawing AI pipeline'ni bir vaqtda RAM'ga yuklamaydi. 140 ta
upload durable tarzda qabul qilinib, worker capacity bo'yicha navbat bilan
baholanadi.

Throughput kerak bo'lsa worker sonini alohida process/instance sifatida
oshirish mumkin. Bunda `SKIP LOCKED` duplicate processingni oldini oladi.
