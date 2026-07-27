# RAG chat excerpt polishing

AI Mentor assistant xabarining `metadata_json.rag.sources[].excerpt` maydoni endi qat’iy `[:500]` kesishdan foydalanmaydi.

Yangi algoritm:

1. matn limitdan qisqa bo‘lsa o‘zgartirmaydi;
2. avval paragraf yoki to‘liq gap chegarasini izlaydi;
3. gap chegarasi topilmasa so‘z chegarasidan kesadi;
4. qisqartirilgan matn oxiriga `…` qo‘shadi;
5. natija belgilangan limitdan oshmaydi.

Xuddi shu boundary-aware yordamchi RAG kontekstining umumiy character budgeti tugayotgan oxirgi chunk uchun ham ishlatiladi. Bu LLM promptida va frontenddagi manba parchalarida so‘zning yarim qolishini kamaytiradi.

Mavjud eski chat metadata yozuvlari o‘zgarmaydi. Yangi assistant xabarlari yangi excerpt formatida saqlanadi.
