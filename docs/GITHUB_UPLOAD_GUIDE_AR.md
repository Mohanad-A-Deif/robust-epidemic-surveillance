# خطوات رفع الحزمة على GitHub

1. فك ضغط ملف `Robust_Epidemic_Surveillance_Supplementary_GitHub.zip`.
2. أنشئ مستودعًا جديدًا باسم مقترح:
   `robust-epidemic-surveillance-reproducibility`
3. لا تضف README أو LICENSE من GitHub عند الإنشاء؛ الحزمة تحتويهما بالفعل.
4. ارفع **كل محتويات المجلد بعد فك الضغط**.
5. نفّذ Commit بعنوان:
   `Initial reproducibility release v1.0.0`
6. من صفحة Releases أنشئ Release باسم `v1.0.0` وارفع ملف ZIP الكامل كأصل إضافي.
7. بعد قبول البحث، أضف DOI المقال إلى `CITATION.cff` و`README.md`.

## قبل جعل المستودع Public

- راجع أسماء جميع المؤلفين في `CITATION.cff`؛ تم إدراج Mohanad A. Deif بصفته مسؤول الحزمة، لأن قائمة المؤلفين الكاملة لم تكن متاحة داخل الملفات.
- تأكد من موافقة جميع المؤلفين على نشر الكود والنتائج.
- لا ترفع قرار المجلة أو تعليقات المحكمين أو خطاب الرد.
- لا تغيّر ملفات النتائج بعد النشر دون إصدار نسخة جديدة موثقة.
