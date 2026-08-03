# الحزمة التكميلية وإعادة الإنتاج للبحث

هذه هي الحزمة الكاملة الجاهزة للرفع على GitHub للبحث:

> **A Robust Analytics Approach to Graph-Based Epidemic Surveillance Under Missing and Delayed Data**

## محتوى الحزمة

- بيانات RKI الخام الموثقة وبصمتها الرقمية.
- البيانات بعد حساب incidence لكل 100 ألف نسمة ثم `log1p`.
- تقسيم زمني Train/Validation/Test دون تسريب Test.
- 14 سيناريو فساد اتصال × 20 بذرة ثابتة = 280 ملف رسائل مضغوط.
- قوالب ثابتة للفقد والتأخير والقيم الشاذة لكل seed.
- الكود الكامل للنموذج والـbaselines والتحليل الإحصائي.
- النتائج الخام على مستوى كل seed.
- الجداول النهائية والأشكال PNG بدقة 600 DPI.
- ملفات LaTeX للمنهجية والنتائج والمناقشة والخاتمة الجديدة.

## التنبيه العلمي الأساسي

المسارات الوبائية حقيقية من RKI، لكن الفقد والتأخير والقيم الشاذة محقونة اصطناعيًا بطريقة محكومة. الجراف الجغرافي مرجع ناعم فقط وليس شبكة انتقال عدوى حقيقية. النتائج لا تثبت تفوق النموذج في كل الظروف، كما أن causal nowcasting واسترداد اتجاهات الجراف ما زالا نقطتي ضعف واضحتين.

## التشغيل

```bash
python -m pip install -r requirements.txt
python scripts/run_tests.py
python scripts/validate_repository.py
```

إعادة توليد البيانات كاملة:

```bash
python scripts/reproduce_data.py
```

اختبار سريع:

```bash
python scripts/run_quick_smoke.py
```

## الرفع على GitHub

فك ضغط الملف النهائي، ثم ارفع **محتويات المجلد** إلى مستودع GitHub، وليس ملف ZIP نفسه داخل المستودع. يمكن رفع ملف ZIP أيضًا ضمن GitHub Release باسم `v1.0.0`.

راجع [`docs/GITHUB_UPLOAD_GUIDE_AR.md`](docs/GITHUB_UPLOAD_GUIDE_AR.md) للخطوات الدقيقة.
