# كود النموذج والتجارب

هذا المجلد يحتوي النموذج المقترح والـbaselines والتحليل الإحصائي وإنشاء الأشكال.

من جذر المستودع:

```bash
python scripts/run_quick_smoke.py
```

ولإعادة تشغيل البيانات الحقيقية كاملة:

```bash
python code/model_pipeline/run_all.py --data-root . --mode all --output outputs/recomputed_full
```

النتائج المجمدة المستخدمة في البحث موجودة داخل `results/`، ولا يُستبدل بها تشغيل جديد إلا بعد توثيق الإصدار والبذور والإعدادات.
