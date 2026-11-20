import json
import re
from pathlib import Path

p = next(Path('data/synthetic_heavy_truck_research').glob('*corpus*.jsonl'))
words = ('cause', 'caused', 'causes', 'prove', 'proves', 'proof', 'guarantee', 'guarantees')
for i, ln in enumerate(p.read_text(encoding='utf-8').splitlines()):
    doc = json.loads(ln)
    text = (doc.get('title', '') + ' ' + ' '.join(doc.get('chunks', []) if isinstance(doc.get('chunks'), list) else [doc.get('content', '')])).lower()
    hits = [w for w in words if w in text]
    if hits:
        print(doc.get('id'), doc.get('title'), hits)
