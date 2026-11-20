import json
import re
from src.agent.response_validation import _CAUSAL_PATTERNS

d = json.load(open('reports/m2/response_evaluation/evaluation.json', encoding='utf-8'))
for r in d['results']:
    if r['id'] in ('K01', 'C01', 'I01'):
        text = r['driver_postview_preview']
        print(r['id'], 'FULL:', repr(text))
        for p in _CAUSAL_PATTERNS:
            m = p.search(text)
            if m:
                print('   MATCHED pattern=', p.pattern, 'at', m.group(0))
