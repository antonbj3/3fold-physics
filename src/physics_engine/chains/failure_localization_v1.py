"""Deterministic delta debugging for PASS/FAIL/INVALID subset oracles."""
import math


def minimize_failure(items,oracle):
    """Find a deletion-minimal witness, not a globally smallest physical cause."""
    items=tuple(items)
    if len(set(items))!=len(items):raise ValueError('Unique identifiers required')
    cache={};trace=[]
    def evaluate(candidate):
        candidate=tuple(candidate)
        if candidate not in cache:
            verdict=oracle(candidate)
            if verdict not in ('PASS','FAIL','INVALID'):raise ValueError('Unknown verdict')
            cache[candidate]=verdict;trace.append(dict(inputs=list(candidate),verdict=verdict))
        return cache[candidate]
    if evaluate(items)!='FAIL':raise ValueError('Original input must fail')
    current=items;n=2
    while len(current)>=2:
        width=math.ceil(len(current)/n);chunks=[current[i:i+width] for i in range(0,len(current),width)]
        candidates=chunks+[tuple(x for x in current if x not in chunk) for chunk in chunks]
        reduced=next((c for c in candidates if c and c!=current and evaluate(c)=='FAIL'),None)
        if reduced is not None:current=reduced;n=max(2,n-1)
        elif n>=len(current):break
        else:n=min(len(current),n*2)
    checks=[dict(removed=x,verdict=evaluate(tuple(y for y in current if y!=x))) for x in current]
    return dict(witness=list(current),verdict=evaluate(current),deletion_minimal=all(c['verdict']!='FAIL' for c in checks),
                deletion_checks=checks,trace=trace,scope='Deletion-minimal witness, not unique root cause or global minimum')


def bisect_interval(length,oracle,min_samples=256):
    """Retain the first failing time half and report both sibling verdicts."""
    if length<min_samples or min_samples<2:raise ValueError('Insufficient samples')
    lo,hi=0,length;trace=[]
    if oracle(lo,hi)!='FAIL':raise ValueError('Full interval must fail')
    while hi-lo>=2*min_samples:
        mid=(lo+hi)//2
        rows=[dict(start=a,stop=b,verdict=oracle(a,b)) for a,b in ((lo,mid),(mid,hi))];trace.extend(rows)
        if any(r['verdict'] not in ('PASS','FAIL','INVALID') for r in rows):raise ValueError('Unknown verdict')
        failed=[r for r in rows if r['verdict']=='FAIL']
        if not failed:break
        lo,hi=failed[0]['start'],failed[0]['stop']
    return dict(start=lo,stop=hi,trace=trace,min_samples=min_samples,
                scope='First retained failing half; sibling failures remain evidence against a unique time site')
