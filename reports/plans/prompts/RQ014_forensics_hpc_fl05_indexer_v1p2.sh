#!/usr/bin/env bash
# RQ014 FL05 v1.2 — fail-closed structured indexer of ALL recorded correlation statistics
# in RQ010B intermediate outputs. Produces the registered CSV schema, untruncated.
# Canonical CSV goes to STDOUT; audit/summary goes to STDERR. Nonzero exit on ANY failure.
# Run from Mac (repo root):
#   ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'bash -s' \
#     < reports/plans/prompts/RQ014_forensics_hpc_fl05_indexer_v1p2.sh \
#     >  reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/historical_stats_index.csv \
#     2> reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/fl05_audit_20260711.log
set -euo pipefail

python3 - <<'PYEOF'
import csv, hashlib, io, json, os, re, sys, datetime

ROOTS = ["/share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis",
         "/share/home/u25310231/ZXC/RQ010B_wod_e2e/results"]
LOGIN_NODE_BYTE_BUDGET = 200 * 1024 * 1024  # frozen: above this, abort -> run as zxc- sbatch
STAT_PAT = re.compile(r"(spearman|pearson|kendall|rho|corr(elation)?)", re.I)
NUM_PAT  = re.compile(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?")
N_KEYS   = re.compile(r"^(n|count|n_scenes|n_segments|n_rows|n_candidates|num[_a-z]*)$", re.I)

def fail(msg):
    sys.stderr.write("FL05_FATAL: %s\n" % msg); sys.exit(2)

for r in ROOTS:
    if not os.path.isdir(r):
        fail("input root missing: %s (missing tree must NOT look like a negative scan)" % r)

files, total = [], 0
for r in ROOTS:
    for dp, _, fns in os.walk(r):
        for fn in fns:
            if fn.lower().endswith((".csv", ".json", ".md")):
                p = os.path.join(dp, fn)
                try: st = os.stat(p)
                except OSError as e: fail("stat failed: %s (%s)" % (p, e))
                files.append((p, st.st_size, st.st_mtime)); total += st.st_size
sys.stderr.write("FL05 file manifest: %d files, %d bytes\n" % (len(files), total))
if total > LOGIN_NODE_BYTE_BUDGET:
    fail("tree %d bytes exceeds frozen login-node budget %d; rerun as a zxc- CPU sbatch job"
         % (total, LOGIN_NODE_BYTE_BUDGET))

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""): h.update(ch)
    return h.hexdigest()

w = csv.writer(sys.stdout, lineterminator="\n")
w.writerow(["statistic_name","value","n","unit","config_fields_as_recorded",
            "source_file","source_file_sha256","mtime","parse_status","raw_locator"])
rows = unparsed = 0

def emit(name, value, n, unit, cfg, path, sha, mt, status, loc):
    global rows, unparsed
    w.writerow([name, value, n, unit, cfg, path, sha, mt, status, loc])
    rows += 1
    if status.startswith("UNPARSED"): unparsed += 1

def walk_json(obj, path, ctx, meta):
    if isinstance(obj, dict):
        sibs = {k: v for k, v in obj.items() if isinstance(v, (int, float, str))}
        for k, v in obj.items():
            if STAT_PAT.search(str(k)) and isinstance(v, (int, float)):
                n = next((sibs[s] for s in sibs if N_KEYS.match(s)), "")
                cfg = json.dumps({a: b for a, b in sibs.items()
                                  if not STAT_PAT.search(a)}, ensure_ascii=False)
                emit(k, v, n, "as_recorded", cfg, *meta, "PARSED_JSON", path + "/" + k)
            walk_json(v, path + "/" + str(k), sibs, meta)
    elif isinstance(obj, list):
        for i, v in enumerate(obj): walk_json(v, "%s[%d]" % (path, i), ctx, meta)

for p, size, mtime in sorted(files):
    sha = sha256(p)
    mt  = datetime.datetime.fromtimestamp(mtime).isoformat()
    meta = (p, sha, mt)
    try:
        raw = open(p, encoding="utf-8", errors="replace").read()
    except OSError as e:
        fail("read failed: %s (%s)" % (p, e))
    ext = p.lower().rsplit(".", 1)[-1]
    if ext == "json":
        try: walk_json(json.loads(raw), "$", {}, meta)
        except Exception:
            for i, ln in enumerate(raw.splitlines(), 1):
                if STAT_PAT.search(ln):
                    emit("json_parse_failed_line", "", "", "", ln.strip(), *meta,
                         "UNPARSED_CANDIDATE", "line:%d" % i)
    elif ext == "csv":
        try:
            rdr = list(csv.reader(io.StringIO(raw)))
            if rdr:
                hdr = rdr[0]
                sc = [i for i, h in enumerate(hdr) if STAT_PAT.search(h)]
                ncol = next((i for i, h in enumerate(hdr) if N_KEYS.match(h)), None)
                if sc:
                    other = [i for i in range(len(hdr)) if i not in sc]
                    for rno, row in enumerate(rdr[1:], 2):
                        if len(row) != len(hdr): continue
                        cfg = json.dumps({hdr[i]: row[i] for i in other}, ensure_ascii=False)
                        for i in sc:
                            if row[i].strip():
                                emit(hdr[i], row[i], row[ncol] if ncol is not None else "",
                                     "as_recorded", cfg, *meta, "PARSED_CSV", "row:%d" % rno)
        except Exception as e:
            fail("csv parse crashed (fail-closed): %s (%s)" % (p, e))
    else:  # md
        for i, ln in enumerate(raw.splitlines(), 1):
            if STAT_PAT.search(ln):
                m = NUM_PAT.search(ln)
                emit(STAT_PAT.search(ln).group(0).lower(), m.group(0) if m else "",
                     "", "as_recorded", ln.strip(), *meta,
                     "PARSED_MD_LINE" if m else "UNPARSED_CANDIDATE", "line:%d" % i)

sys.stderr.write("FL05 completeness: files=%d stat_rows=%d unparsed_candidates=%d\n"
                 % (len(files), rows, unparsed))
if rows == 0:
    fail("zero statistic rows extracted from a non-empty tree; refusing silent negative")
sys.stderr.write("FL05_OK\n")
PYEOF
