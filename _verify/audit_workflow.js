export const meta = {
  name: 'knn-project-audit',
  description: 'Audit the K-NN Architetture project vs the PDF spec and adversarially verify findings',
  phases: [
    { title: 'Audit' },
    { title: 'Verify' },
  ],
}

const SPEC = `
PROJECT SPEC (Progetto Architetture 2025 - Calcolo approssimato dei K vicini):
- Approximate K-NN. Dataset DS (n x D), queries Q. Find k nearest per query.
- Quantization quantizing(v,x): vp,vn are D-dim 0/1 vectors. Pick the x components of v with
  largest |value|; set vp[i]=1 if v[i]>=0 else vn[i]=1.
- Approx distance d~(v,w) = (vp.wp)+(vn.wn)-(vp.wn)-(vn.wp)  (dot products of 0/1 vectors).
- Pivot indexing: choose h pivots at positions floor(n/h)*j for j=0..h-1. Index stores d~(v,p) for all v,p.
- Querying: per query, compute d~(q,p) for each pivot p. For each v:
    d_star = max_p |d~(v,p) - d~(q,p)|  (claimed lower bound via triangle inequality);
    if d_star >= current k-th worst, skip; else compute d~(q,v); if < worst, insert into K-NN.
  After loop, recompute the REAL euclidean distance (Eq.1) for the chosen k neighbours; return <id,delta>.
- Params: DS, Q, h (pivots), k (neighbours), x (quantization). If a param is not applicable the
  program must print a message and terminate.
- Required deliverables: C (gcc); 32-bit Single Precision (float) using SSE; 64-bit Double Precision
  (double) using AVX; assembly (the text says nasm) for the optimized routine; an OpenMP version of
  the AVX/64-bit solution; the program must be made available as a Python package; a written report.
`

const ESTABLISHED = `
ALREADY-ESTABLISHED GROUND TRUTH (verified by the orchestrator; treat as context, re-check if cited):
- An independent NumPy reimplementation reproduces data/results_ids_* and results_dst_* EXACTLY
  (2000/2000 queries, identical ordered ids + matching distances) for BOTH 32-bit and 64-bit, with
  h=16, k=8, x=64. So the committed C standalone algorithm is numerically faithful to the golden files.
- The pivot pruning is NOT sound vs a no-pruning baseline of the same approximate-KNN (pruned result
  differs from no-prune in 1992/2000 queries) because d~ behaves like a SIMILARITY (max when v=w, can be
  negative), not a metric — so |d~(v,p)-d~(q,p)| is not a valid lower bound. This is inherent to the SPEC;
  the golden files were generated the same (pruned) way, so the implementation still matches the assignment.
- No C compiler (gcc/MinGW/MSVC) is installed on the machine; only nasm 3.01 and Python 3.14.
`

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    area: { type: 'string' },
    summary: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          title: { type: 'string' },
          severity: { type: 'string', enum: ['high', 'medium', 'low', 'positive'] },
          claim: { type: 'string', description: 'precise claim being asserted' },
          evidence: { type: 'string', description: 'file:line references and quoted snippets proving it' },
          spec_or_report_ref: { type: 'string' },
        },
        required: ['id', 'title', 'severity', 'claim', 'evidence'],
      },
    },
  },
  required: ['area', 'summary', 'findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'partial'] },
    corrected_claim: { type: 'string', description: 'if partial/refuted, the accurate statement' },
    evidence: { type: 'string', description: 'file:line proof from re-reading the cited code' },
  },
  required: ['id', 'verdict', 'evidence'],
}

const auditors = [
  {
    key: 'python-pkg',
    prompt: `Audit the Python packaging for BUILDABILITY and whether it actually implements the K-NN algorithm.
Read: python/setup.py, python/pyproject.toml, src/quantpivot32_py.c, src/quantpivot64_py.c,
src/quantpivot64omp_py.c, src/quantpivot32.c, src/quantpivot64.c, src/quantpivot64omp.c,
python/Gruppo_Ferrari_DeFusco_Cuconato/__init__.py and the sub-package __init__.py files,
python/Gruppo_Ferrari_DeFusco_Cuconato.egg-info/SOURCES.txt and PKG-INFO, python/README.md.
Determine and report as findings: (1) does common.h exist anywhere (grep the repo)? (2) is the symbol
'prova' defined anywhere or only declared extern + called? (3) do fit()/predict() in quantpivot*.c
actually implement indexing/querying, or are they stubs? (4) does setup.py name match pyproject name;
are the source paths in setup.py valid relative to where build runs (python/ vs repo root: src/ lives at
repo ROOT); do the compile flags (-O3 -msse2 -mavx2 -fopenmp, GAS .S sources) require gcc/MinGW (vs MSVC)?
(5) any other reason 'pip install .' would fail. Be concrete with file:line evidence.`,
  },
  {
    key: 'report-accuracy',
    prompt: `Verify the written report's technical claims against the ACTUAL code. Read the report chapters
doc/file_relazione/chapters/capitolo1.tex..capitolo4.tex and report_completo.txt, then read the code they
describe: src/distance.c, src/distance32ASSEMBLY.c, src/distance64ASSEMBLY.c, src/distance_sse2.S,
src/distance_avx2.S, src/quantization.c, src/matrix.c, python/knn_interface.py.
For EACH of these report claims, decide if it is TRUE or FALSE with file:line evidence:
 (a) the Euclidean distance is vectorized in Assembly using SUBPS/MULPS and HADDPS;
 (b) the code uses POPCNT for the approximate distance;
 (c) the code uses aligned loads MOVAPS/VMOVAPD (vs unaligned movdqu/vmovdqu);
 (d) quantization is implemented in Assembly using CMPPS/VCMPPD against a precomputed threshold;
 (e) matrix.c uses aligned allocation rather than plain malloc;
 (f) knn_interface.py is an argparse CLI supporting .ds2/.csv and variant 32/64/64omp selection;
 (g) assembly receives a single struct pointer with offsets like [RDI+64]/[RDI+68] (vs 4 pointers in RCX/RDX/R8/R9);
 (h) do the speedup numbers in capitolo4 (e.g. scalar 6746ms, ASM 400ms ~17x, OpenMP 46ms ~146x, build 157->77ms)
     match the committed report_completo.txt values? Compute the actual ratios from report_completo.txt.
Report each as a finding (severity high if it materially misrepresents the implementation).`,
  },
  {
    key: 'build-system',
    prompt: `Read progetto_knn.cbp (Code::Blocks project) and README.md. Enumerate every build target, and for
each list: the preprocessor defines (USE_SSE2, USE_AVX, USE_SSE2_ASM, USE_AVX_ASM, _OPENMP/-fopenmp),
the source files compiled (which main*.c, which distance*.c, whether .S files are assembled), and the
compiler/flags. Map them to the 8 configurations in README/report_completo.txt (32/64 x Scalar/Intrinsic/
Assembly/OpenMP). Findings to surface: (1) is there a 32-bit SSE2+OpenMP target and a 64-bit AVX2+OpenMP
target (spec requires OpenMP for the AVX/64-bit solution)? (2) the assembly is GAS .S (.intel_syntax),
assembled by gcc — NOT nasm as the spec wording says; (3) is the build Code::Blocks/MinGW-on-Windows-specific
vs the spec's Linux/gcc/nasm environment? Give evidence.`,
  },
  {
    key: 'c-correctness',
    prompt: `Audit the standalone C algorithm for spec-compliance and robustness (NOT performance). Read:
src/index.c, src/query.c, src/query64.c, src/quantization.c, src/distance.c, src/matrix.c, src/config.c,
include/config.h, include/index.h, include/query.h, include/query64.h, src/main.c, src/main64.c.
Confirm each step matches the spec (quantize top-x by |v|; pivots floor(n/h)*j; index stores d~(v,p);
query lower-bound pruning; final REAL euclidean recompute; output as <id,delta>). Then surface findings on
ROBUSTNESS / spec param-handling: the spec says 'if a parameter value is not applicable, print a message and
terminate'. Check: what happens for h<=0 or h>n (select_pivots does step=n/h -> divide-by-zero if h=0; pivots
out of range if h>n), k<=0 or k>n, x<=0 or x>D, missing -d/-q, non-existent files, dataset/query D mismatch.
Note that main.c hardcodes the comparison file path data/results_ids_2000x8_k8_x64_32.ds2 and requires k to
match. Also note d_star/d~ are ints stored into float/double dist_approx. Report findings with file:line.`,
  },
  {
    key: 'asm-correctness',
    prompt: `Audit the two assembly routines for correctness and ABI compliance. Read src/distance_sse2.S and
src/distance_avx2.S (and src/distance32ASSEMBLY.c, src/distance64ASSEMBLY.c that call them, and how
src/quantpivot*_py.c reference an undefined 'prova'). Verify: (1) the computed value equals pp+nn-pn-np where
pp=popcount(vp&wp) etc.; (2) Win64 ABI non-volatile register save/restore is correct (XMM6-15 saved? GPR
r12-r15?); SPECIFICALLY check whether the AVX2 .reduce_generic path writes r12 (movq r12, xmm9) WITHOUT saving
it — r12 is non-volatile in Win64 — and whether the D=256 fast path avoids that; (3) the stack offset used to
read D ([rsp+176] in SSE2 after sub 104 + 4 pushes; [rsp+128] in AVX after sub 88) is correct; (4) loads are
unaligned (movdqu/vmovdqu) so there is no 16/32-byte alignment requirement; (5) the D=256 fully-unrolled path
is equivalent to the generic loop. Report findings with file:line and say which are real bugs vs latent
(not hit at D=256) vs cosmetic/dead-code.`,
  },
]

phase('Audit')
const results = await pipeline(
  auditors,
  a => agent(`${SPEC}\n${ESTABLISHED}\n\nYou are auditing area "${a.key}".\n${a.prompt}\n\nReturn a structured findings object. Quote real file:line evidence; do not speculate. Working directory is the repo root.`,
    { label: `audit:${a.key}`, phase: 'Audit', schema: FINDINGS_SCHEMA }),
  (res, a) => {
    if (!res || !res.findings) return []
    // adversarially verify the non-trivial findings (skip pure 'low' notes)
    const toVerify = res.findings.filter(f => f.severity === 'high' || f.severity === 'medium' || f.severity === 'positive')
    return parallel(toVerify.map(f => () =>
      agent(`${SPEC}\n\nAdversarially verify this audit finding by RE-READING the cited files yourself. Try to REFUTE it. ` +
            `If the evidence does not hold up, mark refuted/partial and give the accurate statement.\n\n` +
            `AREA: ${a.key}\nFINDING ${f.id}: ${f.title}\nCLAIM: ${f.claim}\nCITED EVIDENCE: ${f.evidence}\n`,
        { label: `verify:${a.key}:${f.id}`, phase: 'Verify', schema: VERDICT_SCHEMA })
        .then(v => ({ area: a.key, finding: f, verdict: v }))
        .catch(() => ({ area: a.key, finding: f, verdict: null }))
    ))
  }
)

const flat = results.flat().filter(Boolean)
const confirmed = flat.filter(r => r.verdict && r.verdict.verdict === 'confirmed')
const partial = flat.filter(r => r.verdict && r.verdict.verdict === 'partial')
const refuted = flat.filter(r => r.verdict && r.verdict.verdict === 'refuted')
log(`Audit complete: ${flat.length} findings checked — ${confirmed.length} confirmed, ${partial.length} partial, ${refuted.length} refuted`)

return {
  byArea: results.map((r, i) => ({ area: auditors[i].key })),
  confirmed: confirmed.map(r => ({ ...r.finding, area: r.area, verdict_evidence: r.verdict.evidence })),
  partial: partial.map(r => ({ ...r.finding, area: r.area, corrected: r.verdict.corrected_claim, verdict_evidence: r.verdict.evidence })),
  refuted: refuted.map(r => ({ ...r.finding, area: r.area, corrected: r.verdict.corrected_claim })),
}
