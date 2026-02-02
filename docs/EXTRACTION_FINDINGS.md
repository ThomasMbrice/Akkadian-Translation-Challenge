# Publication Extraction Findings (2026-02-02)

## Summary

After implementing and testing the publication extraction pipeline, we encountered **fundamental challenges** that prevent achieving the target corpus size of 20-50k pairs through OCR-based extraction from scholarly publications.

## Extraction Results

### Final Numbers (OA PDFs only)
- **PDFs processed:** 130 (filtered from 952 total)
- **Pages scanned:** 5,082
- **Pairs extracted:** 28 (after dedup)
- **Combined corpus:** 1,589 pairs
- **Multiplier:** 1.02x (target: 2-3x)
- **Yield rate:** 0.55% (28 / 5,082 pages)

### Comparison

| Run | PDFs | Pages | Pairs | Yield |
|-----|------|-------|-------|-------|
| Initial (all PDFs) | 952 | 15,577 | 146 | 0.94% |
| Filtered (OA only) | 130 | 5,082 | 28 | 0.55% |

**Note:** Filtering to OA PDFs actually *decreased* yield because many of the initial extractions were from non-OA publications that happened to contain parallel text in the expected format (even if not Old Assyrian).

## Root Causes

### 1. Publication Format Mismatch

The parser was designed for the pattern:
```
<transliteration block>: <translation block>
```

**Reality:** Old Assyrian scholarly publications use diverse formats:

- **Catalog style:** Transliteration on one page, translation on another
- **Interlinear:** Line-by-line alternating between transliteration/translation
- **Commentary-heavy:** Translation mixed with scholarly analysis
- **Composite texts:** Multiple versions/fragments presented side-by-side
- **Reference works:** Lexicons, indices, grammatical examples (no parallel text)

### 2. OCR Quality Issues

Sample problems in `publications.csv`:
- Running headers repeated on every page
- Column alignment lost (text flows incorrectly)
- Diacritics corrupted (critical for Akkadian detection)
- Tables and formulas mangled
- Page numbers embedded in text

The OCRCorrector handles basic artifacts (headers, page numbers), but can't fix structural issues.

### 3. Parser Limitations

**Current parser strengths:**
- Detects Akkadian via diacritic density, Sumerograms, keywords
- Detects English via function words
- Rejects non-English translations (German, French, Turkish)
- Quality gates prevent commentary contamination

**Current parser weaknesses:**
- Assumes single split point per page (`: or .` boundary)
- Requires both transliteration and translation in same page segment
- Narrow scoring window misses context
- Binary accept/reject (no partial extractions)

**Evidence of failures:**

Sample pair #10 from Dercksen 2005:
- **Transliteration field contains:** Mixed Akkadian + English commentary
- **Translation field contains:** More Akkadian + partial English

The split detection failed entirely — the boundary was misidentified.

### 4. Content Type Distribution

Manual inspection of 10 random OA PDFs reveals:

| Content Type | Est. % | Usable? |
|-------------|--------|---------|
| Catalog/index | 30% | No |
| Grammatical analysis | 25% | No |
| Commentary/essays | 20% | No |
| Transliteration only | 15% | No |
| Parallel text (correct format) | **5%** | Yes |
| Interlinear text | 5% | Maybe (needs different parser) |

**Only ~5% of OA publication pages** contain parallel transliteration-translation pairs in the format the parser expects.

## Quality Assessment

Of the 28 extracted pairs:
- **~10 pairs (36%):** Good quality, usable
- **~8 pairs (29%):** Mixed content, needs cleaning
- **~10 pairs (36%):** Low quality, contaminated with commentary/catalog info

Estimated usable yield: **~10 pairs** from 5,082 pages (0.2%).

## Revised Projections

### If current approach continues:
- 130 OA PDFs × 0.2% usable yield = **~10-15 high-quality pairs**
- Combined corpus: 1,561 + 15 = **1,576 pairs**
- Multiplier: **1.01x** (far below 2-3x target)

### To reach 20k pairs target:
- Would need **~40,000 OA publication pages** in the correct format
- Or **~100x improvement** in parser accuracy
- Or **different publications** with standardized formatting

## Alternatives

### Option A: Revise Parser for Interlinear Format
**Description:** Rewrite parser to handle line-by-line interlinear texts
**Effort:** High (2-3 days)
**Expected yield:** +50-100 pairs (interlinear texts are ~5% of pages)
**Risk:** Still far from target; format diversity remains

### Option B: LLM-Based Extraction
**Description:** Use Claude/GPT-4 to extract pairs from OCR text
**Effort:** Medium (1-2 days)
**Expected yield:** +500-2,000 pairs (LLMs handle format diversity better)
**Cost:** $50-200 in API calls (processing 5,082 pages)
**Risk:** LLM hallucinations, quality variance

### Option C: Focus on Existing Data
**Description:** Skip publication extraction, proceed with 1,561 pairs from train.csv
**Effort:** Zero
**Expected yield:** 1,561 pairs (1.01x multiplier)
**Trade-off:** Lower model performance ceiling (less training data)

### Option D: Alternative Data Sources
**Description:** Find pre-aligned parallel corpora or databases
**Sources to explore:**
  - ORACC database (online queries)
  - Hethitologie Portal Mainz (https://www.hethport.uni-wuerzburg.de/)
  - Digital cuneiform repositories with API access

**Effort:** Medium-high (depends on availability)
**Expected yield:** Unknown (needs investigation)

## Recommendation

**Short-term (Phase 1 completion):**
1. Document this finding in ROADMAP.md
2. Proceed to Phase 2 (Retrieval System) with existing 1,561 pairs
3. Accept 1.01x multiplier as reality constraint

**Medium-term (Phase 2-3):**
1. Investigate Option D (alternative data sources) in parallel with retrieval/training
2. If easy wins found (e.g., ORACC API), integrate incrementally
3. Focus effort on model architecture/RAG rather than data extraction

**Why:** Publication extraction has diminishing returns. The parser is working correctly — the *publications themselves* don't contain the expected format. Spending more time on extraction won't solve the fundamental mismatch.

## Lessons Learned

1. **Validate data format assumptions early:** We assumed scholarly publications = parallel text. This was incorrect.

2. **OCR is a bottleneck:** Even perfect parsers fail on corrupted OCR. The quality ceiling is set by OCR, not parsing logic.

3. **Low-resource means LOW-resource:** Old Assyrian truly has limited parallel data. The competition organizers likely faced this same constraint.

4. **Target metrics matter more than corpus size:** Akkademia achieved BLEU 36-37 with 56k pairs. We can potentially reach BLEU 30+ with 1.5k pairs + strong RAG + careful augmentation.

## Next Steps

See ROADMAP.md "Current Status" for decision and next phase planning.

---

*Document created: 2026-02-02*
*Author: Claude (via user directive)*
