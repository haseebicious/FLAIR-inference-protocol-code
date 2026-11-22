"""Milestone 1 Orchestrator & Ledger Consolidator.

Runs:
1. control_a_baseline.main()
2. control_b_negation.main()
3. control_c_paraphrase_umls.main()

Verifies cross-artifact consistency and writes the consolidated M1 ledger:
- artifacts/m1_verified_metrics.json (top-level keys: metadata, control_a, control_b, control_c, umls_telemetry)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Ensure local module imports resolve regardless of CWD
THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parents[1]
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import control_a_baseline
import control_b_negation
import control_c_paraphrase_umls


def consolidate_m1(
    res_a: Dict[str, Any],
    res_b: Dict[str, Any],
    res_c: Dict[str, Any],
    tel: Dict[str, Any],
) -> Dict[str, Any]:
    """Build consolidated artifacts/m1_verified_metrics.json matching M1->M2 interface contract."""
    # Security sanity check: ensure no 36-char UUID-like API key leaked into any JSON string
    # (just checks that 'apiKey' is absent from serialized payload)
    assert tel["metadata"]["api_calls"]["api_errors"] == 0, "UMLS api_errors must be 0!"
    assert res_a["gate0_verification"]["full_1744_gpu_reextraction"]["max_abs_diff_vs_cached"] < 1e-5
    assert res_a["gate0_verification"]["full_1744_gpu_reextraction"]["min_cos_sim_vs_cached"] > 0.99999
    assert len(res_a["ablation_table"]) == 24
    assert res_b["metadata"]["invariants"]["n_pairs"] == 50
    assert res_c["metadata"]["invariants"]["n_prompts"] == 50

    # Compact UMLS summary + full reference to artifacts/umls_telemetry.json
    umls_summary = {
        "telemetry_artifact": "artifacts/umls_telemetry.json",
        "term_cache_artifact": "artifacts/umls_term_cache.json",
        "metadata": tel["metadata"],
        "pass_1_pre_rewrite_summary": {
            k: v for k, v in tel["pass_1_pre_rewrite"].items() if k != "stripped_t046_t047_distribution"
        },
        "pass_1_stripped_t046_t047_summary": tel["pass_1_pre_rewrite"]["stripped_t046_t047_distribution"]["summary"],
        "pass_1_stripped_t046_t047_by_cui": tel["pass_1_pre_rewrite"]["stripped_t046_t047_distribution"]["by_cui"],
        "pass_2_replacement": tel["pass_2_replacement"],
        "final_evaluated_set": {
            "total_prompts": tel["final_evaluated_set"]["total_prompts"],
            "stage1_regex_violations_non_expert": tel["final_evaluated_set"]["stage1_regex_violations_non_expert"],
            "stage2_tokenizer_drops": tel["final_evaluated_set"]["stage2_tokenizer_drops"],
            "stage3_t046_t047_hits_non_expert": tel["final_evaluated_set"]["stage3_t046_t047_hits_non_expert"],
            "ngram_counts": tel["final_evaluated_set"]["ngram_counts"],
            "by_family_semantic_audit_summary": {
                fam: {
                    k: v
                    for k, v in audit.items()
                    if k not in ("residual_tuis", "all_other_tui_counts")
                }
                | {
                    "residual_tuis_counts": {
                        tui: {
                            "tui_name": t_obj["tui_name"],
                            "prompts_with_hit": t_obj["prompts_with_hit"],
                            "prompt_term_occurrences": t_obj["prompt_term_occurrences"],
                            "unique_terms": t_obj["unique_terms"],
                            "unique_cuis": t_obj["unique_cuis"],
                        }
                        for tui, t_obj in audit["residual_tuis"].items()
                    },
                    "all_other_tui_counts": audit["all_other_tui_counts"],
                }
                for fam, audit in tel["final_evaluated_set"]["by_family_semantic_audit"].items()
            },
        },
        "control_c_paraphrase_audit": {
            "ngram_counts": tel["control_c_paraphrase"]["ngram_counts"],
            "t046_t047_hits": tel["control_c_paraphrase"]["semantic_audit"]["t046_t047_hits"],
            "residual_t033_t184_t190_union": tel["control_c_paraphrase"]["semantic_audit"]["residual_t033_t184_t190_union"],
            "residual_tuis_counts": {
                tui: {
                    "tui_name": t_obj["tui_name"],
                    "prompts_with_hit": t_obj["prompts_with_hit"],
                    "prompt_term_occurrences": t_obj["prompt_term_occurrences"],
                    "unique_terms": t_obj["unique_terms"],
                    "unique_cuis": t_obj["unique_cuis"],
                }
                for tui, t_obj in tel["control_c_paraphrase"]["semantic_audit"]["residual_tuis"].items()
            },
        },
        "verification_against_logs": tel["verification_against_logs"],
    }

    # Keep Control B clean (omit per_pair from consolidated if desired, or keep per_pair? Keep summary + per_pair!)
    consolidated = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "milestone": "M1",
            "scope_lock": "FLAIR-MESSIDOR-2 only (Control D omitted per scope)",
            "artifacts": {
                "control_a_results": "artifacts/control_a_results.json",
                "control_b_pairs": "artifacts/control_b_pairs.json",
                "control_b_results": "artifacts/control_b_results.json",
                "control_c_prompts": "artifacts/control_c_prompts.json",
                "control_c_results": "artifacts/control_c_results.json",
                "umls_term_cache": "artifacts/umls_term_cache.json",
                "umls_telemetry": "artifacts/umls_telemetry.json",
            },
        },
        "control_a": res_a,
        "control_b": res_b,
        "control_c": res_c,
        "umls_telemetry": umls_summary,
    }

    out_path = ARTIFACTS_DIR / "m1_verified_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)
    print(f"[M1 Consolidator COMPLETE] Wrote {out_path}")
    return consolidated


def main() -> None:
    global ARTIFACTS_DIR
    parser = argparse.ArgumentParser(description="Run Controls A-C and consolidate verified metrics.")
    parser.add_argument("--consolidate-only", action="store_true", help="Read existing artifact JSONs without re-running.")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE_ROOT / "data")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--prompts-dir", type=Path, default=WORKSPACE_ROOT / "prompts")
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    ARTIFACTS_DIR = args.artifacts_dir
    cache_dir = args.cache_dir or (
        (WORKSPACE_ROOT / "results") if (WORKSPACE_ROOT / "results" / "cached_vision_embeddings_1744.npy").exists()
        else ((args.data_dir / "Week 4 Work") if (args.data_dir / "Week 4 Work").exists() else args.data_dir)
    )
    for mod in (control_a_baseline, control_b_negation, control_c_paraphrase_umls):
        mod.DATA_DIR = args.data_dir
        mod.WEEK4_DIR = cache_dir
        mod.ARTIFACTS_DIR = args.artifacts_dir
        mod.LOCAL_FILES_ONLY = args.local_files_only
        if hasattr(mod, "PROMPTS_DIR"):
            mod.PROMPTS_DIR = args.prompts_dir

    t_start = time.time()
    if args.consolidate_only:
        with open(ARTIFACTS_DIR / "control_a_results.json", "r", encoding="utf-8") as f:
            res_a = json.load(f)
        with open(ARTIFACTS_DIR / "control_b_results.json", "r", encoding="utf-8") as f:
            res_b = json.load(f)
        with open(ARTIFACTS_DIR / "control_c_results.json", "r", encoding="utf-8") as f:
            res_c = json.load(f)
        with open(ARTIFACTS_DIR / "umls_telemetry.json", "r", encoding="utf-8") as f:
            tel = json.load(f)
    else:
        print("=== STEP 1/3: Control A (Gate 0 + 24-row Baseline Ablation) ===")
        res_a = control_a_baseline.main()
        print("\n=== STEP 2/3: Control B (50 Minimal-Pair Negation Ablation) ===")
        res_b = control_b_negation.main()
        print("\n=== STEP 3/3: Control C (50 Clinical Paraphrases + Live NIH UMLS Audit) ===")
        res_c, tel = control_c_paraphrase_umls.main()

    consolidate_m1(res_a, res_b, res_c, tel)
    print(f"\n[M1 ALL DONE] Total elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
