#!/usr/bin/env python3
from pathlib import Path
import pycolmap
from src.evaluator.evaluator import eval_multiview

BASE_DIR = Path("SfM_dataset/eth3d_triangulation_dataset")
SCENES = ["kicker", "office", "delivery_area", "facade", "meadow", "relief",
          "electro", "terrace", "terrains", "courtyard", "relief_2", "pipes", "playground"]
METHOD_DIR = "DetectorFreeSfM_loftr_official_coarse_only__pri_pose"
EVAL_TOOL = Path("third_party/multi-view-evaluation/build/ETH3DMultiViewEvaluation")

results = {}
for scene in SCENES:
    recon_dir = BASE_DIR / scene / METHOD_DIR / "colmap_refined"
    if not recon_dir.exists():
        print(f"⚠️ {scene}: not found")
        continue

    model = pycolmap.Reconstruction(str(recon_dir))
    ply_path = recon_dir / "points.ply"
    model.export_PLY(str(ply_path))

    gt_scan = BASE_DIR / scene / "dslr_scan_eval" / "scan_alignment.mlp"
    result = eval_multiview(EVAL_TOOL, ply_path, gt_scan, [0.01, 0.02, 0.05])
    results[scene] = result

# 输出表格
print("\n" + "=" * 80)
print(f"{'Scene':<15} {'Acc@1cm':>8} {'Acc@2cm':>8} {'Acc@5cm':>8} {'Comp@1cm':>9} {'Comp@2cm':>9} {'Comp@5cm':>9}")
print("=" * 80)

for scene, r in results.items():
    print(f"{scene:<15} "
          f"{r['accuracy']['accuracy@0.01'] * 100:>7.2f}% "
          f"{r['accuracy']['accuracy@0.02'] * 100:>7.2f}% "
          f"{r['accuracy']['accuracy@0.05'] * 100:>7.2f}% "
          f"{r['completeness']['completeness@0.01'] * 100:>8.2f}% "
          f"{r['completeness']['completeness@0.02'] * 100:>8.2f}% "
          f"{r['completeness']['completeness@0.05'] * 100:>8.2f}%")

if results:
    avg_acc_1 = sum(r['accuracy']['accuracy@0.01'] for r in results.values()) / len(results)
    avg_acc_2 = sum(r['accuracy']['accuracy@0.02'] for r in results.values()) / len(results)
    avg_acc_5 = sum(r['accuracy']['accuracy@0.05'] for r in results.values()) / len(results)
    avg_comp_1 = sum(r['completeness']['completeness@0.01'] for r in results.values()) / len(results)
    avg_comp_2 = sum(r['completeness']['completeness@0.02'] for r in results.values()) / len(results)
    avg_comp_5 = sum(r['completeness']['completeness@0.05'] for r in results.values()) / len(results)

    print("=" * 80)
    print(f"{'Average':<15} "
          f"{avg_acc_1 * 100:>7.2f}% "
          f"{avg_acc_2 * 100:>7.2f}% "
          f"{avg_acc_5 * 100:>7.2f}% "
          f"{avg_comp_1 * 100:>8.2f}% "
          f"{avg_comp_2 * 100:>8.2f}% "
          f"{avg_comp_5 * 100:>8.2f}%")