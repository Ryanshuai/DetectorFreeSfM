#!/usr/bin/env python3
"""评估已生成的ETH3D重建结果"""
from pathlib import Path
import pycolmap
from src.evaluator.evaluator import eval_multiview

# 配置
BASE_DIR = Path("SfM_dataset/eth3d_triangulation_dataset")
SCENES = ["kicker", "office", "delivery_area", "facade", "meadow", "relief",
          "electro", "terrace", "terrains", "courtyard", "relief_2", "pipes", "playground"]
METHOD_DIR = "DetectorFreeSfM_loftr_official_coarse_only__pri_pose"
EVAL_TOOL = Path("third_party/multi-view-evaluation/build/ETH3DMultiViewEvaluation")

results = {}
for scene in SCENES:
    recon_dir = BASE_DIR / scene / METHOD_DIR / "colmap_refined"
    if not recon_dir.exists():
        print(f"⚠️  {scene}: not found")
        continue

    # 导出PLY
    model = pycolmap.Reconstruction(str(recon_dir))
    ply_path = recon_dir / "points.ply"
    model.export_PLY(str(ply_path))

    # 评估
    gt_scan = BASE_DIR / scene / "dslr_scan_eval" / "scan_alignment.mlp"
    result = eval_multiview(EVAL_TOOL, ply_path, gt_scan, [0.01, 0.02, 0.05])

    acc1 = result['accuracy']['accuracy@0.01']
    comp1 = result['completeness']['completeness@0.01']
    print(f"{scene:15s} Acc@1cm: {acc1:5.2f}  Comp@1cm: {comp1:5.2f}")
    results[scene] = result

# 平均
if results:
    avg_acc = sum(r['accuracy']['accuracy@0.01'] for r in results.values()) / len(results)
    avg_comp = sum(r['completeness']['completeness@0.01'] for r in results.values()) / len(results)
    print(f"\n{'Average':15s} Acc@1cm: {avg_acc:5.2f}  Comp@1cm: {avg_comp:5.2f}")