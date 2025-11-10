#!/usr/bin/env python3
from pathlib import Path
from src.evaluator.evaluator import Evaluator
from src.utils.colmap.eval_helper import get_best_colmap_index

# ----------------- 配置 -----------------
BASE_DIR = Path("SfM_dataset/eth3d_triangulation_dataset")
SCENES = ["door"]
METHOD_DIR = "DetectorFreeSfM_loftr_official_coarse_only__scratch_no_intrin"


def main():
    for scene in SCENES:
        scene_dir = BASE_DIR / scene
        method_root = scene_dir / METHOD_DIR
        coarse_root = method_root / "colmap_coarse"
        refined_root = method_root / "colmap_refined"
        pose_gt_path = scene_dir / "poses"
        image_dir = scene_dir / "images"

        if not pose_gt_path.exists() or not image_dir.exists():
            print(f"⚠️ {scene}: missing GT poses or images")
            continue

        img_list = sorted(
            str(p)
            for p in image_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        )

        print(f"===================================")
        print(f"Metrics of: {scene}")

        evaluator = Evaluator(
            image_list=img_list,
            pose_gt_path=str(pose_gt_path),
            triangulate_mode=False,
            verbose=False,
        )

        model_paths = []

        # 1. coarse 模型（选最好的那个子文件夹）
        if coarse_root.exists():
            try:
                best_id = get_best_colmap_index(str(coarse_root))
                model_paths.append(coarse_root / best_id)
            except Exception as e:
                print(f"⚠️ {scene}: get_best_colmap_index 失败: {e}")

        # 2. 中间的 model_refined_i
        if method_root.exists():
            refined_tmp = [
                d for d in method_root.iterdir()
                if d.is_dir() and d.name.startswith("model_refined_")
            ]
            refined_tmp = sorted(
                refined_tmp,
                key=lambda p: int(p.name.split("_")[-1])
            )
            model_paths.extend(refined_tmp)

        # 3. 最终的 colmap_refined
        if refined_root.exists():
            model_paths.append(refined_root)

        if not model_paths:
            print(f"⚠️ {scene}: 没有找到任何模型目录")
            continue

        # 顺序评估，每次结果写进 metric_buffer
        for mp in model_paths:
            evaluator.eval_metric(str(mp))

        # 按官方的 prepare_output_from_buffer 方式整理
        metrics_all = evaluator.prepare_output_from_buffer()

        # 按官方格式打印：
        # ===================================
        # Metrics of: door
        # *******************
        # aucs_0
        # auc@1: ...
        # ...
        # *******************
        # *******************
        # aucs_1
        # ...

        keys = sorted(metrics_all.keys())  # 比如 ['aucs_0', 'aucs_1', 'aucs_2']
        for i, k in enumerate(keys):
            print("*******************")
            print(k)
            auc_dict = metrics_all[k]  # 里面是 {'auc@1': v1, 'auc@3': v3, ...}

            # 按数字大小排序阈值
            def _key_fn(name: str):
                try:
                    return float(name.split("@")[1])
                except Exception:
                    return 1e9

            for name in sorted(auc_dict.keys(), key=_key_fn):
                print(f"{name}: {auc_dict[name]}")
            print("*******************")
            if i != len(keys) - 1:
                print("*******************")


if __name__ == "__main__":
    main()
