"""Batch t-SNE visualization for all 8 paper datasets of DMR-ABSA.

Each dataset gets a preset figure title (the four paradigms in the paper).
Outputs land in visualization/picture/<parent>_<stem>.png (+ .pdf / _density.*).
Two-level cache in visualization/cache makes --skip-existing / re-render cheap.
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import plot_tsne

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'picture')

DATASETS = [
    ('5%-shot Traing Data',
     os.path.join(REPO_ROOT, 'data', 'lap', 'sample5_all.json')),
    ('5%-shot Traing Data',
     os.path.join(REPO_ROOT, 'data', 'res', 'sample5_all.json')),
    ('Attribute-Grounded Synthesis',
     os.path.join(REPO_ROOT, 'property-driven', 'lap', 'direct_lap_sample5.json')),
    ('Attribute-Grounded Synthesis',
     os.path.join(REPO_ROOT, 'property-driven', 'res', 'direct_res_sample5.json')),
    ('Seed-Driven Reconstruction',
     os.path.join(REPO_ROOT, 'seed-driven', 'lap', 'direct_lap_sample5.json')),
    ('Seed-Driven Reconstruction',
     os.path.join(REPO_ROOT, 'seed-driven', 'res', 'direct_res_sample5.json')),
    ('Dual-Constrained Deduplication',
     os.path.join(REPO_ROOT, 'dual-constrained-dedup', 'rag_outputs', 'norm_res_final_dataset_RAG.json')),
    ('Dual-Constrained Deduplication',
     os.path.join(REPO_ROOT, 'dual-constrained-dedup', 'rag_outputs', 'refined_res_final_dataset_RAG.json')),
]

def make_output_path(data_path, used_names):
    """输出命名 = 父目录_文件名；重名时自动加爷爷目录末段前缀（如 refined_）。"""
    stem = os.path.splitext(os.path.basename(data_path))[0]
    parent = os.path.basename(os.path.dirname(data_path))
    name = f"{parent}_{stem}"
    if name in used_names:
        gp = os.path.basename(os.path.dirname(os.path.dirname(data_path)))
        name = f"{gp.split('_')[-1]}_{parent}_{stem}"
        print(f"  [命名] 检测到重名，加目录层级前缀: {name}")
    used_names.add(name)
    return os.path.join(PICTURE_DIR, name + '.png')

def run_one(title, data_path, gpus, skip_existing):
    plot_tsne.DATA_PATH = data_path
    plot_tsne.PLOT_TITLE = title

    stem = os.path.splitext(os.path.basename(data_path))[0]
    fp = plot_tsne._data_fingerprint(data_path, plot_tsne.MAX_SAMPLES)
    ekey = plot_tsne._emb_key(fp)
    tkey = plot_tsne._tsne_key(ekey, plot_tsne.TSNE_PARAMS)
    emb_cache = plot_tsne._cache_path(stem, ekey, 'emb')
    tsne_cache = plot_tsne._cache_path(stem, tkey, '2d')

    if skip_existing and os.path.exists(tsne_cache):
        print("[skip] t-SNE 缓存已存在，跳过（删除缓存可强制重跑）")
        return

    print("[1/4] 加载数据...")
    texts, labels = plot_tsne.load_data()

    print("[2/4] 生成 INSTRUCTOR 嵌入...")
    embeddings = plot_tsne.generate_embeddings(texts, gpus=gpus)
    plot_tsne.evaluate_semantic_diversity(embeddings, labels)
    plot_tsne.save_cache(emb_cache, {'texts': texts, 'labels': labels,
                                     'embeddings': embeddings,
                                     'fingerprint': fp})

    print("[3/4] t-SNE 降维...")
    embeddings_2d, kl = plot_tsne.perform_tsne(embeddings)
    plot_tsne.save_cache(tsne_cache, {'embeddings_2d': embeddings_2d,
                                      'labels': labels,
                                      'kl_divergence': kl,
                                      'tsne_params': plot_tsne.TSNE_PARAMS})

    print("[4/4] 渲染图片...")
    plot_tsne.visualize(embeddings_2d, labels)

def main():
    p = argparse.ArgumentParser(description='批量 t-SNE 出图：8 个论文数据集，命名=父目录_文件名')
    p.add_argument('--gpus', type=str, default=None, help='GPU卡号，如 0 或 0,1')
    p.add_argument('--skip-existing', action='store_true',
                   help='t-SNE 缓存已存在的数据集直接跳过')
    p.add_argument('--fake-embeddings', action='store_true',
                   help='模型加载失败时退化为随机嵌入（仅冒烟测试绘图流程，图形无语义意义）')
    args = p.parse_args()

    plot_tsne.ALLOW_FAKE = args.fake_embeddings

    gpus = None
    if args.gpus:
        try:
            gpus = [int(g.strip()) for g in args.gpus.split(',')]
        except ValueError:
            print(f"无效GPU卡号: {args.gpus}")

    os.makedirs(PICTURE_DIR, exist_ok=True)

    used_names = set()
    ok, fail = [], []

    for i, (title, data_path) in enumerate(DATASETS, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(DATASETS)}] 标题: {title}")
        print(f"数据: {data_path}")
        print("=" * 80)

        if not os.path.exists(data_path):
            print(f"!! 数据文件不存在，跳过: {data_path}")
            fail.append(data_path)
            continue

        plot_tsne.OUTPUT_IMAGE = make_output_path(data_path, used_names)
        print(f"输出: {plot_tsne.OUTPUT_IMAGE}（另存 .pdf / _density.png / _density.pdf）")

        try:
            run_one(title, data_path, gpus, args.skip_existing)
            ok.append(data_path)
        except Exception as e:
            print(f"!! 处理失败: {e}")
            fail.append(data_path)

    print("\n" + "=" * 80)
    print(f"批量出图完成: 成功 {len(ok)} / 失败 {len(fail)}")
    for d in fail:
        print(f"  失败: {d}")
    print("=" * 80)

if __name__ == '__main__':
    main()
