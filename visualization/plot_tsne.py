"""DMR-ABSA visualization: INSTRUCTOR embeddings + t-SNE paper figures.

Generates two publication-style figures per dataset:
  Fig.1  sentiment scatter (color-blind-safe Wong 2011 palette)
  Fig.2  point-density hexbin (deep Blues)

Two-level cache (embeddings -> 2D coords) makes re-styling instant via --viz-only.

Example:
    python plot_tsne.py --data ../data/lap/sample5_all.json \
                        --output picture/lap_sample5.png \
                        --title "5%-shot Traing Data" --gpus 0
"""
import json
import random
import os
import sys
import pickle
import hashlib
import argparse
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns

# ==================== 全局配置 ====================
# 论文级字体配置（无衬线，专业感）
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'SimHei'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,   # 让字体在 PDF 里可编辑（投稿必备）
    'ps.fonttype': 42,
    'axes.linewidth': 1.4,
    'axes.edgecolor': '#000000',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'xtick.major.width': 1.4,
    'ytick.major.width': 1.4,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
})

# 数据/输出路径（默认指向仓库内数据，均可用命令行参数 --data / --output 覆盖）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(REPO_ROOT, 'data', 'lap', 'sample5_all.json')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'picture')
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, 'lap_sample5_v2_paper.png')
MAX_SAMPLES = 5000
PLOT_TITLE = "5%-shot Traing Data"
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
ALLOW_FAKE = False   # True 时允许在模型加载失败后退化为随机嵌入（仅用于冒烟测试绘图流程）

# Nature 期刊色盲友好配色（Wong 2011）
SENTIMENT_COLORS = {
    'positive': '#E69F00',   # 橙色（柔和）
    'negative': '#009E73',   # 绿色（柔和）
    'neutral':  '#0072B2',   # 蓝色（柔和）
}
SENTIMENT_LABELS = {'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'}
AXIS_TICKS = [-4, -2, 0, 2, 4]
AXIS_LIMIT = (-4, 4)
FIG_SIZE = (10, 10)   # 稍微调小，配合留白更好看

INSTRUCTOR_MODEL = os.getenv('INSTRUCTOR_MODEL', 'hkunlp/instructor-large')
INSTRUCTION = "Represent the restaurant review for sentiment analysis:"
TSNE_PARAMS = {'n_components': 2, 'random_state': 42, 'perplexity': 50,
               'max_iter': 3000, 'learning_rate': 'auto', 'init': 'pca', 'verbose': 1}

# ==================== 缓存工具（与原版一致） ====================
def _data_fingerprint(data_path, max_samples):
    stat = os.stat(data_path)
    return {'path': os.path.abspath(data_path), 'size': stat.st_size,
            'mtime': int(stat.st_mtime), 'max_samples': max_samples}

def _emb_key(fp):
    s = json.dumps({**fp, 'model': INSTRUCTOR_MODEL, 'instruction': INSTRUCTION}, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]

def _tsne_key(emb_key, tsne_params):
    s = json.dumps({'emb_key': emb_key, 'tsne': tsne_params}, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]

def _cache_path(stem, key, suffix):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{stem}_{key}_{suffix}.pkl")

def save_cache(path, payload):
    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [cache] 已写入: {path}")

# ==================== 数据加载（与原版一致） ====================
def load_data():
    random.seed(42)   # 固定随机种子，保证子采样可复现
    print(f"正在加载数据: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")
    with open(DATA_PATH, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    print(f"原始数据条数: {len(data)}")
    if len(data) > MAX_SAMPLES:
        data = random.sample(data, MAX_SAMPLES)
        print(f"随机采样后数据条数: {len(data)}")

    texts, labels = [], []
    for item in data:
        text_content = None
        for k in ('Sentence', 'sentence', 'text', 'content'):
            if k in item:
                text_content = item[k]
                break
        if not text_content:
            continue
        cnt = {'positive': 0, 'negative': 0, 'neutral': 0, 'conflict': 0}
        if 'aspects' in item and isinstance(item['aspects'], list):
            for a in item['aspects']:
                if isinstance(a, dict) and 'polarity' in a and a['polarity'] in cnt:
                    cnt[a['polarity']] += 1
        elif 'Label' in item and isinstance(item['Label'], list):
            for it in item['Label']:
                if isinstance(it, (list, tuple)) and len(it) >= 2 and it[1] in cnt:
                    cnt[it[1]] += 1
        if max(cnt.values()) == 0:
            for lf in ('label', 'Labels', 'labels', 'sentiment', 'Sentiment', 'polarity', 'Polarity'):
                if lf in item:
                    ld = item[lf]
                    if isinstance(ld, str) and ld in cnt:
                        cnt[ld] += 1
                        break
                    elif isinstance(ld, list):
                        for it in ld:
                            if isinstance(it, str) and it in cnt:
                                cnt[it] += 1
                            elif isinstance(it, dict):
                                s = it.get('sentiment') or it.get('polarity')
                                if s in cnt:
                                    cnt[s] += 1
                        break
        main = max(cnt, key=cnt.get) if max(cnt.values()) > 0 else 'neutral'
        if main != 'conflict':
            texts.append(text_content)
            labels.append(main)
    print(f"有效数据条数: {len(texts)}")
    if texts:
        print(f"情感标签分布: {dict(sorted({l: labels.count(l) for l in labels}.items()))}")
    return texts, labels

def evaluate_semantic_diversity(embeddings, labels):
    random.seed(42)
    print("\n正在评估语义多样性...")
    sample_size = min(1000, len(embeddings))
    idx = random.sample(range(len(embeddings)), sample_size)
    sub = embeddings[idx]
    sim = cosine_similarity(sub)[np.triu_indices(sample_size, k=1)]
    pca = PCA(n_components=10).fit(embeddings)
    res = {'avg_similarity': float(np.mean(sim)), 'std_similarity': float(np.std(sim)),
           'pca_variance': pca.explained_variance_ratio_.tolist(),
           'label_distribution': {l: labels.count(l) for l in set(labels)}}
    print(f"   平均余弦相似度: {res['avg_similarity']:.4f}")
    print(f"   相似度标准差: {res['std_similarity']:.4f}")
    return res

def generate_embeddings(texts, gpus=None):
    if not texts:
        raise ValueError("文本列表为空，无法生成嵌入")
    print("\n正在加载INSTRUCTOR模型...")
    try:
        sys.path.append('.')
        from InstructorEmbedding.instructor import INSTRUCTOR
        import torch
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"GPU设备数量: {gpu_count}")
            for i in range(gpu_count):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            if gpus is not None:
                gpus = [gpus] if isinstance(gpus, int) else gpus
                valid = [g for g in gpus if 0 <= g < gpu_count]
                if valid:
                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, valid))
                    print(f"设置CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            device = torch.device('cuda')
            model = INSTRUCTOR(INSTRUCTOR_MODEL).to(device)
            if torch.cuda.device_count() > 1:
                print(f"使用 {torch.cuda.device_count()} 个GPU进行并行计算")
                model = torch.nn.DataParallel(model)
        else:
            print("未检测到GPU，使用CPU")
            device = torch.device('cpu')
            model = INSTRUCTOR(INSTRUCTOR_MODEL)
        print(f"使用设备: {device}")
        texts_with_instructions = [[INSTRUCTION, t if (t and isinstance(t, str)) else "No content"] for t in texts]
        print("正在生成嵌入...")
        bs = 64 * (torch.cuda.device_count() if torch.cuda.is_available() else 1) if torch.cuda.is_available() else 32
        print(f"使用批处理大小: {bs}")
        embeddings = model.encode(texts_with_instructions, batch_size=bs, show_progress_bar=True, device=device)
        print(f"嵌入生成完成，形状: {embeddings.shape}")
        return embeddings
    except Exception as e:
        if ALLOW_FAKE:
            print(f"加载INSTRUCTOR模型失败: {e}\n[ALLOW_FAKE=True] 将使用随机嵌入（仅冒烟测试，图形无语义意义！）")
            return np.random.rand(len(texts), 768)
        raise RuntimeError(
            f"INSTRUCTOR 模型加载失败: {e}\n"
            "请检查: 1) 已 pip install InstructorEmbedding sentence-transformers; "
            "2) INSTRUCTOR_MODEL 环境变量或默认 'hkunlp/instructor-large' 可访问; "
            "3) 如仅需测试绘图流程，请加 --fake-embeddings。") from e

def perform_tsne(embeddings):
    print("\n正在执行t-SNE降维...")
    try:
        tsne = TSNE(**TSNE_PARAMS)
        emb2d = tsne.fit_transform(embeddings)
        print(f"t-SNE降维完成，形状: {emb2d.shape}")
        print(f"t-SNE KL散度: {tsne.kl_divergence_:.4f}")
        return emb2d, float(tsne.kl_divergence_)
    except Exception as e:
        print(f"t-SNE降维失败: {e}")
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=2000, init='random')
        except TypeError:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=2000, init='random')
        emb2d = tsne.fit_transform(embeddings)
        print(f"t-SNE降维完成，形状: {emb2d.shape}")
        return emb2d, float(getattr(tsne, 'kl_divergence_', 0.0))

# ==================== 论文级可视化 ====================
def _apply_paper_axis_style(ax):
    """论文风格的坐标轴：干净、无网格、无轴标签文字"""
    ax.set_xticks(AXIS_TICKS)
    ax.set_yticks(AXIS_TICKS)
    ax.set_xlim(AXIS_LIMIT)
    ax.set_ylim(AXIS_LIMIT)
    ax.set_aspect('equal', adjustable='box')
    # 去掉顶部和右侧边框，只保留左下两边（论文常见做法）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    # 刻度朝内（学术规范），大字号保证审稿人阅读舒适
    ax.tick_params(axis='both', which='major', direction='in', labelsize=29,
                   top=False, right=False, width=1.8, length=8)
    # 次刻度线：顶刊图的标志性细节
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction='in',
                   top=False, right=False, width=1.0, length=3.5)
    # 极淡虚线网格：辅助读数又不喧宾夺主
    ax.grid(True, which='major', linestyle='--', linewidth=0.6,
            color='#B0B0B0', alpha=0.35)
    ax.set_axisbelow(True)
    # 不画 t-SNE 1 / t-SNE 2 文字
    ax.set_xlabel('')
    ax.set_ylabel('')

def _make_legend(ax, present_labels, labels):
    """论文级图例：大标记 + 纯类别名，不显示样本计数"""
    handles = [Line2D([], [], marker='o', linestyle='none',
                      markerfacecolor=SENTIMENT_COLORS[l], markeredgecolor='white',
                      markeredgewidth=0.8, markersize=12,
                      label=SENTIMENT_LABELS[l])
               for l in present_labels]
    leg = ax.legend(handles=handles, loc='upper right', fontsize=19,
                    frameon=True, framealpha=0.92, edgecolor='#CCCCCC', fancybox=False,
                    handletextpad=0.5, borderpad=0.5, labelspacing=0.45,
                    borderaxespad=0.6)
    leg.set_title('Sentiment', prop={'size': 21, 'weight': 'bold'})
    leg.get_frame().set_linewidth(0.8)
    return leg

def visualize(embeddings_2d, labels):
    """两张图：主情感散点图 + 密度热力图，论文级风格"""
    print("\n正在生成可视化（论文级）...")
    sns.set(style="white")
    out_dir = os.path.dirname(OUTPUT_IMAGE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        print(f"  输出目录已确保存在: {out_dir}")

    present = [l for l in ['positive', 'negative', 'neutral'] if l in labels]

    # ==================== 图1：主情感散点图 ====================
    fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
    _apply_paper_axis_style(ax1)
    # 原点参考线（极淡，帮助定位）
    ax1.axhline(0, color='#999999', linewidth=0.8, linestyle=':', alpha=0.6)
    ax1.axvline(0, color='#999999', linewidth=0.8, linestyle=':', alpha=0.6)
    # 按数量从少到多画（少的在上层，避免被多的覆盖）
    order = sorted(present, key=lambda x: -labels.count(x))
    for lab in order:
        idx = [j for j, l in enumerate(labels) if l == lab]
        e = embeddings_2d[idx]
        ax1.scatter(e[:, 0], e[:, 1], c=SENTIMENT_COLORS[lab],
                    alpha=0.7, s=24, edgecolors='white', linewidths=0.4,
                    label=SENTIMENT_LABELS[lab], rasterized=True)
    _make_legend(ax1, present, labels)
    # 画布级居中：相对整张图片（含未来可能附加的元素）水平居中
    fig1.suptitle(PLOT_TITLE, fontsize=32, fontweight='bold', x=0.5, y=0.97)
    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    fig1.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight', facecolor='white')
    fig1.savefig(OUTPUT_IMAGE.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"图1（主情感散点图）已保存: {OUTPUT_IMAGE}（另存矢量 .pdf）")
    plt.close(fig1)

    # ==================== 图2：密度热力图 ====================
    fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
    # 自适应颜色范围（用 99 分位避免极端值压缩色阶）
    counts = np.histogram2d(embeddings_2d[:, 0], embeddings_2d[:, 1],
                             bins=50, range=[AXIS_LIMIT, AXIS_LIMIT])[0]
    vmax = max(10, int(np.percentile(counts[counts > 0], 99)))
    # 密度图配色：Blues 加深一档（跳过色带最淡的前 25%）
    blues_deep = mpl.colors.LinearSegmentedColormap.from_list(
        'Blues_deep', mpl.colormaps['Blues'](np.linspace(0.25, 1.0, 256)))
    hb = ax2.hexbin(embeddings_2d[:, 0], embeddings_2d[:, 1], gridsize=50,
                    cmap=blues_deep, mincnt=1, vmin=0, vmax=vmax,
                    edgecolors='none', rasterized=True)
    cbar = fig2.colorbar(hb, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_visible(False)
    # 画布级居中：标题以「坐标轴+colorbar」整张图为参照水平居中
    fig2.suptitle(PLOT_TITLE, fontsize=32, fontweight='bold', x=0.5, y=0.97)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    _apply_paper_axis_style(ax2)
    d_out = OUTPUT_IMAGE.replace('.png', '_density.png')
    fig2.savefig(d_out, dpi=300, bbox_inches='tight', facecolor='white')
    fig2.savefig(d_out.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"图2（密度热力图）已保存: {d_out}（另存矢量 .pdf）")
    plt.close(fig2)

def parse_args():
    p = argparse.ArgumentParser(description='论文级 INSTRUCTOR+t-SNE 可视化')
    p.add_argument('--gpus', type=str, default=None, help='GPU卡号，多卡用逗号分隔')
    p.add_argument('--data', type=str, default=DATA_PATH, help='数据文件路径')
    p.add_argument('--output', type=str, default=OUTPUT_IMAGE, help='输出图片路径')
    p.add_argument('--max-samples', type=int, default=MAX_SAMPLES, help='最大样本数量')
    p.add_argument('--cache-dir', type=str, default=CACHE_DIR, help='缓存目录')
    p.add_argument('--force', action='store_true', help='强制重算，覆盖缓存')
    p.add_argument('--viz-only', action='store_true', help='只读缓存画图（调样式专用）')
    p.add_argument('--title', type=str, default=PLOT_TITLE, help='图表主标题')
    p.add_argument('--fake-embeddings', action='store_true',
                   help='模型加载失败时退化为随机嵌入（仅用于测试绘图流程，图形无语义意义）')
    return p.parse_args()

def main():
    global DATA_PATH, OUTPUT_IMAGE, MAX_SAMPLES, CACHE_DIR, PLOT_TITLE, ALLOW_FAKE
    args = parse_args()
    DATA_PATH = args.data
    OUTPUT_IMAGE = args.output
    MAX_SAMPLES = args.max_samples
    CACHE_DIR = args.cache_dir
    PLOT_TITLE = args.title
    ALLOW_FAKE = args.fake_embeddings
    out_dir = os.path.dirname(OUTPUT_IMAGE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    gpus = None
    if args.gpus:
        try:
            gpus = [int(g.strip()) for g in args.gpus.split(',')]
        except ValueError:
            print(f"无效的GPU卡号: {args.gpus}")

    print("=" * 80)
    print("论文级 INSTRUCTOR+t-SNE 可视化")
    print(f"  GPU卡号: {gpus}")
    print(f"  数据文件: {DATA_PATH}")
    print(f"  输出图片: {OUTPUT_IMAGE}")
    print(f"  最大样本: {MAX_SAMPLES}")
    print(f"  缓存目录: {CACHE_DIR}")
    print(f"  force重算: {args.force}")
    print(f"  viz-only: {args.viz_only}")
    print("=" * 80)

    try:
        stem = os.path.splitext(os.path.basename(DATA_PATH))[0]
        fp = _data_fingerprint(DATA_PATH, MAX_SAMPLES)
        ekey = _emb_key(fp)
        tkey = _tsne_key(ekey, TSNE_PARAMS)
        emb_cache = _cache_path(stem, ekey, 'emb')
        tsne_cache = _cache_path(stem, tkey, '2d')

        if args.viz_only:
            print("\n[viz-only] 跳过所有计算，直接读缓存画图...")
            if not os.path.exists(tsne_cache):
                raise FileNotFoundError(f"t-SNE 缓存不存在: {tsne_cache}\n请先不带 --viz-only 跑一次。")
            with open(tsne_cache, 'rb') as f:
                tp = pickle.load(f)
            visualize(tp['embeddings_2d'], tp['labels'])
            print("\n[viz-only] 完成。")
            return

        # 第一级：Embeddings 缓存
        embeddings = None
        if not args.force and os.path.exists(emb_cache):
            print("\n[cache] 命中 Embeddings 缓存，跳过模型推理...")
            with open(emb_cache, 'rb') as f:
                ep = pickle.load(f)
            texts, labels = ep['texts'], ep['labels']
            embeddings = np.asarray(ep['embeddings'])
            diversity_metrics = ep['diversity_metrics']
            print(f"  复用: {len(texts)} 条文本, 嵌入形状 {embeddings.shape}")
        else:
            print("\n[cache] 未命中 Embeddings 缓存，开始全量计算...")
            texts, labels = load_data()
            embeddings = generate_embeddings(texts, gpus=gpus)
            diversity_metrics = evaluate_semantic_diversity(embeddings, labels)
            save_cache(emb_cache, {
                'texts': texts, 'labels': labels,
                'embeddings': np.asarray(embeddings),
                'diversity_metrics': diversity_metrics,
                'fingerprint': fp,
            })

        # 第二级：t-SNE 缓存
        if not args.force and os.path.exists(tsne_cache):
            print("\n[cache] 命中 t-SNE 缓存，跳过降维计算...")
            with open(tsne_cache, 'rb') as f:
                tp = pickle.load(f)
            embeddings_2d = np.asarray(tp['embeddings_2d'])
            print(f"  复用: 2D 嵌入形状 {embeddings_2d.shape}")
        else:
            print("\n[cache] 未命中 t-SNE 缓存，开始降维计算...")
            embeddings_2d, kl = perform_tsne(embeddings)
            save_cache(tsne_cache, {
                'embeddings_2d': np.asarray(embeddings_2d),
                'labels': labels,
                'kl_divergence': kl,
                'tsne_params': TSNE_PARAMS,
            })

        visualize(embeddings_2d, labels)

        print("\n" + "=" * 80)
        print("任务完成！")
        print(f"输出文件:")
        print(f"  图1: {OUTPUT_IMAGE}")
        print(f"  图2: {OUTPUT_IMAGE.replace('.png', '_density.png')}")
        print("=" * 80)
        print("\n语义多样性评估总结:")
        print("-" * 60)
        print(f"1. 平均余弦相似度: {diversity_metrics['avg_similarity']:.4f}")
        print(f"2. 相似度标准差: {diversity_metrics['std_similarity']:.4f}")
        print("-" * 60)
    except Exception as e:
        print(f"\n执行失败: {e}")
        raise

if __name__ == "__main__":
    main()