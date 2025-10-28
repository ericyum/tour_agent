import matplotlib
matplotlib.use('Agg') # UI 백엔드가 없는 환경을 위한 설정
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib import font_manager, rc

def setup_matplotlib_font():
    """한글 폰트를 설정합니다."""
    try:
        rc('font', family='Malgun Gothic')
    except:
        print("Malgun Gothic 폰트를 찾을 수 없습니다. 다른 한글 폰트로 설정해주세요.")
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

# 모듈 로드 시 폰트 설정
setup_matplotlib_font()

def create_donut_chart(total_pos: int, total_neg: int, title: str) -> Figure | None:
    total = total_pos + total_neg
    if total == 0:
        return None

    pos_perc = (total_pos / total) * 100
    neg_perc = (total_neg / total) * 100

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(aspect="equal"))

    labels = ['긍정', '부정']
    sizes = [pos_perc, neg_perc]
    colors = ['#5463FF', '#FF1818']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=None, autopct='%1.1f%%',
                                      startangle=90, colors=colors, pctdistance=0.85,
                                      wedgeprops=dict(width=0.3, edgecolor='w'))

    ax.set_title(title, fontsize=16, pad=20)
    ax.legend(wedges, [f'{l} ({s:.1f}%)' for l, s in zip(labels, sizes)],
              title="감성",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=10, weight="bold", color="white")
    # 범례가 잘리지 않도록 오른쪽 여백을 조정합니다.
    fig.subplots_adjust(right=0.7)
    plt.tight_layout()
    return fig

def create_stacked_bar_chart(total_pos: int, total_neg: int, title: str, figsize=(6, 2.0)) -> Figure | None: # 높이를 약간 늘려 공간 확보
    total = total_pos + total_neg
    if total == 0:
        return None

    pos_perc = (total_pos / total) * 100
    neg_perc = (total_neg / total) * 100

    fig, ax = plt.subplots(figsize=figsize)

    labels = [title]
    pos_data = [pos_perc]
    neg_data = [neg_perc]

    ax.barh(labels, pos_data, color='#5463FF', edgecolor='white', height=0.3, label=f'긍정 ({pos_perc:.1f}%)')
    ax.barh(labels, neg_data, left=pos_data, color='#FF1818', edgecolor='white', height=0.3, label=f'부정 ({neg_perc:.1f}%)')

    ax.set_xlabel('비율 (%)', fontsize=9)
    ax.set_title(f'{title} 긍정/부정 비율', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=10)

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], legend_labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=8)
    
    # tight_layout을 호출하여 전체적으로 레이아웃을 자동 조정합니다.
    fig.tight_layout()
    return fig

def create_sentence_score_bar_chart(judgments: list, title: str) -> Figure | None:
    if not judgments:
        return None

    sentences = [j['sentence'][:30] + '...' if len(j['sentence']) > 30 else j['sentence'] for j in judgments]
    scores = [j['score'] for j in judgments]
    
    colors = ['#5463FF' if s > 0 else '#FF1818' for s in scores]

    # 차트의 높이를 동적으로 조절
    height = max(4, len(sentences) * 0.5)
    fig, ax = plt.subplots(figsize=(8, height))

    y_pos = range(len(sentences))
    ax.barh(y_pos, scores, color=colors, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sentences, fontsize=9)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('감성 점수', fontsize=10)
    ax.set_title(title, fontsize=14, pad=15)

    # 점수 표시
    for i, score in enumerate(scores):
        ax.text(score + (0.01 if score >= 0 else -0.01), i, f'{score:.2f}', 
                va='center', ha='left' if score >= 0 else 'right', fontsize=8)

    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    return fig

def create_score_distribution_histogram(scores: list, boundaries: dict, title: str) -> Figure | None:
    if not scores:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='점수 분포')

    # V-lines for boundaries and mean
    mean_val = boundaries['mean']
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'평균: {mean_val:.2f}')
    
    boundary_colors = {
        'very_dissatisfied': '#FF5733',
        'dissatisfied': '#FF8C33',
        'neutral': '#FFC300',
        'satisfied': '#A2D9A0',
        'very_satisfied': '#4CAF50'
    }
    
    ax.axvline(boundaries['very_dissatisfied_upper'], color=boundary_colors['very_dissatisfied'], linestyle=':', linewidth=2)
    ax.axvline(boundaries['dissatisfied_upper'], color=boundary_colors['dissatisfied'], linestyle=':', linewidth=2)
    ax.axvline(boundaries['neutral_upper'], color=boundary_colors['neutral'], linestyle=':', linewidth=2)
    ax.axvline(boundaries['satisfied_upper'], color=boundary_colors['satisfied'], linestyle=':', linewidth=2)

    # Shading regions
    ax.axvspan(min(scores), boundaries['very_dissatisfied_upper'], alpha=0.2, color=boundary_colors['very_dissatisfied'], label='매우 불만족')
    ax.axvspan(boundaries['very_dissatisfied_upper'], boundaries['dissatisfied_upper'], alpha=0.2, color=boundary_colors['dissatisfied'], label='불만족')
    ax.axvspan(boundaries['dissatisfied_upper'], boundaries['neutral_upper'], alpha=0.2, color=boundary_colors['neutral'], label='보통')
    ax.axvspan(boundaries['neutral_upper'], boundaries['satisfied_upper'], alpha=0.2, color=boundary_colors['satisfied'], label='만족')
    ax.axvspan(boundaries['satisfied_upper'], max(scores), alpha=0.2, color=boundary_colors['very_satisfied'], label='매우 만족')

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('감성 점수', fontsize=12)
    ax.set_ylabel('문장 수', fontsize=12)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def create_outlier_boxplot(scores: list, title: str) -> Figure | None:
    if not scores:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))

    # Box plot
    ax.boxplot(scores, vert=False, patch_artist=True,
                     boxprops=dict(facecolor='lightblue'),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='blue'),
                     capprops=dict(color='black'),
                     flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.6))

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('감성 점수', fontsize=12)
    ax.set_yticks([]) # Hide y-axis ticks as it's a single box plot

    # Calculate and display outlier count
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [s for s in scores if s < lower_bound or s > upper_bound]
    
    outlier_text = f"이상치 개수: {len(outliers)}개"
    ax.text(0.95, 0.05, outlier_text,
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='red', fontsize=12)

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig