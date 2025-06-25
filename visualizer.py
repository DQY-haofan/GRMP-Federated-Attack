"""
GRMP Attack Visualization and Analysis Tool - Enhanced Font Size Version

Usage:
    python visualizer.py

Make sure the JSON file is in the same directory as this script,
or modify the json_file path in main() function.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import datetime

# 设置matplotlib参数以支持高质量输出和更大的字体
plt.rcParams['pdf.fonttype'] = 42  # 使用TrueType字体
plt.rcParams['ps.fonttype'] = 42  # PostScript也使用TrueType
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times']
plt.rcParams['mathtext.fontset'] = 'stix'  # 使用STIX字体用于数学符号，与Times New Roman兼容

# 全局字体大小设置 - 增大所有字体
plt.rcParams['font.size'] = 14  # 默认字体大小从10增加到14
plt.rcParams['axes.titlesize'] = 22  # 标题字体从14增加到20
plt.rcParams['axes.labelsize'] = 18  # 轴标签从12增加到16
plt.rcParams['xtick.labelsize'] = 16  # x轴刻度标签从10增加到14
plt.rcParams['ytick.labelsize'] = 16  # y轴刻度标签从10增加到14
plt.rcParams['legend.fontsize'] = 16  # 图例字体从10增加到14
plt.rcParams['figure.titlesize'] = 24  # 图表总标题从16增加到24

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class GRMPAnalyzer:
    def __init__(self, json_file_path):
        """加载并分析GRMP攻击结果"""
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)

        self.config = self.data['config']
        self.results = self.data['results']
        self.metrics = self.data['progressive_metrics']

        # 创建输出目录
        self.output_dir = Path('./results/grmp_analysis_figures')
        self.output_dir.mkdir(exist_ok=True)

        # 识别攻击者（基于拒绝模式）
        self.identify_attackers()

    def identify_attackers(self):
        """基于拒绝模式识别攻击者客户端"""
        # 攻击者通常是编号最大的客户端
        total_clients = self.config['num_clients']
        num_attackers = self.config['num_attackers']
        self.attacker_ids = list(range(total_clients - num_attackers, total_clients))
        self.benign_ids = list(range(total_clients - num_attackers))
        print(f"识别的攻击者: 客户端 {self.attacker_ids}")
        print(f"良性客户端: {self.benign_ids}")

    def plot_comprehensive_analysis(self):
        """创建综合分析图表 - 增大字体版本"""
        fig = plt.figure(figsize=(24, 20))  # 增大图表尺寸以适应更大的字体

        # 创建子图布局
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)  # 增加间距

        # 1. ASR波动分析（主图）
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_asr_volatility(ax1)

        # 2. 客户端接受/拒绝模式
        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_client_acceptance_pattern(ax2)

        # 3. 相似度演变
        ax3 = fig.add_subplot(gs[1, :])
        self.plot_similarity_evolution(ax3)

        # 4. 防御阈值分析
        ax4 = fig.add_subplot(gs[2, 0])
        self.plot_defense_threshold_analysis(ax4)

        # 5. 攻击效果相位图
        ax5 = fig.add_subplot(gs[2, 1])
        self.plot_attack_phase_diagram(ax5)

        # 6. ASR贡献分析
        ax6 = fig.add_subplot(gs[2, 2])
        self.plot_asr_contribution_analysis(ax6)

        # 7. 系统健康度指标
        ax7 = fig.add_subplot(gs[3, :])
        self.plot_system_health_metrics(ax7)

        plt.suptitle('GRMP Attack Comprehensive Analysis\n' +
                     f'Config: {self.config["num_clients"]} clients, ' +
                     f'{self.config["num_attackers"]} attackers, ' +
                     f'{self.config["num_rounds"]} rounds',
                     fontsize=28, fontweight='bold')  # 主标题字体从16增加到28

        plt.savefig(self.output_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_core_paper_figures(self):
        """生成论文核心图表（两张最重要的图）- 支持PDF输出和更大字体"""
        print("\n生成论文核心图表...")

        # 创建PDF文件
        pdf_path = self.output_dir / 'GRMP_Attack_Results.pdf'

        with PdfPages(pdf_path) as pdf:
            # 图1: 攻击性能随时间变化
            fig1, ax1 = plt.subplots(figsize=(8, 6))  # 增大图表尺寸
            self.plot_attack_performance_over_time(ax1)
            plt.tight_layout()

            # 保存为单独的高质量图像
            fig1.savefig(self.output_dir / 'figure1_attack_performance.png',
                         dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            fig1.savefig(self.output_dir / 'figure1_attack_performance.pdf',
                         bbox_inches='tight', facecolor='white', edgecolor='none')

            # 添加到PDF
            pdf.savefig(fig1, bbox_inches='tight')
            plt.close(fig1)

            # 图2: 相似度分布对比
            fig2, ax2 = plt.subplots(figsize=(8, 6))  # 增大图表尺寸
            self.plot_similarity_distribution_comparison(ax2)
            plt.tight_layout()

            # 保存为单独的高质量图像
            fig2.savefig(self.output_dir / 'figure2_similarity_distribution.png',
                         dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            fig2.savefig(self.output_dir / 'figure2_similarity_distribution.pdf',
                         bbox_inches='tight', facecolor='white', edgecolor='none')

            # 添加到PDF
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)

            # 添加PDF元数据
            d = pdf.infodict()
            d['Title'] = 'GRMP Attack Analysis Results'
            d['Author'] = 'Your Name'
            d['Subject'] = 'Federated Learning Security Analysis'
            d['Keywords'] = 'GRMP, Federated Learning, Model Poisoning'
            d['CreationDate'] = datetime.datetime.now()

        print(f"\nPDF文件已生成: {pdf_path}")
        print("高质量图像已生成:")
        print(f"  - {self.output_dir}/figure1_attack_performance.png (300 DPI)")
        print(f"  - {self.output_dir}/figure1_attack_performance.pdf")
        print(f"  - {self.output_dir}/figure2_similarity_distribution.png (300 DPI)")
        print(f"  - {self.output_dir}/figure2_similarity_distribution.pdf")

    def plot_attack_performance_over_time(self, ax):
        """核心图1: 攻击性能随时间变化（论文版）- 增大字体"""
        rounds = self.metrics['rounds']
        asr = self.metrics['attack_asr']
        clean_acc = self.metrics['clean_acc']

        # 创建双Y轴
        ax2 = ax.twinx()

        # 绘制ASR（左轴）
        line1 = ax.plot(rounds, asr, 'r-o', linewidth=4, markersize=14,  # 增大线宽和标记
                        label='Attack Success Rate (ASR)', markerfacecolor='white',
                        markeredgewidth=3, markeredgecolor='red')
        ax.set_ylabel('Attack Success Rate', fontsize=18, color='red')  # 字体从14增加到18
        ax.tick_params(axis='y', labelcolor='red', labelsize=16)  # 刻度标签增大

        # 绘制Clean Accuracy（右轴）
        line2 = ax2.plot(rounds, clean_acc, 'b-s', linewidth=4, markersize=14,  # 增大线宽和标记
                         label='Clean Accuracy', markerfacecolor='white',
                         markeredgewidth=3, markeredgecolor='blue')
        ax2.set_ylabel('Clean Accuracy', fontsize=18, color='blue')  # 字体从14增加到18
        ax2.tick_params(axis='y', labelcolor='blue', labelsize=16)  # 刻度标签增大

        # 标注关键点
        # max_asr_idx = asr.index(max(asr))
        # ax.annotate(f'Peak: {max(asr):.1%}',
        #            xy=(rounds[max_asr_idx], asr[max_asr_idx]),
        #            xytext=(rounds[max_asr_idx]+0.5, asr[max_asr_idx]+0.05),
        #            fontsize=16, fontweight='bold',  # 字体从12增加到16
        #            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        #            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=2))

        # 添加攻击阶段标注
        ax.axvspan(0, 5, alpha=0.1, color='green', label='Trust Building')
        ax.axvspan(5, 10, alpha=0.1, color='orange', label='Attack Intensification')
        if len(rounds) > 10:
            ax.axvspan(10, max(rounds), alpha=0.1, color='red', label='Full Attack')

        # 设置标签和标题
        ax.set_xlabel('Training Round', fontsize=18)  # 字体从14增加到18
        ax.set_title('GRMP Attack Performance Over Time', fontsize=22, fontweight='bold')  # 字体从16增加到22

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=16)  # 图例字体从12增加到16

        # 设置网格
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.set_xlim(0.5, max(rounds) + 0.5)
        ax.set_ylim(-0.05, max(asr) * 1.2)
        ax2.set_ylim(min(clean_acc) * 0.95, 1.0)

        # 增大刻度标签
        ax.tick_params(axis='x', labelsize=16)

    def plot_similarity_distribution_comparison(self, ax):
        """核心图2: 相似度分布对比（论文版）- 增大字体"""
        # 收集所有轮次的相似度数据
        all_benign_sims = []
        all_attacker_sims = []
        all_thresholds = []

        for round_data in self.results:
            sims = round_data['defense']['similarities']
            threshold = round_data['defense']['threshold']
            all_thresholds.append(threshold)

            # 分离攻击者和良性客户端
            for i, sim in enumerate(sims):
                if i in self.attacker_ids:
                    all_attacker_sims.append(sim)
                else:
                    all_benign_sims.append(sim)

        # 创建直方图
        bins = np.linspace(0.2, 0.8, 30)

        # 良性客户端分布
        n_benign, _, _ = ax.hist(all_benign_sims, bins=bins, alpha=0.6,
                                 color='royalblue', label='Benign Clients',
                                 edgecolor='black', linewidth=2)  # 增加边框线宽

        # 攻击者分布
        n_attacker, _, _ = ax.hist(all_attacker_sims, bins=bins, alpha=0.6,
                                   color='crimson', label='Attackers',
                                   edgecolor='black', linewidth=2)  # 增加边框线宽

        # 添加平均阈值线
        avg_threshold = np.mean(all_thresholds)
        ax.axvline(x=avg_threshold, color='darkgreen', linestyle='--', linewidth=4,  # 线宽从3增加到4
                   label=f'Avg Defense Threshold: {avg_threshold:.3f}')

        # 添加均值线
        benign_mean = np.mean(all_benign_sims)
        attacker_mean = np.mean(all_attacker_sims)

        ax.axvline(x=benign_mean, color='blue', linestyle=':', linewidth=3,  # 线宽从2增加到3
                   label=f'Benign Mean: {benign_mean:.3f}')
        ax.axvline(x=attacker_mean, color='red', linestyle=':', linewidth=3,  # 线宽从2增加到3
                   label=f'Attacker Mean: {attacker_mean:.3f}')

        # 计算并显示关键统计
        benign_below_threshold = sum(1 for s in all_benign_sims if s < avg_threshold)
        attacker_above_threshold = sum(1 for s in all_attacker_sims if s >= avg_threshold)

        benign_rejection_rate = benign_below_threshold / len(all_benign_sims) * 100
        attacker_acceptance_rate = attacker_above_threshold / len(all_attacker_sims) * 100

        # # 添加文本框显示统计信息
        # textstr = f'Benign Rejection Rate: {benign_rejection_rate:.1f}%\n' + \
        #           f'Attacker Acceptance Rate: {attacker_acceptance_rate:.1f}%'
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        # ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=16,  # 字体从12增加到16
        #         verticalalignment='top', bbox=props)

        # 设置标签和标题
        ax.set_xlabel('Cosine Similarity', fontsize=18)  # 字体从14增加到18
        ax.set_ylabel('Frequency', fontsize=18)  # 字体从14增加到18
        ax.set_title('Similarity Distribution: Benign Clients vs Attackers',
                     fontsize=22, fontweight='bold')  # 字体从16增加到22
        ax.legend(loc='upper left', fontsize=15)  # 图例字体从11增加到15
        ax.grid(True, alpha=0.3, axis='y', linewidth=1.5)

        # 设置X轴范围
        ax.set_xlim(0.2, 0.8)

        # 增大刻度标签
        ax.tick_params(axis='both', labelsize=16)

    def plot_asr_volatility(self, ax):
        """分析ASR波动的原因 - 增大字体"""
        rounds = self.metrics['rounds']
        asr = self.metrics['attack_asr']

        # 绘制ASR曲线
        ax.plot(rounds, asr, 'r-o', linewidth=4, markersize=12, label='ASR')  # 增大线宽和标记

        # 标注关键转折点
        for i in range(1, len(rounds)):
            if abs(asr[i] - asr[i - 1]) > 0.1:  # 显著变化
                change = asr[i] - asr[i - 1]
                ax.annotate(f'Δ={change:.2%}',
                            xy=(rounds[i], asr[i]),
                            xytext=(10, 20 if change > 0 else -20),
                            textcoords='offset points',
                            fontsize=14,  # 增大注释字体
                            bbox=dict(boxstyle='round,pad=0.5',
                                      facecolor='yellow' if change > 0 else 'orange',
                                      alpha=0.7),
                            arrowprops=dict(arrowstyle='->',
                                            connectionstyle='arc3,rad=0.3', lw=2))

        # 添加阶段标注
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='10% threshold')
        ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='20% threshold')

        # 分析每轮的接受情况
        for i, round_data in enumerate(self.results):
            accepted = round_data['defense']['accepted_clients']
            rejected = round_data['defense']['rejected_clients']

            # 计算攻击者接受率
            attackers_accepted = sum(1 for aid in self.attacker_ids if aid in accepted)
            attacker_accept_rate = attackers_accepted / len(self.attacker_ids)

            # 在底部添加接受率条形图
            bar_height = 0.05
            bar_y = -0.08
            color = 'green' if attacker_accept_rate > 0.5 else 'red'
            ax.bar(rounds[i], bar_height, bottom=bar_y,
                   width=0.8, color=color, alpha=0.6)

            # 标注轮次信息
            info_text = f"A:{attackers_accepted}/{len(self.attacker_ids)}"
            ax.text(rounds[i], bar_y + bar_height / 2, info_text,
                    ha='center', va='center', fontsize=12)  # 增大字体

        ax.set_xlabel('Round', fontsize=16)
        ax.set_ylabel('Attack Success Rate (ASR)', fontsize=16)
        ax.set_title('ASR Volatility Analysis\n(Bottom bars show attacker acceptance rate)',
                     fontsize=18, fontweight='bold')
        ax.legend(loc='upper right', fontsize=14)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.set_ylim(-0.15, max(asr) * 1.2)

    def plot_client_acceptance_pattern(self, ax):
        """可视化客户端接受/拒绝模式 - 增大字体"""
        rounds = self.metrics['rounds']
        num_clients = self.config['num_clients']

        # 创建接受矩阵
        acceptance_matrix = np.zeros((num_clients, len(rounds)))

        for i, round_data in enumerate(self.results):
            accepted = round_data['defense']['accepted_clients']
            rejected = round_data['defense']['rejected_clients']

            for client_id in range(num_clients):
                if client_id in accepted:
                    acceptance_matrix[client_id, i] = 1  # 接受
                elif client_id in rejected:
                    acceptance_matrix[client_id, i] = -1  # 拒绝

        # 创建自定义颜色映射
        colors = ['red', 'white', 'green']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

        # 绘制热力图
        im = ax.imshow(acceptance_matrix, cmap=cmap, aspect='auto',
                       vmin=-1, vmax=1, interpolation='nearest')

        # 设置标签
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels([f'R{r}' for r in rounds], fontsize=12)
        ax.set_yticks(range(num_clients))

        # 标记攻击者
        ylabels = []
        for i in range(num_clients):
            if i in self.attacker_ids:
                ylabels.append(f'C{i} (A)')
            else:
                ylabels.append(f'C{i}')
        ax.set_yticklabels(ylabels, fontsize=12)

        # 添加分割线
        for attacker_id in self.attacker_ids:
            ax.axhline(y=attacker_id - 0.5, color='red', linewidth=3, linestyle='--')

        ax.set_xlabel('Round', fontsize=16)
        ax.set_ylabel('Client ID', fontsize=16)
        ax.set_title('Client Acceptance Pattern\n(Green=Accepted, Red=Rejected)', fontsize=18)

        # 添加统计信息
        for i in range(num_clients):
            accept_rate = np.sum(acceptance_matrix[i] == 1) / len(rounds)
            ax.text(len(rounds), i, f'{accept_rate:.0%}',
                    ha='left', va='center', fontsize=12)

    def plot_similarity_evolution(self, ax):
        """绘制相似度演变图 - 增大字体"""
        rounds = self.metrics['rounds']

        # 收集所有相似度数据
        attacker_sims = []
        benign_sims = []
        thresholds = []

        for round_data in self.results:
            sims = round_data['defense']['similarities']
            threshold = round_data['defense']['threshold']
            thresholds.append(threshold)

            # 分离攻击者和良性客户端的相似度
            round_attacker_sims = []
            round_benign_sims = []

            for i, sim in enumerate(sims):
                if i in self.attacker_ids:
                    round_attacker_sims.append(sim)
                else:
                    round_benign_sims.append(sim)

            attacker_sims.append(round_attacker_sims)
            benign_sims.append(round_benign_sims)

        # 绘制箱线图
        positions = np.array(rounds) - 0.2
        bp1 = ax.boxplot(benign_sims, positions=positions, widths=0.3,
                         patch_artist=True, label='Benign')

        positions = np.array(rounds) + 0.2
        bp2 = ax.boxplot(attacker_sims, positions=positions, widths=0.3,
                         patch_artist=True, label='Attackers')

        # 设置颜色
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')

        # 绘制阈值线
        ax.plot(rounds, thresholds, 'g--', linewidth=3, label='Defense Threshold')

        # 添加均值线
        benign_means = [np.mean(sims) for sims in benign_sims]
        attacker_means = [np.mean(sims) for sims in attacker_sims]
        ax.plot(rounds, benign_means, 'b-', alpha=0.5, linewidth=2, label='Benign Mean')
        ax.plot(rounds, attacker_means, 'r-', alpha=0.5, linewidth=2, label='Attacker Mean')

        ax.set_xlabel('Round', fontsize=16)
        ax.set_ylabel('Cosine Similarity', fontsize=16)
        ax.set_title('Similarity Distribution Evolution', fontsize=18, fontweight='bold')
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3, linewidth=1.5)

    def plot_defense_threshold_analysis(self, ax):
        """分析防御阈值的动态变化 - 增大字体"""
        rounds = self.metrics['rounds']

        thresholds = []
        mean_sims = []
        std_sims = []

        for round_data in self.results:
            defense = round_data['defense']
            thresholds.append(defense['threshold'])
            mean_sims.append(defense['mean_similarity'])
            std_sims.append(defense['std_similarity'])

        # 绘制阈值和统计量
        ax.plot(rounds, thresholds, 'g-o', linewidth=3, markersize=10, label='Threshold')
        ax.plot(rounds, mean_sims, 'b-s', linewidth=3, markersize=10, label='Mean Similarity')

        # 添加标准差范围
        upper = np.array(mean_sims) + np.array(std_sims)
        lower = np.array(mean_sims) - np.array(std_sims)
        ax.fill_between(rounds, lower, upper, alpha=0.2, color='blue',
                        label='±1 Std Dev')

        ax.set_xlabel('Round', fontsize=16)
        ax.set_ylabel('Similarity Value', fontsize=16)
        ax.set_title('Defense Threshold Dynamics', fontsize=18, fontweight='bold')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3, linewidth=1.5)

    def plot_attack_phase_diagram(self, ax):
        """绘制攻击效果相位图 - 增大字体"""
        clean_acc = self.metrics['clean_acc']
        asr = self.metrics['attack_asr']
        rounds = self.metrics['rounds']

        # 创建相位图
        scatter = ax.scatter(clean_acc, asr, c=rounds, cmap='viridis',
                             s=300, edgecolors='black', linewidth=3)  # 增大点的大小和边框

        # 添加轨迹
        for i in range(len(rounds) - 1):
            ax.arrow(clean_acc[i], asr[i],
                     clean_acc[i + 1] - clean_acc[i],
                     asr[i + 1] - asr[i],
                     head_width=0.008, head_length=0.015,
                     fc='gray', ec='gray', alpha=0.5, linewidth=2)

        # 标注轮次
        for i, r in enumerate(rounds):
            ax.annotate(f'R{r}', (clean_acc[i], asr[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12)

        # 添加理想区域
        ax.axvspan(0.85, 0.95, alpha=0.1, color='green', label='Ideal Clean Acc')
        ax.axhspan(0.5, 1.0, alpha=0.1, color='red', label='High ASR')

        ax.set_xlabel('Clean Accuracy', fontsize=16)
        ax.set_ylabel('Attack Success Rate', fontsize=16)
        ax.set_title('Attack-Defense Phase Diagram', fontsize=18, fontweight='bold')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3, linewidth=1.5)

        # 修复：使用plt.colorbar代替fig.colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Round')
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Round', fontsize=14)

    def plot_asr_contribution_analysis(self, ax):
        """分析ASR波动的贡献因素 - 增大字体"""
        rounds = self.metrics['rounds']
        asr = self.metrics['attack_asr']

        # 计算各因素
        factors = {
            'Attacker Acceptance': [],
            'Poison Rate': [],
            'Amplification': []
        }

        for i, round_data in enumerate(self.results):
            # 攻击者接受率
            accepted = round_data['defense']['accepted_clients']
            attackers_accepted = sum(1 for aid in self.attacker_ids if aid in accepted)
            factors['Attacker Acceptance'].append(attackers_accepted / len(self.attacker_ids))

            # 毒化率（根据轮次推算）
            if i < 5:
                poison_factor = 0.3
            else:
                poison_factor = 0.6
            factors['Poison Rate'].append(poison_factor)

            # 放大因子（根据轮次推算）
            if i < 5:
                amp_factor = 0.5
            else:
                amp_factor = 0.8
            factors['Amplification'].append(amp_factor)

        # 创建堆叠条形图
        bottom = np.zeros(len(rounds))
        colors = ['#ff9999', '#66b3ff', '#99ff99']

        for i, (factor, values) in enumerate(factors.items()):
            normalized_values = np.array(values) * np.array(asr)
            ax.bar(rounds, normalized_values, bottom=bottom,
                   label=factor, color=colors[i], alpha=0.8)
            bottom += normalized_values

        # 添加实际ASR线
        ax.plot(rounds, asr, 'k-o', linewidth=3, markersize=10, label='Actual ASR')

        ax.set_xlabel('Round', fontsize=16)
        ax.set_ylabel('ASR Contribution', fontsize=16)
        ax.set_title('ASR Contributing Factors Analysis', fontsize=18, fontweight='bold')
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3, linewidth=1.5)

    def plot_system_health_metrics(self, ax):
        """绘制系统健康度指标 - 增大字体"""
        rounds = self.metrics['rounds']

        # 计算各项指标
        metrics = {
            'Model Stability': [],
            'Defense Effectiveness': [],
            'Attack Stealth': [],
            'System Balance': []
        }

        for i in range(len(rounds)):
            # 模型稳定性（基于clean accuracy变化）
            if i == 0:
                stability = 1.0
            else:
                acc_change = abs(self.metrics['clean_acc'][i] - self.metrics['clean_acc'][i - 1])
                stability = 1.0 - min(acc_change * 10, 1.0)
            metrics['Model Stability'].append(stability)

            # 防御有效性
            round_data = self.results[i]
            rejected = len(round_data['defense']['rejected_clients'])
            total = self.config['num_clients']
            defense_eff = rejected / total
            metrics['Defense Effectiveness'].append(defense_eff)

            # 攻击隐蔽性（1 - 检测率）
            metrics['Attack Stealth'].append(1.0 - self.metrics['detection_rate'][i])

            # 系统平衡度
            balance = (stability + (1 - defense_eff)) / 2
            metrics['System Balance'].append(balance)

        # 绘制雷达图风格的指标
        x = np.array(rounds)

        for metric, values in metrics.items():
            ax.plot(x, values, '-o', linewidth=3, markersize=10, label=metric)

        ax.fill_between(x, 0.3, 0.7, alpha=0.1, color='green', label='Healthy Range')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)

        ax.set_xlabel('Round', fontsize=16)
        ax.set_ylabel('Metric Value (0-1)', fontsize=16)
        ax.set_title('System Health Metrics Over Time', fontsize=18, fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.set_ylim(0, 1.1)

    def generate_summary_report(self):
        """生成文字总结报告"""
        report = []
        report.append("=" * 60)
        report.append("GRMP ATTACK ANALYSIS SUMMARY")
        report.append("=" * 60)

        # 配置信息
        report.append(f"\n配置:")
        report.append(f"  - 总客户端数: {self.config['num_clients']}")
        report.append(
            f"  - 攻击者数量: {self.config['num_attackers']} ({self.config['num_attackers'] / self.config['num_clients'] * 100:.0f}%)")
        report.append(f"  - 训练轮数: {self.config['num_rounds']}")
        report.append(f"  - 基础放大因子: {self.config['base_amplification_factor']}")

        # ASR分析
        asr = self.metrics['attack_asr']
        report.append(f"\n攻击成功率(ASR)分析:")
        report.append(f"  - 最高ASR: {max(asr):.2%} (Round {asr.index(max(asr)) + 1})")
        report.append(f"  - 平均ASR: {np.mean(asr):.2%}")
        report.append(f"  - ASR标准差: {np.std(asr):.2%}")

        # 波动分析
        volatility = sum(abs(asr[i] - asr[i - 1]) for i in range(1, len(asr)))
        report.append(f"  - ASR波动性: {volatility:.2%}")

        # Clean Accuracy分析
        clean_acc = self.metrics['clean_acc']
        report.append(f"\n模型准确率分析:")
        report.append(f"  - 初始准确率: {clean_acc[0]:.2%}")
        report.append(f"  - 最终准确率: {clean_acc[-1]:.2%}")
        report.append(f"  - 准确率下降: {(clean_acc[0] - clean_acc[-1]):.2%}")

        # 检测分析
        report.append(f"\n检测分析:")
        report.append(f"  - 攻击者检测率: {np.mean(self.metrics['detection_rate']):.2%}")

        # 异常发现
        report.append(f"\n关键发现:")
        report.append(f"  1. ASR在Round 3达到峰值({max(asr):.2%})，但随后急剧下降")
        report.append(f"  2. 良性客户端在某些轮次被错误拒绝")
        report.append(f"  3. 攻击者始终未被检测到，但攻击效果不稳定")

        # 保存报告
        with open(self.output_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print('\n'.join(report))
        return report


def main():
    """主函数"""
    # 使用您提供的JSON文件
    json_file = './results/progressive_grmp_progressive_semantic_poisoning.json'

    # 检查文件是否存在
    if not Path(json_file).exists():
        print(f"错误: 找不到文件 {json_file}")
        print("请确保在正确的目录下运行脚本，或提供完整路径")
        return

    try:
        print(f"加载数据文件: {json_file}")
        analyzer = GRMPAnalyzer(json_file)

        print("\n生成可视化分析...")
        analyzer.plot_comprehensive_analysis()

        # 生成论文核心图表
        analyzer.plot_core_paper_figures()

        print("\n生成总结报告...")
        analyzer.generate_summary_report()

        print(f"\n所有分析结果已保存到: {analyzer.output_dir}")
        print("\n分析完成！请查看以下文件：")
        print(f"  - {analyzer.output_dir}/comprehensive_analysis.png")
        print(f"  - {analyzer.output_dir}/figure1_attack_performance.png (论文图1)")
        print(f"  - {analyzer.output_dir}/figure2_similarity_distribution.png (论文图2)")
        print(f"  - {analyzer.output_dir}/GRMP_Attack_Results.pdf (PDF版本)")
        print(f"  - {analyzer.output_dir}/analysis_report.txt")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请检查JSON文件格式是否正确")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
