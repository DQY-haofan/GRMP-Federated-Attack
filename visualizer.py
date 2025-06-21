import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class GRMPVisualizer:
    def __init__(self, results_path):
        """
        初始化可视化器
        Args:
            results_path: JSON结果文件路径
        """
        self.results_path = Path(results_path)
        self.data = self._load_results()
        self.config = self.data['config']
        self.results = self.data['results']

        # 创建输出目录
        self.output_dir = self.results_path.parent / 'figures'
        self.output_dir.mkdir(exist_ok=True)

    def _load_results(self):
        """加载JSON结果文件"""
        with open(self.results_path, 'r') as f:
            return json.load(f)

    def plot_accuracy_evolution(self):
        """绘制准确率和后门成功率随轮次的变化"""
        rounds = [r['round'] for r in self.results]
        clean_acc = [r['clean_accuracy'] for r in self.results]
        backdoor_asr = [r['backdoor_success_rate'] for r in self.results]

        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制主线
        ax.plot(rounds, clean_acc, 'b-o', label='Clean Accuracy',
                linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
        ax.plot(rounds, backdoor_asr, 'r-s', label='Backdoor Success Rate',
                linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)

        # 添加阴影区域
        ax.fill_between(rounds, clean_acc, alpha=0.2, color='blue')
        ax.fill_between(rounds, backdoor_asr, alpha=0.2, color='red')

        # 标注关键点
        ax.annotate(f'{clean_acc[-1]:.2%}',
                    xy=(rounds[-1], clean_acc[-1]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3))
        ax.annotate(f'{backdoor_asr[-1]:.2%}',
                    xy=(rounds[-1], backdoor_asr[-1]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Accuracy / Success Rate', fontsize=12)
        ax.set_title('GRMP Attack: Performance Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_similarity_distribution(self):
        """绘制相似度分布对比图"""
        # 收集所有轮次的相似度数据
        all_benign_sims = []
        all_attacker_sims = []

        for round_data in self.results:
            sims = round_data['defense']['similarities']
            client_ids = round_data['defense']['accepted_clients'] + round_data['defense']['rejected_clients']

            # 假设最大的client_id是攻击者
            max_client_id = max(client_ids)

            for i, (client_id, sim) in enumerate(zip(client_ids, sims)):
                if client_id == max_client_id:
                    all_attacker_sims.append(sim)
                else:
                    all_benign_sims.append(sim)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 箱线图
        bp = ax1.boxplot([all_benign_sims, all_attacker_sims],
                         labels=['Benign Clients', 'Attacker'],
                         patch_artist=True)

        # 设置箱线图颜色
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 添加阈值线
        threshold = self.results[0]['defense']['threshold']
        ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
        ax1.set_ylabel('Cosine Similarity', fontsize=12)
        ax1.set_title('Similarity Distribution Comparison', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 时间序列图
        rounds = []
        attacker_sims_by_round = []
        benign_mean_sims = []
        thresholds = []

        for round_data in self.results:
            rounds.append(round_data['round'])
            sims = round_data['defense']['similarities']

            # 分离良性和攻击者
            attacker_sim = sims[-1]  # 假设最后一个是攻击者
            benign_sims = sims[:-1]

            attacker_sims_by_round.append(attacker_sim)
            benign_mean_sims.append(np.mean(benign_sims))
            thresholds.append(round_data['defense']['threshold'])

        ax2.plot(rounds, attacker_sims_by_round, 'r-o', label='Attacker', linewidth=2.5, markersize=8)
        ax2.plot(rounds, benign_mean_sims, 'b-s', label='Benign (mean)', linewidth=2.5, markersize=8)
        ax2.plot(rounds, thresholds, 'g--', label='Dynamic Threshold', linewidth=2)

        ax2.fill_between(rounds, attacker_sims_by_round, thresholds,
                         where=(np.array(attacker_sims_by_round) >= np.array(thresholds)),
                         alpha=0.3, color='green', label='Accepted Region')

        ax2.set_xlabel('Training Round', fontsize=12)
        ax2.set_ylabel('Cosine Similarity', fontsize=12)
        ax2.set_title('Similarity Evolution Over Time', fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'similarity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_detection_analysis(self):
        """绘制检测分析图"""
        rounds = []
        detection_status = []

        num_attackers = self.config['num_attackers']
        total_clients = self.config['num_clients']
        attacker_ids = list(range(total_clients - num_attackers, total_clients))

        for round_data in self.results:
            rounds.append(round_data['round'])
            rejected = round_data['defense']['rejected_clients']

            # 检查攻击者是否被检测
            attacker_detected = any(aid in rejected for aid in attacker_ids)
            detection_status.append(1 if attacker_detected else 0)

        fig, ax = plt.subplots(figsize=(10, 4))

        # 创建热力图风格的检测状态图
        detection_matrix = np.array(detection_status).reshape(1, -1)

        im = ax.imshow(detection_matrix, cmap='RdYlGn_r', aspect='auto', alpha=0.8)

        # 设置标签
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels([f'R{r}' for r in rounds])
        ax.set_yticks([0])
        ax.set_yticklabels(['Attacker Status'])

        # 添加文本标注
        for i, status in enumerate(detection_status):
            text = 'Detected' if status else 'Undetected'
            color = 'red' if status else 'green'
            ax.text(i, 0, text, ha='center', va='center',
                    fontweight='bold', color=color)

        ax.set_title(f'GRMP Attack Detection Status Across Rounds\n'
                     f'Detection Rate: {sum(detection_status)}/{len(rounds)} rounds',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'detection_status.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_attack_impact_summary(self):
        """绘制攻击影响总结图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 准确率变化柱状图
        categories = ['Initial', 'Final']
        clean_values = [self.results[0]['clean_accuracy'], self.results[-1]['clean_accuracy']]
        backdoor_values = [self.results[0]['backdoor_success_rate'], self.results[-1]['backdoor_success_rate']]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, clean_values, width, label='Clean Accuracy', color='skyblue')
        bars2 = ax1.bar(x + width / 2, backdoor_values, width, label='Backdoor Success', color='lightcoral')

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.2%}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom')

        ax1.set_ylabel('Rate', fontsize=12)
        ax1.set_title('Attack Impact: Before vs After', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. 关键指标饼图
        metrics = {
            'Attack Success': self.results[-1]['backdoor_success_rate'],
            'Clean Performance': self.results[-1]['clean_accuracy'],
            'Detection Evasion': 1 - sum(1 for r in self.results
                                         if any(aid in r['defense']['rejected_clients']
                                                for aid in
                                                range(self.config['num_clients'] - self.config['num_attackers'],
                                                      self.config['num_clients']))) / len(self.results)
        }

        colors = ['#ff9999', '#66b3ff', '#99ff99']
        wedges, texts, autotexts = ax2.pie(metrics.values(), labels=metrics.keys(), colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        ax2.set_title('GRMP Attack Success Metrics', fontsize=13, fontweight='bold')

        # 3. 损失值变化（模拟数据，实际应从日志中提取）
        ax3.text(0.5, 0.5, 'Loss Evolution\n(Placeholder for actual loss data)',
                 ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. 攻击配置信息
        config_text = f"""Attack Configuration:

Total Clients: {self.config['num_clients']}
Attackers: {self.config['num_attackers']}
Poison Rate: {self.config['poison_rate']:.1%}
Defense Threshold: {self.config['defense_threshold']}
Local Epochs: {self.config.get('local_epochs', 2)}
Learning Rate: {self.config['client_lr']}

Final Results:
Clean Accuracy: {self.results[-1]['clean_accuracy']:.2%}
Backdoor Success: {self.results[-1]['backdoor_success_rate']:.2%}
Detection Rate: {sum(1 for r in self.results if any(aid in r['defense']['rejected_clients'] for aid in range(self.config['num_clients'] - self.config['num_attackers'], self.config['num_clients'])))}/{len(self.results)} rounds
"""
        ax4.text(0.1, 0.9, config_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')

        plt.suptitle(f"GRMP Attack Analysis: {self.config['experiment_name']}",
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attack_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_all_figures(self):
        """生成所有图表"""
        print("Generating visualizations...")

        self.plot_accuracy_evolution()
        print("✓ Accuracy evolution plot saved")

        self.plot_similarity_distribution()
        print("✓ Similarity distribution plot saved")

        self.plot_detection_analysis()
        print("✓ Detection analysis plot saved")

        self.plot_attack_impact_summary()
        print("✓ Attack impact summary saved")

        print(f"\nAll figures saved to: {self.output_dir}")


# 使用示例
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <path_to_json_results>")
        print("Example: python visualizer.py ../results/grmp_attack_results_grmp_demo_fixed.json")
        sys.exit(1)

    results_file = sys.argv[1]

    # 创建可视化器并生成所有图表
    visualizer = GRMPVisualizer(results_file)
    visualizer.generate_all_figures()