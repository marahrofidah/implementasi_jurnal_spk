import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style yang aesthetic dan profesional
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TOPSISProfessional:
    def __init__(self):
        self.title = "AI Generatif Selector - TOPSIS Method"
        self.alternatives = ['ChatGPT', 'Gemini', 'Claude', 'Copilot']
        self.criteria = ['Akurasi Teks', 'Kecepatan Respons', 'Kemudahan Penggunaan', 'Fleksibilitas Integrasi']
        self.emojis = {'ChatGPT': '', 'Gemini': '', 'Claude': '', 'Copilot': ''}
        
    def print_header(self):
        print("\n" + "="*80)
        print(f"{self.title:^80}")
        print("="*80)
        print("Analisis Pemilihan Model AI Generatif Menggunakan Metode TOPSIS\n")
        
    def create_decision_matrix(self):
        """Data dari jurnal yang sudah ada"""
        print("Step 1: Decision Matrix (Data mentah dari research)")
        print("-" * 60)
        
        # Data dari jurnal
        data = {
            'Alternatif': self.alternatives,
            'C1 (Akurasi)': [95, 92, 90, 88],
            'C2 (Kecepatan)': [80, 85, 82, 75],
            'C3 (Kemudahan)': [90, 85, 88, 84],
            'C4 (Fleksibilitas)': [85, 80, 88, 86]
        }
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        
        # Extract decision matrix
        self.decision_matrix = df.iloc[:, 1:].values
        print(f"\nMatrix shape: {self.decision_matrix.shape}\n")
        
        return df
        
    def set_weights(self):
        """Bobot kriteria dari survei (data jurnal)"""
        print("Step 2: Criteria Weights (Dari survei 19 responden)")
        print("-" * 60)
        
        # Data bobot dari jurnal (sudah dinormalisasi)
        self.weights = np.array([0.251, 0.232, 0.257, 0.260])
        
        weight_df = pd.DataFrame({
            'Kriteria': self.criteria,
            'Bobot': self.weights,
            'Persentase': [f"{w*100:.1f}%" for w in self.weights]
        })
        
        print(weight_df.to_string(index=False))
        print(f"Total bobot: {sum(self.weights):.3f}\n")
        
    def normalize_matrix(self):
        """Normalisasi matriks keputusan"""
        print("Step 3: Normalisasi Matrix")
        print("-" * 60)
        
        # Hitung normalisasi menggunakan euclidean distance
        normalized = np.zeros_like(self.decision_matrix, dtype=float)
        
        for j in range(self.decision_matrix.shape[1]):
            sum_squares = np.sum(self.decision_matrix[:, j] ** 2)
            sqrt_sum = np.sqrt(sum_squares)
            normalized[:, j] = self.decision_matrix[:, j] / sqrt_sum
            
        self.normalized_matrix = normalized
        
        # Tampilkan hasil normalisasi
        norm_df = pd.DataFrame(
            self.normalized_matrix, 
            columns=[f'C{i+1}' for i in range(len(self.criteria))],
            index=self.alternatives
        )
        
        print("Normalized Decision Matrix:")
        print(norm_df.round(3).to_string())
        print("\n")
        
    def weighted_normalized_matrix(self):
        """Matriks ternormalisasi terbobot"""
        print("Step 4: Weighted Normalized Matrix")
        print("-" * 60)
        
        # Kalikan dengan bobot
        self.weighted_matrix = self.normalized_matrix * self.weights
        
        weighted_df = pd.DataFrame(
            self.weighted_matrix,
            columns=[f'C{i+1}' for i in range(len(self.criteria))],
            index=self.alternatives
        )
        
        print("Weighted Normalized Matrix:")
        print(weighted_df.round(3).to_string())
        print("\n")
        
    def ideal_solutions(self):
        """Menentukan solusi ideal positif dan negatif"""
        print("Step 5: Ideal Solutions")
        print("-" * 60)
        
        # Solusi ideal positif (max dari setiap kolom)
        self.ideal_positive = np.max(self.weighted_matrix, axis=0)
        # Solusi ideal negatif (min dari setiap kolom)
        self.ideal_negative = np.min(self.weighted_matrix, axis=0)
        
        ideal_df = pd.DataFrame({
            'Kriteria': [f'C{i+1}' for i in range(len(self.criteria))],
            'Ideal Positive A+': self.ideal_positive.round(3),
            'Ideal Negative A-': self.ideal_negative.round(3)
        })
        
        print(ideal_df.to_string(index=False))
        print("\n")
        
    def calculate_distances(self):
        """Hitung jarak ke solusi ideal"""
        print("Step 6: Distance Calculation")
        print("-" * 60)
        
        n_alternatives = self.weighted_matrix.shape[0]
        
        # Jarak ke solusi ideal positif
        self.distance_positive = np.zeros(n_alternatives)
        # Jarak ke solusi ideal negatif  
        self.distance_negative = np.zeros(n_alternatives)
        
        for i in range(n_alternatives):
            # Euclidean distance ke A+
            self.distance_positive[i] = np.sqrt(
                np.sum((self.weighted_matrix[i] - self.ideal_positive) ** 2)
            )
            # Euclidean distance ke A-
            self.distance_negative[i] = np.sqrt(
                np.sum((self.weighted_matrix[i] - self.ideal_negative) ** 2)
            )
            
        distance_df = pd.DataFrame({
            'Alternatif': self.alternatives,
            'D+ (ke Ideal)': self.distance_positive.round(3),
            'D- (ke Anti-Ideal)': self.distance_negative.round(3)
        })
        
        print(distance_df.to_string(index=False))
        print("\n")
        
    def calculate_preference_scores(self):
        """Hitung skor preferensi dan ranking final"""
        print("Step 7: Preference Scores & Final Ranking")
        print("-" * 60)
        
        # Hitung preference score
        self.preference_scores = self.distance_negative / (
            self.distance_positive + self.distance_negative
        )
        
        # Buat ranking
        ranking_indices = np.argsort(-self.preference_scores)  # Descending order
        
        results = []
        for i, idx in enumerate(ranking_indices):
            results.append({
                'Rank': i + 1,
                'AI Model': self.alternatives[idx],
                'Preference Score': round(self.preference_scores[idx], 3),
                'Status': self.get_status(i + 1)
            })
            
        self.results_df = pd.DataFrame(results)
        print(self.results_df.to_string(index=False))
        
        # Print winner announcement
        winner = self.alternatives[ranking_indices[0]]
        winner_score = self.preference_scores[ranking_indices[0]]
        
        print(f"\nWINNER: {winner}")
        print(f"Score: {winner_score:.3f}\n")
        
    def get_status(self, rank):
        """Status berdasarkan ranking"""
        if rank == 1:
            return "Main C"
        elif rank == 2:
            return "Side C"  
        elif rank == 3:
            return "Support"
        else:
            return "Bckground"
            
    def create_visualization(self):
        """Buat visualisasi yang profesional"""
        print("Creating visualizations...\n")
        
        # Create figure dengan multiple subplots
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('AI Generatif Performance Analysis - TOPSIS Method', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Bar chart preference scores
        ax1 = plt.subplot(2, 3, 1)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726']
        bars = ax1.bar(range(len(self.alternatives)), self.preference_scores, 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_title('Preference Scores Ranking', fontsize=14, fontweight='bold')
        ax1.set_xlabel('AI Models')
        ax1.set_ylabel('Preference Score')
        ax1.set_xticks(range(len(self.alternatives)))
        ax1.set_xticklabels(self.alternatives)
        
        # Add value labels on bars
        for bar, score in zip(bars, self.preference_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Radar chart untuk criteria comparison
        ax2 = plt.subplot(2, 3, 2, projection='polar')
        angles = np.linspace(0, 2*np.pi, len(self.criteria), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, alt in enumerate(self.alternatives):
            values = self.decision_matrix[i].tolist()
            values += values[:1]  # Complete the circle
            ax2.plot(angles, values, 'o-', linewidth=2, label=alt, 
                     color=colors[i], alpha=0.8)
            ax2.fill(angles, values, alpha=0.1, color=colors[i])
            
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(['Akurasi', 'Kecepatan', 'Kemudahan', 'Fleksibilitas'])
        ax2.set_title('Multi-Criteria Comparison', fontsize=12, fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 3. Heatmap weighted matrix
        ax3 = plt.subplot(2, 3, 3)
        im = ax3.imshow(self.weighted_matrix, cmap='YlOrRd', aspect='auto')
        ax3.set_title('Weighted Matrix Heatmap', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(self.criteria)))
        ax3.set_xticklabels([f'C{i+1}' for i in range(len(self.criteria))])
        ax3.set_yticks(range(len(self.alternatives)))
        ax3.set_yticklabels(self.alternatives)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Weighted Score', rotation=270, labelpad=15)
        
        # 4. Distance comparison
        ax4 = plt.subplot(2, 3, 4)
        x = np.arange(len(self.alternatives))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, self.distance_positive, width, 
                        label='Distance to Ideal (+)', color='lightcoral', alpha=0.8)
        bars2 = ax4.bar(x + width/2, self.distance_negative, width,
                        label='Distance to Anti-Ideal (-)', color='lightblue', alpha=0.8)
        
        ax4.set_title('Distance Analysis', fontsize=12, fontweight='bold')
        ax4.set_xlabel('AI Models')
        ax4.set_ylabel('Distance')
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.alternatives)
        ax4.legend()
        
        # 5. Criteria weights pie chart
        ax5 = plt.subplot(2, 3, 5)
        colors_pie = ['#FFB6C1', '#98FB98', '#87CEEB', '#DDA0DD']
        wedges, texts, autotexts = ax5.pie(self.weights, labels=self.criteria, autopct='%1.1f%%',
                                           colors=colors_pie, startangle=90, explode=[0.05]*len(self.criteria))
        ax5.set_title('Criteria Weights Distribution', fontsize=12, fontweight='bold')
        
        # 6. Final ranking table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create table data
        table_data = []
        sorted_indices = np.argsort(-self.preference_scores)
        for i, idx in enumerate(sorted_indices):
            table_data.append([
                f"#{i+1}",
                self.alternatives[idx],
                f"{self.preference_scores[idx]:.3f}",
                self.get_status(i+1)
            ])
            
        table = ax6.table(cellText=table_data,
                          colLabels=['Rank', 'AI Model', 'Score', 'Status'],
                          cellLoc='center',
                          loc='center',
                          colColours=['#E6E6FA']*4)
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax6.set_title('Final Ranking Table', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        # hm
    def generate_insights(self):
        """Generate insights dan rekomendasi"""
        print("Step 8: AI Insights & Recommendations")
        print("="*60)
        
        sorted_indices = np.argsort(-self.preference_scores)
        winner = self.alternatives[sorted_indices[0]]
        runner_up = self.alternatives[sorted_indices[1]]
        
        print(f"KEY INSIGHTS:")
        print(f"  - {winner} muncul sebagai model terbaik dengan kinerja yang seimbang.")
        print(f"  - {runner_up} menjadi pilihan kedua yang kuat.")
        print(f"  - Fleksibilitas Integrasi (26.0%) adalah kriteria paling penting berdasarkan survei.")
        print(f"  - Kemudahan Penggunaan (25.7%) juga sangat krusial untuk pengalaman pengguna.")
        
        print(f"\nDETAILED ANALYSIS:")
        for i, idx in enumerate(sorted_indices):
            model = self.alternatives[idx]
            score = self.preference_scores[idx]
            if i == 0:
                print(f"  - {model}: Pilihan utama. Skor {score:.3f}. Keseimbangan sempurna dari semua kriteria.")
            elif i == 1:
                print(f"  - {model}: Pilihan solid. Skor {score:.3f}. Kontender yang kuat.")
            elif i == 2:
                print(f"  - {model}: Opsi yang layak. Skor {score:.3f}. Bagus untuk kebutuhan spesifik.")
            else:
                print(f"  - {model}: Perlu peningkatan. Skor {score:.3f}. Pertimbangkan alternatif lain.")
                
        print(f"\nRECOMMENDATIONS:")
        print(f"  - Untuk kinerja keseluruhan terbaik -> Pilih {winner}.")
        print(f"  - Untuk kasus penggunaan spesifik -> Periksa skor kriteria individual.")
        print(f"  - Untuk penelitian di masa depan -> Pertimbangkan untuk menambahkan kriteria lain seperti biaya dan ketersediaan API.")
        print(f"  - Perbarui data secara berkala karena model AI terus berkembang.")
        
    def run_complete_analysis(self):
        """Jalankan analisis lengkap"""
        self.print_header()
        
        # Step by step analysis
        df = self.create_decision_matrix()
        self.set_weights()
        self.normalize_matrix()
        self.weighted_normalized_matrix()
        self.ideal_solutions()
        self.calculate_distances()
        self.calculate_preference_scores()
        
        # Visualizations
        self.create_visualization()
        
        # Final insights
        self.generate_insights()
        
        print("\n" + "="*80)
        print("ANALISIS SELESAI!")
        print("="*80)
        
        return self.results_df

# Main execution
if __name__ == "__main__":
    # Create TOPSIS analyzer
    topsis = TOPSISProfessional()
    
    # Run complete analysis
    final_results = topsis.run_complete_analysis()
    
    # Optional: Save results to CSV
    final_results.to_csv('topsis_ai_results.csv', index=False)
    print(f"\nResults saved to 'topsis_ai_results.csv'")