import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator
import os

def plot_sampling_rate_distribution(df_hist, save_path):
    sr = None
    for c in ['fs_inferred','fs_target','fs','sampling_rate']:
        if c in df_hist.columns:
            sr = pd.to_numeric(df_hist[c], errors='coerce'); break
    length = None
    for c in ['数据长度','data_length']:
        if c in df_hist.columns:
            length = pd.to_numeric(df_hist[c], errors='coerce'); break
    fig_w = 6 if (length is None or not length.notna().any()) else 12
    fig, axes = plt.subplots(1, 1 if fig_w==6 else 2, figsize=(fig_w, 4))
    if fig_w==6: axes=[axes]
    if sr is not None and sr.notna().any():
        counts = {12000: int((sr==12000).sum()), 48000: int((sr==48000).sum())}
        axes[0].bar(['12kHz','48kHz'], [counts[12000], counts[48000]], color=['#87CEFA','#FFB347'])
        axes[0].set_title('Sampling Rate Distribution', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Sampling Rate (Hz)', fontsize=11); axes[0].set_ylabel('Sample Count', fontsize=11)
        axes[0].tick_params(axis='both', labelsize=10)
        for x,y in zip(['12kHz','48kHz'], [counts[12000], counts[48000]]):
            axes[0].text(x, y+0.5, str(y), ha='center', va='bottom', fontsize=9)
    else:
        axes[0].axis('off'); axes[0].text(0.5,0.5,'No sampling rate info', ha='center', va='center')
    if fig_w==12:
        if length is not None and length.notna().any():
            axes[1].hist(length.dropna(), bins=10, color='#90EE90', edgecolor='white')
            axes[1].set_title('Data Length Distribution', fontsize=13, fontweight='bold')
            axes[1].set_xlabel('Length (samples)', fontsize=11); axes[1].set_ylabel('Sample Count', fontsize=11)
            axes[1].tick_params(axis='both', labelsize=10)
        else:
            axes[1].axis('off'); axes[1].text(0.5,0.5,'No data length info', ha='center', va='center')
    plt.tight_layout();
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight');
    plt.close()

def plot_sample_distribution(features_df, save_path):
    if 'fault_type' in features_df.columns:
        labels = features_df['fault_type'].astype(str)
    elif 'label_cls' in features_df.columns:
        map_ft = {'N':'Normal','OR':'Outer Race','IR':'Inner Race','B':'Ball'}
        labels = features_df['label_cls'].map(map_ft).fillna(features_df['label_cls'].astype(str))
    else:
        labels = pd.Series(['Unknown']*len(features_df))
    order = [c for c in ['Normal','Outer Race','Inner Race','Ball'] if (labels==c).any()]
    counts = labels.value_counts().reindex(order).fillna(0).astype(int)
    fig, axes = plt.subplots(1,2, figsize=(11,3.8))
    colors = ['#FF7F7F','#87CEFA','#90EE90','#FFD700'][:len(counts)]
    bars = axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_title('Sample Distribution by Fault Type', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Fault Type', fontsize=11)
    axes[0].set_ylabel('Sample Count', fontsize=11)
    axes[0].tick_params(axis='x', labelsize=10, rotation=0)
    axes[0].tick_params(axis='y', labelsize=10)
    axes[0].grid(True, axis='y', alpha=0.3)
    for b,v in zip(bars, counts.values):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.5, str(v), ha='center', va='bottom', fontsize=9)
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'fontsize':10})
    axes[1].set_title('Class Proportion', fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_signal_waveforms(signals, save_path):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    step = 100
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), facecolor='#f7f7f7')
    fig.suptitle('DE Signal Waveforms for Different Fault Types',
                 fontsize=18, fontweight='bold', color='#333333', y=0.95)
    fault_labels = ['B', 'IR', 'OR', 'N']
    fault_names = {'B': 'Ball Fault', 'IR': 'Inner Race Fault', 'OR': 'Outer Race Fault', 'N': 'Normal'}
    for ax, fault_type, color in zip(axs.flatten(), fault_labels, colors):
        if fault_type in signals:
            signal_data = signals[fault_type]
            t = np.linspace(0, 1000, len(signal_data))
            ax.plot(t[::step], signal_data[::step], color=color, linewidth=1.2, 
                    label=f'Fault Type: {fault_type}')
            ax.set_title(f'{fault_names[fault_type]} ({fault_type})', 
                        fontsize=14, fontweight='bold', color=color)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)
            ax.grid(True, which='major', alpha=0.3)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)
            ax.tick_params(axis='both', which='minor', length=3, width=1.0)
        else:
            ax.text(0.5, 0.5, f'No data for {fault_type}', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Fault Type: {fault_type}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_rms_kurtosis_scatter(features_df, save_path):
    df = features_df
    need_cols = ["label_cls", "DE_rms", "DE_kurtosis"]
    missing_cols = [c for c in need_cols if c not in df.columns]
    if missing_cols:
        return
    label_order = ["N", "B", "IR", "OR"]
    markers = {"N": "o", "B": "x", "IR": "s", "OR": "D"}
    colors_map = {"N": "#2E8B57", "B": "#FF6347", "IR": "#4169E1", "OR": "#FFD700"}
    plt.figure(figsize=(10, 7))
    for lab in label_order:
        part = df[df["label_cls"] == lab]
        if len(part) == 0:
            continue
        plt.scatter(
            part["DE_rms"], part["DE_kurtosis"],
            s=60, alpha=0.8,
            marker=markers.get(lab, "o"),
            c=colors_map.get(lab, "gray"),
            label=f'{lab} ({len(part)} samples)',
            linewidths=0.8, edgecolors="black"
        )
    plt.title("DE Channel: RMS vs Kurtosis Feature Space\n(Fault Type Classification)", 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("RMS (Root Mean Square)", fontsize=12)
    plt.ylabel("Kurtosis (Impact Sensitivity)", fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(title="Fault Types", frameon=True, fontsize=10, 
               title_fontsize=11, loc='upper right')
    plt.text(0.02, 0.98, 'Low Energy\nLow Impact', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             fontsize=9, verticalalignment='top')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_correlation_heatmap(features_df, save_path):
    df = features_df
    key_features = ['DE_rms', 'DE_kurtosis', 'DE_crest_factor', 'DE_spec_centroid',
                    'FE_rms', 'FE_kurtosis', 'FE_crest_factor', 'FE_spec_centroid']
    available_features = [feat for feat in key_features if feat in df.columns]
    if len(available_features) >= 4:
        corr_matrix = df[available_features].corr()
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, center=0,
                    square=True, linewidths=0.5, fmt='.2f',
                    cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix\n(Key Bearing Fault Features)',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Features', fontweight='bold', fontsize=12)
        plt.ylabel('Features', fontweight='bold', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_load_stratified_distribution(features_df, save_path):
    df = features_df
    if 'label_load_hp' in df.columns:
        load_df = df[df['label_load_hp'].notna()]
        if len(load_df) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            load_counts = load_df['label_load_hp'].value_counts().sort_index()
            axes[0,0].bar(load_counts.index, load_counts.values, 
                         color='steelblue', alpha=0.7, edgecolor='black')
            axes[0,0].set_xlabel('Load (HP)', fontweight='bold')
            axes[0,0].set_ylabel('Sample Count', fontweight='bold')
            axes[0,0].set_title('Load Distribution\n(Load Distribution Pattern)', 
                               fontweight='bold', pad=15)
            axes[0,0].grid(True, alpha=0.3)
            if 'DE_rms' in df.columns:
                for load in sorted(load_df['label_load_hp'].unique()):
                    load_data = load_df[load_df['label_load_hp'] == load]
                    axes[0,1].scatter(load_data['label_load_hp'], load_data['DE_rms'],
                                    alpha=0.6, s=50, label=f'{load} HP')
                
                axes[0,1].set_xlabel('Load (HP)', fontweight='bold')
                axes[0,1].set_ylabel('DE RMS', fontweight='bold')
                axes[0,1].set_title('Load vs DE RMS Relationship\n(Load vs RMS Relationship)', 
                                   fontweight='bold', pad=15)
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].legend()
            if 'DE_kurtosis' in df.columns:
                load_groups = []
                kurtosis_groups = []
                
                for load in sorted(load_df['label_load_hp'].unique()):
                    load_data = load_df[load_df['label_load_hp'] == load]['DE_kurtosis'].dropna()
                    if len(load_data) > 0:
                        load_groups.append(str(load))
                        kurtosis_groups.append(load_data.values)
                
                if len(kurtosis_groups) > 1:
                    bp = axes[1,0].boxplot(kurtosis_groups, labels=load_groups, patch_artist=True)
                    colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                
                axes[1,0].set_xlabel('Load (HP)', fontweight='bold')
                axes[1,0].set_ylabel('DE Kurtosis', fontweight='bold')
                axes[1,0].set_title('Load vs DE Kurtosis Distribution\n(Load vs Kurtosis Pattern)', 
                                   fontweight='bold', pad=15)
                axes[1,0].grid(True, alpha=0.3)
            load_fault_cross = pd.crosstab(load_df['label_load_hp'], load_df['label_cls'])
            load_fault_cross.plot(kind='bar', stacked=True, ax=axes[1,1], 
                                 color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'])
            axes[1,1].set_xlabel('Load (HP)', fontweight='bold')
            axes[1,1].set_ylabel('Sample Count', fontweight='bold')
            axes[1,1].set_title('Load vs Fault Type Distribution\n(Fault Distribution by Load)', 
                               fontweight='bold', pad=15)
            axes[1,1].legend(title='Fault Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].tick_params(axis='x', rotation=0)
            
            plt.tight_layout()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

def plot_fault_type_distribution(features_df, save_path):
    df = features_df
    fault_counts = df['label_cls'].value_counts()
    fault_names = {'OR': 'Outer RaceFault', 'IR': 'Inner RaceFault', 'B': 'BallFault', 'N': 'Normal'}
    plt.figure(figsize=(10, 6))
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    bars = plt.bar(range(len(fault_counts)), fault_counts.values, 
                   color=colors[:len(fault_counts)], alpha=0.8, edgecolor='black', linewidth=1)
    plt.xlabel('Fault Type', fontsize=12, fontweight='bold')
    plt.ylabel('Sample Count', fontsize=12, fontweight='bold')
    plt.title('Fault Type Distribution', 
              fontsize=14, fontweight='bold', pad=20)
    fault_labels = [f"{idx}\n({fault_names.get(idx, idx)})" for idx in fault_counts.index]
    plt.xticks(range(len(fault_counts)), fault_labels, fontsize=11)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_rpm_distribution(features_df, save_path):
    df = features_df
    if 'rpm_mean' in df.columns:
        rpm_data = df[df['rpm_mean'].notna()]['rpm_mean']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.hist(rpm_data, bins=15, color='skyblue', alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_xlabel('RPM (Rotational Speed)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('RPM Distribution Histogram', 
                      fontsize=13, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='--')
        rpm_stats = f'Mean: {rpm_data.mean():.1f}\nStd: {rpm_data.std():.1f}\nRange: {rpm_data.min():.0f}-{rpm_data.max():.0f}'
        ax1.text(0.02, 0.98, rpm_stats, transform=ax1.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                 verticalalignment='top')
        fault_types = df[df['rpm_mean'].notna()]['label_cls'].unique()
        rpm_by_fault = [df[df['label_cls'] == fault]['rpm_mean'].dropna().values 
                        for fault in fault_types]
        box_plot = ax2.boxplot(rpm_by_fault, labels=fault_types, patch_artist=True)
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
        for patch, color in zip(box_plot['boxes'], colors[:len(fault_types)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_xlabel('Fault Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('RPM (Rotational Speed)', fontsize=12, fontweight='bold')
        ax2.set_title('RPM Distribution by Fault Type', 
                      fontsize=13, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_vibration_mean_comparison(features_df, save_path):
    df = features_df
    mean_features = [col for col in ['DE_mean', 'FE_mean', 'BA_mean'] if col in df.columns]
    if len(mean_features) >= 2:
        fault_types = df['label_cls'].unique()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        x = np.arange(len(fault_types))
        width = 0.25
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        channel_names = {'DE_mean': 'DE_point', 'FE_mean': 'FE_point', 'BA_mean': 'BA_point'}
        for i, feature in enumerate(mean_features):
            means_by_fault = []
            for fault in fault_types:
                fault_data = df[df['label_cls'] == fault][feature].dropna()
                if len(fault_data) > 0:
                    means_by_fault.append(fault_data.mean())
                else:
                    means_by_fault.append(0)
            ax1.bar(x + i * width, means_by_fault, width, 
                   label=channel_names.get(feature, feature), 
                   color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Fault Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Vibration Mean', fontsize=12, fontweight='bold')
        ax1.set_title('Vibration Signal Mean by Fault Type', 
                      fontsize=13, fontweight='bold', pad=15)
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(fault_types)
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        sample_indices = range(min(30, len(df)))
        for i, feature in enumerate(mean_features):
            values = df[feature].iloc[:len(sample_indices)].fillna(0)
            ax2.plot(sample_indices, values, marker='o', linewidth=2, markersize=4,
                    label=channel_names.get(feature, feature), color=colors[i], alpha=0.8)
        ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Vibration Mean', fontsize=12, fontweight='bold')
        ax2.set_title('Vibration Signal Mean Sequence', 
                      fontsize=13, fontweight='bold', pad=15)
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_vibration_std_comparison(features_df, save_path):
    df = features_df
    std_features = [col for col in ['DE_std', 'FE_std', 'BA_std'] if col in df.columns]
    if len(std_features) >= 2:
        fault_types = df['label_cls'].unique()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        x = np.arange(len(fault_types))
        width = 0.25
        colors = ['#E74C3C', '#3498DB', '#2ECC71']
        channel_names = {'DE_std': 'DE_point', 'FE_std': 'FE_point', 'BA_std': 'BA_point'}
        for i, feature in enumerate(std_features):
            stds_by_fault = []
            for fault in fault_types:
                fault_data = df[df['label_cls'] == fault][feature].dropna()
                if len(fault_data) > 0:
                    stds_by_fault.append(fault_data.mean())
                else:
                    stds_by_fault.append(0)
            ax1.bar(x + i * width, stds_by_fault, width, 
                   label=channel_names.get(feature, feature), 
                   color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Fault Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Vibration Standard Deviation', fontsize=12, fontweight='bold')
        ax1.set_title('Vibration Signal Std by Fault Type', 
                      fontsize=13, fontweight='bold', pad=15)
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(fault_types)
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        sample_indices = range(min(30, len(df)))
        for i, feature in enumerate(std_features):
            values = df[feature].iloc[:len(sample_indices)].fillna(0)
            ax2.plot(sample_indices, values, marker='s', linewidth=2, markersize=4,
                    label=channel_names.get(feature, feature), color=colors[i], alpha=0.8)
        ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Vibration Standard Deviation', fontsize=12, fontweight='bold')
        ax2.set_title('Vibration Signal Std Sequence', 
                      fontsize=13, fontweight='bold', pad=15)
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_vibration_amplitude_range(features_df, save_path):
    df = features_df
    if 'DE_rms' in df.columns and 'DE_std' in df.columns:
        n_samples = min(40, len(df))
        sample_indices = range(n_samples)
        fault_colors = {'OR': '#E74C3C', 'IR': '#3498DB', 'B': '#2ECC71', 'N': '#F39C12'}
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        de_means = df['DE_rms'].iloc[:n_samples].values
        de_stds = df['DE_std'].iloc[:n_samples].values
        fault_types = df['label_cls'].iloc[:n_samples].values
        colors = [fault_colors.get(fault, 'gray') for fault in fault_types]
        ax1.errorbar(sample_indices, de_means, yerr=de_stds, 
                    fmt='o', capsize=3, capthick=1, elinewidth=1.5,
                    markersize=6, alpha=0.8)
        for i, (x, y, color, fault) in enumerate(zip(sample_indices, de_means, colors, fault_types)):
            ax1.scatter(x, y, c=color, s=50, alpha=0.9, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('DE Vibration Amplitude', fontsize=12, fontweight='bold')
        ax1.set_title('DE Channel Vibration Amplitude Range', 
                      fontsize=13, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='--')
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                             markersize=8, label=f'{fault}Fault') 
                  for fault, color in fault_colors.items() if fault in fault_types]
        ax1.legend(handles=handles, loc='upper right')
        if 'FE_rms' in df.columns and 'FE_std' in df.columns:
            fe_means = df['FE_rms'].iloc[:n_samples].fillna(0).values
            fe_stds = df['FE_std'].iloc[:n_samples].fillna(0).values
            x_offset = 0.2
            ax2.errorbar([x - x_offset for x in sample_indices], de_means, yerr=de_stds,
                        fmt='o', capsize=3, capthick=1, elinewidth=1.5,
                        markersize=5, alpha=0.8, color='#E74C3C', label='DE_point')
            
            ax2.errorbar([x + x_offset for x in sample_indices], fe_means, yerr=fe_stds,
                        fmt='s', capsize=3, capthick=1, elinewidth=1.5,
                        markersize=5, alpha=0.8, color='#3498DB', label='FE_point')
        else:
            if 'DE_peak' in df.columns:
                de_peaks = df['DE_peak'].iloc[:n_samples].values
                de_minimums = de_means - de_stds
                ax2.fill_between(sample_indices, de_minimums, de_peaks, 
                                alpha=0.3, color='#E74C3C', label='振动Range')
                ax2.plot(sample_indices, de_means, 'o-', color='#E74C3C', 
                        markersize=4, linewidth=1.5, label='Mean', alpha=0.8)
                ax2.plot(sample_indices, de_peaks, '^-', color='#FF6B6B', 
                        markersize=4, linewidth=1.5, label='Peak', alpha=0.8)
        ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Vibration Amplitude', fontsize=12, fontweight='bold')
        ax2.set_title('Multi-Channel Vibration Amplitude Comparison', 
                      fontsize=13, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_fault_type_vs_vibration_features(features_df, save_path):
    df = features_df
    vibration_features = ['de_rms', 'fe_rms', 'ba_rms', 'de_std', 'fe_std', 'ba_std']
    available_features = [f for f in vibration_features if f in df.columns]
    if len(available_features) > 0 and 'fault_type' in df.columns:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for i, feature in enumerate(available_features[:6]):
            if i < len(axes):
                fault_types = df['fault_type'].unique()
                data_for_plot = [df[df['fault_type']==ft][feature].dropna() for ft in fault_types]
                bp = axes[i].boxplot(data_for_plot, labels=fault_types, patch_artist=True)
                colors = plt.cm.Set3(np.linspace(0, 1, len(fault_types)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                axes[i].set_title(f'{feature.upper()} by Fault Type', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Fault Type', fontsize=10)
                axes[i].set_ylabel(f'{feature.upper()}', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='x', rotation=45)
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        plt.suptitle('Fault Type vs Vibration Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_rpm_vs_vibration_intensity(features_df, save_path):
    df = features_df
    rpm_col = None
    for col in ['rpm_mean', 'rpm', 'RPM']:
        if col in df.columns:
            rpm_col = col
            break
    vibration_features = ['de_rms', 'fe_rms', 'ba_rms']
    if 'fault_type' not in df.columns and 'label_cls' in df.columns:
        df = df.copy(); df['fault_type'] = df['label_cls']
    available_features = [f for f in vibration_features if f in df.columns]
    if rpm_col and len(available_features) > 0:
        fig, axes = plt.subplots(1, len(available_features), figsize=(6*len(available_features), 5))
        if len(available_features) == 1:
            axes = [axes]
        colors = plt.cm.Set1(np.linspace(0, 1, len(df['fault_type'].unique())))
        fault_color_map = dict(zip(df['fault_type'].unique(), colors))
        for i, feature in enumerate(available_features):
            for fault_type in df['fault_type'].unique():
                mask = df['fault_type'] == fault_type
                axes[i].scatter(df[mask][rpm_col], df[mask][feature],
                              c=[fault_color_map[fault_type]], label=fault_type, alpha=0.7, s=50)
            axes[i].set_xlabel('RPM', fontsize=12)
            axes[i].set_ylabel(f'{feature.upper()}', fontsize=12)
            axes[i].set_title(f'RPM vs {feature.upper()}', fontsize=14, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            x_all = pd.to_numeric(df[rpm_col], errors='coerce')
            y_all = pd.to_numeric(df[feature], errors='coerce')
            msk = x_all.notna() & y_all.notna()
            if msk.sum() >= 2:
                z = np.polyfit(x_all[msk].values, y_all[msk].values, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_all[msk].min(), x_all[msk].max(), 100)
                axes[i].plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2)
        from matplotlib.lines import Line2D
        ft_unique = list(df['fault_type'].unique())
        handles = [Line2D([0],[0], marker='o', linestyle='', color=fault_color_map[ft], markersize=6, alpha=0.8) for ft in ft_unique]
        labels = ft_unique
        for ax in axes:
            leg = ax.get_legend()
            if leg is not None:
                try: leg.remove()
                except Exception: pass
        fig.legend(handles, labels, title='Fault Type', loc='upper left', bbox_to_anchor=(0.01, 0.99),
                   frameon=True, fontsize=9, title_fontsize=10, markerscale=0.8, borderpad=0.2,
                   labelspacing=0.3, handletextpad=0.4)
        plt.suptitle('RPM vs Vibration Intensity Relationship', fontsize=16, fontweight='bold')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_three_channel_feature_correlation(features_df, save_path):
    df = features_df
    de_features = [col for col in df.columns if col.startswith('de_')][:3]
    fe_features = [col for col in df.columns if col.startswith('fe_')][:3]
    ba_features = [col for col in df.columns if col.startswith('ba_')][:3]
    all_channel_features = de_features + fe_features + ba_features
    available_features = [f for f in all_channel_features if f in df.columns]
    if len(available_features) >= 6 and 'fault_type' in df.columns:
        sample_size = min(100, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_data = df_sample[available_features[:9]]
        feature_data_scaled = pd.DataFrame(
            scaler.fit_transform(feature_data),
            columns=feature_data.columns,
            index=feature_data.index
        )
        feature_data_scaled['fault_type'] = df_sample['fault_type'].values
        fig, ax = plt.subplots(figsize=(15, 8))
        fault_types = feature_data_scaled['fault_type'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(fault_types)))
        for i, fault_type in enumerate(fault_types):
            mask = feature_data_scaled['fault_type'] == fault_type
            subset = feature_data_scaled[mask]
            for idx, row in subset.iterrows():
                values = row[available_features[:9]].values
                ax.plot(range(len(values)), values, color=colors[i], alpha=0.6, linewidth=1)
        ax.set_xticks(range(len(available_features[:9])))
        ax.set_xticklabels([f.upper() for f in available_features[:9]], rotation=45, ha='right')
        ax.set_ylabel('Standardized Values', fontsize=12)
        ax.set_title('Three-Channel Feature Correlation', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        legend_elements = [plt.Line2D([0], [0], color=colors[i], label=fault_type)
                          for i, fault_type in enumerate(fault_types)]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()