#!/usr/bin/env python3
"""
Script to download WandB data and create uncertainty comparison plots 
for ORX ablation study.
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Optional

# Set plot style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

class WandBDataDownloader:
    def __init__(self, entity: str, project: str):
        self.entity = entity
        self.project = project
        self.api = wandb.Api()
        
    def download_runs_data(self, filters: Dict = None) -> pd.DataFrame:
        """Download all runs data from WandB project."""
        print(f"Downloading data from {self.entity}/{self.project}...")
        
        # Get all runs from the project
        runs = self.api.runs(f"{self.entity}/{self.project}", filters=filters)
        
        all_data = []
        
        for run in runs:
            print(f"Processing run: {run.name} ({run.state})")
            
            # Skip failed runs
            if run.state != "finished":
                print(f"  Skipping {run.name} - state: {run.state}")
                continue
                
            # Get run config and metrics
            config = run.config
            history = run.history()
            
            if history.empty:
                print(f"  No history data for {run.name}")
                continue
                
            # Add metadata to each row
            history['run_name'] = run.name
            
            # Extract strategy from run name (format: Strategy-noise0.1-seed834)
            strategy = config.get('exploration_strategy', 'Unknown')
            noise_level = config.get('noise_level', 1.0)
            seed = config.get('seed', 0)
            
            if strategy == 'Unknown' and run.name:
                # Parse from run name format: "ORX-noise0.1-seed834" or "Optimistic-noise0.1-seed834"
                parts = run.name.split('-')
                if len(parts) >= 1:
                    strategy = parts[0]
                    
                # Extract noise level from run name
                for part in parts:
                    if part.startswith('noise'):
                        try:
                            noise_level = float(part.replace('noise', ''))
                        except ValueError:
                            pass
                    elif part.startswith('seed'):
                        try:
                            seed = int(part.replace('seed', ''))
                        except ValueError:
                            pass
            
            history['strategy'] = strategy
            history['noise_level'] = noise_level
            history['seed'] = seed
            history['beta'] = config.get('beta', 1.0)
            history['use_log'] = config.get('use_log', False)
            history['use_al'] = config.get('use_al', False)
            
            all_data.append(history)
            
        if not all_data:
            print("No data found!")
            return pd.DataFrame()
            
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Downloaded {len(combined_df)} data points from {len(all_data)} runs")
        
        return combined_df

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save dataframe to CSV."""
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

class UncertaintyPlotter:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def create_uncertainty_plot(self, save_path: str = "uncertainty_comparison.png"):
        """Create uncertainty comparison plot with mean lines and confidence intervals."""
        
        # Find the uncertainty column to use
        uncertainty_col = None
        if 'validation_eps_std_max' in self.data.columns:
            uncertainty_col = 'validation_eps_std_max'
        elif 'val_eps_std' in self.data.columns:
            uncertainty_col = 'val_eps_std'
        else:
            uncertainty_columns = [col for col in self.data.columns if 'uncertainty' in col.lower() or 'std' in col.lower()]
            if uncertainty_columns:
                uncertainty_col = uncertainty_columns[0]
            else:
                print("No uncertainty columns found!")
                return
        
        print(f"Using uncertainty column: {uncertainty_col}")
        
        # Determine episode column
        episode_col = None
        if 'learning_step' in self.data.columns:
            episode_col = 'learning_step'
        elif '_step' in self.data.columns:
            episode_col = '_step'
        else:
            print("No episode column found!")
            return
            
        print(f"Using episode column: {episode_col}")
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Focus on ORX for the main comparison
        orx_data = self.data[self.data['strategy'] == 'ORX'].copy()
        
        if orx_data.empty:
            print("No ORX data found!")
            return
            
        print(f"Found {len(orx_data)} ORX data points")
        print(f"ORX noise levels: {sorted(orx_data['noise_level'].unique())}")
        print(f"ORX seeds: {sorted(orx_data['seed'].unique())}")
        
        # Colors and styles for different noise levels
        noise_level_styles = {
            0.1: {'color': '#2E86AB', 'linestyle': '--', 'label': 'ORX'},      # Dashed for ORX
            1.0: {'color': '#2E86AB', 'linestyle': '--', 'label': 'ORX'},     # Dashed for ORX
            10.0: {'color': '#2E86AB', 'linestyle': '--', 'label': 'ORX'}    # Dashed for ORX
        }
        
        # Process each noise level separately
        for noise_level in sorted(orx_data['noise_level'].unique()):
            noise_data = orx_data[orx_data['noise_level'] == noise_level].copy()
            
            print(f"\nProcessing noise level {noise_level}:")
            print(f"  Seeds: {sorted(noise_data['seed'].unique())}")
            
            # Collect data by episode across all seeds
            episode_data = {}
            
            for seed in sorted(noise_data['seed'].unique()):
                seed_data = noise_data[noise_data['seed'] == seed]
                
                if uncertainty_col not in seed_data.columns or seed_data[uncertainty_col].isna().all():
                    continue
                    
                # Only use rows where learning_step is not NaN (evaluation points)
                eval_data = seed_data[seed_data['learning_step'].notna()]
                clean_data = eval_data[['learning_step', '_step', uncertainty_col]].dropna()
                
                if len(clean_data) == 0:
                    continue
                    
                print(f"    Seed {seed}: {len(clean_data)} data points")
                
                # For each learning_step, take only the first occurrence (earliest _step)
                for learning_step in clean_data['learning_step'].unique():
                    step_data = clean_data[clean_data['learning_step'] == learning_step]
                    # Take the first occurrence (smallest _step)
                    first_occurrence = step_data.loc[step_data['_step'].idxmin()]
                    
                    episode = int(first_occurrence['learning_step'])
                    uncertainty = float(first_occurrence[uncertainty_col])
                    
                    if episode not in episode_data:
                        episode_data[episode] = []
                    episode_data[episode].append(uncertainty)
            
            if not episode_data:
                print(f"  No valid data for noise level {noise_level}")
                continue
                
            # Convert to arrays for plotting
            episodes = sorted(episode_data.keys())
            means = []
            stds = []
            
            for episode in episodes:
                values = episode_data[episode]
                means.append(np.mean(values))
                stds.append(np.std(values))
            
            episodes = np.array(episodes)
            means = np.array(means)
            stds = np.array(stds)
            
            print(f"  Final: {len(episodes)} episode points, mean uncertainty: {np.mean(means):.3f}")
            
            # Get style for this noise level - use simplified label since we only have one noise level
            style = noise_level_styles.get(noise_level, {
                'color': '#2E86AB', 'linestyle': '--', 'label': 'ORX'
            })
            
            # Plot mean line
            ax.plot(episodes, means, 
                   color=style['color'], 
                   linestyle=style['linestyle'], 
                   linewidth=3, 
                   label=style['label'],
                   alpha=0.9)
            
            # Add shaded confidence interval (mean ± std)
            ax.fill_between(episodes, 
                           means - stds, 
                           means + stds,
                           color=style['color'], 
                           alpha=0.25)
        
        # Also add OpAx and Random baselines if available
        other_strategies = ['Optimistic', 'Uniform']
        other_styles = {
            'Optimistic': {'color': '#F18F01', 'linestyle': '-', 'label': 'OpAX'},  # Solid for OpAX
            'Uniform': {'color': '#C73E1D', 'linestyle': '-.', 'label': 'Random'}   # Dash-dot for Random
        }
        
        for strategy in other_strategies:
            strategy_data = self.data[self.data['strategy'] == strategy].copy()
            
            if strategy_data.empty:
                continue
                
            # Combine all noise levels and seeds for baseline strategies
            episode_data = {}
            
            # Process each seed separately to get first occurrence per seed
            for seed in strategy_data['seed'].unique():
                seed_data = strategy_data[strategy_data['seed'] == seed]
                
                # Only use evaluation points
                eval_data = seed_data[seed_data['learning_step'].notna()]
                clean_data = eval_data[['learning_step', '_step', uncertainty_col]].dropna()
                
                if len(clean_data) == 0:
                    continue
                
                # For each learning_step in this seed, take only the first occurrence
                for learning_step in clean_data['learning_step'].unique():
                    step_data = clean_data[clean_data['learning_step'] == learning_step]
                    # Take the first occurrence (smallest _step)
                    first_occurrence = step_data.loc[step_data['_step'].idxmin()]
                    
                    episode = int(first_occurrence['learning_step'])
                    uncertainty = float(first_occurrence[uncertainty_col])
                    
                    if episode not in episode_data:
                        episode_data[episode] = []
                    episode_data[episode].append(uncertainty)
            
            if not episode_data:
                continue
                
            episodes = sorted(episode_data.keys())
            means = []
            stds = []
            
            for episode in episodes:
                values = episode_data[episode]
                means.append(np.mean(values))
                stds.append(np.std(values))
            
            episodes = np.array(episodes)
            means = np.array(means)
            stds = np.array(stds)
            
            style = other_styles[strategy]
            
            # Plot mean line
            ax.plot(episodes, means,
                   color=style['color'],
                   linestyle=style['linestyle'],
                   linewidth=3,
                   label=style['label'],
                   alpha=0.9)
            
            # Add shaded confidence interval
            ax.fill_between(episodes,
                           means - stds,
                           means + stds,
                           color=style['color'],
                           alpha=0.25)
        
        # Use a linear scale for the y-axis
        # ax.set_yscale('log')
        
        # Set custom tick marks for linear scale if needed, otherwise auto
        # ax.set_xticks([0, 5, 10, 15, 20])
        # ax.set_yticks([1, 10])  # 10^0 = 1, 10^1 = 10
        # ax.set_yticklabels(['10⁰', '10¹'])
        
        # Labels and formatting
        ax.set_xlabel('Episodes', fontsize=14)
        ax.set_ylabel('max σₙ', fontsize=14)
        ax.set_title('Pendulum-v1 - uncertainty', fontsize=16, pad=15)
        
        # Add square border around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')
        
        # Legend
        legend = ax.legend(fontsize=12, loc='upper right', fancybox=False, shadow=False)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_edgecolor('grey')
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Styling
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Statistical plot saved to {save_path}")
        
        plt.close(fig)
        
    def create_all_plots(self, results_dir: Path):
        """Create multiple types of plots for comprehensive analysis."""
        
        # Main uncertainty comparison plot
        self.create_uncertainty_plot(str(results_dir / "uncertainty_comparison.png"))
        
        # Reward comparison plots for different tasks
        if 'reward_task_1' in self.data.columns:
            self.create_reward_comparison_plot(
                reward_col_key='reward_task_1',
                plot_title='Pendulum-v1 - Swing-Up Task Reward',
                save_path=str(results_dir / "reward_swing_up_comparison.png"),
                yscale_type='linear',
                add_inset=False  # No inset for this plot
            )
        else:
            print("Column 'reward_task_1' not found. Skipping Swing-Up reward plot.")

        if 'reward_task_0' in self.data.columns:
            self.create_reward_comparison_plot(
                reward_col_key='reward_task_0',
                plot_title='Pendulum-v1 - Keep-Down Task Reward',
                save_path=str(results_dir / "reward_keep_down_comparison.png"),
                yscale_type='linear',
                add_inset=True  # Add the inset zoom for this plot
            )
        else:
            print("Column 'reward_task_0' not found. Skipping Keep-Down reward plot.")
        
        print(f"All plots saved to {results_dir}")
    
    def create_reward_comparison_plot(self, reward_col_key: str, plot_title: str, save_path: str, yscale_type: str = 'linear', yscale_kwargs: Optional[Dict] = None, add_inset: bool = False):
        """Create reward comparison plot with optional y-axis scaling and inset zoom."""
        
        if reward_col_key not in self.data.columns:
            print(f"Reward column {reward_col_key} not found in data. Skipping this plot.")
            return
            
        print(f"Creating reward plot: {plot_title} using column: {reward_col_key}, yscale: {yscale_type}")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Set y-axis scale before plotting
        if yscale_kwargs is None:
            yscale_kwargs = {}
        ax.set_yscale(yscale_type, **yscale_kwargs)
        
        # Colors and styles (consistent with uncertainty plot)
        strategy_styles = {
            'ORX': {'color': '#2E86AB', 'linestyle': '--', 'label': 'ORX'},
            'Optimistic': {'color': '#F18F01', 'linestyle': '-', 'label': 'OpAX'},
            'Uniform': {'color': '#C73E1D', 'linestyle': '-.', 'label': 'Random'}
        }
        
        processed_data = [] # Store data for inset
        for strategy in sorted(self.data['strategy'].unique()):
            strategy_data = self.data[self.data['strategy'] == strategy].copy()
            if strategy_data.empty:
                continue

            episode_data = {}
            # Process each seed separately to get first occurrence per seed
            for seed in strategy_data['seed'].unique():
                seed_data_for_strat = strategy_data[strategy_data['seed'] == seed]
                eval_data = seed_data_for_strat[seed_data_for_strat['learning_step'].notna()]
                # Use the passed reward_col_key here
                clean_data = eval_data[['learning_step', '_step', reward_col_key]].dropna()
                
                if len(clean_data) == 0:
                    continue
                
                for learning_step_val in clean_data['learning_step'].unique():
                    step_data = clean_data[clean_data['learning_step'] == learning_step_val]
                    first_occurrence = step_data.loc[step_data['_step'].idxmin()]
                    episode = int(first_occurrence['learning_step'])
                    # Use the passed reward_col_key here
                    reward_value = float(first_occurrence[reward_col_key])
                    
                    if episode not in episode_data:
                        episode_data[episode] = []
                    episode_data[episode].append(reward_value)
            
            if not episode_data:
                continue
                
            episodes = sorted(episode_data.keys())
            means = []
            stds = []
            
            for ep in episodes:
                values = episode_data[ep]
                means.append(np.mean(values))
                stds.append(np.std(values))
            
            episodes = np.array(episodes)
            means = np.array(means)
            stds = np.array(stds)
            
            style = strategy_styles.get(strategy, {'color': 'black', 'linestyle': '--', 'label': strategy})
            
            # Plot main lines
            ax.plot(episodes, means,
                   label=style['label'], 
                   color=style['color'], 
                   linestyle=style['linestyle'], 
                   linewidth=3,
                   alpha=0.9)
            
            ax.fill_between(episodes, 
                           means - stds, 
                           means + stds,
                           color=style['color'], 
                           alpha=0.25)
            
            processed_data.append({'episodes': episodes, 'means': means, 'stds': stds, 'style': style})
        
        if add_inset:
            # More centered and higher position for the inset
            ax_inset = ax.inset_axes([0.35, 0.25, 0.55, 0.45])  # Position: [left, bottom, width, height]
            
            y_min, y_max = np.inf, -np.inf

            for item in processed_data:
                # Plot on inset
                ax_inset.plot(item['episodes'], item['means'], color=item['style']['color'], linestyle=item['style']['linestyle'], linewidth=2)
                ax_inset.fill_between(item['episodes'], item['means'] - item['stds'], item['means'] + item['stds'], color=item['style']['color'], alpha=0.25)

                # Determine y-limits from data within the x-range [1, 11]
                mask = (item['episodes'] >= 1) & (item['episodes'] <= 11)
                if np.any(mask):
                    y_min = min(y_min, np.min(item['means'][mask] - item['stds'][mask]))
                    y_max = max(y_max, np.max(item['means'][mask] + item['stds'][mask]))

            ax_inset.set_xlim(1, 11)
            y_padding = (y_max - y_min) * 0.1
            ax_inset.set_ylim(y_min - y_padding, y_max + y_padding)
            
            # Make indicator lines thicker
            ax.indicate_inset_zoom(ax_inset, edgecolor="black", linewidth=1.5)

        ax.set_xlabel('Episodes', fontsize=14)
        ax.set_ylabel(r'$\sum_{t=0}^{T-1} r_t$', fontsize=14)
        ax.set_title(plot_title, fontsize=16, pad=15)
        
        legend_loc = 'lower right' if add_inset else 'best'
        legend = ax.legend(fontsize=12, loc=legend_loc, fancybox=False, shadow=False)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_edgecolor('grey')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('white')
        
        # Add square border around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')
        
        # Set custom x-axis tick marks
        # Let Matplotlib decide ticks for clarity with inset
        # ax.set_xticks([0, 5, 10, 15, 20])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Reward plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Download WandB data and create uncertainty plots")
    parser.add_argument("--entity", default="bbullinger", help="WandB entity name")
    parser.add_argument("--project", default="ORX-Ablation-Long-Single-Seed", help="WandB project name")
    parser.add_argument("--results-dir", default="results", help="Directory to save results")
    parser.add_argument("--output-data", default="wandb_data.csv", help="Output CSV filename")
    parser.add_argument("--output-plot", default="uncertainty_comparison.png", help="Output plot filename")
    parser.add_argument("--download", action="store_true", help="Download fresh data from WandB")
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    print(f"Using results directory: {results_dir}")
    
    # Set full paths for output files
    data_path = results_dir / args.output_data
    plot_path = results_dir / args.output_plot
    
    # Download or load data
    if args.download or not data_path.exists():
        print("Downloading data from WandB...")
        downloader = WandBDataDownloader(args.entity, args.project)
        df = downloader.download_runs_data()
        downloader.save_data(df, str(data_path))
    else:
        print(f"Loading existing data from {data_path}")
        df = pd.read_csv(data_path)
    
    if df.empty:
        print("No data available!")
        return
    
    # Create plots
    plotter = UncertaintyPlotter(df)
    plotter.create_all_plots(results_dir)

if __name__ == "__main__":
    main() 