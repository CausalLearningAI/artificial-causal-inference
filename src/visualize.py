import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

def plot_outcome_distribution(dataset, save=False, total=True, results_dir="./results"):
    T = dataset['T']
    Y = dataset['Y']
    fig, axs = plt.subplots(1, 2+total, figsize=(12+6*total, 5))
    colors = ['Y2F', 'B2F']
    for i, color in enumerate(colors):
        p_Y_0_T_0 = (Y[T==0, i]==0).sum() / Y.shape[0]
        p_Y_1_T_0 = (Y[T==0, i]==1).sum() / Y.shape[0]
        p_Y_0_T_1 = (Y[T==1, i]==0).sum() / Y.shape[0]
        p_Y_1_T_1 = (Y[T==1, i]==1).sum() / Y.shape[0]
        p_Y_0_T_2 = (Y[T==2, i]==0).sum() / Y.shape[0]
        p_Y_1_T_2 = (Y[T==2, i]==1).sum() / Y.shape[0]
        axs[i].bar(['0', '1'], [p_Y_0_T_0, p_Y_1_T_0], alpha=0.75, label='T=0 (control)')
        axs[i].bar(['0', '1'], [p_Y_0_T_1, p_Y_1_T_1], bottom=[p_Y_0_T_0, p_Y_1_T_0], alpha=0.75, label='T=1 (beads)')
        axs[i].bar(['0', '1'], [p_Y_0_T_2, p_Y_1_T_2], bottom=[p_Y_0_T_0+p_Y_0_T_1, p_Y_1_T_0+p_Y_1_T_1], alpha=0.75, label='T=2 (infused beads)')
        axs[i].set_xlabel('Y')
        axs[i].set_ylabel('p(Y)')
        axs[i].set_ylim(0, 1)
        axs[i].legend()
        axs[i].set_title(f'Grooming ({color})')
    if total:
        Y_tot = Y.sum(axis=1)
        p_Y_0_T_0 = (Y_tot[T==0]==0).sum() / Y.shape[0]
        p_Y_1_T_0 = (Y_tot[T==0]==1).sum() / Y.shape[0]
        p_Y_2_T_0 = (Y_tot[T==0]==2).sum() / Y.shape[0]
        p_Y_0_T_1 = (Y_tot[T==1]==0).sum() / Y.shape[0]
        p_Y_1_T_1 = (Y_tot[T==1]==1).sum() / Y.shape[0]
        p_Y_2_T_1 = (Y_tot[T==1]==2).sum() / Y.shape[0]
        p_Y_0_T_2 = (Y_tot[T==2]==0).sum() / Y.shape[0]
        p_Y_1_T_2 = (Y_tot[T==2]==1).sum() / Y.shape[0]
        p_Y_2_T_2 = (Y_tot[T==2]==2).sum() / Y.shape[0]
        axs[2].bar(['0', '1', '2'], [p_Y_0_T_0, p_Y_1_T_0, p_Y_2_T_0], alpha=0.75, label='T=0 (control)')
        axs[2].bar(['0', '1', '2'], [p_Y_0_T_1, p_Y_1_T_1, p_Y_2_T_1], bottom=[p_Y_0_T_0, p_Y_1_T_0, p_Y_2_T_0], alpha=0.75, label='T=1 (beads)')
        axs[2].bar(['0', '1', '2'], [p_Y_0_T_2, p_Y_1_T_2, p_Y_2_T_2], bottom=[p_Y_0_T_0+p_Y_0_T_1, p_Y_1_T_0+p_Y_1_T_1, p_Y_2_T_0+p_Y_2_T_1], alpha=0.75, label='T=2 (infused beads)')
        axs[2].set_xlabel('Y')
        axs[2].set_ylabel('p(Y)')
        axs[2].set_ylim(0, 1)
        axs[2].legend()
        axs[2].set_title(f'Grooming (total)')
    if save:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        title = "outcome_distribution.png"
        path_fig = os.path.join(results_dir, title)
        fig.savefig(path_fig)
    else:
        plt.show();

def plot_error_all(df, model_name, results_dir):
    poss = np.arange(1, 10)  
    batchs = [0, 1, 2, 3, 4]
    metrics = ["Y", "Y_hat", "bias", "accuracy", "recall", "precision"]
    #metrics = ["duration", "Y", "Y_hat", "bias", "accuracy", "recall", "precision"]
    fps = 2

    evals = df.evaluate()
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(
        f"Accuracy: {evals['acc']:.2f}, Balanced Accuracy: {evals['bacc']:.2f}, PP-ATE: {evals['PPATE']:.1f} ± {evals['PPATE_std']:.1f} (ATE: {evals['ATE']:.1f} ± {evals['ATE_std']:.1f})",
        fontsize=16,
        x=0.5,
        y=0.95  # higher number = closer to top
    )
    plt.subplots_adjust(top=0.92)  # higher number = plots move up
    gs = gridspec.GridSpec(len(metrics), len(batchs) + 1, 
                       width_ratios=[1]*len(batchs) + [0.05],
                       wspace=0.02, hspace=0.05)  # Reduce hspace to tighten vertical spacing   
    axes = np.empty((len(metrics), len(batchs)), dtype=object)

    for row_idx, metric in enumerate(metrics):
        all_values = []

        if metric == "bias":
            df.supervised["metric"] = df.supervised["Y_hat"] - df.supervised["Y"]
        elif metric == "accuracy":
            df.supervised["metric"] = (df.supervised["Y"] == df.supervised["Y_hat"].round()).float()
        elif metric == "Y":
            df.supervised["metric"] = df.supervised["Y"].float()
        elif metric == "Y_hat":
            df.supervised["metric"] = df.supervised["Y_hat"]
        elif metric == "duration":
            df.supervised["metric"] = df.supervised["Y"] + (1 - df.supervised["Y"])
        elif metric in ["recall", "precision"]:
            pass
        else:
            raise ValueError("Unknown metric")

        # Set colormap and normalization
        if metric == "bias":
            for batch in batchs:
                filter_batch = df.supervised["source_data"]["experiment"] == batch
                values = [df.supervised["metric"][(df.supervised["source_data"]["position"] == pos) & filter_batch].sum() for pos in poss]
                all_values.extend(values)
            vmax = max(abs(min(all_values)), abs(max(all_values)))
            norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cmap = 'seismic'
        elif metric in ["accuracy", "recall", "precision"]:
            norm = colors.Normalize(vmin=0, vmax=1)
            cmap = 'RdYlGn'
        elif metric in ["Y_hat", "Y", "duration"]:
            for batch in batchs:
                filter_batch = df.supervised["source_data"]["experiment"] == batch
                values = [df.supervised["metric"][(df.supervised["source_data"]["position"] == pos) & filter_batch].sum() for pos in poss]
                all_values.extend(values)
            max_value = max(all_values) / fps
            norm = colors.Normalize(vmin=0, vmax=max_value)
            cmap = 'Blues'

        last_cax = None

        for col_idx, batch in enumerate(batchs):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            axes[row_idx, col_idx] = ax
            filter_batch = df.supervised["source_data"]["experiment"] == batch
            data = []
            is_training = []

            for pos in poss:
                mask = (df.supervised["source_data"]["position"] == pos) & filter_batch
                if metric == "recall":
                    y_true = df.supervised["Y"][mask]
                    y_pred = df.supervised["Y_hat"][mask].round()
                    TP = ((y_pred == 1) & (y_true == 1)).sum()
                    FN = ((y_pred == 0) & (y_true == 1)).sum()
                    value = TP / (TP + FN) if (TP + FN) > 0 else np.nan if mask.sum() > 0 else np.nan
                    data.append(value)
                elif metric == "precision":
                    y_true = df.supervised["Y"][mask]
                    y_pred = df.supervised["Y_hat"][mask].round()
                    TP = ((y_pred == 1) & (y_true == 1)).sum()
                    FP = ((y_pred == 1) & (y_true == 0)).sum()
                    value = TP / (TP + FP) if (TP + FP) > 0 else np.nan if mask.sum() > 0 else np.nan
                    data.append(value)
                elif metric in ["bias","Y", "Y_hat","duration"]:
                    values = df.supervised["metric"][mask]
                    data.append(values.sum().float()/fps if len(values) > 0 else np.nan)
                else:
                    values = df.supervised["metric"][mask]
                    data.append(values.mean() if len(values) > 0 else np.nan)

                train_flag = df.supervised["split"][mask].any()
                is_training.append(train_flag)

            data = np.array(data).reshape(3, 3)
            markers = np.array(is_training).reshape(3, 3)

            cax = ax.matshow(data, cmap=cmap, norm=norm)
            last_cax = cax

            for (i, j), val in np.ndenumerate(data):
                if "hq" in df.data_dir and batch == 1 and i == 0 and j == 2:
                    ax.text(j, i, "--", va='center', ha='center', color='black')
                elif "lq" in df.data_dir and batch == 2 and i == 2 and j == 2:
                    ax.text(j, i, "--", va='center', ha='center', color='black')
                elif metric in ["Y", "duration"]:
                    ax.text(j, i, f"{int(val)}", va='center', ha='center',
                            color='black' if (val < norm.vmax * 0.5) else 'white')
                elif metric == "bias":
                    ax.text(j, i, f"{val:.1f}", va='center', ha='center',
                            color='black' if (val < norm.vmax * 0.5) else 'white')
                else:
                    ax.text(j, i, f"{val:.2f}", va='center', ha='center',
                            color='black' if (metric in ["accuracy", "recall", "precision", "Y"] or abs(val) < norm.vmax * 0.3) else 'white')

                if markers[i, j]:
                    ax.plot(j, i, marker='o', markersize=35, markerfacecolor='none',
                            markeredgecolor='black', markeredgewidth=2)

            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(f"Batch {batch}", fontsize=12)

        # Add row label as subtitle
        axes[row_idx, 0].set_ylabel(metric.capitalize(), rotation=0, size=14, labelpad=60 + len(metric) * 2, va='center')

        # Add colorbar to the last column
        cbar_ax = fig.add_subplot(gs[row_idx, -1])
        fig.colorbar(last_cax, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig(f"{results_dir}/{model_name}.png", dpi=300, bbox_inches='tight')