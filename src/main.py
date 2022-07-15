# Run this script to initiate a training sequence. Configuration of the training and desired output depend on
# config.yaml, so make sure that is correct before execution.

from src.utils import get_configs
from src.visualization import plots as plot
import matplotlib.pyplot as plt
import json


def main(cfg):
    model_name = cfg['training']['model']

    # temporary -- error out if doesn't exist (yet)
    if model_name != 'GrIS_HybridFlow':
        raise NotImplementedError(f'{model_name} either does not exist or is not supported yet.')
    if model_name == 'GrIS_HybridFlow' and cfg['model']['conditional']:
        raise NotImplementedError('Conditional GrIS_HybridFlow not supported yet.')

    if model_name == 'GrIS_HybridFlow':
        from src.training.train_GrIS_HF import train_GrIS_HF
        print('Training...')
        data, metrics = train_GrIS_HF(cfg)

        if cfg['training']['save_metrics']:
            save_metrics(metrics)
            print('')
            print(f'Metrics saved to: \"./results/metrics.json\"')
        print(metrics)

    print('')
    print('Generating Plots...')
    make_plots(cfg, data)

    print('')
    print('Done!')


def save_metrics(metrics):
    metrics_save_path = "./results/metrics.json"
    with open(metrics_save_path, "w") as outfile:
        json.dump(metrics, outfile)


def make_plots(cfg, data):
    plot_list = cfg['visualizations']['generate_plots']
    if plot_list is not None:
        inputs = data['X_test']
        predictions = data['predictions']
        true_values = data['y_test']
        uncertainties = data['uncertainties']
        save = cfg['visualizations']['save_plots']

        if 'prediction_uncertainty' in plot_list:
            for layout in ['side-by-side', 'side-by-side hist', 'overlay']:
                plot.prediction_uncertainty(
                    inputs=inputs,
                    predictions=predictions,
                    true_values=true_values,
                    uncertainties=uncertainties,
                    layout=layout
                )
                if save:
                    plt.savefig("./results/plots/prediction_uncertainty_{layout}.png")
                plt.show()

        if 'residuals_histogram' in plot_list:
            plot.residuals_histogram(
                predictions=predictions,
                true_values=true_values,
                bins=30
            )
            if save:
                plt.savefig("./results/plots/residuals.png")
            plt.show()

        if 'cross_validation_plots' in plot_list:
            plot.cross_validation_plot(
                predictions=predictions,
                true_values=true_values,
                uncertainties=uncertainties,
            )
            if save:
                plt.savefig("./results/plots/cross_val_plot.png")
            plt.show()

            plot.hist_cross_validation_plot(
                predictions=predictions,
                true_values=true_values,
                bins=(30, 30))

            if save:
                plt.savefig("./results/plots/cross_val_hist.png")
            plt.show()

        if save:
            print('Plots saved to: ./results/plots/')


if __name__ == '__main__':
    cfg = get_configs()
    main(cfg)
