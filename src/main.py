# Run this script to initiate a training sequence. Configuration of the training and desired output depend on
# config.yaml, so make sure that is correct before execution.

from src.utils import get_configs
from src.visualization import plots as plot
import matplotlib.pyplot as plt
import json
from src.training.Trainer import Trainer
import time


def main(cfg):
    times = {'start': time.time()}

    # Train the model while keeping track of times
    print('Training...')
    trainer = Trainer(cfg)
    trainer = trainer.load_data()
    times['training_start'] = time.time()
    trainer = trainer.train()
    times['training_end'] = time.time()
    data, metrics = trainer.evaluate()
    times['evaluation_end'] = time.time()

    if cfg['training']['plot_loss']:
        trainer.plot_loss()

    if cfg['training']['save_metrics']:
        save_metrics(metrics, model=cfg['training']['model'])
        print('')
        print(f'Metrics saved to: \"./results/metrics_{cfg["training"]["model"]}.json\"')
    print(metrics)

    # Make plots listed in config
    print('')
    print('Generating Plots...')
    make_plots(cfg, data)

    times['end'] = time.time()

    print('')
    print('Done!')
    print(f"""__ Runtimes __ 
Total Time: {round(times['end'] - times['start'], 6)} seconds
Training Time: {round(times['training_end'] - times['training_start'], 6)} seconds
Evaluation Time: {round(times['evaluation_end'] - times['training_end'], 6)} seconds""")


def save_metrics(metrics, model):
    metrics_save_path = f"./results/metrics_{model}.json"
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

    # TODO: Implement export-model boolean config with filetype config
