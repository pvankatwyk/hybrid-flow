import matplotlib.pyplot as plt
from src.utils import check_input


def residuals_histogram(predictions, true_values, bins=30):
    plt.hist((predictions - true_values), bins=bins)
    plt.title('Neural Network Residuals')
    plt.xlabel('Absolute Error (Predicted - True)')
    plt.ylabel('Frequency')


def cross_validation_plot(predictions, true_values, uncertainties=None):
    plt.scatter(true_values, predictions, c=uncertainties, s=3)
    plt.plot(true_values, true_values, 'r-')
    plt.xlabel('Simulated SLE (cm)')
    plt.ylabel('Emulated SLE (cm)')
    plt.title('HybridFlow Emulator')


def hist_cross_validation_plot(predictions, true_values, bins=(30, 30)):
    plt.hist2d(true_values, predictions, bins=bins)
    plt.plot(true_values, true_values, 'r-', )
    plt.xlabel('Simulated (cm SLE) -- y_test')
    plt.ylabel('Emulated (cm SLE) -- predicted')
    plt.title('HybridFlow Emulator')


def prediction_uncertainty(inputs, predictions, true_values, uncertainties, layout='side-by-side'):
    check_input(layout, ['side-by-side', 'overlay', 'side-by-side hist'])

    normalized_uncertainties = (uncertainties - min(uncertainties)) / (max(uncertainties) - min(uncertainties))

    if layout == 'overlay':
        try:
            plt.scatter(inputs[:, 0], true_values, c='red', s=3, label='Simulations')
            plt.scatter(inputs[:, 0], predictions, c=normalized_uncertainties, marker='+',
                        label='Prediction & Uncertainty')
        except IndexError:
            plt.scatter(inputs, true_values, c='red', s=3, label='Simulations')
            plt.scatter(inputs, predictions, c=normalized_uncertainties, marker='+',
                        label='Prediction & Uncertainty')

        plt.title('Simulation vs HybridFlow Emulation')
        plt.xlabel('Global mean air temperature')
        plt.ylabel('Sea level contribution (cm)')
        plt.colorbar()
        plt.legend()

    if layout == 'side-by-side':
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5))
        fig.suptitle('GCM Simulation vs HybridFlow Emulation')
        try:  # (2D+ inputs)
            ax1.scatter(inputs[:, 0], true_values, c='red', s=3, label='Simulations')
            c = ax2.scatter(inputs[:, 0], predictions, c=normalized_uncertainties, marker='+',
                            label='Prediction & Uncertainty')
        except IndexError:  # (1D input)
            ax1.scatter(inputs, true_values, c='red', s=3, label='Simulations')
            c = ax2.scatter(inputs, predictions, c=normalized_uncertainties, marker='+',
                            label='Prediction & Uncertainty')

        ax1.set_title("GCM Simulations")
        ax2.set_title("HybridFlow")
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.08, 0.0275, .78])
        fig.colorbar(c, cax=cbar_ax)

    if layout == 'side-by-side hist':
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5))
        fig.suptitle('GCM Simulation vs HybridFlow Emulation')
        try:
            ax1.hist2d(inputs[:, 0], true_values, bins=(30, 30))
            c = ax2.scatter(inputs[:, 0], predictions, c=normalized_uncertainties, marker='+',
                            label='Prediction & Uncertainty')
        except IndexError:
            ax1.hist2d(inputs, true_values, bins=(30, 30))
            c = ax2.scatter(inputs, predictions, c=normalized_uncertainties, marker='+',
                            label='Prediction & Uncertainty')

        ax1.set_title("GCM Simulations")
        ax2.set_title("HybridFlow")
        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.08, 0.0275, .78])
        fig.colorbar(c, cax=cbar_ax)
