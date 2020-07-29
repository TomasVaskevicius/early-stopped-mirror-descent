from .. import rprml_path  # noqa

from rprml import Experiment
from rprml.utils.plotting import get_new_fig_and_ax, save_fig
from rprml.utils.io import load_from_disk, save_to_disk
from rprml.core.executor import _iteration_level_event
import numpy as np
import torch
from os import path

from ..mirror_descent_simulation_factory import MirrorDescentSimulationFactory
from ..mirror_descent_optimizer import PreconditionedL2MirrorMap

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 20.0
mpl.rcParams['lines.linewidth'] = 2.0

# Set up the simulation setting.
d = 2
# Define the optimal/data generating parameter w*.
w_star = torch.tensor([1.5, 0.5], dtype=torch.float32).view(-1, 1)
simulation_kwargs = {
    'd': d,
    'n_train': 100,
    'data': 'gaussian',
    'data_factory_kwargs': {
            'w_star': w_star,
            'noise_std': np.sqrt(0.5),
            'covariance_matrix': torch.tensor(
                [[1.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    },
    '_learning_rate': 0.001
}

# L2 mirror map pre-conditioned with the true covariance matrix will
# approximatelly take the shortest/straightest path to the empirical risk
# minimizer. Since the path of optimal parameters w_star_R (with varying
# radius R) is chosen to be curved, this will create an example where
# The range of early-stopped mirror descent iterates violate Bernstein's
# assumption.
simulation_kwargs['mirror_map'] = PreconditionedL2MirrorMap(
    simulation_kwargs['data_factory_kwargs']['covariance_matrix'])


# w_star_1 denotes an optimal parameter in a l2 ball of radius 1.
w_star_1_fname = './outputs/EarlyStoppedEstimatorsRangeExperiment/w_star_1'


def get_w_star_1():
    # Loads saved w_star_1 (optimal parameter in an l2 ball of radius 1)
    # from disk. The parameter w_star_1 is computed and saved to disk
    # by the function compute_w_star_1.
    w_star_1 = load_from_disk(w_star_1_fname + '.pickle')
    return w_star_1


def compute_and_plot_w_star_path():
    Sigma = simulation_kwargs['data_factory_kwargs']['covariance_matrix']
    if Sigma is None:
        Sigma = torch.diag(torch.ones(d))

    EXY = Sigma @ w_star

    # Now compute the regularization path.
    Id = torch.diag(torch.ones(d))
    mus = np.exp(np.linspace(
        start=np.log(10**(-7)), stop=np.log(1e5), num=10000))
    mus = np.insert(mus, 0, 0.0)
    w_star_path = np.zeros((len(mus), d))
    for idx, mu in enumerate(mus):
        Sigma_mu = Sigma + mu * Id
        Sigma_mu_inverse = torch.inverse(Sigma_mu)
        w_star_mu = (Sigma_mu_inverse @ EXY).cpu().detach().numpy()
        w_star_path[idx, :] = w_star_mu.flatten()

    w_star_norms = np.sum(w_star_path**2, axis=1)

    # Find optimal parameter of norm 1.
    norm_1_idx = np.argmin(w_star_norms > 1.0)
    w_star_1 = torch.tensor(w_star_path[norm_1_idx, :],
                            dtype=torch.float32).view(-1, 1)
    # Save it to disk so that spawned jobs do not need to recompute it.
    save_to_disk(w_star_1, w_star_1_fname)

    fig, ax = get_new_fig_and_ax(0, width=9.0, height=6.0)
    ax.plot(w_star_path[:, 0], w_star_path[:, 1])
    ax.scatter(w_star_path[norm_1_idx, 0], w_star_path[norm_1_idx, 1],
               s=50, color='C1', zorder=10)
    return fig, ax


# Below we implement a handler that signals once the training loss has reached
# the level of the training loss of w_star_1 and terminates the simulation.


def terminate_training_at_w_star_1_train_loss_level(simulation):
    """ Terminates the training as soon as the train loss gets lower than the
      train loss of the parameter w_star_1. """

    w_star_1 = get_w_star_1()
    # Convert the reference parameter to pytorch tensor if needed.
    if not isinstance(w_star_1, torch.Tensor):
        w_star_1 = torch.tensor(w_star_1, dtype=torch.float32)
    w_star_1 = w_star_1.to(device=simulation.device)

    # Compute the train loss of the reference_parameter.
    with torch.no_grad():
        w_t = simulation.model.get_w()
        simulation.model.set_w(w_star_1)
        evaluator = simulation.evaluator
        evaluator.run(simulation.train_dl)
        w_star_1_loss = evaluator.state.metrics['loss']
        simulation.model.set_w(w_t)

    # For debugging purposes, store the w_star_1_loss in the executor's
    # history.
    simulation.executor.history['w_star_1_loss'] = w_star_1_loss

    @simulation.trainer.on(_iteration_level_event)
    def terminate(engine):
        executor = simulation.executor
        train_loss = executor.history['train_loss'][-1]
        if train_loss <= w_star_1_loss:
            simulation.trainer.terminate()


simulation_kwargs['custom_handlers'] = [
    terminate_training_at_w_star_1_train_loss_level]


class EarlyStoppedEstimatorsRangeExperiment(Experiment):
    def __init__(self):
        """ :**simulation_kwargs: Will be forwarded to the constructor of
            SimulationFactoryClass. """
        simulation_factories = [
            MirrorDescentSimulationFactory(**simulation_kwargs)]
        experiment_name = 'EarlyStoppedEstimatorsRangeExperiment'
        super().__init__(experiment_name, simulation_factories)

    def construct_simulation_identifier(self, simulation):
        return simulation.seed

    def handle_simulation_output(self, history):
        w_t = np.array(history['w_t']).squeeze()
        # The simulation terminates at the first point such that
        # R_{n}(w_t) <= R_n(w_star_1), which we return below.
        return w_t[-1, :]


def compute_excess_risk(w_hat, w_ref):
    """ Computes the excess risk of w_hat with respect to the reference point
    w_ref. """
    Sigma = simulation_kwargs['data_factory_kwargs']['covariance_matrix']
    # Note that R(w) = E (<X, w> - Y)^{2} = w^t Sigma w - 2 E[<X,w>Y] + E[Y^2]
    # E[Y^{2}] will cancel hence does not need to be compute.
    # Further, E[<X,w>Y]=E[<X,w>(<X,w*> + xi)] = E[<X,w><X,w*>] = w^t Sigma w*
    # (since xi is independent and 0 mean).
    excess_risk = w_hat.t() @ Sigma @ w_hat
    excess_risk -= 2 * w_hat.t() @ Sigma @ w_star
    # Now subtract the risk of w_ref (without the E[Y^2] term).
    excess_risk -= w_ref.t() @ Sigma @ w_ref
    excess_risk += 2 * w_ref.t() @ Sigma @ w_star
    return excess_risk


def load_simulation_outputs(experiment_name):
    fname = './outputs/'+experiment_name+'/experiment_0.pickle'
    saved_outputs = load_from_disk(fname)
    simulation_outputs = []
    for (output, used_seed, _) in saved_outputs[1]:
        simulation_outputs.append(output)
    return simulation_outputs


if __name__ == '__main__':
    fig, ax = compute_and_plot_w_star_path()
    w_star_1 = get_w_star_1()

    n_runs = 100
    experiment = EarlyStoppedEstimatorsRangeExperiment()
    if not path.exists('./outputs/'+experiment.name+'/experiment_0.pickle'):
        print("Running experiment: ", experiment.name + ".")
        experiment.full_run(
            n_runs_per_device=n_runs,
            n_processes_per_device=8,
            devices_list=[torch.device('cpu')],
            epochs_per_simulation=int(1e10))
    else:
        print("Experiment " + experiment.name + " was already performed.")

    simulation_outputs = load_simulation_outputs(experiment.name)
    circle = plt.Circle((0, 0), torch.sqrt(torch.sum(w_star_1**2)),
                        color='gray', alpha=0.25)
    ax.add_artist(circle)
    for output in simulation_outputs:
        w_hat = torch.tensor(output).view(-1, 1)
        # Compute excess risk with respect to the point w_star_1.
        excess_risk = compute_excess_risk(w_hat, w_star_1)
        if excess_risk < 0:
            # Bernstein assumption was violated.
            marker = 'x'
            color = 'C3'
        else:
            # Bernstein assumption is not necessarily violated.
            marker = '^'
            color = 'C2'
        ax.scatter(output[0], output[1], color=color, marker=marker, s=50.0,
                   zorder=9)

    fig_path = './figures/range_of_early_stopped_estimator.pdf'
    ax.legend([
        r'$(\alpha_{\mathcal{F}_{R}})_{R \geq 0}$',
        r'$\alpha_{\mathcal{F}_{1}}$',
        r'$\alpha_{t^{\star}} : R(\alpha_{t^{\star}}) - '
        r'R(\alpha_{\mathcal{F}_{1}}) \geq 0$',
        r'$\alpha_{t^{\star}} : R(\alpha_{t^{\star}}) - '
        r'R(\alpha_{\mathcal{F}_{1}}) < 0$',
    ],
        bbox_to_anchor=(1, 0.5),
        loc='center left')
    ax.title.set_text('Violation of the Bernstein Condition')
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, 1.55)
    save_fig(fig, fig_path)
