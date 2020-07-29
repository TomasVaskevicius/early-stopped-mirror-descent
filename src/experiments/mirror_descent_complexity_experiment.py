from .. import rprml_path  # noqa

from rprml import Experiment
from rprml.utils.plotting import get_new_fig_and_ax, save_fig
from rprml.utils.io import load_from_disk
import numpy as np
import torch
from os import path

from ..mirror_descent_simulation_factory import MirrorDescentSimulationFactory
from ..mirror_descent_optimizer import L2MirrorMap, HypentropyMirrorMap
from ..elastic_net_simulation import ElasticNetSimulationFactory

import matplotlib as mpl

mpl.rcParams['font.size'] = 20.0
mpl.rcParams['lines.linewidth'] = 2.5


class _ExperimentBase(Experiment):
    def __init__(self, experiment_name, SimulationFactoryClass,
                 **simulation_kwargs):
        """ :**simulation_kwargs: Will be forwarded to the constructor of
            SimulationFactoryClass. """
        simulation_factories = [
            SimulationFactoryClass(**simulation_kwargs)]
        super().__init__(experiment_name, simulation_factories)

    def construct_simulation_identifier(self, simulation):
        return simulation.seed

    def handle_simulation_output(self, history):
        # Convert complexity parameters to 1d numpy arrays.
        delta = np.array(history['delta_t']).flatten()  # Loss(wt) - Loss(w*).
        r = np.array(history['r_t']).flatten()  # In-sample prediction error.
        eps = delta + r
        excess_risk = np.array(history['excess_risk_t']).flatten()
        breg = np.array(history['bregman_t']).flatten()
        train_loss = np.array(history['train_loss']).flatten()

        t_hat = np.argmin(eps >= 0.0)  # Empirical stopping time.
        t_star = np.argmin(excess_risk)  # Best iterate among the path.
        t_breg = np.argmin(breg)  # Minimum D(w*, wt) along the path.

        xaxis = history['iter']  # Number of iterations.
        # If explicit regularization scheme was used, change x axis to the
        # regularization parameter.
        if 'lambdas' in history:
            xaxis = history['lambdas']

        return {
            'eps': eps,
            'excess_risk': excess_risk,
            'breg': breg,
            't_hat': t_hat,
            't_star': t_star,
            't_breg': t_breg,
            'train_loss': train_loss,
            'xaxis': xaxis
        }


class MirrorDescentComplexityExperiment(_ExperimentBase):
    """ Runs mirror descent and records its complexity parameters. """

    def __init__(self, experiment_name, **simulation_kwargs):
        """ :**simulation_kwargs: Will be forwarded to
            MirrorDescentSimulationFactory. """
        super().__init__(experiment_name, MirrorDescentSimulationFactory,
                         **simulation_kwargs)


class ElasticNetComplexityExperiment(_ExperimentBase):
    """ Runs elastic net and records its complexity parameters. """

    def __init__(self, experiment_name, **simulation_kwargs):
        """ :**simulation_kwargs: Will be forwarded to
            ElasticNetSimulationFactory. """
        super().__init__(experiment_name, ElasticNetSimulationFactory,
                         **simulation_kwargs)


# The below set up is shared across all the experiments.
d = 100
np.random.seed(0)
w_star = np.random.binomial(1, p=0.5, size=(d, 1))*2 - 1
w_star[10:, 0] = 0  # Make w_star (alpha' in the paper) 10-sparse.
simulation_kwargs = {
    'd': d,
    'n_train': 200,
    'data': 'gaussian',
    'data_factory_kwargs': {
            'w_star': w_star,
            'noise_std': np.sqrt(5.0),
            'covariance_matrix': None  # Identity covariance.
    },
}

# The following values of regularization parameters will be shared for
# ridge and lasso simulations.
lambdas = np.exp(np.linspace(start=np.log(10**(-5)), stop=np.log(10.0),
                             num=1000))

# L2 Experiment setup.
###############################################################################


def get_l2_experiment():
    experiment = MirrorDescentComplexityExperiment(
        experiment_name="l2",
        _learning_rate=0.01,
        mirror_map=L2MirrorMap(),
        **simulation_kwargs)
    return experiment

# Hypentropy experiment setup.
###############################################################################


def get_hypentropy_experiment():
    experiment = MirrorDescentComplexityExperiment(
        experiment_name="hypentropy",
        _learning_rate=0.1,
        mirror_map=HypentropyMirrorMap(gamma=1e-6),
        **simulation_kwargs)
    return experiment

# Ridge regression experiment setup.
###############################################################################


def get_ridge_experiment():
    experiment = ElasticNetComplexityExperiment(
        experiment_name="ridge",
        alpha=0.0,  # alpha=0 reduces elastic net to ridge.
        lambdas=lambdas,
        **simulation_kwargs)
    return experiment


# Lasso regression experiment setup.
###############################################################################
def get_lasso_experiment():
    experiment = ElasticNetComplexityExperiment(
        experiment_name="lasso",
        alpha=1.0,  # alpha=1 reduces elastic net to lasso.
        lambdas=lambdas,
        **simulation_kwargs)
    return experiment


# Plotting
###############################################################################

# Will be incremented after each call to one of the plotting functions.
fig_id = -1


def stack_parameters_over_multiple_runs(simulation_outputs):
    """ Stacks the complexity parameters of handle_simulation_output over
    multiple runs. """
    n_runs = len(simulation_outputs)
    n_iters = len(simulation_outputs[0]['eps'])

    xaxis = None
    eps = np.zeros((n_runs, n_iters))
    excess_risk = np.zeros((n_runs, n_iters))
    breg = np.zeros((n_runs, n_iters))
    train_loss = np.zeros((n_runs, n_iters))

    for i in range(n_runs):
        xaxis = simulation_outputs[i]['xaxis']
        eps[i, :] = simulation_outputs[i]['eps']
        excess_risk[i, :] = simulation_outputs[i]['excess_risk']
        breg[i, :] = simulation_outputs[i]['breg']
        train_loss[i, :] = simulation_outputs[i]['train_loss']

    return xaxis, eps, excess_risk, breg, train_loss


def plot_dynamics_of_complexity_parameters(simulation_outputs):
    """ :simulation_outputs: A list of simulation outputs generated by the
        handle_simulation_output function of class
        MirrorDescentComplexityExperiment.
    """
    global fig_id
    fig_id += 1
    fig, ax = get_new_fig_and_ax(fig_id)

    xaxis, eps, excess_risk, breg, _ = stack_parameters_over_multiple_runs(
        simulation_outputs)

    lower_p = 10
    upper_p = 90

    eps_means = np.mean(eps, axis=0)
    eps_lower = np.percentile(eps, lower_p, axis=0)
    eps_upper = np.percentile(eps, upper_p, axis=0)

    excess_risk_means = np.mean(excess_risk, axis=0)
    excess_risk_lower = np.percentile(excess_risk, lower_p, axis=0)
    excess_risk_upper = np.percentile(excess_risk, upper_p, axis=0)

    breg_means = np.mean(breg, axis=0)
    breg_lower = np.percentile(breg, lower_p, axis=0)
    breg_upper = np.percentile(breg, upper_p, axis=0)

    alpha = 0.25
    linestyles = ['-', '-.', '--']
    idx = 0
    for (means, lower, upper) in [(excess_risk_means, excess_risk_lower,
                                   excess_risk_upper),
                                  (eps_means, eps_lower, eps_upper),
                                  (breg_means, breg_lower, breg_upper)]:
        ax.plot(xaxis, means, linestyle=linestyles[idx])
        ax.fill_between(xaxis, lower, upper, alpha=alpha)
        idx += 1

    t_star = np.argmin(excess_risk_means)
    ax.scatter(t_star, excess_risk_means[t_star], color='C0')
    t_hat = np.argmin(eps_means >= 0)
    ax.scatter(t_hat, eps_means[t_hat], color='C1')
    t_breg = np.argmin(breg_means)
    ax.scatter(t_breg, breg_means[t_breg], color='C2')

    ax.hlines(y=0.0, xmin=0, xmax=len(xaxis), linestyle='dotted', linewidth=2)
    ax.vlines(x=t_hat, ymin=-100, ymax=100, linestyle='dotted', linewidth=2)

    ax.set_xlabel(r'Number of Iterations $t$')
    ax.legend([r'Excess Risk',
               r'$\varepsilon_{t}$',
               r'$D_{\psi}(\alpha^\prime, \alpha_{t})$'])

    return fig, ax


def plot_excess_risks_on_axis(ax, simulation_outputs, **plot_kwargs):
    xaxis, _, excess_risk, _, _ = stack_parameters_over_multiple_runs(
        simulation_outputs)

    lower_p = 10
    upper_p = 90

    excess_risk_means = np.mean(excess_risk, axis=0)
    excess_risk_lower = np.percentile(excess_risk, lower_p, axis=0)
    excess_risk_upper = np.percentile(excess_risk, upper_p, axis=0)

    if len(xaxis) == len(lambdas):
        print(len(xaxis))
        # This is an elastic net simulation. We will plot the regularization
        # parameter lambda on the log axis.
        xaxis = np.log(xaxis)
        # Next, we trim the initial part of the regularization range, which
        # returns the same model (i.e., for very small lambda, the solution
        # is approximately equal to the solution of unconstrained least
        # squares).
        lambda_start_idx = 250
        # Note that lambdas are ordered in decreasing order, hence to remove
        # the smallest lambdas we need to clip the end of all the arrays.
        xaxis = xaxis[:-lambda_start_idx]
        excess_risk_means = excess_risk_means[:-lambda_start_idx]
        excess_risk_upper = excess_risk_upper[:-lambda_start_idx]
        excess_risk_lower = excess_risk_lower[:-lambda_start_idx]

    ax.plot(xaxis, excess_risk_means, **plot_kwargs)
    ax.fill_between(xaxis, excess_risk_lower, excess_risk_upper, alpha=0.25)


# Main
###############################################################################

experiments = {
    'l2': get_l2_experiment(),
    'hypentropy': get_hypentropy_experiment(),
    'ridge': get_ridge_experiment(),
    'lasso': get_lasso_experiment()


}


def run_experiments():
    """ Runs the experiments defined above if an output directory with a
    given experiment name does not exist. """
    n_runs = 100
    for experiment in experiments.values():
        if not path.exists('./outputs/'+experiment.name):
            print("Running experiment: ", experiment.name + ".")
            experiment.full_run(
                n_runs_per_device=n_runs,
                n_processes_per_device=8,
                devices_list=[torch.device('cpu')],
                epochs_per_simulation=300)
        else:
            print("Experiment " + experiment.name + " was already performed.")


def load_simulation_outputs(experiment_name):
    fname = './outputs/'+experiment_name+'/experiment_0.pickle'
    saved_outputs = load_from_disk(fname)
    simulation_outputs = []
    for (output, used_seed, _) in saved_outputs[1]:
        simulation_outputs.append(output)
    return simulation_outputs


if __name__ == '__main__':
    run_experiments()

    # Plot complexity parameters of l2 mirror descent.
    simulation_outputs = load_simulation_outputs('l2')
    fig, ax = plot_dynamics_of_complexity_parameters(simulation_outputs)
    ax.set_ylim(-1.0, 11.0)
    ax.title.set_text(r'Mirror Descent With $\ell_{2}$ Mirror Map')
    fig_path = './figures/l2_complexity_parameters.pdf'
    save_fig(fig, fig_path)

    # Plot complexity parameters of hypentropy mirror descent.
    simulation_outputs = load_simulation_outputs('hypentropy')
    fig, ax = plot_dynamics_of_complexity_parameters(simulation_outputs)
    ax.set_ylim(-1.0, 11.0)
    ax.title.set_text(r'Mirror Descent With Hyperbolic Entropy Mirror Map')
    fig_path = './figures/hypentropy_complexity_parameters.pdf'
    save_fig(fig, fig_path)

    # Plot excess risk vs iterations for l2 and hypentropy mirror descent.
    fig_id += 1
    fig, ax = get_new_fig_and_ax(fig_id)
    simulation_outputs = load_simulation_outputs('l2')
    plot_excess_risks_on_axis(ax, simulation_outputs)
    simulation_outputs = load_simulation_outputs('hypentropy')
    plot_excess_risks_on_axis(ax, simulation_outputs, linestyle='-.')
    ax.set_ylim(-1.0, 11.0)
    ax.title.set_text(r'Excess Risks along the Optimization Paths '
                      r'$(\alpha_{t})_{t \geq 0}$'+'\n'
                      r'traced by Mirror Descent Algorithms')
    ax.set_xlabel(r'Number of Iterations $t$')
    ax.set_ylabel('Excess Risk')
    ax.legend([r'$\ell_{2}$ Mirror Map',
               'Hyperbolic Entropy Mirror Map'])
    fig_path = './figures/l2_vs_hypentropy_excess_risk.pdf'
    save_fig(fig, fig_path)

    # Plot excess risk vs regularization parameter for ridge and lasso.
    fig_id += 1
    fig, ax = get_new_fig_and_ax(fig_id)
    simulation_outputs = load_simulation_outputs('ridge')
    plot_excess_risks_on_axis(ax, simulation_outputs)
    simulation_outputs = load_simulation_outputs('lasso')
    plot_excess_risks_on_axis(ax, simulation_outputs, linestyle='-.')
    ax.set_ylim(-1.0, 11.0)
    ax.title.set_text(r'Excess Risks along the Regularization Paths '
                      r'$(\alpha_{\lambda})_{\lambda \geq 0}$,'+'\n'
                      r'where $\alpha_{\lambda} \in'
                      r'\operatorname{argmin}_{\alpha} '
                      r'R_{n}(\alpha) + \lambda \psi(\alpha)$')
    ax.set_xlabel(r'$\log \lambda$')
    ax.set_ylabel('Excess Risk')
    ax.legend([r'$\psi(\alpha) = \|\alpha\|_{2}^{2}$ (Ridge Regression)',
               r'$\psi(\alpha) = \|\alpha\|_{1}$ (Lasso)'])
    fig_path = './figures/ridge_vs_lasso_excess_risk.pdf'
    save_fig(fig, fig_path)
