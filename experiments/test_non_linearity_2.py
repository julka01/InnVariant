import argparse
import os
import pickle
import time
import multiprocessing as mp
import numpy as np
from multiprocessing import Pool
from experiments.utils.imported_utils import *

from functools import partial

from disentangling.metrics import (
    z_min_var,
    mig,
    mig_sup,
    dcimig,
    sap,
    modularity,
    dci,
    edi,
    mig_ksg,
mig_sup_ksg,
dcimig_ksg, dci_xgb
)


# set variables for the experiment
NB_JOBS = get_nb_jobs('max')
NB_RANDOM_REPRESENTATIONS = 50
NB_RUNS = 1
NB_EXAMPLES = 20000
NB_FACTORS = 6
DISTRIBUTION = [np.random.uniform, {'low': 0., 'high': 1.}]
NON_LINEARITY_STRENGTH = [i / 10 for i in range(11) if i % 2 == 0]
METRICS =  [ edi, sap, dcimig, mig, mig_sup, mig_sup_ksg, dci_xgb, z_min_var]
METRIC_DICT = {dcimig: 'DCIMIG', edi: 'EDI', dci: 'DCI', mig: 'MIG', mig_ksg: 'MIG-ksg', mig_sup: 'MIG-sup', mig_sup_ksg: 'MIG-sup-ksg', dcimig_ksg: 'dcimig-ksg', z_min_var: 'FactorVAE', sap: 'SAP', modularity: 'Modularity'}

# config parameters for plots
loosely_dashed = (0, (5, 10))
densely_dashdotdotted = (0, (3, 1, 1, 1, 1, 1))
densely_dotted = (0, (1, 1))
PLOTS = {

    'FAMILIES': {
        'Metrics': [  'Z-diff', 'Z-min Variance', 'Modularity' , 'DCI Mod', 'DCI Comp',  'DCI Expl', 'SAP'],

        'Information-based': ['EDI Expl', 'EDI Comp', 'EDI Mod', 'MIG', 'MIG-ksg', 'DCIMIG-ksg', 'DCIMIG' , 'MIG-sup']
    },
    'COLORS': ['blue', 'green', 'red', 'darkturquoise', 'orange', 'black', 'violet', 'indigo', 'gray', 'maroon'],
    'LINE STYLES': ['--', '-.', loosely_dashed, densely_dashdotdotted, ':', densely_dotted],
    'LEGEND POSITIONS': [(0.5, -0.32), (0.5, -0.32)],
    'NB LEGEND COLUMNS': [4, 4]
}


def get_factors_codes_dataset(non_linearity_strength):
    ''' Create factors-codes dataset

    :param non_linearity_strength:     strength of non-linearity relation between factors and codes
    '''
    # create factors dataset
    dist, dist_kwargs = DISTRIBUTION
    factors = get_artificial_factors_dataset(nb_examples=NB_EXAMPLES, nb_factors=NB_FACTORS,
                                             distribution=dist, dist_kwargs=dist_kwargs)
    # compute codes from continuous factors
    alpha = non_linearity_strength * 10
    codes = 1 / (1 + np.exp(-alpha * factors))
    return factors, codes

def run_non_linearity_experiment(sub_parser_args):
    ''' Run non-linearity experiment using several metrics and save score results

    :param sub_parser_args:     arguments of "run" sub-parser command
                                metrics (list):         metrics to use in the experiment
                                output_dir (string):    directory to save metric scores
    '''
    # extract sub-parser arguments
    metrics = sub_parser_args.metrics
    output_dir = sub_parser_args.output_dir

    # seeds to use for the experiment
    seeds = get_experiment_seeds(nb_representations=NB_RANDOM_REPRESENTATIONS, nb_runs=NB_RUNS)

    # iterate over metrics
    for metric in METRICS:
        # track time
        begin = time.time()
        metric_name = METRIC_DICT[metric]
        print(f'Running {metric_name} metric')

        # initialize arrays
        non_linearity_array = np.asarray(NON_LINEARITY_STRENGTH)
        scores_array = np.zeros((len(non_linearity_array), NB_RANDOM_REPRESENTATIONS, NB_RUNS)).squeeze()

        # depending on the metric, we can have several scores per representation
        if 'DCI' in metric_name or 'EDI' in metric_name and 'MIG' not in metric_name:
        #
        #     # DCI metric returns Modularity, Compactness and Explicitness scores
            metric_scores = {f'{metric_name} Mod': scores_array.copy(),
                              f'{metric_name} Comp': scores_array.copy(),
                              f'{metric_name} Expl': scores_array.copy()}
        else:
        #     # only one score is returned
            metric_scores = {f'{metric_name}': scores_array}
        #
        # # set metric function and its hyper-params
        # metric_func = METRICS[metric]['function']
        # metric_kwargs = METRICS[metric]['kwargs']
        # metric_func = partial(metric_func, **metric_kwargs)

        # run metric for each non-linearity level
        for line_idx, non_linearity_strength in enumerate(NON_LINEARITY_STRENGTH):
            # get scores using multi-processing
            factors_codes_dataset = partial(get_factors_codes_dataset, non_linearity_strength=non_linearity_strength)

            # print("debug")
            scores = launch_multi_process(iterable=seeds, func=get_score, n_jobs=NB_JOBS, timer_verbose=False,
                                           metric=metric, factors_codes_dataset=factors_codes_dataset)
            print(scores)
            # fill arrays for current non-linear level
            for idx, key in enumerate(metric_scores):
                 if len(metric_scores) == 1:
                     metric_scores[key][line_idx] = [score for score in scores]
                 else:
                     metric_scores[key][line_idx] = [score[idx] for score in scores]
            #
            # # display remaining time
            estimate_required_time(nb_items_in_list=len(seeds) * len(NON_LINEARITY_STRENGTH),
                                   current_index=(line_idx + 1) * len(seeds) - 1,
                                   time_elapsed=time.time() - begin)

        # save dictionaries
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)  # create output directory
            for key in metric_scores:
                with open(os.path.join(output_dir, key), 'wb') as output:
                    pickle.dump({f'{key}': (non_linearity_array, metric_scores[key])}, output)

        # # display time
        duration = (time.time() - begin) / 60
        print(f'\nTotal time to run experiment on {metric_name} metric -- {duration:.2f} min')


def plot_scores(sub_parser_args):
    ''' Plot metric scores

    :param sub_parser_args:     arguments of "plot" sub-parser command
                                output_dir (string):    directory to save plots
    '''
    # extract sub-parser arguments
    output_dir = sub_parser_args.output_dir

    # extract metric scores
    scores = {}
    metrics = [os.path.join(output_dir, x) for x in os.listdir(output_dir)
               if not x.endswith('.eps') and not x.endswith('.png')]
    for metric in metrics:
        with open(metric, 'rb') as input:
            metric_scores = pickle.load(input)
        scores.update(metric_scores)

    # compute means and standard deviations
    for metric, values in scores.items():
        non_linearity_array, scores_array = values[0], values[1]
        scores[metric] = (non_linearity_array,
                          np.mean(scores_array, axis=tuple(range(1, scores_array.ndim))),
                          np.std(scores_array, axis = tuple(range(1, scores_array.ndim)))
                          )

        print(scores[metric])
    # plot graphs in .png and .eps formats
    plots = {}
    for family, metrics in PLOTS['FAMILIES'].items():
        plots[family] = {}
        for metric in metrics:
            if metric in scores:
                plots[family][metric] = scores[metric]

    output_file = os.path.join(output_dir, f'{os.path.basename(output_dir)}')
    for format in ['png', 'eps']:
        plot_curves(plots, output_file, format=format,
                    x_label='α', y_label='aggregated score',
                    colors=PLOTS['COLORS'], line_styles=PLOTS['LINE STYLES'],
                    legend_positions=PLOTS['LEGEND POSITIONS'],
                    legend_columns=PLOTS['NB LEGEND COLUMNS'])





if __name__ == "__main__":
    # project ROOT
    FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.realpath(os.path.dirname(FILE_ROOT))

    # # default metrics and default output directory
    metrics = [metric for metric in METRICS]
    output_dir = os.path.join(PROJECT_ROOT, 'results', 'experiment_non_linearity_2')
    #
    # # create parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    #
    # parser for the "run" command -- run non-linearity experiment
    parser_run = subparsers.add_parser('run', help='compute scores')
    parser_run.set_defaults(func=run_non_linearity_experiment)
    parser_run.add_argument('--metrics', nargs='+', default=metrics, required=False,
                            help='metrics to use to compute scores: "metric_1" ... "metric_N"')
    parser_run.add_argument('--output_dir', type=str, default=output_dir, required=False,
                            help='output directory to store scores results')

    # parser fot the "plot" command -- plot metric scores
    parser_plot = subparsers.add_parser('plot', help='plot scores')
    parser_plot.set_defaults(func=plot_scores)
    parser_plot.add_argument('--output_dir', type=str, default=output_dir, required=False,
                             help='output directory to store plots')

    args = parser.parse_args()
    args.func(args)
