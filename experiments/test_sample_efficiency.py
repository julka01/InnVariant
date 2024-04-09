import argparse
import os
import pickle
import time
import multiprocessing as mp
import numpy as np
from multiprocessing import Pool
from experiments.utils.imported_utils import *
from disentangling.metrics import *
import pandas as pd
from experiments.utils.seed_everything import seed_everything
from disentangling.metrics.utils import *


def generate_factors(batch_size, n_factors=2):
    """Generate factor samples of size as `batch_size`, and the dimension of factor is n_factors."""
    factors = [
        np.random.randint(0, n_classes, size=(batch_size, 1))
        for i in range(n_factors)
    ]
    factors = np.hstack(factors)
    return factors


def sample_factors(
        disentanglement, completeness, informativeness=True, batch_size=100
):
    """Generate factor samples according to the presence of disentanglement, completeness and informativeness.

    When it is not disentangled, the dimension of factor will be set to 3, otherwise 2.
    Since in the not disentangled case, the first two factor will be encoded into one code. Set the dimension to 3 so the dimension of codes can be more than one.
    """
    n_factors = 3 if not disentanglement and completeness else 2
    factors = generate_factors(batch_size, n_factors=n_factors)
    return factors


def get_representation_function(
        factors: np.ndarray,
        disentanglement: bool,
        completeness: bool,
        informativeness: bool,
):
    """Get the encode process according to the presence of disentanglement, completeness, informativeness.

    Args:
        factors (np.ndarray): The global factors.
        disentanglement (bool): Indicate the encoding process disentangle or not.
        completeness (bool): Indicate the encoding process complete or not.
        informativeness (bool): Indicate the encoding process informative or not.

    Returns:
        Callable[np.ndarray, np.ndarray]: The desired encoding process.
    """
    if disentanglement and completeness:
        encode = m1c1i1(factors)
    elif disentanglement and not completeness:
        encode = m1c0i1(factors)
    elif not disentanglement and completeness:
        encode = m0c1i1(factors)
    elif not disentanglement and not completeness:
        encode = m0c0i1(factors)
    if informativeness:
        return encode
    return lambda factors: encode(information_pipeline(factors))


def get_scores(
        sample_size: int,
        metric,
        disentanglement: bool,
        completeness: bool,
        informativeness: bool,
        discrete_factors: False,
        batch_size=50000,
):
    """Run the metric of a desired encoding process indicated by disentanglement, completeness, informativeness.

    Args:
        metric (Callable): The target metric.
        disentanglement (bool): Indicate the encoding process disentangle or not.
        completeness (bool): Indicate the encoding process complete or not.
        informativeness (bool): Indicate the encoding process informative or not.
        discrete_factors (Union[List[bool], bool]): It implies if each factor is discrete. Default: True.
        batch_size (int): The batch size of generated factor samples.

    Returns:
        Any: The return values of the target metric where the input is the random generative factors and the latent codes encoded by the desired encoding process.
    """

    if metric.__name__ == "z_min_var":
        sample_func = lambda batch_size=10000: sample_factors(
            disentanglement, completeness, informativeness, batch_size
        )
        encode_func = get_representation_function(
            sample_func(10000),
            disentanglement,
            completeness,
            informativeness,
        )
        scores = metric(sample_factors=sample_func, factor2code=encode_func)
        return scores
    factors = sample_factors(disentanglement, completeness, informativeness, batch_size=batch_size)
    encode = get_representation_function(
        factors, disentanglement, completeness, informativeness
    )
    codes = encode(factors)
    scores = metric(factors, codes, discrete_factors=discrete_factors)
    return scores


def plot_curves(plots, output_file, format='png', figsize=(25, 6), x_scale='linear',
                y_scale='linear', x_lims=None, y_lims=(-0.05, 1.05), x_label="", y_label="",
                title_font_size=22, axis_font_size=16, legend_font_size=13, colors=None,
                line_styles=None, legend_positions=None, legend_columns=None):
    ''' Plot curves from a dictionary into the same figure

    :param plots:                   {title: {curves_to_plot}} dictionary
                                    the following format is assumed:
                                    {
                                        title_plot_1:
                                        {
                                            <legend_curve_1>: ([x0, x1, ..., xN], [y0, y1, ..., yN]),
                                            ...
                                        }
                                        ...
                                    }
    :param output_file:             where and under which name the plot is saved
    :param format:                  format to save the plot
    :param figsize:                 size of the figure
    :param x_scale:                 scale on x axis
    :param y_scale:                 scale on y axis
    :param x_lims:                  limit values on x axis
    :param y_lims:                  limit values on y axis
    :param x_label:                 label for the x axis
    :param y_label:                 label for the y axis
    :param title_font_size:         font size for title of each plot
    :param axis_font_size:          font size of axis labels
    :param legend_font_size:        font size of legend labels
    :param colors:                  colors to use for the curves in the plots
    :param line_styles:             line styles to use for the curves in the plot
    :param legend_positions:        positioning of the legend for each plot
    :param legend_columns:          number of columns in the legend for each plot
    '''
    _, axes = plt.subplots(nrows=1, ncols=len(plots), figsize=figsize)

    for idx, title in enumerate(plots):
        # extract legend labels
        curves = plots[title]
        legends = [legend for legend in curves]

        # cycle through colors
        colors_cycle = cycle(colors) if colors is not None else cycle(['blue', 'green', 'red'])
        colors_cycle = [next(colors_cycle) for _ in range(len(curves))]

        # cycle through line styles
        lines_cycle = cycle(line_styles) if line_styles is not None else cycle(['-'])
        lines_cycle = [next(lines_cycle) for _ in range(len(curves))]

        # plot curves
        ax = axes.ravel()[idx]
        for datapoints, legend, color, line_style in zip(curves.values(), legends, colors_cycle, lines_cycle):
            x, y = datapoints  # x: x-values, y: mean values, z: standard deviation
            ax.plot(x, y, label=legend, color=color, linestyle=line_style)

#             # Calculate the lower and upper bounds for the standard deviation area
#             lower_bound = y - z
#             upper_bound = y + z

#             # Add the shaded area
#             ax.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.4)  # Adjust alpha for transparency

        # set scales on each axis
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

        # set limits on x and y axis
        if x_lims is not None:
            ax.set_xlim(x_lims[0], x_lims[1])
        if y_lims is not None:
            ax.set_ylim(y_lims[0], y_lims[1])

        # set title and labels on axis
        ax.set_title(title, fontsize=title_font_size, pad=20)
        ax.set_xlabel(x_label, fontsize=axis_font_size)
        ax.set_ylabel(y_label, fontsize=axis_font_size, labelpad=10)

        # set x-ticks and y-ticks font size
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(axis_font_size)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(axis_font_size)

            # set legends
        if legend_positions is not None and legend_columns is not None:
            ax.legend(handlelength=2, loc='lower center', bbox_to_anchor=legend_positions[idx],
                      fontsize=legend_font_size, ncol=legend_columns[idx])
        elif legend_positions is not None:
            ax.legend(handlelength=2, loc='lower center', bbox_to_anchor=legend_positions[idx],
                      fontsize=legend_font_size)
        elif legend_columns is not None:
            ax.legend(handlelength=2, loc='lower center', fontsize=legend_font_size, ncol=legend_columns[idx])
        else:
            ax.legend(handlelength=2, loc='lower center', fontsize=legend_font_size)

    # save the plot
    save_dir = os.path.dirname(output_file)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{output_file}.{format}', bbox_inches='tight', format=format)
    plt.close()



def test_metrics(metrics , n_times, samples=[10, 100,1000, 10000, 10000]):
    cases = [
        [False, False, False],
        [False, False, True],
        [False, True, False],
        [False, True, True],
        [True, False, False],
        [True, False, True],
        [True, True, False],
        [True, True, True],
    ]
    dfs = []  # Stores DataFrames from all runs
    base_path = "results/experiment_sample_efficiency"
    os.makedirs(base_path, exist_ok=True)

    for n in range(n_times):
        seed_everything(n)  # Ensure this is defined elsewhere
        run_data = []

        for sample_size in samples:
            all_data = []
            for c in cases:
                case_results = {}
                for fn in metrics:
                    res = get_scores(sample_size, fn, *c, discrete_factors=True, batch_size=sample_size)
                    if isinstance(res, tuple):
                        for i, suffix in enumerate(['Mod', 'Comp', 'Expl']):
                            case_results[f'{fn.__name__} {suffix}'] = res[i]
                    else:
                        case_results[fn.__name__] = res

                row = {'seed': n, 'case': str(c), 'sample_size': sample_size, **case_results}
                all_data.append(row)

            df = pd.DataFrame(all_data)
            run_data.append(df)

        run_df = pd.concat(run_data, ignore_index=True)
        dfs.append(run_df)

    final_df = pd.concat(dfs, ignore_index=True)
    # Process the differences and save them
    output_dir = base_path  # Use the base_path as the output directory
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    for metric_fn in metrics:
        begin = time.time()

        metric_scores = {}
        for suffix in ['', '_Mod', '_Comp', '_Expl']:  # Include base and suffixed metrics
            metric_name = f'{metric_fn.__name__}{suffix}'.strip()
            if metric_name in final_df.columns:
                # Aggregate scores by sample size and case, then calculate mean
                aggregated_scores = final_df.groupby(['sample_size', 'case'])[metric_name].mean().reset_index()
                # Pivot to have sample sizes as columns for easier subtraction
                pivot_df = aggregated_scores.pivot(index='case', columns='sample_size', values=metric_name)
                # Calculate differences: last sample size - current, then mean over cases
                last_values = pivot_df[max(samples)]
                differences = pivot_df.apply(lambda x: np.abs(last_values - x))
                mean_differences = differences.mean(axis=0)  # Mean over all cases
                metric_scores[metric_name] = mean_differences.to_dict()
                # Save dictionaries
    for key, value in metric_scores.items():
        noise_array = np.array(list(value.keys()))
        differences_array = np.array(list(value.values()))
        keyname = METRIC_DICT[globals()[key]]
        with open(os.path.join(output_dir, f'{keyname}'), 'wb') as output:
            pickle.dump({f'{keyname}': (noise_array, differences_array)}, output)

    # display time
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

    # compute means
    for metric, values in scores.items():
        num_samples, scores_array = values[0], values[1]
        scores[metric] = (num_samples,
                          np.mean(scores_array, axis=tuple(range(1, scores_array.ndim))),
                          )

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
                    x_label='num_samples', y_label='mean score',
                    colors=PLOTS['COLORS'], line_styles=PLOTS['LINE STYLES'],
                    legend_positions=PLOTS['LEGEND POSITIONS'],
                    legend_columns=PLOTS['NB LEGEND COLUMNS'])



NB_JOBS = get_nb_jobs('max')
NB_RANDOM_REPRESENTATIONS = 50
NB_RUNS = 1
NB_EXAMPLES = 20000
NB_FACTORS = 6
DISTRIBUTION = [np.random.uniform, {'low': 0., 'high': 1.}]
NON_LINEARITY_STRENGTH = [i / 10 for i in range(11) if i % 2 == 0]
METRICS =  [edi]
#METRIC_DICT = {dcimig: 'DCIMIG', dcii: 'DCII', dci: 'DCI', mig: 'MIG', mig_ksg: 'MIG-ksg', mig_sup: 'MIG-sup', mig_sup_ksg: 'MIG-sup-ksg', dcimig_ksg: 'dcimig-ksg', z_min_var: 'FactorVAE', sap: 'SAP', modularity: 'Modularity'}
METRIC_DICT = {dcimig: 'DCIMIG', edi: 'EDI', dci: 'DCI', mig: 'MIG', mig_ksg: 'MIG-ksg', mig_sup: 'MIG-sup', mig_sup_ksg: 'MIG-sup-ksg', dcimig_ksg: 'DCIMIG-ksg', z_min_var: 'Z-min Var', sap: 'SAP', dci_Mod: 'DCI Mod',  dci_Comp: 'DCI Comp', dci_Expl: 'DCI Expl', edi_Mod: 'EDI Mod', edi_Comp: 'EDI Comp', edi_Expl: 'EDI Expl'  }

# config parameters for plots
loosely_dashed = (0, (5, 10))
densely_dashdotdotted = (0, (3, 1, 1, 1, 1, 1))
densely_dotted = (0, (1, 1))
PLOTS = {

    'FAMILIES': {
        'Metrics': [ 'DCII Expl', 'DCII Comp', 'DCII Mod' , 'DCI Mod', 'DCI Comp',  'DCI Expl', 'SAP'],

        'Information-based': ['DCII Expl', 'DCII Comp', 'DCII Mod', 'MIG', 'MIG-ksg', 'DCIMIG-ksg', 'DCIMIG' , 'MIG-sup']
    },
    'COLORS': ['blue', 'green', 'red', 'darkturquoise', 'magenta', 'orange', 'black', 'violet', 'yellow'],
    'LINE STYLES': ['--', '-.', loosely_dashed, densely_dashdotdotted, ':', densely_dotted],
    'LEGEND POSITIONS': [(0.5, -0.25), (0.5, -0.25)],
    'NB LEGEND COLUMNS': [ 5, 5]
}



if __name__ == "__main__":
    # project ROOT
    FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.realpath(os.path.dirname(FILE_ROOT))

    # # default metrics and default output directory
    metrics = [metric for metric in METRICS]
    output_dir = os.path.join(PROJECT_ROOT, 'results', 'experiment_sample_efficiency')
    #
    # # create parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    #
    # parser for the "run" command -- run non-linearity experiment
    parser_run = subparsers.add_parser('run', help='compute scores')
    parser_run.set_defaults(func=test_metrics(METRICS, n_times= 10))
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
