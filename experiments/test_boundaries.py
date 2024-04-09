import pandas as pd
import os


from disentangling.metrics import (
    z_min_var,
    mig,
    mig_sup,
    dcimig,
    sap,
    modularity,
    dci,
    dcii,
    mig_ksg
)



from disentangling.metrics.utils import *

from experiments.utils.seed_everything import seed_everything




def generate_factors(batch_size, n_factors=2):
    """Generate factor samples of size as `batch_size`, and the dimension of factor is n_factors."""
    factors = [
        np.random.randint(0, n_classes, size=(batch_size, 1))
        for i in range(n_factors)
    ]
    factors = np.hstack(factors)
    return factors


def sample_factors(
        disentanglement, completeness, informativeness=True, batch_size=50000
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
        metric: Callable,
        disentanglement: bool,
        completeness: bool,
        informativeness: bool,
        discrete_factors:Union[List[bool], bool] = False,
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

def test_metrics(n_times):
    #metrics = [mig,  mig_sup, mig_ksg, modularity, sap, dcimig, z_min_var, dci, dcii]
    metrics = [dcii]
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

    dfs = []  # Store DataFrames from each run here

    # Create the base results folder if it doesn't already exist
    base_path = "results/experiment_boundary_cases"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for i in range(n_times):
        seed_everything(i)
        all_data = []

        for c in cases:
            all_results = {}
            for fn in metrics:

                res = get_scores(fn, *c, discrete_factors=True)

                # Check if res is a tuple
                if isinstance(res, tuple):
                    # DCI metric returns Modularity, Compactness and Explicitness scores
                    all_results.update({f'{fn.__name__} Mod': res[0],
                                     f'{fn.__name__} Comp': res[1],
                                     f'{fn.__name__} Expl': res[2]})
                elif isinstance(res, float):
                    all_results.update({fn.__name__: res})

            row = {'seed': i, 'case': str(c)}
            row.update(all_results)
            all_data.append(row)

        # Convert collected data to DataFrame for this run
        df = pd.DataFrame(all_data)
        dfs.append(df)

        # Create subfolder for this run and save the DataFrame
        run_path = os.path.join(base_path, f"test_properties_{i+1}")  # Adjusted index for human-readable naming
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        df.to_csv(os.path.join(run_path, "grouped_results.csv"), index=False)

    # Combine all run DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    # Drop the 'seeds' column from the combined DataFrame
    #combined_df = combined_df.drop(columns=['seeds'])

    # Calculate mean and standard deviation for each metric across all runs
    # Assuming 'case' should be kept as a grouping variable
    grouped = combined_df.groupby('case').agg(['mean', 'std'])

    # Save the combined and grouped results in the base path
    grouped.to_csv(os.path.join(base_path, "combined_grouped_results.csv"))

    # Printing the result
    print(grouped.to_string())

    print("end")


test_metrics(25)
