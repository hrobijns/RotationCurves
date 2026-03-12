import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def date_file_to_df(path: str) -> pd.DataFrame:
    """Load in a SPARC .dat file from a given file path into a dataframe."""
    columns = [
        "Rad_kpc", "Vobs_km/s", "errV_km/s", "Vgas_km/s",
        "Vdisk_km/s", "Vbul_km/s", "SBdisk_L/pc^2", "SBbul_L/pc^2"
    ]
    df = pd.read_csv(path, comment="#", sep='\s+', names = columns)
    df["galaxy"] = os.path.basename(path).replace("_rotmod.dat", "")
    return df

def produce_SPARC_df(
    data_dir: str,
    reduced_load: bool = False,
    n: int = 50,
    random_state: int = 42,
    quality: int | None = None,
    galaxies: list[str] | None = None,
) -> pd.DataFrame:
    """
    Combine all SPARC .dat files from the data folder into one DataFrame.

    Parameters
    ----------
    data_dir : str
        Path to directory holding all the SPARC _rotmod.dat files.
    reduced_load : bool
        If True, randomly sample at most n galaxies from the (filtered) set.
    n : int
        Number of galaxies to sample when reduced_load=True.
    random_state : int
        RNG seed for reproducible sampling.
    quality : int | None
        If provided, keep only galaxies whose Q column in galaxy_parameters.csv
        equals this value (e.g. quality=1 for highest-quality curves).
        Expects galaxy_parameters.csv one directory above data_dir.
    galaxies : list[str] | None
        If provided, keep only the named galaxies (e.g. ["NGC2403", "DDO154"]).
        Can be combined with quality; the intersection is used.

    Returns
    -------
    pd.DataFrame with columns:
        'Rad_kpc', 'Vobs_km/s', 'errV_km/s', 'Vgas_km/s', 'Vdisk_km/s',
        'Vbul_km/s', 'SBdisk_L/pc^2', 'SBbul_L/pc^2', 'galaxy'
    """
    all_files = os.listdir(data_dir)
    allowed = {f.replace("_rotmod.dat", "") for f in all_files}

    if quality is not None:
        params_path = os.path.join(
            os.path.dirname(os.path.normpath(data_dir)), "galaxy_parameters.csv"
        )
        galaxy_params = pd.read_csv(params_path)
        q_names = set(galaxy_params.loc[galaxy_params["Q"] == quality, "galaxy"])
        allowed &= q_names

    if galaxies is not None:
        allowed &= set(galaxies)

    selected_files = [f for f in all_files if f.replace("_rotmod.dat", "") in allowed]

    if reduced_load:
        rng = np.random.default_rng(random_state)
        selected_files = list(
            rng.choice(selected_files, size=min(n, len(selected_files)), replace=False)
        )

    dfs = [date_file_to_df(os.path.join(data_dir, f)) for f in selected_files]

    out = pd.concat(dfs, ignore_index=True)
    print(f"Taken data from {len(out['galaxy'].unique())} galaxies, making one DataFrame with {len(out)} rows.")
    return out

