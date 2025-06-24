import pathlib
import os

import click
from .parallel import run_parallel_sampling

from synformer.chem.mol import Molecule, read_mol_file


def _input_mols_option(p):
    return list(read_mol_file(p))


@click.command()
@click.option("--input", "-i", type=_input_mols_option, required=True)
@click.option("--output", "-o", type=click.Path(exists=False, path_type=pathlib.Path), required=True)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/trained_weights/default.ckpt",
)
@click.option("--search-width", type=int, default=24)
@click.option("--exhaustiveness", type=int, default=64)
@click.option("--num-gpus", type=int, default=-1)
@click.option("--num-workers-per-gpu", type=int, default=1)
@click.option("--task-qsize", type=int, default=0)
@click.option("--result-qsize", type=int, default=0)
@click.option("--time-limit", type=int, default=180)
@click.option("--dont-sort", is_flag=True)
@click.option("--use-desert", is_flag=True, help="Use DESERT encoder for molecule generation")
@click.option("--no-adapter", is_flag=True, help="Run the model without the fragment encoder (raw DESERT output).")
@click.option("--minimal-processing", is_flag=True, help="Run with minimal processing (just dimension matching and basic feature placement).")
@click.option(
    "--desert-model-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to the DESERT model file",
)
@click.option(
    "--vocab-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to the vocabulary file for DESERT",
)
@click.option(
    "--smiles-checkpoint-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to the pretrained SMILES checkpoint for the decoder",
)
@click.option(
    "--shape-patches-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to the shape patches file for DESERT",
)
@click.option(
    "--receptor-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to the receptor PDBQT file for docking",
)
@click.option(
    "--receptor-center",
    type=(float, float, float),
    help="Center coordinates (x,y,z) for docking box",
)
def main(
    input: list[Molecule],
    output: pathlib.Path,
    model_path: pathlib.Path,
    search_width: int,
    exhaustiveness: int,
    num_gpus: int,
    num_workers_per_gpu: int,
    task_qsize: int,
    result_qsize: int,
    time_limit: int,
    dont_sort: bool,
    use_desert: bool,
    no_adapter: bool,
    minimal_processing: bool,
    desert_model_path: pathlib.Path,
    vocab_path: pathlib.Path,
    smiles_checkpoint_path: pathlib.Path,
    shape_patches_path: pathlib.Path,
    receptor_path: pathlib.Path,
    receptor_center: tuple[float, float, float],
):
    # Validate DESERT options
    if use_desert:
        if not desert_model_path:
            raise ValueError("--desert-model-path must be provided when --use-desert is set")
        if not vocab_path:
            raise ValueError("--vocab-path must be provided when --use-desert is set")
        if not smiles_checkpoint_path:
            raise ValueError("--smiles-checkpoint-path must be provided when --use-desert is set")
        if not shape_patches_path:
            print("Warning: --shape-patches-path not provided, using default shape patches")
    
    run_parallel_sampling(
        input=input,
        output=output,
        model_path=model_path,
        search_width=search_width,
        exhaustiveness=exhaustiveness,
        num_gpus=num_gpus,
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        time_limit=time_limit,
        sort_by_scores=not dont_sort,
        use_desert=use_desert,
        no_adapter=no_adapter,
        minimal_processing=minimal_processing,
        desert_model_path=desert_model_path,
        vocab_path=vocab_path,
        smiles_checkpoint_path=smiles_checkpoint_path,
        shape_patches_path=shape_patches_path,
        receptor_path=receptor_path,
        receptor_center=receptor_center,
    )


if __name__ == "__main__":
    main()
