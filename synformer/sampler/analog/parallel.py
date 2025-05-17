import multiprocessing as mp
import os
import pathlib
import pickle
import subprocess
from multiprocessing import synchronize as sync
from typing import TypeAlias

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from synformer.chem.fpindex import FingerprintIndex
from synformer.chem.matrix import ReactantReactionMatrix
from synformer.chem.mol import FingerprintOption, Molecule
from synformer.models.synformer import Synformer
from synformer.sampler.analog.state_pool import StatePool, TimeLimit

TaskQueueType: TypeAlias = "mp.JoinableQueue[Molecule | None]"
ResultQueueType: TypeAlias = "mp.Queue[tuple[Molecule, pd.DataFrame]]"


class Worker(mp.Process):
    def __init__(
        self,
        model_path: pathlib.Path,
        task_queue: TaskQueueType,
        result_queue: ResultQueueType,
        gpu_id: str,
        gpu_lock: sync.Lock,
        state_pool_opt: dict | None = None,
        max_evolve_steps: int = 12,
        max_results: int = 100,
        time_limit: int = 120,
        use_desert: bool = False,
        desert_model_path: pathlib.Path = None,
        vocab_path: pathlib.Path = None,
        smiles_checkpoint_path: pathlib.Path = None,
        shape_patches_path: pathlib.Path = None,
        receptor_path: pathlib.Path = None,
        receptor_center: tuple[float, float, float] = None,
    ):
        super().__init__()
        self._model_path = model_path
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._gpu_id = gpu_id
        self._gpu_lock = gpu_lock

        self._state_pool_opt = state_pool_opt or {}
        self._max_evolve_steps = max_evolve_steps
        self._max_results = max_results
        self._time_limit = time_limit
        
        # DESERT options
        self._use_desert = use_desert
        self._desert_model_path = desert_model_path
        self._vocab_path = vocab_path
        self._smiles_checkpoint_path = smiles_checkpoint_path
        self._shape_patches_path = shape_patches_path
        
        # Docking options
        self._receptor_path = receptor_path
        self._receptor_center = receptor_center

    def run(self) -> None:
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_id
        device = "cuda"

        ckpt = torch.load(self._model_path, map_location="cpu")
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        
        # Modify config if using DESERT
        if self._use_desert:
            print(f"\nUsing DESERT encoder with model path: {self._desert_model_path}")
            config.model.encoder_type = "desert"
            config.model.encoder.desert_model_path = str(self._desert_model_path)
            config.model.encoder.vocab_path = str(self._vocab_path)
            config.model.encoder.shape_patches_path = str(self._shape_patches_path) if self._shape_patches_path else None
            print(f"Shape patches path set to: {self._shape_patches_path}")
        
        model = Synformer(config.model).to(device)
        
        # Load model weights
        model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        model.eval()
        
        # Load pretrained decoder if using DESERT
        if self._use_desert and self._smiles_checkpoint_path:
            print(f"\nLoading pretrained decoder from: {self._smiles_checkpoint_path}")
            model.load_pretrained_decoder(self._smiles_checkpoint_path, device=device)
        
        self._model = model

        self._fpindex: FingerprintIndex = pickle.load(open(config.chem.fpindex, "rb"))
        self._rxn_matrix: ReactantReactionMatrix = pickle.load(open(config.chem.rxn_matrix, "rb"))

        try:
            while True:
                next_task = self._task_queue.get()
                if next_task is None:
                    self._task_queue.task_done()
                    break
                result_df = self.process(next_task)
                self._task_queue.task_done()
                self._result_queue.put((next_task, result_df))
                if len(result_df) == 0:
                    print(f"{self.name}: No results for {next_task.smiles}")
                else:
                    max_sim = result_df["score"].max()
                    print(f"{self.name}: {max_sim:.3f} {next_task.smiles}")
        except KeyboardInterrupt:
            print(f"{self.name}: Exiting due to KeyboardInterrupt")
            return

    def process(self, mol: Molecule):
        # Create state pool options including shape_patches_path and docking options
        state_pool_options = {
            **self._state_pool_opt,
            'shape_patches_path': self._shape_patches_path,
            'receptor_path': self._receptor_path,
            'receptor_center': self._receptor_center
        }
        
        sampler = StatePool(
            fpindex=self._fpindex,
            rxn_matrix=self._rxn_matrix,
            mol=mol,
            model=self._model,
            **state_pool_options,
        )
        tl = TimeLimit(self._time_limit)
        for _ in range(self._max_evolve_steps):
            sampler.evolve(gpu_lock=self._gpu_lock, show_pbar=False, time_limit=tl)
            
            # Get products and calculate max combined score
            products = list(sampler.get_products())
            if not products:
                continue
                
            # Check if we've found a perfect match (combined score of 1.0)
            # This is now based on the combined score (docking*0.9 + tanimoto*0.1)
            max_combined_score = -float('inf')
            for p in products:
                # Get RDKit molecule for docking
                rdkit_mol = None
                try:
                    from rdkit import Chem
                    rdkit_mol = Chem.MolFromSmiles(p.molecule.smiles)
                except Exception as e:
                    print(f"Error converting molecule to RDKit: {str(e)}")
                
                tanimoto_score = p.molecule.sim(mol, FingerprintOption.morgan_for_tanimoto_similarity())
                
                # Get docking score if possible
                docking_score = None
                if rdkit_mol is not None and self._receptor_path and self._receptor_center:
                    from synformer.sampler.analog.state_pool import dock_best_molecule
                    docking_score = dock_best_molecule(rdkit_mol, self._receptor_path, self._receptor_center)
                
                # Calculate combined score
                if docking_score is not None:
                    # Normalize docking score (lower is better, typically -12 to 0)
                    normalized_docking_score = max(0, min(1, (-docking_score) / 12.0))
                    combined_score = 0.9 * normalized_docking_score + 0.1 * tanimoto_score
                else:
                    combined_score = tanimoto_score
                
                max_combined_score = max(max_combined_score, combined_score)
            
            # If we found a perfect match, stop evolving
            if max_combined_score >= 0.99:
                break

        # Get the final dataframe of results
        df = sampler.get_dataframe()[: self._max_results]
        
        # Print a message if no results were found
        if len(df) == 0:
            print(f"{self.name}: No results for {mol.smiles}")
            # Create a minimal DataFrame with just the target molecule
            df = pd.DataFrame([{
                "target": mol.smiles,
                "smiles": "",
                "tanimoto_score": 0.0,
                "docking_score": None,
                "score": 0.0,
                "synthesis": "",
                "num_steps": 0
            }])
        else:
            max_sim = df["score"].max()
            print(f"{self.name}: {max_sim:.3f} {mol.smiles}")
            
        return df


class WorkerPool:
    def __init__(
        self,
        gpu_ids: list[int | str],
        num_workers_per_gpu: int,
        task_qsize: int,
        result_qsize: int,
        **worker_opt,
    ) -> None:
        super().__init__()
        self._task_queue: TaskQueueType = mp.JoinableQueue(task_qsize)
        self._result_queue: ResultQueueType = mp.Queue(result_qsize)
        self._gpu_ids = [str(d) for d in gpu_ids]
        self._gpu_locks = [mp.Lock() for _ in gpu_ids]
        num_gpus = len(gpu_ids)
        num_workers = num_workers_per_gpu * num_gpus
        self._workers = [
            Worker(
                task_queue=self._task_queue,
                result_queue=self._result_queue,
                gpu_id=self._gpu_ids[i % num_gpus],
                gpu_lock=self._gpu_locks[i % num_gpus],
                **worker_opt,
            )
            for i in range(num_workers)
        ]

        for w in self._workers:
            w.start()

    def submit(self, task: Molecule, block: bool = True, timeout: float | None = None):
        self._task_queue.put(task, block=block, timeout=timeout)

    def fetch(self, block: bool = True, timeout: float | None = None):
        return self._result_queue.get(block=block, timeout=timeout)

    def kill(self):
        for w in self._workers:
            w.kill()
        self._result_queue.close()
        self._task_queue.close()

    def end(self):
        for _ in self._workers:
            self._task_queue.put(None)
        self._task_queue.join()
        for w in tqdm(self._workers, desc="Terminating"):
            w.terminate()
        self._result_queue.close()
        self._task_queue.close()


class WorkerNoStop(mp.Process):
    def __init__(
        self,
        model_path: pathlib.Path,
        task_queue: TaskQueueType,
        result_queue: ResultQueueType,
        gpu_id: str,
        gpu_lock: sync.Lock,
        state_pool_opt: dict | None = None,
        max_evolve_steps: int = 12,
        max_results: int = 100,
        time_limit: int = 120,
        use_desert: bool = False,
        desert_model_path: pathlib.Path = None,
        vocab_path: pathlib.Path = None,
        smiles_checkpoint_path: pathlib.Path = None,
        shape_patches_path: pathlib.Path = None,
        receptor_path: pathlib.Path = None,
        receptor_center: tuple[float, float, float] = None,
    ):
        super().__init__()
        self._model_path = model_path
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._gpu_id = gpu_id
        self._gpu_lock = gpu_lock

        self._state_pool_opt = state_pool_opt or {}
        self._max_evolve_steps = max_evolve_steps
        self._max_results = max_results
        self._time_limit = time_limit
        
        # DESERT options
        self._use_desert = use_desert
        self._desert_model_path = desert_model_path
        self._vocab_path = vocab_path
        self._smiles_checkpoint_path = smiles_checkpoint_path
        self._shape_patches_path = shape_patches_path
        
        # Docking options
        self._receptor_path = receptor_path
        self._receptor_center = receptor_center

    def run(self) -> None:
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_id
        device = "cuda"

        ckpt = torch.load(self._model_path, map_location="cpu")
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        
        # Modify config if using DESERT
        if self._use_desert:
            print(f"\nUsing DESERT encoder with model path: {self._desert_model_path}")
            config.model.encoder_type = "desert"
            config.model.encoder.desert_model_path = str(self._desert_model_path)
            config.model.encoder.vocab_path = str(self._vocab_path)
            config.model.encoder.shape_patches_path = str(self._shape_patches_path) if self._shape_patches_path else None
            print(f"Shape patches path set to: {self._shape_patches_path}")
        
        model = Synformer(config.model).to(device)
        
        # Load model weights
        model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        model.eval()
        
        # Load pretrained decoder if using DESERT
        if self._use_desert and self._smiles_checkpoint_path:
            print(f"\nLoading pretrained decoder from: {self._smiles_checkpoint_path}")
            model.load_pretrained_decoder(self._smiles_checkpoint_path, device=device)
        
        self._model = model

        self._fpindex: FingerprintIndex = pickle.load(open(config.chem.fpindex, "rb"))
        self._rxn_matrix: ReactantReactionMatrix = pickle.load(open(config.chem.rxn_matrix, "rb"))

        try:
            while True:
                next_task = self._task_queue.get()
                if next_task is None:
                    self._task_queue.task_done()
                    break
                result_df = self.process(next_task)
                self._task_queue.task_done()
                self._result_queue.put((next_task, result_df))
                if len(result_df) == 0:
                    print(f"{self.name}: No results for {next_task.smiles}")
                else:
                    max_sim = result_df["score"].max()
                    print(f"{self.name}: {max_sim:.3f} {next_task.smiles}")
        except KeyboardInterrupt:
            print(f"{self.name}: Exiting due to KeyboardInterrupt")
            return

    def process(self, mol: Molecule):
        # Create state pool options including shape_patches_path and docking options
        state_pool_options = {
            **self._state_pool_opt,
            'shape_patches_path': self._shape_patches_path,
            'receptor_path': self._receptor_path,
            'receptor_center': self._receptor_center
        }
        
        sampler = StatePool(
            fpindex=self._fpindex,
            rxn_matrix=self._rxn_matrix,
            mol=mol,
            model=self._model,
            **state_pool_options,
        )
        tl = TimeLimit(self._time_limit)
        for _ in range(self._max_evolve_steps):
            sampler.evolve(gpu_lock=self._gpu_lock, show_pbar=False, time_limit=tl)
            
            # Get products and calculate max combined score
            products = list(sampler.get_products())
            if not products:
                continue
                
            # Check if we've found a perfect match (combined score of 1.0)
            # This is now based on the combined score (docking*0.9 + tanimoto*0.1)
            max_combined_score = -float('inf')
            for p in products:
                # Get RDKit molecule for docking
                rdkit_mol = None
                try:
                    from rdkit import Chem
                    rdkit_mol = Chem.MolFromSmiles(p.molecule.smiles)
                except Exception as e:
                    print(f"Error converting molecule to RDKit: {str(e)}")
                
                tanimoto_score = p.molecule.sim(mol, FingerprintOption.morgan_for_tanimoto_similarity())
                
                # Get docking score if possible
                docking_score = None
                if rdkit_mol is not None and self._receptor_path and self._receptor_center:
                    from synformer.sampler.analog.state_pool import dock_best_molecule
                    docking_score = dock_best_molecule(rdkit_mol, self._receptor_path, self._receptor_center)
                
                # Calculate combined score
                if docking_score is not None:
                    # Normalize docking score (lower is better, typically -12 to 0)
                    normalized_docking_score = max(0, min(1, (-docking_score) / 12.0))
                    combined_score = 0.9 * normalized_docking_score + 0.1 * tanimoto_score
                else:
                    combined_score = tanimoto_score
                
                max_combined_score = max(max_combined_score, combined_score)
            
            # If we found a perfect match, stop evolving
            if max_combined_score >= 0.99:
                break

        # Get the final dataframe of results
        df = sampler.get_dataframe()[: self._max_results]
        
        # Print a message if no results were found
        if len(df) == 0:
            print(f"{self.name}: No results for {mol.smiles}")
            # Create a minimal DataFrame with just the target molecule
            df = pd.DataFrame([{
                "target": mol.smiles,
                "smiles": "",
                "tanimoto_score": 0.0,
                "docking_score": None,
                "score": 0.0,
                "synthesis": "",
                "num_steps": 0
            }])
        else:
            max_sim = df["score"].max()
            print(f"{self.name}: {max_sim:.3f} {mol.smiles}")
            
        return df


class WorkerPoolNoStop:
    def __init__(
        self,
        gpu_ids: list[int | str],
        num_workers_per_gpu: int,
        task_qsize: int,
        result_qsize: int,
        **worker_opt,
    ) -> None:
        super().__init__()
        self._task_queue: TaskQueueType = mp.JoinableQueue(task_qsize)
        self._result_queue: ResultQueueType = mp.Queue(result_qsize)
        self._gpu_ids = [str(d) for d in gpu_ids]
        self._gpu_locks = [mp.Lock() for _ in gpu_ids]
        num_gpus = len(gpu_ids)
        num_workers = num_workers_per_gpu * num_gpus
        self._workers = [
            WorkerNoStop(
                task_queue=self._task_queue,
                result_queue=self._result_queue,
                gpu_id=self._gpu_ids[i % num_gpus],
                gpu_lock=self._gpu_locks[i % num_gpus],
                **worker_opt,
            )
            for i in range(num_workers)
        ]

        for w in self._workers:
            w.start()

    def submit(self, task: Molecule, block: bool = True, timeout: float | None = None):
        self._task_queue.put(task, block=block, timeout=timeout)

    def fetch(self, block: bool = True, timeout: float | None = None):
        return self._result_queue.get(block=block, timeout=timeout)

    def kill(self):
        for w in self._workers:
            w.kill()
        self._result_queue.close()
        self._task_queue.close()

    def end(self):
        for _ in self._workers:
            self._task_queue.put(None)
        self._task_queue.join()
        for w in tqdm(self._workers, desc="Terminating"):
            w.terminate()
        self._result_queue.close()
        self._task_queue.close()


def _count_gpus():
    return int(
        subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l", shell=True, text=True
        ).strip()
    )


def run_parallel_sampling(
    input: list[Molecule],
    output: pathlib.Path,
    model_path: pathlib.Path,
    search_width: int = 24,
    exhaustiveness: int = 64,
    num_gpus: int = -1,
    num_workers_per_gpu: int = 2,
    task_qsize: int = 0,
    result_qsize: int = 0,
    time_limit: int = 180,
    sort_by_scores: bool = True,
    use_desert: bool = False,
    desert_model_path: pathlib.Path = None,
    vocab_path: pathlib.Path = None,
    smiles_checkpoint_path: pathlib.Path = None,
    shape_patches_path: pathlib.Path = None,
    receptor_path: pathlib.Path = None,
    receptor_center: tuple[float, float, float] = None,
) -> None:
    num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
    pool = WorkerPool(
        gpu_ids=list(range(num_gpus)),
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        model_path=model_path,
        state_pool_opt={
            "factor": search_width,
            "max_active_states": exhaustiveness,
            "sort_by_score": sort_by_scores,
        },
        time_limit=time_limit,
        use_desert=use_desert,
        desert_model_path=desert_model_path,
        vocab_path=vocab_path,
        smiles_checkpoint_path=smiles_checkpoint_path,
        shape_patches_path=shape_patches_path,
        receptor_path=receptor_path,
        receptor_center=receptor_center,
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    total = len(input)
    for mol in input:
        pool.submit(mol)

    df_all: list[pd.DataFrame] = []
    with open(output, "w") as f:
        for _ in tqdm(range(total)):
            _, df = pool.fetch()
            # We always have at least an empty DataFrame now
            df_with_results = df[df["smiles"] != ""]  # Filter out empty results
            if len(df_with_results) > 0:
                df_with_results.to_csv(f, float_format="%.3f", index=False, header=f.tell() == 0)
                df_all.append(df_with_results)

    # Check if we have any results before concatenating
    if not df_all:
        print("No valid results found for any input molecules.")
        pool.end()
        return

    df_merge = pd.concat(df_all, ignore_index=True)
    print(df_merge.loc[df_merge.groupby("target").idxmax()["score"]].select_dtypes(include="number").sum() / total)

    count_success = len(df_merge["target"].unique())
    print(f"Success rate: {count_success}/{total} = {count_success / total:.3f}")

    # Update the reconstruction criteria to use the combined score
    high_score_targets: set[str] = set()
    for _, row in df_merge.iterrows():
        if row["score"] >= 0.99:  # Using combined score threshold of 0.99
            high_score_targets.add(row["target"])
    count_high_score = len(high_score_targets)
    print(f"High score rate (score >= 0.99): {count_high_score}/{total} = {count_high_score / total:.3f}")

    pool.end()


def run_parallel_sampling_return_smiles(
    input: list[Molecule],
    model_path: pathlib.Path,
    search_width: int = 24,
    exhaustiveness: int = 64,
    num_gpus: int = -1,
    num_workers_per_gpu: int = 2,
    task_qsize: int = 0,
    result_qsize: int = 0,
    time_limit: int = 180,
    sort_by_scores: bool = True,
    use_desert: bool = False,
    desert_model_path: pathlib.Path = None,
    vocab_path: pathlib.Path = None,
    smiles_checkpoint_path: pathlib.Path = None,
    shape_patches_path: pathlib.Path = None,
    receptor_path: pathlib.Path = None,
    receptor_center: tuple[float, float, float] = None,
) -> None:
    num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
    pool = WorkerPool(
        gpu_ids=list(range(num_gpus)),
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        model_path=model_path,
        state_pool_opt={
            "factor": search_width,
            "max_active_states": exhaustiveness,
            "sort_by_score": sort_by_scores,
        },
        time_limit=time_limit,
        use_desert=use_desert,
        desert_model_path=desert_model_path,
        vocab_path=vocab_path,
        smiles_checkpoint_path=smiles_checkpoint_path,
        shape_patches_path=shape_patches_path,
        receptor_path=receptor_path,
        receptor_center=receptor_center,
    )

    total = len(input)
    for mol in input:
        pool.submit(mol)

    df_all: list[pd.DataFrame] = []
    for _ in tqdm(range(total)):
        _, df = pool.fetch()
        if len(df) == 0:
            continue
        df_all.append(df)

    pool.end()
    return pd.concat(df_all, ignore_index=True)


def run_parallel_sampling_return_smiles_no_early_stop(
    input: list[Molecule],
    model_path: pathlib.Path,
    search_width: int = 100,
    exhaustiveness: int = 100,
    num_gpus: int = -1,
    num_workers_per_gpu: int = 2,
    task_qsize: int = 0,
    result_qsize: int = 0,
    time_limit: int = 180,
    sort_by_scores: bool = True,
    use_desert: bool = False,
    desert_model_path: pathlib.Path = None,
    vocab_path: pathlib.Path = None,
    smiles_checkpoint_path: pathlib.Path = None,
    shape_patches_path: pathlib.Path = None,
    receptor_path: pathlib.Path = None,
    receptor_center: tuple[float, float, float] = None,
) -> None:
    num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
    pool = WorkerPoolNoStop(
        gpu_ids=list(range(num_gpus)),
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        model_path=model_path,
        state_pool_opt={
            "factor": search_width,
            "max_active_states": exhaustiveness,
            "sort_by_score": sort_by_scores,
        },
        time_limit=time_limit,
        use_desert=use_desert,
        desert_model_path=desert_model_path,
        vocab_path=vocab_path,
        smiles_checkpoint_path=smiles_checkpoint_path,
        shape_patches_path=shape_patches_path,
        receptor_path=receptor_path,
        receptor_center=receptor_center,
    )

    total = len(input)
    for mol in input:
        pool.submit(mol)

    df_all: list[pd.DataFrame] = []

    for _ in tqdm(range(total)):
        _, df = pool.fetch()
        if len(df) == 0:
            continue
        df_all.append(df)

    df_merge = pd.concat(df_all, ignore_index=True)
    pool.end()

    return df_merge

def run_sampling_one_cpu(
    input: Molecule,
    model_path: pathlib.Path,
    mat_path: pathlib.Path,
    fpi_path: pathlib.Path,
    search_width: int = 24,
    exhaustiveness: int = 64,
    time_limit: int = 180,
    max_results: int = 100,
    max_evolve_steps: int = 12,
    sort_by_scores: bool = True,
    use_desert: bool = False,
    desert_model_path: pathlib.Path = None,
    vocab_path: pathlib.Path = None,
    smiles_checkpoint_path: pathlib.Path = None,
    receptor_path: pathlib.Path = None,
    receptor_center: tuple[float, float, float] = None,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(model_path, map_location=device)
    config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
    
    # Modify config if using DESERT
    if use_desert:
        print(f"\nUsing DESERT encoder with model path: {desert_model_path}")
        config.model.encoder_type = "desert"
        config.model.encoder.desert_model_path = str(desert_model_path)
        config.model.encoder.vocab_path = str(vocab_path)
    
    model = Synformer(config.model).to(device)
    
    # Load model weights
    model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
    model.eval()
    
    # Load pretrained decoder if using DESERT
    if use_desert and smiles_checkpoint_path:
        print(f"\nLoading pretrained decoder from: {smiles_checkpoint_path}")
        model.load_pretrained_decoder(smiles_checkpoint_path, device=device)
    
    fpindex: FingerprintIndex = pickle.load(open(fpi_path, "rb"))
    rxn_matrix: ReactantReactionMatrix = pickle.load(open(mat_path, "rb"))

    sampler = StatePool(
        fpindex=fpindex,
        rxn_matrix=rxn_matrix,
        mol=input,
        model=model,
        factor=search_width,
        max_active_states=exhaustiveness,
        sort_by_score=sort_by_scores,
    )
    tl = TimeLimit(time_limit)
    for _ in range(max_evolve_steps):
        sampler.evolve(show_pbar=False, time_limit=tl)
        
        # Get products and calculate max combined score
        products = list(sampler.get_products())
        if not products:
            continue
            
        # Check if we've found a perfect match (combined score of 1.0)
        # This is now based on the combined score (docking*0.9 + tanimoto*0.1)
        max_combined_score = -float('inf')
        for p in products:
            # Get RDKit molecule for docking
            rdkit_mol = None
            try:
                from rdkit import Chem
                rdkit_mol = Chem.MolFromSmiles(p.molecule.smiles)
            except Exception as e:
                print(f"Error converting molecule to RDKit: {str(e)}")
            
            tanimoto_score = p.molecule.sim(input, FingerprintOption.morgan_for_tanimoto_similarity())
            
            # Get docking score if possible
            docking_score = None
            if rdkit_mol is not None:
                from synformer.sampler.analog.state_pool import dock_best_molecule
                docking_score = dock_best_molecule(rdkit_mol)
            
            # Calculate combined score
            if docking_score is not None:
                # Normalize docking score (lower is better, typically -12 to 0)
                normalized_docking_score = max(0, min(1, (-docking_score) / 12.0))
                combined_score = 0.9 * normalized_docking_score + 0.1 * tanimoto_score
            else:
                combined_score = tanimoto_score
            
            max_combined_score = max(max_combined_score, combined_score)
        
        # If we found a perfect match, stop evolving
        if max_combined_score >= 0.99:
            break

    # Get the final dataframe of results
    df = sampler.get_dataframe()[:max_results]
    
    # Print a message if no results were found
    if len(df) == 0:
        print(f"No valid results found for molecule: {input.smiles}")
        # Create a minimal DataFrame with just the target molecule
        df = pd.DataFrame([{
            "target": input.smiles,
            "smiles": "",
            "tanimoto_score": 0.0,
            "docking_score": None,
            "score": 0.0,
            "synthesis": "",
            "num_steps": 0
        }])
    else:
        max_sim = df["score"].max()
        print(f"Best score for {input.smiles}: {max_sim:.3f}")
        
    return df
    
