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
        mixture_weight: float = 0.8,
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
        self._mixture_weight = mixture_weight
        
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

        self._fpindex: FingerprintIndex = pickle.load(open("/workspace/data/processed/comp_2048/fpindex.pkl", "rb"))
        self._rxn_matrix: ReactantReactionMatrix = pickle.load(open("/workspace/data/processed/comp_2048/matrix.pkl", "rb"))

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
            factor=self._state_pool_opt.get("factor", 32),
            max_active_states=self._state_pool_opt.get("max_active_states", 256),
            sort_by_score=self._state_pool_opt.get("sort_by_score", True),
            shape_patches_path=self._shape_patches_path,
            receptor_path=self._receptor_path,
            receptor_center=self._receptor_center,
            mixture_weight=self._mixture_weight,
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
        mixture_weight: float = 0.8,
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
        self._mixture_weight = mixture_weight
        
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

        self._fpindex: FingerprintIndex = pickle.load(open("/workspace/data/processed/comp_2048/fpindex.pkl", "rb"))
        self._rxn_matrix: ReactantReactionMatrix = pickle.load(open("/workspace/data/processed/comp_2048/matrix.pkl", "rb"))

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
            factor=self._state_pool_opt.get("factor", 32),
            max_active_states=self._state_pool_opt.get("max_active_states", 256),
            sort_by_score=self._state_pool_opt.get("sort_by_score", True),
            shape_patches_path=self._shape_patches_path,
            receptor_path=self._receptor_path,
            receptor_center=self._receptor_center,
            mixture_weight=self._mixture_weight,
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
    max_retries_on_no_results: int = 1,
    initial_desert_mixture_weight: float = 0.8,
    retry_desert_mixture_weight: float = 0.2,
) -> None:
    num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
    output.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries_on_no_results + 1):
        print(f"Starting sampling attempt {attempt + 1} of {max_retries_on_no_results + 1}...")
        
        current_mixture_weight = initial_desert_mixture_weight if attempt == 0 else retry_desert_mixture_weight
        if use_desert:
            print(f"Attempt {attempt + 1}: Using DESERT mixture_weight = {current_mixture_weight}")

        # Worker options dictionary
        worker_constructor_options = {
            "model_path": model_path,
            "state_pool_opt": {
                "factor": search_width,
                "max_active_states": exhaustiveness,
                "sort_by_score": sort_by_scores,
            },
            "max_evolve_steps": 12,
            "max_results": 100,
            "time_limit": time_limit,
            "use_desert": use_desert,
            "desert_model_path": desert_model_path,
            "vocab_path": vocab_path,
            "smiles_checkpoint_path": smiles_checkpoint_path,
            "shape_patches_path": shape_patches_path,
            "receptor_path": receptor_path,
            "receptor_center": receptor_center,
            "mixture_weight": current_mixture_weight if use_desert else None,
        }

        pool = WorkerPool(
            gpu_ids=list(range(num_gpus)),
            num_workers_per_gpu=num_workers_per_gpu,
            task_qsize=task_qsize,
            result_qsize=result_qsize,
            **worker_constructor_options,
        )

        total = len(input)
        if total == 0:
            print("Input list of molecules is empty. Nothing to process.")
            pool.end() # End the created pool
            if output.exists(): # Clean up if output file exists
                try:
                    output.unlink()
                except OSError:
                    pass # Ignore if unlinking fails
            return

        for mol in input:
            pool.submit(mol)

        collected_dfs_for_this_attempt: list[pd.DataFrame] = []
        pbar_desc = f"Attempt {attempt + 1}/{max_retries_on_no_results + 1} - Fetching results"
        
        all_tasks_fetched_successfully = True
        for i in tqdm(range(total), desc=pbar_desc, dynamic_ncols=True):
            try:
                # Using a timeout related to task's own time_limit could be beneficial
                # For now, matches original blocking fetch.
                mol_task, df_result = pool.fetch(block=True, timeout=None)
                
                # Filter for results with actual smiles based on original logic
                df_with_results = df_result[df_result["smiles"] != ""]
                if not df_with_results.empty:
                    collected_dfs_for_this_attempt.append(df_with_results)
            except Exception as e: # Includes potential queue.Empty on timeout, or other errors
                print(f"Error or timeout fetching result for task {i+1}/{total} on attempt {attempt + 1}: {e}")
                all_tasks_fetched_successfully = False
                # Depending on severity, might want to break or ensure pool is properly ended later.

        pool.end() # Shutdown workers for this attempt

        if not collected_dfs_for_this_attempt:
            print(f"Attempt {attempt + 1} completed: No valid results found for any input molecules.")
            if output.exists(): # Clear any previous output file if this attempt failed
                try:
                    output.unlink()
                    print(f"Cleared existing output file: {output}")
                except OSError as e:
                    print(f"Error clearing existing output file {output}: {e}")

            if attempt < max_retries_on_no_results:
                print(f"Retrying... ({max_retries_on_no_results - attempt} retries left)")
            else:
                print("Max retries reached. Still no valid results after all attempts.")
                # The original print "No valid results found for any input molecules." is covered.
                return # Exit the function
        else:
            # Successfully got some results in this attempt
            print(f"Attempt {attempt + 1} successful. Writing {len(collected_dfs_for_this_attempt)} collected DataFrames to {output}...")
            try:
                with open(output, "w") as f:
                    header_written_to_file = False
                    for df_piece in collected_dfs_for_this_attempt:
                        is_header_for_this_piece = not header_written_to_file
                        df_piece.to_csv(f, float_format="%.3f", index=False, header=is_header_for_this_piece)
                        if is_header_for_this_piece:
                            header_written_to_file = True
                print(f"Results successfully written to {output}.")
            except IOError as e:
                print(f"Error writing results to {output}: {e}")
                # Decide if to proceed with stats if writing failed
            
            df_merge = pd.concat(collected_dfs_for_this_attempt, ignore_index=True)
            
            if df_merge.empty:
                print("Warning: Collected DataFrames were non-empty, but merged DataFrame is empty. No stats to calculate.")
            else:
                print("Calculating and printing statistics...")
                try:
                    # Ensure 'target' and 'score' columns exist before groupby and idxmax
                    if "target" in df_merge.columns and "score" in df_merge.columns:
                        # Handle cases where groupby might be empty or idxmax returns empty series
                        if not df_merge.empty:
                             grouped_max_score_indices = df_merge.groupby("target")["score"].idxmax(skipna=True)
                             if not grouped_max_score_indices.empty:
                                 sum_stats = df_merge.loc[grouped_max_score_indices].select_dtypes(include="number").sum()
                                 if total > 0:
                                     print("Average stats for best score per target:", sum_stats / total)
                                 else: # Should have been caught by total == 0 earlier
                                     print("No input tasks to calculate average stats for.")
                             else:
                                 print("Could not determine best scores per target (e.g., all scores are NaN or no targets).")
                        else:
                            print("Merged DataFrame is empty, skipping sum stats.")

                    else:
                        print("Missing 'target' or 'score' columns in merged DataFrame, cannot calculate sum stats.")

                    count_success = len(df_merge["target"].unique()) if "target" in df_merge.columns else 0
                    if total > 0:
                        print(f"Success rate: {count_success}/{total} = {count_success / total:.3f}")
                    else:
                         print(f"Success rate: {count_success}/0 (No input tasks)")

                    high_score_targets: set[str] = set()
                    if "score" in df_merge.columns and "target" in df_merge.columns:
                        for _, row in df_merge.iterrows():
                            # Ensure row["score"] is not NaN before comparison
                            if pd.notna(row["score"]) and row["score"] >= 0.99:
                                high_score_targets.add(row["target"])
                    count_high_score = len(high_score_targets)
                    if total > 0:
                        print(f"High score rate (score >= 0.99): {count_high_score}/{total} = {count_high_score / total:.3f}")
                    else:
                        print(f"High score rate: {count_high_score}/0 (No input tasks)")
                except Exception as e:
                    print(f"An error occurred during statistics calculation: {e}")

            return # Exit function after successful attempt and processing

    # Fallback if loop finishes without returning (should not happen with current logic)
    print("Exited retry loop unexpectedly.")


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
    initial_desert_mixture_weight: float = 0.8,
    retry_desert_mixture_weight: float = 0.2,
) -> None:
    num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
    
    # Determine mixture weight (although this function doesn't have retries, so it will always be initial)
    # For consistency, we can set it, though it won't change.
    current_mixture_weight = initial_desert_mixture_weight
    if use_desert:
        print(f"Using DESERT mixture_weight = {current_mixture_weight}")

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
        mixture_weight=current_mixture_weight if use_desert else None,
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
    initial_desert_mixture_weight: float = 0.8,
    retry_desert_mixture_weight: float = 0.2,
) -> None:
    num_gpus = num_gpus if num_gpus > 0 else _count_gpus()

    # Determine mixture weight (although this function doesn't have retries, so it will always be initial)
    current_mixture_weight = initial_desert_mixture_weight
    if use_desert:
        print(f"Using DESERT mixture_weight = {current_mixture_weight}")
        
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
        mixture_weight=current_mixture_weight if use_desert else None,
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
    desert_mixture_weight: float = 0.8,
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
        mixture_weight=desert_mixture_weight if use_desert else None,
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
    
