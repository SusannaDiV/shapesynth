import argparse
import pathlib
import pandas as pd
from rdkit import Chem

from synformer.chem.mol import Molecule, read_mol_file
from synformer.sampler.analog.docking_parallel import run_docking_sampling

def main():
    parser = argparse.ArgumentParser(description="Sample molecules optimized for docking scores")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input molecule file (CSV with SMILES column)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output file path")
    parser.add_argument("--receptor", "-r", type=str, required=True, help="Path to receptor PDBQT file")
    parser.add_argument("--center", "-c", type=float, nargs=3, required=True, help="Docking box center (x y z)")
    parser.add_argument("--box-size", "-b", type=float, nargs=3, required=True, help="Docking box size (x y z)")
    parser.add_argument("--qvina-path", type=str, default="qvina2.1", help="Path to QVina executable")
    parser.add_argument("--obabel-path", type=str, default="obabel", help="Path to OpenBabel executable")
    parser.add_argument("--search-width", type=int, default=24, help="Search width")
    parser.add_argument("--exhaustiveness", type=int, default=64, help="Exhaustiveness")
    parser.add_argument("--time-limit", type=int, default=180, help="Time limit in seconds")
    parser.add_argument("--max-results", type=int, default=100, help="Maximum number of results")
    parser.add_argument("--max-evolve-steps", type=int, default=12, help="Maximum number of evolution steps")
    parser.add_argument("--smiles-col", type=str, default="SMILES", help="Name of SMILES column in input file")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs for docking (-1 for all cores)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing molecules")
    
    args = parser.parse_args()
    
    # Load input molecules
    mols = read_mol_file(args.input, smiles_col=args.smiles_col)
    
    # Process molecules in batches
    all_results = []
    for i in range(0, len(mols), args.batch_size):
        batch = mols[i:i+args.batch_size]
        print(f"\nProcessing batch {i//args.batch_size + 1}/{(len(mols) + args.batch_size - 1)//args.batch_size}")
        
        for j, mol in enumerate(batch):
            print(f"\nProcessing molecule {i+j+1}/{len(mols)}: {mol.smiles}")
            
            # Run docking-based sampling
            results = run_docking_sampling(
                input=mol,
                model_path=pathlib.Path(args.model_path),
                mat_path=pathlib.Path(args.model_path).parent / "matrix.pkl",
                fpi_path=pathlib.Path(args.model_path).parent / "fpindex.pkl",
                receptor_path=args.receptor,
                center=tuple(args.center),
                box_size=tuple(args.box_size),
                qvina_path=args.qvina_path,
                obabel_path=args.obabel_path,
                search_width=args.search_width,
                exhaustiveness=args.exhaustiveness,
                time_limit=args.time_limit,
                max_results=args.max_results,
                max_evolve_steps=args.max_evolve_steps,
                n_jobs=args.n_jobs,
            )
            
            if not results.empty:
                all_results.append(results)
    
    # Combine results and save
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(args.output, index=False)
        print(f"\nSaved {len(combined_df)} results to {args.output}")
    else:
        print("\nNo results generated")

if __name__ == "__main__":
    main()    return pd.DataFrame()