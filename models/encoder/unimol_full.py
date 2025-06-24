import os
import numpy as np
import pandas as pd
import lmdb
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import pickle
import glob
from multiprocessing import Pool
from collections import defaultdict
import time
import shutil
smi_list = [
'O=C1c2ccccc2C(=O)c2c1ccc(C(=O)n1nc3c4c(cccc41)C(=O)c1ccccc1-3)c2[N+](=O)[O-]'
]

def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2_3Dcoords(smi,cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi) 
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def inner_smi2coords(content):
    smi = content
    cnt = 10 # conformer num,all==11, 10 3d + 1 2d

    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
        print("atom num >400,use 2D coords",smi)
    else:
        coordinate_list = smi2_3Dcoords(smi,cnt)
        # add 2d conf
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 
    return pickle.dumps({'atoms': atoms, 'coordinates': coordinate_list, 'smi': smi }, protocol=-1)


def smi2coords(content):
    try:
        return inner_smi2coords(content)
    except:
        print("failed smiles: {}".format(content[0]))
        return None


def write_lmdb(smiles_list, job_name, seed=42, outpath='./results', nthreads=8):
    os.makedirs(outpath, exist_ok=True)
    output_name = os.path.join(outpath,'{}.lmdb'.format(job_name))
    try:
        os.remove(output_name)
    except:
        pass
    env_new = lmdb.open(
        output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    with Pool(nthreads) as pool:
        i = 0
        for inner_output in tqdm(pool.imap(smi2coords, smiles_list)):
            if inner_output is not None:
                txn_write.put(f'{i}'.encode("ascii"), inner_output)
                i += 1
        print('{} process {} lines'.format(job_name, i))
        txn_write.commit()
        env_new.close()
        
def main():
    seed = 42
    job_name = 'get_mol_repr'   # replace to your custom name
    data_path = './results'  # replace to your data path
    import subprocess
    # Update path to checkpoint - you'll need to provide the correct path to the model weights
    weight_path = '/home/luost_local/sdivita/synformer/data/mol_pre_no_h_220816.pt'  # UPDATE THIS PATH
    dict_source = '/home/luost_local/sdivita/synformer/synformer/models/encoder/Uni-Mol/unimol/example_data/molecule/dict.txt'  # UPDATE THIS PATH
    if not os.path.exists(dict_source):
        raise FileNotFoundError(f"Dictionary file not found at: {dict_source}")

    only_polar = 0  # no h
    dict_name = 'dict.txt'
    batch_size = 1
    conf_size = 11  # default 10 3d + 1 2d
    results_path = data_path   # replace to your save path

    os.makedirs(data_path, exist_ok=True)
    start_time = time.time()

    write_lmdb(smi_list, job_name=job_name, seed=seed, outpath=data_path)

    
    # Copy dictionary file
    shutil.copy(dict_source, os.path.join(data_path, dict_name))

    # Run inference using subprocess
    
    cmd = [
        'python', '/home/luost_local/sdivita/synformer/synformer/models/encoder/Uni-Mol/unimol/unimol/infer.py',
        '--user-dir', '/home/luost_local/sdivita/synformer/synformer/models/encoder/Uni-Mol/unimol/unimol',
        data_path,
        '--valid-subset', job_name,
        '--results-path', results_path,
        '--num-workers', '8',
        '--ddp-backend=c10d',
        '--batch-size', str(batch_size),
        '--task', 'unimol',
        '--loss', 'unimol_infer',
        '--arch', 'unimol_base',
        '--path', weight_path,
        '--only-polar', str(only_polar),
        '--dict-name', dict_name,
        '--conf-size', str(conf_size),
        '--log-interval', '50',
        '--log-format', 'simple',
        '--random-token-prob', '0',
        '--leave-unmasked-prob', '1.0',
        '--mode', 'infer'
    ]
    
    result = subprocess.run(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Inference failed with error:")
        print(result.stderr)
        return

    # Process results
    pkl_files = glob.glob(f'{results_path}/*_{job_name}.out.pkl')
    if not pkl_files:
        print("No output pickle files found!")
        return
        
    get_csv_results(pkl_files[0], results_path)
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")



def get_csv_results(predict_path, results_path):
    predict = pd.read_pickle(predict_path)
    mol_repr_dict = defaultdict(list)
    atom_repr_dict = defaultdict(list)
    pair_repr_dict = defaultdict(list)
    for batch in predict:
        sz = batch["bsz"]
        for i in range(sz):
            smi = batch["data_name"][i]
            mol_repr_dict[smi].append(batch["mol_repr_cls"][i])
            atom_repr_dict[smi].append(batch["atom_repr"][i])
            pair_repr_dict[smi].append(batch["pair_repr"][i])
    # get mean repr for each molecule with multiple conf
    smi_list, avg_mol_repr_list, avg_atom_repr_list, avg_pair_repr_list = [], [], [], []
    for smi in mol_repr_dict.keys():
        smi_list.append(smi)
        avg_mol_repr_list.append(np.mean(mol_repr_dict[smi], axis=0))
        avg_atom_repr_list.append(np.mean(atom_repr_dict[smi], axis=0))
        avg_pair_repr_list.append(np.mean(pair_repr_dict[smi], axis=0))
    predict_df = pd.DataFrame({
    "SMILES": smi_list,
    "mol_repr": avg_mol_repr_list,
    "atom_repr": avg_atom_repr_list,
    "pair_repr": avg_pair_repr_list
    })
    output_path = os.path.join(results_path, 'molecular_representations.csv')
    predict_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    main()
