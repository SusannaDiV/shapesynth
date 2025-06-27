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
'O=C1c2ccccc2C(=O)'
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


def inner_smi2coords(smi: str):
    """Generate 10 3D conformers and 1 2D conformer for a SMILES string."""
    cnt = 10  # number of 3D conformers to generate
    
    # Generate 3D conformers
    coordinate_list = smi2_3Dcoords(smi, cnt)
    
    # Add one 2D conformer
    coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    
    # Get atom information (only needs to be done once)
    mol = Chem.MolFromSmiles(smi)
    atoms = []
    for atom in mol.GetAtoms():
        atoms.append(atom.GetSymbol())
    
    # Package everything
    data = {
        'atoms': atoms,
        'coordinates': coordinate_list
    }
    return pickle.dumps(data)


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
    start_time = time.time()
    
    # Test SMILES
    test_smi = 'CCCS(=O)(=O)N1CCCC(C(=O)OC(c2nnc([C@@H]3CC[C@H](CO)O3)o2)(C(C)c2nc(CC(C)NC(=O)OC(C)(C)C)nn2C(=O)CNC(=O)c2ccco2)C(CNC(=O)C2CCOCC2)c2nc(CSc3ccc(C(C)C)cc3)nn2C(=O)c2cc3c(s2)CCCCCC3)C1'
    
    print("Generating conformers and computing representations...")
    result = inner_smi2coords(test_smi)
    data = pickle.loads(result)
    
    os.makedirs('./results', exist_ok=True)
    
    conformer_data = {
        'SMILES': test_smi,
        'num_conformers': len(data['coordinates']),
        'num_atoms': len(data['atoms']),
        'atoms': data['atoms'],
        'coordinates': data['coordinates']
    }
    
    output_path = './results/conformer_data.csv'
    pd.DataFrame([conformer_data]).to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()

    
