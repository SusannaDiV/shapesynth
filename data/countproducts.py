import pickle
import numpy as np
from tqdm import tqdm

def count_possible_reactions(matrix_path, batch_size=10000, pairs_per_template=8000):
    # Load the matrix
    with open(matrix_path, 'rb') as f:
        matrix_obj = pickle.load(f)
    
    matrix = matrix_obj._matrix
    reactions = matrix_obj._reactions
    reactants = matrix_obj._reactants
    
    # Only use first half of building blocks
    n_reactants = len(reactants) // 10
    print(f"\nUsing first {n_reactants:,} building blocks out of {len(reactants):,}")
    
    total_products = 0
    counts = {1: 0, 2: 0, 3: 0, "3+": 0}
    products_by_type = {1: 0, 2: 0, 3: 0, "3+": 0}
    
    print("\nCounting possible reactions...")
    for rxn_idx, rxn in enumerate(tqdm(reactions)):
        rxn_col = matrix[:, rxn_idx][:n_reactants]  # Only use first half of building blocks
        num_reactants = rxn.num_reactants
        counts[num_reactants if num_reactants <= 3 else "3+"] += 1
        
        if num_reactants == 2:
            first_reactants = np.nonzero(rxn_col & 0b01)[0]
            second_reactants = np.nonzero(rxn_col & 0b10)[0]
            '''
            print(f"\nTemplate {rxn_idx} ({rxn.smarts})")
            print(f"First position reactants: {len(first_reactants):,}")
            print(f"Second position reactants: {len(second_reactants):,}")
            '''
            # Calculate total products using median 3.5 products per pair
            total_pairs = len(first_reactants) * len(second_reactants) * 2  # *2 for both orderings
            template_products = total_pairs * 3.5  # Using median products per pair
            
            products_by_type[2] += template_products
            total_products += template_products
    
    print("\nReaction template counts:")
    for n_reactants, count in counts.items():
        print(f"{n_reactants} reactant(s): {count:,} templates")
    
    print("\nEstimated unique products by reaction type:")
    for n_reactants, count in products_by_type.items():
        print(f"{n_reactants} reactant(s): {count:,.0f} possible unique products")
    
    return total_products, products_by_type

def probability_of_k_matches(precomputed_count, training_generations, total_possible, k):
    from scipy.stats import binom
    p_hit = precomputed_count / total_possible
    return binom.sf(k-1, training_generations, p_hit)

if __name__ == "__main__":
    matrix_path = "/home/luost_local/sdivita/synformer/data/processed/comp_2048/matrix.pkl"
    total, products_by_type = count_possible_reactions(matrix_path)
    print(f"\nTotal unique products: {total:,.0f}")
    
    precomputed = 1_000_000  # We precompute shapes for 800K SMILES
    training_gens = 189_000_000  # We generate 189M during training
    total_possible = products_by_type[2]  # Total possible unique SMILES
    
    thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
    print(f"\nWith:")
    print(f"- {precomputed:,} precomputed SMILES")
    print(f"- {training_gens:,} training generations")
    print(f"- {total_possible:,.0f} total possible SMILES")
    print(f"\nProbabilities of matching at least:")
    for threshold in thresholds:
        k = int(precomputed * threshold)
        p = probability_of_k_matches(precomputed, training_gens, total_possible, k)
        print(f"- {threshold*100:.0f}% ({k:,} SMILES): {p:.2%}")
