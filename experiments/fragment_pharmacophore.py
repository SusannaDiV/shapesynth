import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Draw
from rdkit import RDLogger
import threading
import time
import os
import random
import py3Dmol
from collections import Counter
import re

# Disable RDKit logging
RDLogger.DisableLog('rdApp.*')

# Two hardcoded valid SMILES strings (medium length to avoid processing issues)
SMILES1 = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
SMILES2 = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen

def timeout_handler(smiles, func, args=(), timeout=3):
    """Thread-based timeout handler for molecule processing"""
    result = [None]
    error = [None]
    
    def worker():
        try:
            result[0] = func(*args)
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"Timeout for {smiles}")
        return None
    if error[0] is not None:
        print(f"Error processing {smiles}: {str(error[0])}")
        return None
    return result[0]

def process_mol(mol, smiles):
    """Process molecule without timeout"""
    if mol is None:
        return None
        
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'S' and atom.GetFormalCharge() != 0:
            return None
    
    mol = Chem.AddHs(mol)
    
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        Chem.SanitizeMol(mol)
        
        if AllChem.EmbedMolecule(mol, useRandomCoords=True) != 0:
            return None

        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        return mol
        
    except Exception as e:
        print(f"Error in process_mol: {str(e)}")
        return None

def process_smiles(smiles):
    """Generate a 3D conformer for a given SMILES."""
    # Early rejection criteria
    if any([
        len(smiles) > 180,  # Very long SMILES strings
        smiles.count('(') > 8,  # Too many branches
        smiles.count('@') > 6,  # Too many chiral centers
        smiles.count('1') + smiles.count('2') + smiles.count('3') > 6,  # Too many rings
    ]):
        print(f"Early rejection of complex molecule: {smiles}")
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    return timeout_handler(smiles, process_mol, args=(mol, smiles))

def identify_pharmacophore_features(mol):
    """
    Identify pharmacophore features in a molecule
    
    Returns a dictionary mapping atom indices to feature types:
    - H-bond donor (D)
    - H-bond acceptor (A)
    - Positive ionizable (P)
    - Negative ionizable (N)
    - Hydrophobic (H)
    - Aromatic (R)
    """
    features = {}
    
    # Define SMARTS patterns for pharmacophore features
    smarts_patterns = {
        'D': ['[#7!H0]', '[#8!H0]'],  # H-bond donors (N-H, O-H)
        'A': ['[#7]', '[#8]'],        # H-bond acceptors (N, O)
        'P': ['[+,#7;!$(N~[!#6]);!$(*~[#7,#8,#15,#16])]'],  # Positive ionizable (basic N)
        'N': ['[-,#8;$(*~[#6,#7])]'],  # Negative ionizable (acidic O)
        'H': ['[#6;+0]~[#6;+0]', '[#6]~[F,Cl,Br,I]', '[S;D2;+0]'],  # Hydrophobic (C-C, C-halogen, sulfide)
        'R': ['a5', 'a6']             # Aromatic (5 and 6-membered rings)
    }
    
    # Identify features using SMARTS patterns
    for feature_type, patterns in smarts_patterns.items():
        for pattern in patterns:
            patt = Chem.MolFromSmarts(pattern)
            if patt:
                matches = mol.GetSubstructMatches(patt)
                for match in matches:
                    for atom_idx in match:
                        if atom_idx not in features:
                            features[atom_idx] = []
                        features[atom_idx].append(feature_type)
    
    # Refine features (remove duplicates, handle special cases)
    refined_features = {}
    for atom_idx, feature_list in features.items():
        # Remove duplicates
        unique_features = list(set(feature_list))
        
        # Special case: carboxylic acids are strong H-bond acceptors
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'O' and atom.GetFormalCharge() == -1:
            if 'A' not in unique_features:
                unique_features.append('A')
        
        # Special case: amines can be both donors and positive ionizable
        if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() > 0:
            if 'D' not in unique_features:
                unique_features.append('D')
            if 'P' not in unique_features and atom.GetFormalCharge() >= 0:
                # Check if it's a basic amine (not amide, etc.)
                is_basic = True
                for bond in atom.GetBonds():
                    other_atom = bond.GetOtherAtom(atom)
                    if other_atom.GetSymbol() in ['C', 'S'] and other_atom.GetIsAromatic():
                        is_basic = False
                    if other_atom.GetSymbol() in ['O', 'S'] and bond.GetBondType() == Chem.BondType.DOUBLE:
                        is_basic = False
                if is_basic:
                    unique_features.append('P')
        
        # Special case: aromatic rings are hydrophobic
        if 'R' in unique_features and 'H' not in unique_features:
            unique_features.append('H')
        
        refined_features[atom_idx] = unique_features
    
    return refined_features

def find_rotatable_bonds(mol):
    """Find rotatable bonds in a molecule that can be used for fragmentation"""
    rotatable_bond_smarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    pattern = Chem.MolFromSmarts(rotatable_bond_smarts)
    rotatable_bonds = mol.GetSubstructMatches(pattern)
    return rotatable_bonds

def fragment_molecule(mol):
    """
    Fragment a molecule by breaking rotatable bonds
    
    Returns a list of fragment molecules and their corresponding atom indices in the original molecule
    """
    # Find rotatable bonds
    rotatable_bonds = find_rotatable_bonds(mol)
    
    if not rotatable_bonds:
        print("No rotatable bonds found, returning the whole molecule as a single fragment")
        return [(mol, list(range(mol.GetNumAtoms())))]
    
    # Randomly select one rotatable bond to break
    bond_idx = random.choice(rotatable_bonds)
    bond = mol.GetBondBetweenAtoms(bond_idx[0], bond_idx[1])
    
    # Create a copy of the molecule and break the bond
    rwmol = Chem.RWMol(mol)
    rwmol.RemoveBond(bond_idx[0], bond_idx[1])
    
    # Get the fragments
    fragments = Chem.GetMolFrags(rwmol, asMols=True, sanitizeFrags=True)
    fragment_atom_indices = Chem.GetMolFrags(rwmol)
    
    # Filter out fragments that are too small (less than 3 atoms)
    valid_fragments = []
    for i, (frag, atom_indices) in enumerate(zip(fragments, fragment_atom_indices)):
        if frag.GetNumAtoms() >= 3:
            try:
                # Try to sanitize the fragment
                Chem.SanitizeMol(frag)
                valid_fragments.append((frag, atom_indices))
            except:
                print(f"Failed to sanitize fragment {i}")
    
    # If no valid fragments, return the original molecule
    if not valid_fragments:
        print("No valid fragments found, returning the whole molecule as a single fragment")
        return [(mol, list(range(mol.GetNumAtoms())))]
    
    return valid_fragments

def count_pharmacophore_features(mol, atom_indices=None):
    """
    Count the number of pharmacophore features in a molecule or a subset of atoms
    
    Args:
        mol: RDKit molecule
        atom_indices: List of atom indices to consider (if None, all atoms are considered)
        
    Returns:
        Dictionary with counts of each pharmacophore feature type
    """
    features = identify_pharmacophore_features(mol)
    
    # If atom_indices is provided, filter features to only include those atoms
    if atom_indices is not None:
        filtered_features = {idx: feats for idx, feats in features.items() if idx in atom_indices}
    else:
        filtered_features = features
    
    # Count the number of each feature type
    feature_counts = Counter()
    for atom_idx, feature_list in filtered_features.items():
        for feature in feature_list:
            feature_counts[feature] += 1
    
    return feature_counts

def visualize_fragments(mol, fragments, filename="fragments.html"):
    """
    Visualize the original molecule and its fragments
    
    Args:
        mol: Original RDKit molecule
        fragments: List of (fragment_mol, atom_indices) tuples
        filename: Output HTML file name
    """
    # Create a simple HTML template for visualization
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Molecule Fragments</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <style>
            .mol-container {{
                width: 100%;
                height: 400px;
                position: relative;
                margin-bottom: 20px;
            }}
            .mol-title {{
                font-family: Arial, sans-serif;
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="mol-title">Original Molecule</div>
        <div id="container-original" class="mol-container"></div>
        
        {fragment_divs}
        
        <script>
            $(document).ready(function() {{
                // Original molecule
                let viewerOriginal = $3Dmol.createViewer($("#container-original"));
                let molOriginal = `{mol_original}`;
                viewerOriginal.addModel(molOriginal, "mol");
                viewerOriginal.setStyle({{}}, {{stick: {{radius: 0.2, color: 'gray'}}}});
                viewerOriginal.zoomTo();
                viewerOriginal.render();
                
                {fragment_scripts}
            }});
        </script>
    </body>
    </html>
    """
    
    # Generate HTML for fragment containers
    fragment_divs = ""
    fragment_scripts = ""
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
    
    # Original molecule as molblock
    mol_original = Chem.MolToMolBlock(mol)
    
    # Generate HTML for each fragment
    for i, (frag, _) in enumerate(fragments):
        color = colors[i % len(colors)]
        frag_block = Chem.MolToMolBlock(frag)
        
        # Add container div
        fragment_divs += f"""
        <div class="mol-title">Fragment {i+1}</div>
        <div id="container-frag-{i}" class="mol-container"></div>
        """
        
        # Add script for this fragment
        fragment_scripts += f"""
                // Fragment {i+1}
                let viewerFrag{i} = $3Dmol.createViewer($("#container-frag-{i}"));
                let molFrag{i} = `{frag_block}`;
                viewerFrag{i}.addModel(molFrag{i}, "mol");
                viewerFrag{i}.setStyle({{}}, {{stick: {{radius: 0.2, color: '{color}'}}}}); 
                viewerFrag{i}.zoomTo();
                viewerFrag{i}.render();
        """
    
    # Fill in the template
    html_content = html_template.format(
        fragment_divs=fragment_divs,
        fragment_scripts=fragment_scripts,
        mol_original=mol_original
    )
    
    # Write to file
    with open(filename, 'w') as f:
        f.write(html_content)
    
    print(f"Visualization saved to {filename}")

def load_reaction_templates(filename):
    """
    Load reaction templates from a file
    
    Args:
        filename: Path to the file containing reaction templates
        
    Returns:
        List of reaction templates as SMARTS strings
    """
    with open(filename, 'r') as f:
        templates = f.readlines()
    
    # Remove empty lines and comments
    templates = [t.strip() for t in templates if t.strip() and not t.startswith('#')]
    
    return templates

def apply_reaction_template(reactant1, reactant2, template):
    """
    Apply a reaction template to two reactants
    
    Args:
        reactant1: First reactant molecule (RDKit mol)
        reactant2: Second reactant molecule (RDKit mol)
        template: Reaction template as SMARTS string
        
    Returns:
        Product molecule (RDKit mol) or None if the reaction fails
    """
    try:
        # Create reaction from template
        rxn = AllChem.ReactionFromSmarts(template)
        
        # Run the reaction
        products = rxn.RunReactants((reactant1, reactant2))
        
        # Check if any products were generated
        if not products or len(products) == 0:
            return None
        
        # Get the first product
        product = products[0][0]
        
        # Sanitize the product
        Chem.SanitizeMol(product)
        
        return product
    except Exception as e:
        print(f"Error applying reaction template: {str(e)}")
        return None

def visualize_reaction(reactant1, reactant2, product, filename="reaction.html"):
    """
    Visualize a chemical reaction
    
    Args:
        reactant1: First reactant molecule (RDKit mol)
        reactant2: Second reactant molecule (RDKit mol)
        product: Product molecule (RDKit mol)
        filename: Output HTML file name
    """
    # Create a simple HTML template for visualization
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chemical Reaction</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <style>
            .mol-container {{
                width: 100%;
                height: 400px;
                position: relative;
                margin-bottom: 20px;
            }}
            .mol-title {{
                font-family: Arial, sans-serif;
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .reaction-arrow {{
                font-size: 24px;
                text-align: center;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="mol-title">Reactant 1</div>
        <div id="container-reactant1" class="mol-container"></div>
        
        <div class="mol-title">Reactant 2</div>
        <div id="container-reactant2" class="mol-container"></div>
        
        <div class="reaction-arrow">↓</div>
        
        <div class="mol-title">Product</div>
        <div id="container-product" class="mol-container"></div>
        
        <script>
            $(document).ready(function() {{
                // Reactant 1
                let viewerReactant1 = $3Dmol.createViewer($("#container-reactant1"));
                let molReactant1 = `{reactant1_block}`;
                viewerReactant1.addModel(molReactant1, "mol");
                viewerReactant1.setStyle({{}}, {{stick: {{radius: 0.2, color: 'blue'}}}});
                viewerReactant1.zoomTo();
                viewerReactant1.render();
                
                // Reactant 2
                let viewerReactant2 = $3Dmol.createViewer($("#container-reactant2"));
                let molReactant2 = `{reactant2_block}`;
                viewerReactant2.addModel(molReactant2, "mol");
                viewerReactant2.setStyle({{}}, {{stick: {{radius: 0.2, color: 'green'}}}});
                viewerReactant2.zoomTo();
                viewerReactant2.render();
                
                // Product
                let viewerProduct = $3Dmol.createViewer($("#container-product"));
                let molProduct = `{product_block}`;
                viewerProduct.addModel(molProduct, "mol");
                viewerProduct.setStyle({{}}, {{stick: {{radius: 0.2, color: 'red'}}}});
                viewerProduct.zoomTo();
                viewerProduct.render();
            }});
        </script>
    </body>
    </html>
    """
    
    # Convert molecules to molblocks
    reactant1_block = Chem.MolToMolBlock(reactant1)
    reactant2_block = Chem.MolToMolBlock(reactant2)
    product_block = Chem.MolToMolBlock(product)
    
    # Fill in the template
    html_content = html_template.format(
        reactant1_block=reactant1_block,
        reactant2_block=reactant2_block,
        product_block=product_block
    )
    
    # Write to file
    with open(filename, 'w') as f:
        f.write(html_content)
    
    print(f"Reaction visualization saved to {filename}")

def analyze_functional_groups(mol):
    """
    Analyze the functional groups present in a molecule
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Dictionary with functional group types and their counts
    """
    functional_groups = {
        'carboxylic_acid': '[C,c](=O)[OH]',
        'alcohol': '[C,c][OH]',
        'phenol': 'c[OH]',
        'amine': '[N;!$(N=*);!$(NC=O);!$(NC=S)]',
        'amide': '[NX3;$(NC=O)]',
        'carbonyl': '[C,c]=O',
        'alkene': '[C;!$(C=O)]=C',
        'alkyne': 'C#C',
        'aromatic': 'a',
        'halogen': '[F,Cl,Br,I]',
        'nitrile': 'C#N',
        'nitro': '[N+](=O)[O-]',
        'ether': '[C,c][O][C,c]',
        'ester': '[C,c](=O)[O][C,c]',
        'sulfide': '[C,c][S][C,c]',
        'sulfoxide': '[C,c][S](=O)[C,c]',
        'sulfone': '[C,c][S](=O)(=O)[C,c]',
        'sulfonamide': '[C,c][S](=O)(=O)[N]'
    }
    
    results = {}
    
    for name, smarts in functional_groups.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                results[name] = len(matches)
    
    return results

def filter_compatible_templates(templates, reactant1, reactant2):
    """
    Filter reaction templates that are likely to be compatible with the given reactants
    
    Args:
        templates: List of reaction templates as SMARTS strings
        reactant1: First reactant molecule (RDKit mol)
        reactant2: Second reactant molecule (RDKit mol)
        
    Returns:
        List of filtered templates that are likely to work with the reactants
    """
    # Analyze functional groups in reactants
    groups1 = analyze_functional_groups(reactant1)
    groups2 = analyze_functional_groups(reactant2)
    
    print(f"\nFunctional groups in reactant 1: {groups1}")
    print(f"Functional groups in reactant 2: {groups2}")
    
    # Define keywords for common functional groups
    functional_group_keywords = {
        'carboxylic_acid': ['COOH', 'C(=O)O', 'acid', 'OH1'],
        'alcohol': ['OH', 'alcohol'],
        'phenol': ['OH', 'phenol'],
        'amine': ['NH2', 'NH1', 'amin'],
        'amide': ['NC=O', 'amide'],
        'carbonyl': ['C=O', 'carbonyl', 'aldehyde', 'ketone'],
        'alkene': ['C=C', 'alkene'],
        'alkyne': ['C#C', 'alkyne'],
        'aromatic': ['aromatic', 'aryl', 'ar', 'c1', 'c:'],
        'halogen': ['Cl', 'Br', 'I', 'F', 'halogen'],
        'nitrile': ['C#N', 'nitrile', 'cyano'],
        'nitro': ['NO2', 'nitro'],
        'ether': ['ether', 'OR', 'OC'],
        'ester': ['ester', 'OC=O', 'C(=O)O'],
        'sulfide': ['sulfide', 'thioether', 'SC'],
        'sulfoxide': ['sulfoxide', 'S=O'],
        'sulfone': ['sulfone', 'SO2'],
        'sulfonamide': ['sulfonamide', 'SO2N']
    }
    
    # Combine all functional groups from both reactants
    all_groups = set(groups1.keys()) | set(groups2.keys())
    
    # Create a list of keywords to look for in templates
    keywords = []
    for group in all_groups:
        if group in functional_group_keywords:
            keywords.extend(functional_group_keywords[group])
    
    # Filter templates based on keywords
    filtered_templates = []
    for template in templates:
        # Check if template contains any of the keywords
        if any(keyword in template for keyword in keywords):
            filtered_templates.append(template)
    
    # If no templates match, return a subset of the original templates
    if not filtered_templates:
        print("No templates matched the functional groups. Using a subset of all templates.")
        return random.sample(templates, min(20, len(templates)))
    
    print(f"Filtered {len(filtered_templates)} compatible templates out of {len(templates)} total templates.")
    return filtered_templates

def main():
    parser = argparse.ArgumentParser(description='Fragment molecules and analyze pharmacophore features')
    parser.add_argument('--smiles1', type=str, default=SMILES1, help='SMILES string for first molecule')
    parser.add_argument('--smiles2', type=str, default=SMILES2, help='SMILES string for second molecule')
    parser.add_argument('--rxn_file', type=str, default='experiments/rxn.txt', help='File containing reaction templates')
    args = parser.parse_args()
    
    # Process the molecules
    print(f"Processing molecule 1: {args.smiles1}")
    mol1 = process_smiles(args.smiles1)
    if mol1 is None:
        print("Failed to process molecule 1")
        return
    
    print(f"Processing molecule 2: {args.smiles2}")
    mol2 = process_smiles(args.smiles2)
    if mol2 is None:
        print("Failed to process molecule 2")
        return
    
    # Fragment the molecules
    print("\nFragmenting molecule 1...")
    fragments1 = fragment_molecule(mol1)
    print(f"Generated {len(fragments1)} fragments for molecule 1")
    
    print("\nFragmenting molecule 2...")
    fragments2 = fragment_molecule(mol2)
    print(f"Generated {len(fragments2)} fragments for molecule 2")
    
    # Analyze pharmacophore features in the fragments
    print("\nPharmacophore features in molecule 1 fragments:")
    for i, (frag, atom_indices) in enumerate(fragments1):
        feature_counts = count_pharmacophore_features(mol1, atom_indices)
        total_features = sum(feature_counts.values())
        print(f"  Fragment {i+1}: {feature_counts}, Total: {total_features}")
    
    print("\nPharmacophore features in molecule 2 fragments:")
    for i, (frag, atom_indices) in enumerate(fragments2):
        feature_counts = count_pharmacophore_features(mol2, atom_indices)
        total_features = sum(feature_counts.values())
        print(f"  Fragment {i+1}: {feature_counts}, Total: {total_features}")
    
    # Visualize the fragments
    print("\nVisualizing fragments...")
    visualize_fragments(mol1, fragments1, filename="molecule1_fragments.html")
    visualize_fragments(mol2, fragments2, filename="molecule2_fragments.html")
    
    # Identify the fragment with the most pharmacophore features for each molecule
    print("\nIdentifying fragments with the most pharmacophore features:")
    
    # For molecule 1
    max_features1 = 0
    best_fragment1 = None
    for i, (frag, atom_indices) in enumerate(fragments1):
        feature_counts = count_pharmacophore_features(mol1, atom_indices)
        total_features = sum(feature_counts.values())
        if total_features > max_features1:
            max_features1 = total_features
            best_fragment1 = (i, frag, atom_indices, feature_counts)
    
    if best_fragment1:
        i, frag, atom_indices, feature_counts = best_fragment1
        print(f"  Molecule 1: Fragment {i+1} has the most pharmacophore features:")
        print(f"    Features: {feature_counts}, Total: {max_features1}")
        print(f"    SMILES: {Chem.MolToSmiles(frag)}")
    
    # For molecule 2
    max_features2 = 0
    best_fragment2 = None
    for i, (frag, atom_indices) in enumerate(fragments2):
        feature_counts = count_pharmacophore_features(mol2, atom_indices)
        total_features = sum(feature_counts.values())
        if total_features > max_features2:
            max_features2 = total_features
            best_fragment2 = (i, frag, atom_indices, feature_counts)
    
    if best_fragment2:
        i, frag, atom_indices, feature_counts = best_fragment2
        print(f"  Molecule 2: Fragment {i+1} has the most pharmacophore features:")
        print(f"    Features: {feature_counts}, Total: {max_features2}")
        print(f"    SMILES: {Chem.MolToSmiles(frag)}")
    
    # Load reaction templates
    print("\nLoading reaction templates...")
    templates = load_reaction_templates(args.rxn_file)
    print(f"Loaded {len(templates)} reaction templates")
    
    # Apply a random reaction template to the best fragments
    if best_fragment1 and best_fragment2:
        print("\nApplying random reaction template to the best fragments...")
        
        # Get the best fragments
        _, frag1, _, _ = best_fragment1
        _, frag2, _, _ = best_fragment2
        
        # Filter compatible templates
        filtered_templates = filter_compatible_templates(templates, frag1, frag2)
        
        # Try reaction templates until a valid product is found
        max_attempts = min(20, len(filtered_templates))
        product = None
        used_templates = []
        
        for attempt in range(max_attempts):
            # Select a random template
            template = random.choice(filtered_templates)
            while template in used_templates and len(used_templates) < len(filtered_templates):
                template = random.choice(filtered_templates)
            
            used_templates.append(template)
            
            print(f"  Attempt {attempt+1}: Trying template: {template}")
            
            # Apply the template
            product = apply_reaction_template(frag1, frag2, template)
            
            # Check if a valid product was generated
            if product is not None:
                try:
                    # Sanitize and check if the product is valid
                    Chem.SanitizeMol(product)
                    product_smiles = Chem.MolToSmiles(product)
                    print(f"  Success! Generated valid product: {product_smiles}")
                    
                    # Visualize the reaction
                    print("  Visualizing the reaction...")
                    visualize_reaction(frag1, frag2, product, filename="reaction.html")
                    
                    break
                except Exception as e:
                    print(f"  Error validating product: {str(e)}")
                    product = None
            else:
                print("  Failed to generate a product with this template")
        
        if product is None:
            print("\nFailed to generate a valid product after multiple attempts")
    
    print("\nFragmentation and analysis complete.")

if __name__ == "__main__":
    main() 