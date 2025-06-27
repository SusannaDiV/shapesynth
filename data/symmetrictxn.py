from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import re

def analyze_reaction_templates(template_file):
    reaction_counts = defaultdict(int)
    symmetric_counts = defaultdict(int)
    asymmetric_counts = defaultdict(int)
    
    with open(template_file, 'r') as f:
        templates = f.readlines()
    
    for template in templates:
        reactants = template.split('>>')[0].strip()
        num_reactants = reactants.count('.') + 1
        reaction_counts[num_reactants] += 1
        
        if num_reactants == 2:
            try:
                r1, r2 = reactants.split('.')
                if r1 == r2:
                    symmetric_counts[2] += 1
                else:
                    asymmetric_counts[2] += 1
            except:
                continue
                
    print(f"Total templates: {len(templates)}")
    print("\nBreakdown by number of reactants:")
    for n, count in sorted(reaction_counts.items()):
        print(f"{n} reactants: {count}")
    
    print("\nFor 2-reactant reactions:")
    print(f"Symmetric: {symmetric_counts[2]}")
    print(f"Asymmetric: {asymmetric_counts[2]}")

if __name__ == "__main__":
    template_file = "/home/luost_local/sdivita/synformer/data/rxn_templates/comprehensive.txt"
    analyze_reaction_templates(template_file)