from powdiffrac.simulation import Powder

def generate_varied_pattern(it, pymatgen_structure, SimClass, max_strain):
    powder = Powder(
        pymatgen_structure,
        vary_strain=True,
        max_strain=max_strain
    )
    powder.roll_variances()
    return SimClass.get_simulation(powder.strained_struct)