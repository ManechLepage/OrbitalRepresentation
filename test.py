from pyscf import gto, scf

# Step 1: Define the molecule
mol = gto.M(
    atom=[
        ["Ca", (0.0, 0.0, 0.0)],  # Hydrogen at origin
    ],
    basis="sto-3g",  # Basis set
    unit="angstrom",  # Units
)

# Step 2: Build the Hartree-Fock object
mf = scf.RHF(mol)



# Step 6: Perform SCF calculation
mf.kernel()

mf.analyze()