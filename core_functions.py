import functools
import inspect
import os
import sys
from datetime import datetime
from typing import Dict, TextIO, Any, Callable
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# Global variable to track the last call signature
last_call_signature = None

# Force field atom/bead types as per https://doi.org/10.1016/j.matdes.2024.112831 (Table 4).
force_field_types = {
    "C-H1-SP2-2-True-True": "A",
    "O-H0-SP2-1-False-False": "B",
    "C-H1-SP3-3-True-False": "C",
    "C-H2-SP3-2-True-False": "D",
    "N-H0-SP2-2-True-True": "E",
    "S-H0-SP2-2-True-True": "F",
    "O-H0-SP2-2-True-False": "G",
    "O-H1-SP2-1-False-False": "H",
    "S-H0-SP3-3-True-False": "I",
    "C-H1-SP3-3-False-False": "J",
    "C-H3-SP3-1-False-False": "K",
    "C-H2-SP3-2-False-False": "L",
    "N-H1-SP2-2-True-True": "M",
    "C-H0-SP3-4-True-False": "N",
    "C-H0-SP2-3-True-False": "O",
    "C-H1-SP2-2-True-False": "P",
    "C-H0-SP2-3-True-True": "Q",
}

# Force field atom/bead types as per https://doi.org/10.1016/j.matdes.2024.112831 (Table 4).
force_field_masses = {
    "A": 13.01864,
    "B": 15.9994,
    "C": 13.01864,
    "D": 14.02658,
    "E": 14.0067,
    "F": 32.065,
    "G": 15.9994,
    "H": 17.00734,
    "I": 32.065,
    "J": 13.01864,
    "K": 15.03452,
    "L": 14.02658,
    "M": 15.01464,
    "N": 12.0107,
    "O": 12.0107,
    "P": 13.01864,
    "Q": 12.0107,
}

# Force field atom/bead types as per https://doi.org/10.1016/j.matdes.2024.112831 (Table 4).
force_field_colors = {
    "A": (0.1216, 0.4667, 0.7059, 1.0),
    "B": (0.6824, 0.7804, 0.9098, 1.0),
    "C": (1.0, 0.498, 0.0549, 1.0),
    "D": (1.0, 0.7333, 0.4706, 1.0),
    "E": (0.1725, 0.6275, 0.1725, 1.0),
    "F": (0.5961, 0.8745, 0.5412, 1.0),
    "G": (0.8392, 0.1529, 0.1569, 1.0),
    "H": (1.0, 0.5961, 0.5882, 1.0),
    "I": (0.5804, 0.4039, 0.7412, 1.0),
    "J": (0.7725, 0.6902, 0.8353, 1.0),
    "K": (0.549, 0.3373, 0.2941, 1.0),
    "L": (0.7686, 0.6118, 0.5804, 1.0),
    "M": (0.8902, 0.4667, 0.7608, 1.0),
    "N": (0.9686, 0.7137, 0.8235, 1.0),
    "O": (0.498, 0.498, 0.498, 1.0),
    "P": (0.7804, 0.7804, 0.7804, 1.0),
    "Q": (0.7373, 0.7412, 0.1333, 1.0),
}


def initialize_log_file(
    filename: str = "calls_log.txt",
):
    """
    Initializes a log file with a header containing a timestamp, title, description,
    and optional additional information including non-native Python libraries,
    Python version, Python executable location, and current working directory.

    Parameters:
    - title (str): The title of the log.
    - description (str): A brief description of the log.
    - additional_info (dict): Optional dictionary containing additional information to include in the header.
    - filename (str): The name of the log file. Defaults to 'calls_log.txt'.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Getting Python details
    python_version = sys.version
    python_executable = sys.executable
    current_working_directory = os.getcwd()

    header = [
        "Log Initialization",
        f"Timestamp: {current_time}",
        f"Current Python Version: {python_version.split()[0]}",
        f"Current Python Executable: {python_executable}",
        f"Current Working Directory: {current_working_directory}",
        "\n--- Log Start ---\n",
    ]

    with open(filename, "w") as f:
        f.write("\n".join(header))


def log_message_to_file(message: str, printmsg: bool = False) -> None:
    """
    Logs a custom message to 'calls_log.txt', prepending it with the current timestamp.

    This function retrieves the current timestamp, formats it together with the provided message,
    and appends this information to the file named 'calls_log.txt'. If the file does not exist,
    it will be created. The timestamp is formatted as 'YYYY-MM-DD_HH:MM:SS'.

    :param message: The text message to log.
    :param printmsg: Whether to also print message to command line. Defaults to False.
    :type printmsg: bool
    :type message: str
    """

    # Get the current timestamp
    now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Construct the log message with timestamp
    formatted_message = f"{now} | {message}\n"

    # Open the log file and append the message
    with open("calls_log.txt", "a") as log_file:
        log_file.write(formatted_message)
    if printmsg:
        print(formatted_message)


def track_call_depth(func: Callable) -> Callable:
    """
    Decorator that logs calls to the decorated function with timestamp and call signature.
    Differentiates consecutive calls to different functions, appending details to `calls_log.txt`.
    Utilizes `inspect` for call signature and `datetime` for timestamps.

    :param func: Function to decorate.
    :type func: Callable
    :return: Wrapper function with logging capability.
    :rtype: Callable
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global last_call_signature

        # Construct the current call signature
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename).replace(".py", "")
        class_name = ""
        if "self" in frame.f_locals:
            class_name = frame.f_locals["self"].__class__.__name__ + "."
        current_call_signature = f"{filename}.{class_name}{func.__name__}"

        # Get the current timestamp
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # Check if the current call signature matches the last one
        if current_call_signature != last_call_signature:
            # Construct the log message
            log_message = f"{now} | {current_call_signature}\n"

            # Open the log file and append the log message
            with open("calls_log.txt", "a") as log_file:
                log_file.write(log_message)

            last_call_signature = current_call_signature

        # Call the original function
        return func(*args, **kwargs)

    return wrapper


@track_call_depth
def read_pdb_file(filename: str) -> list:
    """
    Reads the contents of a PDB file and returns it as a list of lines.

    :param filename: The path to the PDB file.
    :type filename: str
    :return: A list containing the lines of the file.
    :rtype: list
    """
    with open(filename, "r") as file:
        lines = file.read().splitlines()
    return lines


@track_call_depth
def extract_molecules(pdb_lines: list) -> list:
    """
    Extracts and groups molecules from a list of PDB file lines into separate blocks.

    Parses through the given list of lines from a PDB file, identifying and grouping lines
    related to individual molecules. Each molecule block starts with the file's header and includes
    lines describing the molecule's atoms and connections. Blocks are delineated by the presence of
    "HETATM" lines for atom information and "CONECT" lines for simple bonding connectivity.
    The parsing stops upon reaching the "MASTER" record, indicating the end
    of relevant molecule data.

    :param pdb_lines: The lines of a PDB file.
    :type pdb_lines: list
    :return: A list of molecule blocks, each represented as a list of lines.
    :rtype: list
    """
    pdb_molecule_blocks = []
    molecule = []
    header = pdb_lines[0]
    new_molecule = True
    for line in pdb_lines[1:]:
        if line.startswith("MASTER"):
            break
        if line.startswith("HETATM") and new_molecule:
            if molecule:
                pdb_molecule_blocks.append(molecule)
            molecule = [header]
            new_molecule = False
        molecule.append(line)
        if line.startswith("CONECT"):
            new_molecule = True
    if molecule:
        pdb_molecule_blocks.append(molecule)
    return pdb_molecule_blocks


@track_call_depth
def extract_pdb_box_dimensions(pdb_header: str) -> Dict[str, float]:
    """
    Extracts box dimensions from PDB header.

    Parameters:
    - pdb_header (str): PDB header containing box dimensions.

    Returns:
    - dimensions (Dict[str, float]): Dictionary containing box dimensions.
    """
    split_line = pdb_header.split()
    return {
        "a": float(split_line[1]),
        "b": float(split_line[2]),
        "c": float(split_line[3]),
        "box.alpha": float(split_line[4]),
        "box.beta": float(split_line[5]),
        "box.gamma": float(split_line[6]),
    }


@track_call_depth
def compute_atom_label(atom) -> str:
    """
    Compute a detailed label for each atom excluding hydrogens and without neighbor information.

    :param atom: The atom for which to compute the label.
    :type atom:
    :return: A string representing the detailed label of the atom.
    :rtype: str
    """
    n_hydrogens = atom.GetTotalNumHs()
    atom_type = atom.GetSymbol()
    hybridization = str(atom.GetHybridization())
    degree = atom.GetDegree()
    is_in_ring = atom.IsInRing()
    is_aromatic = atom.GetIsAromatic()  # check if the atom is aromatic

    atom_label = f"{atom_type}-H{n_hydrogens}-{hybridization}-{degree}-{is_in_ring}-{is_aromatic}"
    return atom_label


@track_call_depth
def generate_atom_info_list(
    mol: Chem.Mol,
) -> tuple[list[tuple[int, str]], list[Chem.Atom]]:
    """
    Generate a list of atom labels for the molecule excluding hydrogens.

    :param mol: The molecule for which to generate atom labels.
    :type mol: Chem.Mol
    :return: A tuple containing a list of atom index and label pairs, and the list of atoms.
    :rtype: tuple[list[tuple[int, str]], list[Chem.Atom]]
    """
    atoms = mol.GetAtoms()
    atom_info_list = []
    for idx, atom in enumerate(atoms, start=1):
        if atom.GetSymbol() != "H":
            atom_label = compute_atom_label(atom)
            atom_info_list.append((idx, atom_label))
    return atom_info_list, atoms


@track_call_depth
def get_atom_types(
    atom_info_list: list[tuple[int, str]], used_atom_types: set[str]
) -> tuple[dict[int, str], set[str]]:
    """
    Map atom labels to force field atom types and update the set of used atom types.

    :param atom_info_list: A list of tuples where each tuple contains an atom index and its corresponding label.
    :type atom_info_list: list[tuple[int, str]]
    :param used_atom_types: A set of previously used atom types.
    :type used_atom_types: set[str]
    :return: A tuple containing a dictionary mapping atom indices to their force field types, and the updated set of used atom types.
    :rtype: tuple[dict[int, str], set[str]]
    :raises ValueError: If an atom label is not found in the force field types.
    """
    atom_types = {}
    for idx, atom_label in atom_info_list:
        if atom_label in force_field_types:
            atom_type = force_field_types[atom_label]
            atom_types[idx - 1] = atom_type
            used_atom_types.add(atom_type)
        else:
            raise ValueError(f"Atom label {atom_label} not found in force field types")
    return atom_types, used_atom_types


@track_call_depth
def process_bonds(
    bonds: list[Chem.Bond],
    atom_types: dict[int, str],
    bond_type_to_int: dict[tuple[str, str], int],
    bond_type_index: int,
) -> tuple[list[tuple[int, int, int]], dict[tuple[str, str], int], int]:
    """
    Process bonds to determine bond types and create a mapping of bond type keys to integers.

    :param bonds: A list of bonds in the molecule.
    :type bonds: list[Chem.Bond]
    :param atom_types: A dictionary mapping atom indices to their force field types.
    :type atom_types: dict[int, str]
    :param bond_type_to_int: A dictionary mapping bond type keys (tuples of atom types) to integers.
    :type bond_type_to_int: dict[tuple[str, str], int]
    :param bond_type_index: The current index to be assigned to a new bond type.
    :type bond_type_index: int
    :return: A tuple containing a list of bond types with their atom indices, an updated bond type to integer mapping, and the next bond type index.
    :rtype: tuple[list[tuple[int, int, int]], dict[tuple[str, str], int], int]
    """
    bond_types = []
    for bond in bonds:
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        atom_type_1 = atom_types[begin_atom_idx]
        atom_type_2 = atom_types[end_atom_idx]
        bond_type_key = tuple(sorted((atom_type_1, atom_type_2)))

        if bond_type_key not in bond_type_to_int:
            bond_type_to_int[bond_type_key] = bond_type_index
            bond_type_index += 1

        bond_type = bond_type_to_int[bond_type_key]
        bond_types.append((bond_type, begin_atom_idx, end_atom_idx))
    return bond_types, bond_type_to_int, bond_type_index


@track_call_depth
def process_angles(
    bonds: list[Chem.Bond],
    atom_types: dict[int, str],
    angle_type_to_int: dict[tuple[str, str, str], int],
    angle_type_index: int,
) -> tuple[list[tuple[int, int, int, int]], dict[tuple[str, str, str], int], int]:
    """
    Process angles formed by bonds and determine angle types, creating a mapping of angle type keys to integers.

    :param bonds: A list of bonds in the molecule.
    :type bonds: list[Chem.Bond]
    :param atom_types: A dictionary mapping atom indices to their force field types.
    :type atom_types: dict[int, str]
    :param angle_type_to_int: A dictionary mapping angle type keys (tuples of atom types) to integers.
    :type angle_type_to_int: dict[tuple[str, str, str], int]
    :param angle_type_index: The current index to be assigned to a new angle type.
    :type angle_type_index: int
    :return: A tuple containing a list of angle types with their atom indices, an updated angle type to integer mapping, and the next angle type index.
    :rtype: tuple[list[tuple[int, int, int, int]], dict[tuple[str, str, str], int], int]
    """
    angle_types = []
    for bond in bonds:
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        for neighbor in begin_atom.GetNeighbors():
            if neighbor.GetIdx() != end_atom.GetIdx() and neighbor.GetSymbol() != "H":
                angle_atoms = [
                    neighbor.GetIdx(),
                    begin_atom.GetIdx(),
                    end_atom.GetIdx(),
                ]
                angle_atom_types = (
                    atom_types[neighbor.GetIdx()],
                    atom_types[begin_atom.GetIdx()],
                    atom_types[end_atom.GetIdx()],
                )
                # Ensure angles like XAY and YAX are treated the same
                if angle_atom_types[0] > angle_atom_types[2]:
                    angle_atom_types = (
                        angle_atom_types[2],
                        angle_atom_types[1],
                        angle_atom_types[0],
                    )
                    angle_atoms = [
                        end_atom.GetIdx(),
                        begin_atom.GetIdx(),
                        neighbor.GetIdx(),
                    ]

                if angle_atom_types not in angle_type_to_int:
                    angle_type_to_int[angle_atom_types] = angle_type_index
                    angle_type_index += 1
                angle_type = angle_type_to_int[angle_atom_types]
                angle_types.append(
                    (
                        angle_type,
                        angle_atoms[0],
                        angle_atoms[1],
                        angle_atoms[2],
                    )
                )
        for neighbor in end_atom.GetNeighbors():
            if neighbor.GetIdx() != begin_atom.GetIdx() and neighbor.GetSymbol() != "H":
                angle_atoms = [
                    begin_atom.GetIdx(),
                    end_atom.GetIdx(),
                    neighbor.GetIdx(),
                ]
                angle_atom_types = (
                    atom_types[begin_atom.GetIdx()],
                    atom_types[end_atom.GetIdx()],
                    atom_types[neighbor.GetIdx()],
                )
                # Ensure angles like XAY and YAX are treated the same
                if angle_atom_types[0] > angle_atom_types[2]:
                    angle_atom_types = (
                        angle_atom_types[2],
                        angle_atom_types[1],
                        angle_atom_types[0],
                    )
                    angle_atoms = [
                        neighbor.GetIdx(),
                        end_atom.GetIdx(),
                        begin_atom.GetIdx(),
                    ]

                if angle_atom_types not in angle_type_to_int:
                    angle_type_to_int[angle_atom_types] = angle_type_index
                    angle_type_index += 1
                angle_type = angle_type_to_int[angle_atom_types]
                angle_types.append(
                    (
                        angle_type,
                        angle_atoms[0],
                        angle_atoms[1],
                        angle_atoms[2],
                    )
                )

    # Remove duplicate angles
    angle_types = list(set(angle_types))

    return angle_types, angle_type_to_int, angle_type_index


@track_call_depth
def get_particle_positions(
    mol: Chem.Mol, atoms: list[Chem.Atom]
) -> dict[int, Chem.rdGeometry.Point3D]:
    """
    Get the 3D positions of non-hydrogen atoms in a molecule.

    :param mol: The molecule from which to get the atom positions.
    :type mol: Chem.Mol
    :param atoms: A list of atoms in the molecule.
    :type atoms: list[Chem.Atom]
    :return: A dictionary mapping atom indices to their 3D positions.
    :rtype: dict[int, Chem.rdGeometry.Point3D]
    """
    conf = mol.GetConformer()
    return {
        atom.GetIdx(): conf.GetAtomPosition(atom.GetIdx())
        for atom in atoms
        if atom.GetSymbol() != "H"
    }


@track_call_depth
def convert_atom_types_to_int(used_atom_types: set[str]) -> dict[str, int]:
    """
    Convert a set of atom types to a dictionary mapping each type to a unique integer.

    :param used_atom_types: A set of atom types.
    :type used_atom_types: set[str]
    :return: A dictionary mapping each atom type to a unique integer.
    :rtype: dict[str, int]
    """
    return {atom_type: idx + 1 for idx, atom_type in enumerate(sorted(used_atom_types))}


@track_call_depth
def generate_lammps_data(
    molecule_blocks: list[list[str]],
    output_file: str,
    box_dimensions: tuple[float, float, float],
) -> None:
    """
    Generate a LAMMPS data file from molecule blocks.

    :param molecule_blocks: A list of lists, where each inner list represents a molecule block in PDB format.
    :type molecule_blocks: list[list[str]]
    :param output_file: The name of the output file to write the LAMMPS data.
    :type output_file: str
    :param box_dimensions: The dimensions of the simulation box.
    :type box_dimensions: tuple[float, float, float]
    :return: None
    """
    total_atoms = 0
    total_bonds = 0
    total_angles = 0

    molecule_data = []
    used_atom_types = set()
    bond_type_to_int = {}
    angle_type_to_int = {}
    bond_type_index = 1
    angle_type_index = 1

    # Process each molecule independently
    for n, block in enumerate(molecule_blocks):
        # Read molecule object from PDB block
        mol = Chem.MolFromPDBBlock("\n".join(block), sanitize=True, removeHs=False)

        # Determine bonding information
        rdDetermineBonds.DetermineBondOrders(mol)

        # Remove hydrogens from molecule
        mol = Chem.RemoveAllHs(mol)

        # Get atoms and bonds
        atom_info_list, atoms = generate_atom_info_list(mol)
        bonds = mol.GetBonds()

        # Get atom types for molecule and mixture
        atom_types, used_atom_types = get_atom_types(atom_info_list, used_atom_types)

        # Process bonds
        bond_types, bond_type_to_int, bond_type_index = process_bonds(
            bonds, atom_types, bond_type_to_int, bond_type_index
        )

        # Process angles
        angle_types, angle_type_to_int, angle_type_index = process_angles(
            bonds, atom_types, angle_type_to_int, angle_type_index
        )

        # Get positions
        positions = get_particle_positions(mol, atoms)

        # Store molecule data
        molecule_data.append(
            {
                "mol_name": n + 1,
                "atom_types": atom_types,
                "bond_types": bond_types,
                "angle_types": angle_types,
                "positions": positions,
            }
        )

        total_atoms += len(atom_types)
        total_bonds += len(bond_types)
        total_angles += len(angle_types)

    atom_type_to_int = convert_atom_types_to_int(used_atom_types)

    # Write the data to the LAMMPS data file
    with open(output_file, "w") as f:
        write_lammps_file_header(
            f,
            total_atoms,
            total_bonds,
            total_angles,
            used_atom_types,
            bond_type_to_int,
            angle_type_to_int,
            box_dimensions,
        )

        write_lammps_file_masses(f, atom_type_to_int)

        write_lammps_file_atoms(f, molecule_data, atom_type_to_int)

        write_lammps_file_bonds(f, molecule_data)

        write_lammps_file_angles(f, molecule_data)


@track_call_depth
def write_lammps_file_angles(file: TextIO, molecule_data: list[dict]) -> None:
    """
    Write angle information to a LAMMPS data file.

    :param file: The file object to write the angle information to.
    :type file: TextIO
    :param molecule_data: A list of dictionaries containing molecule information.
    :type molecule_data: list[dict]
    :return: None
    """
    file.write("Angles\n\n")

    angle_index = 1
    atom_offset = 0
    mol_idx = 1
    for mol_info in molecule_data:
        angle_types = mol_info["angle_types"]
        for angle_type, start, middle, end in angle_types:
            atom_type_1 = mol_info["atom_types"][start]
            atom_type_2 = mol_info["atom_types"][middle]
            atom_type_3 = mol_info["atom_types"][end]

            # Alphabetically prioritize equivalent angles
            if atom_type_1 > atom_type_3:
                atom_type_1, atom_type_3 = atom_type_3, atom_type_1
                start, end = end, start

            file.write(
                f"{angle_index} {angle_type} {start + atom_offset + 1} {middle + atom_offset + 1} {end + atom_offset + 1} # {atom_type_1}-{atom_type_2}-{atom_type_3}\n"
            )
            angle_index += 1
        atom_offset += len(mol_info["atom_types"])
        mol_idx += 1


@track_call_depth
def write_lammps_file_bonds(file: TextIO, molecule_data: list[dict]) -> None:
    """
    Write bond information to a LAMMPS data file.

    :param file: The file object to write the bond information to.
    :type file: TextIO
    :param molecule_data: A list of dictionaries containing molecule information.
    :type molecule_data: list[dict]
    :return: None
    """
    file.write("Bonds\n\n")
    bond_index = 1
    atom_offset = 0
    mol_idx = 1
    for mol_info in molecule_data:
        bond_types = mol_info["bond_types"]
        for bond_type, start, end in bond_types:
            atom_type_1 = list(mol_info["atom_types"].values())[start]
            atom_type_2 = list(mol_info["atom_types"].values())[end]
            file.write(
                f"{bond_index} {bond_type} {start + atom_offset + 1} {end + atom_offset + 1} # {atom_type_1}-{atom_type_2}\n"
            )
            bond_index += 1
        atom_offset += len(mol_info["atom_types"])
        mol_idx += 1

    file.write("\n")


@track_call_depth
def write_lammps_file_atoms(
    file: TextIO, molecule_data: list[dict], atom_type_to_int: dict[str, int]
) -> None:
    """
    Write atom information to a LAMMPS data file.

    :param file: The file object to write the atom information to.
    :type file: TextIO
    :param molecule_data: A list of dictionaries containing molecule information.
    :type molecule_data: list[dict]
    :param atom_type_to_int: A dictionary mapping atom types to unique integers.
    :type atom_type_to_int: dict[str, int]
    :return: None
    """
    file.write("Atoms\n\n")
    atom_index = 1
    mol_idx = 1
    for mol_info in molecule_data:
        atom_types = mol_info["atom_types"]
        positions = mol_info["positions"]
        for atom_idx in atom_types.keys():
            atom_type = atom_types[atom_idx]
            pos = positions[atom_idx]
            file.write(
                f"{atom_index} {mol_idx} {atom_type_to_int[atom_type]} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f} # {atom_type} \n"
            )
            atom_index += 1
        mol_idx += 1

    file.write("\n")


@track_call_depth
def write_lammps_file_masses(file: TextIO, atom_type_to_int: dict[str, int]) -> None:
    """
    Write atom mass information to a LAMMPS data file.

    :param file: The file object to write the mass information to.
    :type file: TextIO
    :param atom_type_to_int: A dictionary mapping atom types to unique integers.
    :type atom_type_to_int: dict[str, int]
    :return: None
    """
    file.write("Masses\n\n")
    for atom_type, atom_id in atom_type_to_int.items():
        file.write(f"{atom_id} {force_field_masses[atom_type]} # {atom_type}\n")
    file.write("\n")


@track_call_depth
def write_lammps_file_header(
    file: TextIO,
    total_atoms: int,
    total_bonds: int,
    total_angles: int,
    used_atom_types: set[str],
    bond_type_to_int: dict[tuple[str, str], int],
    angle_type_to_int: dict[tuple[str, str, str], int],
    box_dimensions: dict[str, float],
) -> None:
    """
    Write the header information to a LAMMPS data file.

    :param file: The file object to write the header information to.
    :type file: TextIO
    :param total_atoms: The total number of atoms.
    :type total_atoms: int
    :param total_bonds: The total number of bonds.
    :type total_bonds: int
    :param total_angles: The total number of angles.
    :type total_angles: int
    :param used_atom_types: A set of used atom types.
    :type used_atom_types: set[str]
    :param bond_type_to_int: A dictionary mapping bond type keys to integers.
    :type bond_type_to_int: dict[tuple[str, str], int]
    :param angle_type_to_int: A dictionary mapping angle type keys to integers.
    :type angle_type_to_int: dict[tuple[str, str, str], int]
    :param box_dimensions: A dictionary containing the dimensions of the simulation box.
    :type box_dimensions: dict[str, float]
    :return: None
    """
    file.write("LAMMPS data file\n\n")
    file.write(f"{total_atoms} atoms\n")
    file.write(f"{total_bonds} bonds\n")
    file.write(f"{total_angles} angles\n")
    file.write("\n")
    file.write(f"{len(used_atom_types)} atom types\n")
    file.write(f"{len(bond_type_to_int)} bond types\n")
    file.write(f"{len(angle_type_to_int)} angle types\n")

    file.write("\n")
    file.write(
        f"# Cell dimensions: {box_dimensions['a']}, {box_dimensions['b']}, {box_dimensions['c']}\n"
    )
    file.write(f"0.000000\t{box_dimensions['a']} xlo xhi\n")
    file.write(f"0.000000\t{box_dimensions['b']} ylo yhi\n")
    file.write(f"0.000000\t{box_dimensions['c']} zlo zhi\n")

    file.write("\n")
