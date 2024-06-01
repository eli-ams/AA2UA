"""
Main execution script for AA2UA.
If used for research, academic, or corporate purposes, please cite: https://doi.org/10.1016/j.matdes.2024.112831
"""

from core_functions import *


if __name__ == "__main__":
    output_folder = "output"  # changed to results/ from output/ for CodeOcean.
    initialize_log_file()
    log_message_to_file(
        message=f"Successfully initialized AA2UA and its dependencies. Executing now...",
        printmsg=True,
    )
    for input_file in [
        f"input/{i}" for i in os.listdir("input/") if i.endswith(".pdb")
    ]:
        log_message_to_file(message=f"Reading file {input_file}...", printmsg=True)
        pdb_file_lines = read_pdb_file(input_file)

        box_dimensions = extract_pdb_box_dimensions(pdb_header=pdb_file_lines[0])

        pdb_molecule_blocks = extract_molecules(pdb_file_lines)

        output_file = f"{output_folder}/{os.path.splitext(os.path.basename(input_file))[0]}_structure.data"
        generate_lammps_data(pdb_molecule_blocks, output_file, box_dimensions)
        log_message_to_file(
            message=f"Successfully converted file {input_file}...", printmsg=True
        )
