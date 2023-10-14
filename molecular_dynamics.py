"""Adapted version for the molecular dynamics study."""
import pathlib
import re
import subprocess

import ase.geometry.analysis
import numpy as np
import package1
import package2
import pandas as pd
import yaml
from ase.io.lammpsdata import write_lammps_data
from jinja2 import Environment, FileSystemLoader


class Lammps(package1.Node):
    """Lammps Node.

    Node to run MD simulations with LAMMPS using a template file.

    Attributes
    ----------
    data : list
        List of ASE Atoms objects.
    data_id : int
        Index of the data to be used.
    parameter_file : str
        Path to the parameter file.
    thermo : pd.DataFrame
        Dataframe with the thermo output.
    template : str
        Path to the template file.
    lammps_data : pathlib.Path
        Path to the lammps data file.
    lmp_directory : pathlib.Path
        Path to the lammps directory.
    """

    data: list = package1.deps()
    data_id: int = package1.params()

    parameter_file: str = package1.params_path()
    thermo = package1.plots(x="Step", y="Density")
    template: str = package1.deps_path()
    lammps_data: pathlib.Path = package1.outs_path(package1.nwd / "input.lmp_data")
    lmp_directory = package1.outs_path(package1.nwd / "lammps")

    def run(self) -> None:
        """Perform the MD simulation."""
        atoms = self.data[self.data_id]
        write_lammps_data(self.lammps_data, atoms, atom_style="full")

        self.lmp_directory.mkdir(exist_ok=True)

        loader = FileSystemLoader(".")
        env = Environment(loader=loader)
        template = env.get_template(self.template)

        params = yaml.safe_load(pathlib.Path(self.parameter_file).read_text())

        context = template.render(params | {"input_file": self.lammps_data.resolve()})

        input_script = self.lmp_directory / "input.lmp"

        input_script.write_text(context)

        subprocess.check_call(
            ["lmp_serial", "-in", input_script.resolve()], cwd=self.lmp_directory
        )

        with open(self.lmp_directory / "log.lammps") as f:
            # find line that starts with the thermo output
            start_idx = None
            stop_idx = None

            regex = re.compile(r"\s+Step\s+Time\s+Temp\s+Density\s+Press\s+")

            for idx, line in enumerate(f):
                # if regex matches line, set start_idx
                if regex.match(line):
                    start_idx = idx
                if line.startswith("Loop time of"):
                    stop_idx = idx - 1

        print(start_idx, stop_idx)

        # now read the dataframe with pandas
        self.thermo = pd.read_csv(
            self.lmp_directory / "log.lammps",
            skiprows=start_idx,
            nrows=stop_idx - start_idx,
            delim_whitespace=True,
        )


class RDF(package1.Node):
    """Compute the radial distribution function.

    Attributes
    ----------
    data : list
        List of ASE Atoms objects.
    rdf : pd.DataFrame
        Dataframe with the radial distribution function.
    nbins : int
        Number of bins.
    rmax : float
        Maximum distance to be considered.
    """

    data: list = package1.deps()
    rdf = package1.plots(x="r", y="g(r)")

    nbins = package1.params(100)
    rmax = package1.params(20.0)

    def run(self) -> None:
        analysis = ase.geometry.analysis.Analysis(self.data)
        rdf = analysis.get_rdf(nbins=self.nbins, rmax=self.rmax)
        self.rdf = pd.DataFrame(
            {"r": np.linspace(0, self.rmax, self.nbins), "g(r)": np.mean(rdf, axis=0)}
        )


if __name__ == "__main__":
    with package1.Project(
        automatic_node_names=True, remove_existing_graph=True
    ) as project:
        # we use a pre-exiting package which provides nodes specific
        # for molecular dynamics simulations and machine learned interatomic potentials
        # the package is publicly available and also anonymous here.
        na = package2.configuration_generation.SmilesToAtoms(smiles="[Na+]")
        cl = package2.configuration_generation.SmilesToAtoms(smiles="[Cl-]")

        packmol = package2.configuration_generation.Packmol(
            data=[na.atoms, cl.atoms], count=[1000, 1000], density=1486
        )

        md = Lammps(
            data=packmol.atoms,
            data_id=-1,
            template="template.lmp",
            parameter_file="lammps.yaml",
        )

        rdf = RDF(data=md.atoms, nbins=300, rmax=7.0)

    project.build()
