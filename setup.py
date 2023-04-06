import os
from setuptools import setup, find_packages

setup(
    name="phyloformer",
    version="0.0.1a4",
    description="Fast and accurate Phylogeny estimation with self-attention Networks",
    long_description=open("./README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lucanest/Phyloformer",
    author="Luca Nesterenko, Bastien Boussau, Laurent Jacob",
    license="CeCIL",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.1",
        "scipy>=1.7.3",
        "numpy>=1.21.2",
        "ete3>=3.1.2",
        "biopython>=1.79",
        "dendropy>=4.5.2",
        "scikit-bio>=0.5.6",
        "tqdm>=4.65.0",
    ],
    package_data={
        'phyloformer': [
            os.path.join("pretrained_models", "*"),
            "LICENSE",
        ]
    },
    include_package_data=True,
    python_requires=">=3.7, <3.10",
    entry_points = {
        'console_scripts': [
            "train_phyloformer = phyloformer.scripts.train:main",
            "simulate_trees = phyloformer.scripts.simulateTrees:main",
            "simulate_alignments = phyloformer.scripts.simulateAlignments:main",
            "make_tensors = phyloformer.scripts.make_tensors:main",
            "predict = phyloformer.scripts.predict:main",
            "evaluate = phyloformer.scripts.evaluate:main",
        ]
    }
)
