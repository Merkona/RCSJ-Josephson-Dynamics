# Simulating RCSJ Josephson dynamics in singular and coupled systems
### Created by Tyler Sachs, Austin Merkel, Ellie Howard, Chetan Mehta

# Getting Started

We use Python 3 with the following packages:
- `NumPy`
- `Matplotlib`
- `SciPy`

# Introduction
Superconducting circuits have become one of the most powerful platforms for realizing controllable quantum systems. In particular, circuits that utilize the superconducting Josephson Junction (JJ) to provide a non-linear inductive element have formed the backbone of modern quantum technologies, including qubits, superconducting resonators, and macroscopic quantum devices. 

Using a Resistively and Capacitively Shunted Junction (RCSJ) model, we can explore the classical dynamics of Josephson Junctions by observing this system as it exhibits hysteresis, phase slips, Shapiro steps, and anharmonic energy levels. Understanding these classical dynamics provides both physical intuition and a computational foundation for exploring more complex superconducting systems. To that end, this repository was made to: 

- Implement modular RCSJ solvers capable of simulating single junctions, multiple junctions in series, and junction–capacitor networks.
- Demonstrate how these circuits can serve as toy models for artificial atoms, including hydrogen-like and helium-like structures formed from coupled junctions.
- Explore how the JJ’s intrinsic nonlinearity generates anharmonic energy levels—an essential ingredient for isolating qubit states or modeling multi-electron atomic behavior.

# Theory

# Basic Usage
The folder titled `RCSJ Basis` outlines the classes and methods we use that form the basis for exploring JJs. The foundational python file of this project is `RCSJ_Core.py`, which contains the classes `RCSJParams`, `RCSJModel`, and `RCSJSolve`. These classes work sequentially to set up, model, and solve the relevant ODEs respectively.

Also in `RCSJ Basis` are the files `single_junction.py` and `coupled_junction.py`, which lay the groundwork for modeling hydrogen and helium by defining the classes `RCSJ` and `CoupledRCSJ`. These classes describe and extend the single Josephson junction governed by the standard RCSJ differential equation, with parameters such as $I_c, R, C$ and external bias currents specified upon initialization. The `RCSJ` implementation serves as the simplest example of phase dynamics in an anharmonic potential, while `CoupledRCSJ` generalizes the model to two interacting phases through capacitive coupling, enabling multi-degree-of-freedom behavior analogous to multi-electron systems.

Beyond the core RCSJ models, the repository contains higher-level files that connect these junction dynamics to artificial atomic analogues. The files `artificial_hydrogen.py` and `artificial_helium.py` build directly on the `single_junction` and `coupled_junction` classes to construct simplified hydrogen-like and helium-like systems respectively. These scripts not only initialize and solve the corresponding RCSJ models, but also include routines to visualize the time evolution of the phase variables, plot the effective potential landscapes, and compare these classical junction-based potentials to their real atomic counterparts.

In this organization, the physics and numerics of the RCSJ model remain isolated within `RCSJ_Core.py`, the concrete physical systems (single and coupled junctions) are defined in their own dedicated modules, and the artificial atom files act as application-level examples that tie everything together. This structure keeps the codebase clear, extensible, and easy to navigate for anyone wishing to explore Josephson junction dynamics or build upon the artificial-atom analogies developed in this project.
