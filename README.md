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
## 1. The Josephson Junction: 

A Josephson Junction is comprised of two superconducting materials with a thin insulator placed between them. When these two materials become superconducting, a current flows across the insulating barrier without any applied voltage, and when a voltage is applied it causes this current to oscillate.

The behavior of these phenomenon is captured in the **Josephson Equations**:

```math
$$
\begin{align}
&I(t)=I_{c}\sin(\mathcal{\varphi}(t)) \\
&\frac{ \partial \varphi }{ \partial t } = \frac{2eV(t)}{\hbar}
\end{align}
$$
```
Where
- $I(t)$ is the current across the junction
- $V(t)$ is the voltage across the junction
- $I_{c}$ is a parameter known as the critical current

To derive these, we must start with the order parameters of the superconductors. The Ginzburg-Landau order parameter can be taken as the wavefunction of a Cooper pair in superconductor $A$ or $B$, and takes the following form:

```math
$$
\begin{align}
\psi_{A}=\sqrt{ n_{A} }e^{i\phi_{A}}\text{, }\psi_{B}=\sqrt{ n_{B} }e^{i\phi_{B}}
\end{align}
$$
```
Where $n_{A\text{, }B}$ is the density of Cooper pairs, and $\phi_{A\text{, }B}$ is the phase of each wavefunction. 

If we apply a voltage across the junction, then since each Cooper pair consists of 2 electrons, and thus has charge $2e$, we get an energy difference of $2eV$.

Using the Schrodinger equation then gives us two differential equations to describe this system. A system of this form is called a two-level system.

```math
$$
\begin{align}
i\hbar \frac{ \partial  }{ \partial t } \begin{pmatrix}
\sqrt{ n_{A} }e^{i\phi_{A}} \\
\sqrt{ n_{B} }e^{i\phi_{B}}
\end{pmatrix}=\begin{pmatrix}
eV & K \\
K & -eV 
\end{pmatrix}\begin{pmatrix}
\sqrt{ n_{A} }e^{i\phi_{A}} \\
\sqrt{ n_{B} }e^{i\phi_{B}}
\end{pmatrix}
\end{align}
$$
```

Here $K$ is a parameter intrinsic to the junction. 

To solve the first equation, we see that

```math
$$
\begin{align}
\frac{ \partial  }{ \partial t } (\sqrt{ n_{A} }e^{i\phi_{A}})=\dot{\sqrt{ n_{A} }}e^{i\phi_{A}}+i\sqrt{ n_{A} }\dot{\phi_{A}}e^{i\phi_{A}}=e^{i\phi_{A}}(\dot{\sqrt{ n_{A} }}+i\sqrt{ n_{A} }\dot{\phi_{A}})
\end{align}
$$
```

Plugging this into the SE gives

```math
$$
\begin{align}
e^{i\phi_{A}}(\dot{\sqrt{ n_{A} }}+i\sqrt{ n_{A} }\dot{\phi_{A}})&=\frac{1}{i\hbar}(eV\sqrt{ n_{A} }e^{i\phi_{A}}+K\sqrt{ n_{B} }e^{i\phi_{B}}) \\
(\dot{\sqrt{ n_{A} }}+i\sqrt{ n_{A} }\dot{\phi_{A}})&=\frac{1}{i\hbar}(eV\sqrt{ n_{A} }+K\sqrt{ n_{B} }e^{i\varphi})
\end{align}
$$
```

where $\varphi=\phi_{B}-\phi_{A}$ is the Josephson phase.

If we add the conjugate of this equation to itself, we can get an equation for $\dot{\sqrt{ n_{A} }}$,

```math
$$
\begin{align}
&2\dot{\sqrt{n_{A} }}=\frac{2\dot{n}_{A}}{2\sqrt{ n_{A} }}=\frac{K\sqrt{ n_{B} }}{\hbar}2\sin(\varphi) \\
&\dot{n}_{A}= \frac{2K\sqrt{ n_{A}n_{B} }}{\hbar}\sin(\varphi)
\end{align}
$$
```

If we instead subtract the two conjugate equations, we can eliminate $\dot{\sqrt{ n_{A} }}$ to get

```math
$$
\begin{align}
\dot{\phi}_{A}= -\frac{1}{\hbar}\left( eV+K\sqrt{ \frac{n_{B}}{n_{A}} } \cos(\varphi)\right)
\end{align}
$$
```

Similarly, we can find

```math
$$
\begin{align}
&\dot{n}_{B}= -\frac{2K\sqrt{ n_{A}n_{B} }}{\hbar}\sin(\varphi) \\
&\dot{\phi}_{B}= \frac{1}{\hbar}\left( eV-K\sqrt{ \frac{n_{B}}{n_{A}} } \cos(\varphi)\right)
\end{align}
$$
```

Since the time derivative of $n_{A}$ is proportional to current, then if $n_{A}\approx n_{B}$, we get the first Josephson equation, and by combining $\dot{\varphi}=\dot{\phi}_{B}-\dot{\phi}_{A}$ we get the second Josephson equation.

# Basic Usage
The folder titled `RCSJ Basis` outlines the classes and methods we use that form the basis for exploring JJs. The foundational python file of this project is `RCSJ_Core.py`, which contains the classes `RCSJParams`, `RCSJModel`, and `RCSJSolve`. These classes work sequentially to set up, model, and solve the relevant ODEs respectively.

Also in `RCSJ Basis` are the files `single_junction.py` and `coupled_junction.py`, which lay the groundwork for modeling hydrogen and helium by defining the classes `RCSJ` and `CoupledRCSJ`. These classes describe and extend the single Josephson junction governed by the standard RCSJ differential equation, with parameters such as $I_c, R, C$ and external bias currents specified upon initialization. The `RCSJ` implementation serves as the simplest example of phase dynamics in an anharmonic potential, while `CoupledRCSJ` generalizes the model to two interacting phases through capacitive coupling, enabling multi-degree-of-freedom behavior analogous to multi-electron systems.

Beyond the core RCSJ models, the repository contains higher-level files that connect these junction dynamics to artificial atomic analogues. The files `artificial_hydrogen.py` and `artificial_helium.py` build directly on the `single_junction` and `coupled_junction` classes to construct simplified hydrogen-like and helium-like systems respectively. These scripts not only initialize and solve the corresponding RCSJ models, but also include routines to visualize the time evolution of the phase variables, plot the effective potential landscapes, and compare these classical junction-based potentials to their real atomic counterparts.

In this organization, the physics and numerics of the RCSJ model remain isolated within `RCSJ_Core.py`, the concrete physical systems (single and coupled junctions) are defined in their own dedicated modules, and the artificial atom files act as application-level examples that tie everything together. This structure keeps the codebase clear, extensible, and easy to navigate for anyone wishing to explore Josephson junction dynamics or build upon the artificial-atom analogies developed in this project.
