# Simulating RCSJ Josephson dynamics in singular and coupled systems
### Created by Tyler Sachs, Austin Merkel, Ellie Howard, and Chetan Mehta

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

## 1. The Josephson Junction

A Josephson Junction is formed by two superconductors separated by a thin insulating barrier.  
When both materials are in the superconducting state, Cooper pairs are able to tunnel through the barrier, producing a supercurrent even when no voltage is applied.  
When a voltage is applied, this supercurrent oscillates at a well-defined frequency.

The behavior of the junction is captured by the **Josephson Equations**:

```math
$$
\begin{align}
&I(t) = I_c \sin(\varphi(t)) \\
&\frac{d\varphi}{dt} = \frac{2e}{\hbar} V(t)
\end{align}
$$
```

**Where:**
- $I(t)$: current across the junction  
- $V(t)$: voltage across the junction  
- $I_c$: critical current  
- $\varphi = \phi_B - \phi_A$: phase difference between the superconductors

## 2. Order Parameters and the Origin of the Josephson Phase

Each superconductor is described by a macroscopic wavefunction (Ginzburg–Landau / BCS order parameter):

```math
$$
\psi_A = \sqrt{n_A}\, e^{i\phi_A}, \qquad \psi_B = \sqrt{n_B}\, e^{i\phi_B}
$$
```

**Here:**
- $n_A$ and $n_B$: Cooper-pair densities  
- $\phi_A$ and $\phi_B$: phases of the superconductors  

When a voltage $V$ is applied across the junction, a Cooper pair (charge $2e$) experiences an energy shift of $2eV$.  
In electromagnetism, physical observables must remain unchanged when we shift the scalar potential Φ or vector potential 𝐴.
For a charged condensate, this means that if the potentials change, the superconducting phases must shift too, in such a way that measurable quantities remain the same.

This requirement, gauge invariance, forces the direct relationship:

```math
$$
\boxed{\frac{d\varphi}{dt} = \frac{2e}{\hbar} V(t)}
$$
```

This is the **AC Josephson relation**, showing how voltage directly controls the time evolution of the superconducting phase.

## 3. Tunneling and the Supercurrent

Weak tunneling between the superconductors couples the two order parameters.  
In the ideal model, the tunneling current is proportional to the sine of the phase difference:

```math
$$
\boxed{I = I_c \sin(\varphi)}
$$
```


## **4. The RCSJ Model**

The Resistively and Capacitively Shunted Junction (RCSJ) model (also known as the Stewart–McCumber model) is the standard classical model for Josephson junction dynamics.

This model starts by viewing the Josephson junction as having **three parallel current channels**:

1. **The Josephson supercurrent** (Cooper-pair tunneling)
2. **A resistive current** from normal electrons tunneling through the barrier
3. **A capacitive displacement current** due to the junction’s intrinsic capacitance

These appear as:

```math
I_s = I_c \sin(\varphi)
```

```math
I_R = \frac{V}{R}
```

```math
I_C = C \frac{dV}{dt}
```

Using Kirchhoff’s law, the total current through the junction is:

```math
I = I_s + I_R + I_C
```


### **Using the Josephson phase–voltage relation**

From the second Josephson equation,

```math
V(t) = \frac{\hbar}{2e} \frac{d\varphi}{dt}
```


Taking a time derivative gives:

```math
\frac{dV}{dt} = \frac{\hbar}{2e} \frac{d^2\varphi}{dt^2}
```

Substituting these into the current terms yields the classical RCSJ differential equation:

```math
C \frac{d^2\varphi}{dt^2}
+ \frac{1}{R} \frac{d\varphi}{dt}
+ \frac{2e I_c}{\hbar} \sin(\varphi)
=
\frac{2e}{\hbar} I
```

This equation describes a **driven, damped, nonlinear oscillator**, and is the fundamental equation solved throughout this repository.


### **Non-dimensionalizing the equation**

Often the normalized version of the RCSJ equation will be used. This is obtained by using the plasma frequency defined as $\omega_{p}^{2}=\frac{2eI_{c}}{\hbar C}$ to obtain

```math
$$
\begin{align}
\ddot{\varphi}+\frac{1}{RC}\dot{\varphi}+\omega_{p}^{2}\sin(\varphi)=\frac{2eI}{\hbar C}
\end{align}
$$
```

Then, we switch differentiation from $t$ to $\tau=\omega_{p}t$, which is known as dimensionless time, with $\frac{d}{dt}=\omega_{p} \frac{d}{d\tau}$. Making this replacement and then simplifying again yields the normalized RCSJ equation:

```math
$$
\begin{align}
\frac{d^{2}{\varphi}}{d\tau^{2}}+\alpha  \frac{d{\varphi}}{d\tau}+\sin(\varphi)=i
\end{align}
$$
```

Where $\alpha = \frac{1}{RC\omega_{p}}$, and $i = \frac{I}{I_{c}}$. 


## 5. Washboard Potential:
The differential equation we obtained in the RCSJ model is of the form

```math
$$
\begin{align}
m \ddot{x}+\gamma  \dot{x}+ \frac{dU}{dx}=0
\end{align}
$$
```

In this analogy, $\frac{dU}{d\varphi}=\sin(\varphi)-i$. Integrating this gives

```math
$$
\begin{align}
U(\varphi)=1-\cos(\varphi)-i\varphi
\end{align}
$$
```

Where the constant $1$ is arbitrarily chosen.

This potential is known as a washboard potential due to it's downward sloped oscillating features.

The washboard potential provides a convenient feature for understanding the zero-voltage state at $I_{c}>I$ versus the finite-voltage state for $I_{c}< I$.

For the first case, the potential has local minima corresponding to zero voltage via the 2nd Josephson equation. Meanwhile for the second case, there is no such minima, and so there is a always a change in $\varphi$ and therefore finite voltage.



## 6. Shapiro Steps:
Instead of having only a DC driving current $i_{dc}$ (which is $i$ in previous context), we can also drive the junction with an AC current, $i_{ac}$. Including this term gives us the following RCSJ equation:

```math
$$
\begin{align}
\frac{d^{2}{\varphi}}{d\tau^{2}}+\alpha  \frac{d{\varphi}}{d\tau}+\sin(\varphi)=i_{DC}+i_{AC}\sin(\Omega \tau)
\end{align}
$$
```

Where $\Omega=\frac{\omega_{drive}}{\omega_{p}}$ and $\omega_{drive}$ is the frequency of the AC current.

For reasons outside of the scope of this section, the frequency of the Josephson Junction caused by the voltage $V$ becomes locked to integer multiples of the driving frequency $\Omega$. This leads to a quantization of the I-V curve known as Shapiro Steps.


# Basic Usage
The folder titled `RCSJ Basis` outlines the classes and methods we use that form the basis for exploring JJs. The foundational python file of this project is `RCSJ_Core.py`, which contains the classes `RCSJParams`, `RCSJModel`, and `RCSJSolve`. These classes work sequentially to set up, model, and solve the relevant ODEs respectively.

Also in `RCSJ Basis` are the files `single_junction.py` and `coupled_junction.py`, which lay the groundwork for modeling hydrogen and helium by defining the classes `RCSJ` and `CoupledRCSJ`. These classes describe and extend the single Josephson junction governed by the standard RCSJ differential equation, with parameters such as $I_c, R, C$ and external bias currents specified upon initialization. The `RCSJ` implementation serves as the simplest example of phase dynamics in an anharmonic potential, while `CoupledRCSJ` generalizes the model to two interacting phases through capacitive coupling, enabling multi-degree-of-freedom behavior analogous to multi-electron systems.

Beyond the core RCSJ models, the repository contains higher-level files that connect these junction dynamics to artificial atomic analogues. The files `artificial_hydrogen.py` and `artificial_helium.py` build directly on the `single_junction` and `coupled_junction` classes to construct simplified hydrogen-like and helium-like systems respectively. These scripts not only initialize and solve the corresponding RCSJ models, but also include routines to visualize the time evolution of the phase variables, plot the effective potential landscapes, and compare these classical junction-based potentials to their real atomic counterparts.

In this organization, the physics and numerics of the RCSJ model remain isolated within `RCSJ_Core.py`, the concrete physical systems (single and coupled junctions) are defined in their own dedicated modules, and the artificial atom files act as application-level examples that tie everything together. This structure keeps the codebase clear, extensible, and easy to navigate for anyone wishing to explore Josephson junction dynamics or build upon the artificial-atom analogies developed in this project.

# Results


# Resources
- https://www.wmi.badw.de/fileadmin/WMI/Lecturenotes/Applied_Superconductivity/AS_Chapter1.pdf - Chapters 1-3 cover theory of JJ and the LCSJ circuit model.
- [https://www.nobelprize.org/uploads/2025/10/advanced-physicsprize2025.pdf](https://www.nobelprize.org/uploads/2025/10/advanced-physicsprize2025.pdf "https://www.nobelprize.org/uploads/2025/10/advanced-physicsprize2025.pdf") - Information on application of Josephson junctions and related Nobel prize.
- 



