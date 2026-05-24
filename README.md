# ijpe-soybeans-abm-bzl

# Agent-Based Model for Brazilian Soybean Supply Chain with Q-Learning

**A Comprehensive Technical Documentation**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Model Architecture](#4-model-architecture)
5. [Q-Learning Algorithm](#5-q-learning-algorithm)
6. [Agent Specifications](#6-agent-specifications)
7. [Market Clearing Mechanism](#7-market-clearing-mechanism)
8. [Implementation Details](#8-implementation-details)
9. [References](#9-references)

---

## 1. Introduction

This document presents a detailed specification of an **Agent-Based Model (ABM)** for the Brazilian soybean supply chain, incorporating **Q-learning** algorithms for autonomous agent decision-making. The model simulates the interactions between three types of agents—**Farmers**, **Crush Plants (Crushers)**, and **Exporters**—over a 52-week period representing one agricultural year.

### 1.1 Motivation

Traditional supply chain models often rely on:
- **Explicit price functions** (e.g., $P = f(Q)$)
- **Perfect information assumptions**
- **Centralized optimization**

However, real-world supply chains exhibit:
- **Decentralized decision-making** under uncertainty
- **Learning behavior** based on past experiences
- **Strategic interactions** without complete information

This model addresses these realities by employing **reinforcement learning** (specifically Q-learning) to simulate how agents learn optimal trading strategies through repeated interactions (Park & Ryu, 2022; Watkins & Dayan, 1992).

### 1.2 Research Context

The model builds upon:
- **Agent-based modeling** in supply chain management (Gallay & Hongler, 2009)
- **Q-learning in economic systems** (Calvano et al., 2020; Waltman & Kaymak, 2008)
- **Reinforcement learning in operations research** (Charpentier et al., in press)

---

## 2. Theoretical Foundation

### 2.1 Markov Decision Process (MDP)

Q-learning is grounded in the framework of **Markov Decision Processes** (Watkins, 1989). An MDP is defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \delta)$:

- $\mathcal{S}$: Finite set of **states**
- $\mathcal{A}$: Finite set of **actions**
- $\mathcal{P}(s_{t+1} | s_t, a_t)$: **Transition probability** from state $s_t$ to $s_{t+1}$ given action $a_t$
- $\mathcal{R}(s_t, a_t)$: **Reward function** for taking action $a_t$ in state $s_t$
- $\delta \in [0,1]$: **Discount factor** for future rewards

The agent's objective is to find a policy $\pi: \mathcal{S} \rightarrow \mathcal{A}$ that maximizes the expected discounted cumulative reward:

$$
V^\pi(s_t) = \mathbb{E}\left[\sum_{k=0}^{\infty} \delta^k r_{t+k} \mid s_t, \pi\right]
$$

where $r_t = \mathcal{R}(s_t, a_t)$ is the reward at time $t$.

### 2.2 Bellman Optimality Equation

The **optimal value function** $V^*(s)$ satisfies the Bellman equation:

$$
V^*(s_t) = \max_{a_t \in \mathcal{A}} \left\{ \mathcal{R}(s_t, a_t) + \delta \sum_{s_{t+1} \in \mathcal{S}} \mathcal{P}(s_{t+1} | s_t, a_t) V^*(s_{t+1}) \right\}
$$

### 2.3 Q-Function

The **Q-function** (action-value function) represents the expected return of taking action $a$ in state $s$ and following the optimal policy thereafter:

$$
Q^*(s_t, a_t) = \mathcal{R}(s_t, a_t) + \delta \sum_{s_{t+1} \in \mathcal{S}} \mathcal{P}(s_{t+1} | s_t, a_t) \max_{a_{t+1}} Q^*(s_{t+1}, a_{t+1})
$$

The relationship between $V^*$ and $Q^*$ is:

$$
V^*(s_t) = \max_{a_t \in \mathcal{A}} Q^*(s_t, a_t)
$$

### 2.4 Q-Learning Update Rule

Q-learning estimates $Q^*$ **without knowing** $\mathcal{P}$ or $\mathcal{R}$ explicitly. After observing transition $(s_t, a_t, r_t, s_{t+1})$, the Q-value is updated using:

$$
Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha \left[ r_t + \delta \max_{a'} Q_t(s_{t+1}, a') - Q_t(s_t, a_t) \right]
$$

**Parameters:**
- $\alpha \in (0,1]$: **Learning rate** (controls how much new information overrides old)
- $\delta \in [0,1]$: **Discount factor** (importance of future rewards)

**Convergence:** Under certain conditions (all state-action pairs visited infinitely often, appropriate decay of $\alpha$), $Q_t \rightarrow Q^*$ as $t \rightarrow \infty$ (Watkins & Dayan, 1992).

### 2.5 Exploration vs. Exploitation

To ensure convergence, agents must **explore** suboptimal actions. The **$\varepsilon$-greedy policy** balances exploration and exploitation:

$$
a_t = \begin{cases}
\arg\max_{a \in \mathcal{A}} Q_t(s_t, a) & \text{with probability } 1 - \varepsilon_t \\
\text{random action from } \mathcal{A} & \text{with probability } \varepsilon_t
\end{cases}
$$

In this model, we use a **time-decaying** exploration rate (Park & Ryu, 2022):

$$
\varepsilon_t = e^{-t/\tau}
$$

where $\tau$ is a decay constant (e.g., $\tau = 10$). This ensures:
- **Early exploration** ($\varepsilon_0 = 1$, fully random)
- **Late exploitation** ($\varepsilon_t \rightarrow 0$ as $t \rightarrow \infty$)

---

## 3. Exploratory Data Analysis

### 3.1 Data Sources

The model uses three datasets representing the Brazilian soybean supply chain:

1. **`df_farmers.csv`**: 5,563 farmer agents
2. **`df_crushers.csv`**: 102 crusher agents
3. **`df_exporters.csv`**: 31 exporter agents

### 3.2 Farmers

**Key Statistics:**

| Metric | Value |
|--------|-------|
| Total agents | 5,563 |
| Total production | 144.47 M tons |
| Mean production | 25,970 tons/agent |
| Median production | 0 tons (many small/inactive) |
| Total storage capacity | 57.79 M tons |
| Storage/Production ratio | 40% |

**Geographic Distribution (Top 5 States):**

| State | Number of Farmers | Total Production (M tons) |
|-------|-------------------|---------------------------|
| MG (Minas Gerais) | 853 | ~15.2 |
| SP (São Paulo) | 642 | ~12.8 |
| RS (Rio Grande do Sul) | 497 | ~18.5 |
| BA (Bahia) | 417 | ~22.1 |
| PR (Paraná) | 399 | ~14.3 |

**Top 10 Producers:**

1. **São Desidério (BA)**: 2.09 M tons
2. **Sorriso (MT)**: 2.08 M tons
3. **Formosa do Rio Preto (BA)**: 1.96 M tons
4. **Rio Verde (GO)**: 1.57 M tons
5. **Nova Ubiratã (MT)**: 1.50 M tons

**Pareto Analysis:**
- **80% of production** comes from approximately **1,200 farmers** (21.6% of agents)
- This indicates high **concentration** in production

**Storage Constraint:**
- Farmers have storage capacity equal to **40% of annual production**
- This creates a **forced sale** constraint: excess production must be sold immediately

### 3.3 Crushers

**Key Statistics:**

| Metric | Value |
|--------|-------|
| Total agents | 102 |
| Total capacity | 72.05 M tons/year |
| Total minimal processing | 54.04 M tons/year |
| Capacity utilization | 75% |
| Total storage capacity | 14.41 M tons |

**Geographic Distribution (Top 5 States):**

| State | Number of Crushers | Total Capacity (M tons/year) |
|-------|-------------------|------------------------------|
| RS (Rio Grande do Sul) | 22 | ~15.8 |
| PR (Paraná) | 17 | ~12.3 |
| MT (Mato Grosso) | 15 | ~18.9 |
| GO (Goiás) | 15 | ~11.2 |
| MS (Mato Grosso do Sul) | 7 | ~6.5 |

**Top 5 Crushers by Capacity:**

1. **Rondonópolis (MT)**: 2.06 M tons/year
2. **Primavera do Leste (MT)**: 1.72 M tons/year
3. **Uberlândia (MG)**: 1.61 M tons/year
4. **Rondonópolis (MT)**: 1.54 M tons/year
5. **Nova Mutum (MT)**: 1.37 M tons/year

**Demand Analysis:**
- Crushers require **54.04 M tons/year** (37.4% of farmer production)
- **Supply/Demand ratio**: 2.67× (ample supply)

**Weekly Consumption:**
- Minimal processing: $\text{minimal\_process} / 52$ tons/week
- **Forced buy constraint**: If stock < 2 weeks of consumption, must buy

### 3.4 Exporters

**Key Statistics:**

| Metric | Value |
|--------|-------|
| Total agents | 31 |
| Total yearly exports | 98.81 M tons |
| Mean exports | 3.19 M tons/agent |
| Total storage capacity | 24.70 M tons |

**Geographic Distribution (Top 5 States):**

| State | Number of Exporters | Total Exports (M tons/year) |
|-------|---------------------|----------------------------|
| SP (São Paulo) | 4 | ~31.7 |
| RS (Rio Grande do Sul) | 5 | ~15.2 |
| SC (Santa Catarina) | 3 | ~9.8 |
| PR (Paraná) | 3 | ~14.5 |
| RJ (Rio de Janeiro) | 3 | ~2.1 |

**Top 5 Exporters:**

1. **Santos (SP)**: 27.96 M tons/year (28.3% of total!)
2. **São Luís (MA)**: 13.94 M tons/year
3. **Paranaguá (PR)**: 13.72 M tons/year
4. **Rio Grande (RS)**: 10.79 M tons/year
5. **Belém (PA)**: 9.70 M tons/year

**Concentration:**
- **80% of exports** come from **5 exporters** (16.1% of agents)
- **Santos alone** accounts for 28.3% of all exports

**Demand Analysis:**
- Exporters require **98.81 M tons/year** (68.4% of farmer production)
- **Export/Production ratio**: 68.4%

### 3.5 Supply Chain Balance

**Overall Balance:**

| Component | Volume (M tons) | % of Production |
|-----------|-----------------|-----------------|
| **Supply** (Farmer production) | 144.47 | 100% |
| **Demand** (Crusher minimal) | 54.04 | 37.4% |
| **Demand** (Exporter) | 98.81 | 68.4% |
| **Total Demand** | 152.86 | **105.8%** |
| **Surplus/Deficit** | **-8.38** | **-5.8%** |

**Key Insight:**
- The system is **slightly undersupplied** (deficit of 5.8%)
- This creates **competition** between crushers and exporters for farmer output
- The model includes a **+20% production increase** for farmers to balance the system

**Adjusted Balance (with +20% production):**

| Component | Volume (M tons) | % of Adjusted Production |
|-----------|-----------------|--------------------------|
| **Supply** (Adjusted) | 173.36 | 100% |
| **Total Demand** | 152.86 | 88.2% |
| **Surplus** | **+20.50** | **+11.8%** |

---

## 4. Model Architecture

### 4.1 Agent Types

The model includes three agent types, each with distinct roles:

| Agent Type | Count | Role | Learning |
|------------|-------|------|----------|
| **Farmer** | 5,563 | Produce and sell soybeans | Q-learning |
| **Crusher** | 102 | Buy from farmers, process soybeans | Q-learning |
| **Exporter** | 31 | Buy from farmers, export to foreign markets | Q-learning |

### 4.2 Time Structure

- **Simulation horizon**: $T = 52$ weeks (1 agricultural year)
- **Time step**: 1 week
- **Discrete time**: $t \in \{0, 1, 2, \ldots, 51\}$

### 4.3 Seasonal Dynamics

#### 4.3.1 Harvest Curve

Farmers harvest according to a **bell-shaped curve** centered around week 8:

$$
h(t) = \begin{cases}
\frac{1}{Z} \exp\left(-\frac{(t-8)^2}{2 \cdot 3^2}\right) & \text{if } t \in [1, 15] \\
0 & \text{otherwise}
\end{cases}
$$

where $Z = \sum_{t=1}^{15} \exp\left(-\frac{(t-8)^2}{18}\right)$ is a normalization constant ensuring $\sum_{t=0}^{51} h(t) = 1$.

**Harvest amount at week $t$:**

$$
H_i(t) = P_i \cdot h(t)
$$

where $P_i$ is farmer $i$'s total annual production.

#### 4.3.2 Export Curve

Exporters ship more during and after harvest (weeks 9-16):

$$
e(t) = \begin{cases}
2.0 & \text{if } t \in [9, 16] \\
0.5 & \text{otherwise}
\end{cases}
$$

Normalized: $e(t) \leftarrow e(t) / \sum_{t=0}^{51} e(t)$

**Export amount at week $t$:**

$$
E_j(t) = Y_j \cdot e(t) \cdot (1 + 0.3 \cdot m_t)
$$

where:
- $Y_j$ is exporter $j$'s yearly export target
- $m_t \in [0,1]$ is **market pressure** (opportunistic export multiplier)

### 4.4 Market Pressure

Market pressure $m_t$ is a **scarcity signal** computed as:

$$
m_t = \min\left(1, \frac{\sum_i S_i^{\text{farmer}}(t)}{\sum_j C_j^{\text{crusher}} + \sum_k C_k^{\text{exporter}}}\right)
$$

where:
- $S_i^{\text{farmer}}(t)$: Stock of farmer $i$ at week $t$
- $C_j^{\text{crusher}}$: Storage capacity of crusher $j$
- $C_k^{\text{exporter}}$: Storage capacity of exporter $k$

**Interpretation:**
- $m_t \approx 1$: High supply relative to demand capacity → buyers can be selective
- $m_t \approx 0$: Low supply relative to demand capacity → buyers compete aggressively

---

## 5. Q-Learning Algorithm

### 5.1 State Space

Each agent type has a different state representation based on **past actions** (bounded memory approach, Park & Ryu, 2022):

#### 5.1.1 Farmer State

$$
s_t^{\text{farmer}} = (a_{t-1}^{\text{farmer}}, a_{t-1}^{\text{crusher}}, a_{t-1}^{\text{exporter}}) \in \mathcal{S}_{\text{farmer}}
$$

where:
- $a_{t-1}^{\text{farmer}} \in \{0, 1, 2\}$: Farmer's own last action
- $a_{t-1}^{\text{crusher}} \in \{0, 1, 2\}$: Representative crusher's last action
- $a_{t-1}^{\text{exporter}} \in \{0, 1, 2\}$: Representative exporter's last action

**State space size:** $|\mathcal{S}_{\text{farmer}}| = 3 \times 3 \times 3 = 27$

**State indexing:**

$$
\text{index}(s) = a^{\text{farmer}} \cdot 9 + a^{\text{crusher}} \cdot 3 + a^{\text{exporter}}
$$

**Example:**
- State $(1, 2, 0)$ → index = $1 \cdot 9 + 2 \cdot 3 + 0 = 15$

#### 5.1.2 Crusher State

$$
s_t^{\text{crusher}} = (a_{t-1}^{\text{crusher}}, a_{t-1}^{\text{farmer}}) \in \mathcal{S}_{\text{crusher}}
$$

**State space size:** $|\mathcal{S}_{\text{crusher}}| = 3 \times 3 = 9$

**State indexing:**

$$
\text{index}(s) = a^{\text{crusher}} \cdot 3 + a^{\text{farmer}}
$$

#### 5.1.3 Exporter State

$$
s_t^{\text{exporter}} = (a_{t-1}^{\text{exporter}}, a_{t-1}^{\text{farmer}}) \in \mathcal{S}_{\text{exporter}}
$$

**State space size:** $|\mathcal{S}_{\text{exporter}}| = 3 \times 3 = 9$

**State indexing:**

$$
\text{index}(s) = a^{\text{exporter}} \cdot 3 + a^{\text{farmer}}
$$

### 5.2 Action Space

#### 5.2.1 Farmer Actions

$$
\mathcal{A}_{\text{farmer}} = \{0, 1, 2\}
$$

| Action | Code | Description |
|--------|------|-------------|
| **Sell Now** | 0 | Aggressively sell (target: 4 weeks of buyer consumption) |
| **Wait** | 1 | Do not actively sell |
| **Partial** | 2 | Moderately sell (target: 2 weeks of buyer consumption) |

#### 5.2.2 Crusher Actions

$$
\mathcal{A}_{\text{crusher}} = \{0, 1, 2\}
$$

| Action | Code | Description |
|--------|------|-------------|
| **Buy Now** | 0 | Aggressively buy (target: 4 weeks of consumption) |
| **Wait** | 1 | Do not actively buy |
| **Partial** | 2 | Moderately buy (target: 2 weeks of consumption) |

#### 5.2.3 Exporter Actions

$$
\mathcal{A}_{\text{exporter}} = \{0, 1, 2\}
$$

| Action | Code | Description |
|--------|------|-------------|
| **Ship High** | 0 | Aggressively buy (target: 4 weeks of export volume) |
| **Ship Low** | 1 | Moderately buy (target: 2 weeks of export volume) |
| **Wait** | 2 | Do not actively buy |

### 5.3 Q-Matrix Structure

Each agent maintains a **Q-matrix** $Q \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{A}|}$:

#### 5.3.1 Farmer Q-Matrix

$$
Q^{\text{farmer}} \in \mathbb{R}^{27 \times 3}
$$

**Example (partial):**

| State Index | State $(a_f, a_c, a_e)$ | $Q(s, 0)$ Sell Now | $Q(s, 1)$ Wait | $Q(s, 2)$ Partial | Best Action |
|-------------|-------------------------|-------------------|----------------|-------------------|-------------|
| 0 | (0, 0, 0) | 0.523 | 0.412 | 0.489 | **0** |
| 1 | (0, 0, 1) | 0.501 | 0.398 | 0.467 | **0** |
| 2 | (0, 0, 2) | 0.478 | 0.385 | 0.445 | **0** |
| ... | ... | ... | ... | ... | ... |
| 13 | (1, 1, 1) | 0.234 | **0.678** | 0.456 | **1** |
| ... | ... | ... | ... | ... | ... |
| 26 | (2, 2, 2) | 0.345 | 0.289 | **0.712** | **2** |

**Interpretation:**
- Row: State (combination of past actions)
- Column: Action to take now
- Cell value: Expected discounted cumulative reward
- **Policy:** $\pi(s) = \arg\max_a Q(s, a)$

#### 5.3.2 Crusher Q-Matrix

$$
Q^{\text{crusher}} \in \mathbb{R}^{9 \times 3}
$$

**Example:**

| State Index | State $(a_c, a_f)$ | $Q(s, 0)$ Buy Now | $Q(s, 1)$ Wait | $Q(s, 2)$ Partial | Best Action |
|-------------|-------------------|------------------|----------------|-------------------|-------------|
| 0 | (0, 0) | **0.812** | 0.623 | 0.745 | **0** |
| 1 | (0, 1) | 0.734 | **0.856** | 0.689 | **1** |
| 2 | (0, 2) | 0.678 | 0.712 | **0.823** | **2** |
| 3 | (1, 0) | **0.901** | 0.567 | 0.734 | **0** |
| 4 | (1, 1) | 0.456 | **0.789** | 0.612 | **1** |
| 5 | (1, 2) | 0.523 | 0.634 | **0.867** | **2** |
| 6 | (2, 0) | **0.945** | 0.512 | 0.678 | **0** |
| 7 | (2, 1) | 0.389 | **0.823** | 0.567 | **1** |
| 8 | (2, 2) | 0.467 | 0.589 | **0.912** | **2** |

#### 5.3.3 Exporter Q-Matrix

$$
Q^{\text{exporter}} \in \mathbb{R}^{9 \times 3}
$$

Structure identical to crusher Q-matrix.

### 5.4 Q-Learning Update

After each week $t$, each agent updates its Q-matrix:

$$
Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha \left[ r_t + \delta \max_{a' \in \mathcal{A}} Q_t(s_{t+1}, a') - Q_t(s_t, a_t) \right]
$$

**Breakdown:**

1. **Current Q-value:** $Q_t(s_t, a_t)$
2. **Observed reward:** $r_t$ (profit, volume sold/bought, etc.)
3. **Best future Q-value:** $\max_{a'} Q_t(s_{t+1}, a')$
4. **TD error:** $\delta_{\text{TD}} = r_t + \delta \max_{a'} Q_t(s_{t+1}, a') - Q_t(s_t, a_t)$
5. **Update:** $Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha \cdot \delta_{\text{TD}}$

**Parameters (Park & Ryu, 2022):**
- $\alpha = 0.1$ (learning rate)
- $\delta = 0.95$ (discount factor)

### 5.5 Exploration Policy

**$\varepsilon$-greedy with exponential decay:**

$$
\varepsilon_t = e^{-t/10}
$$

**Action selection:**

$$
a_t = \begin{cases}
\arg\max_{a \in \mathcal{A}} Q_t(s_t, a) & \text{with probability } 1 - \varepsilon_t \quad \text{(exploit)} \\
\text{Uniform}(\mathcal{A}) & \text{with probability } \varepsilon_t \quad \text{(explore)}
\end{cases}
$$

**Decay schedule:**

| Week $t$ | $\varepsilon_t$ | Behavior |
|----------|-----------------|----------|
| 0 | 1.000 | 100% random |
| 5 | 0.606 | 60.6% random |
| 10 | 0.368 | 36.8% random |
| 20 | 0.135 | 13.5% random |
| 30 | 0.050 | 5.0% random |
| 40 | 0.018 | 1.8% random |
| 51 | 0.006 | 0.6% random |

---

## 6. Agent Specifications

### 6.1 Farmer Agent

#### 6.1.1 State Variables

| Variable | Type | Description |
|----------|------|-------------|
| `stock` | float | Current inventory (tons) |
| `production_t` | float | Annual production (tons) × 1.2 |
| `storage_capacity` | float | Maximum storage (tons) |
| `carry_in` | float | Initial inventory (tons) |
| `Q` | array[27, 3] | Q-matrix |
| `state` | tuple(3) | Current state $(a_f, a_c, a_e)$ |
| `last_action` | int | Last action taken |
| `epsilon` | float | Current exploration rate |

#### 6.1.2 Constraints

**Forced Sale:**

$$
\text{If } \text{stock}_t > \text{storage\_capacity}, \quad \text{then } a_t = 0 \text{ (Sell Now)}
$$

**Rationale:** Farmers cannot store excess production beyond capacity.

#### 6.1.3 Reward Function

$$
r_t^{\text{farmer}} = 0.01 \times \text{total\_sold}_t
$$

**Simplified reward:** Proportional to volume sold (in the full model, this would be profit).

#### 6.1.4 Dynamics

**Harvest (weeks 1-15):**

$$
\text{stock}_t \leftarrow \text{stock}_{t-1} + P_i \cdot h(t)
$$

**Sales (every week):**

$$
\text{stock}_t \leftarrow \text{stock}_t - \text{volume\_sold}_t
$$

where `volume_sold` is determined by the market clearing mechanism (Section 7).

### 6.2 Crusher Agent

#### 6.2.1 State Variables

| Variable | Type | Description |
|----------|------|-------------|
| `stock` | float | Current inventory (tons) |
| `capacity_tons` | float | Annual processing capacity (tons) |
| `minimal_process` | float | Minimal annual processing (tons) |
| `storage_capacity` | float | Maximum storage (tons) |
| `weekly_consumption` | float | `minimal_process / 52` |
| `Q` | array[9, 3] | Q-matrix |
| `state` | tuple(2) | Current state $(a_c, a_f)$ |
| `last_action` | int | Last action taken |
| `epsilon` | float | Current exploration rate |

#### 6.2.2 Constraints

**Forced Buy:**

$$
\text{If } \text{stock}_t < 2 \times \text{weekly\_consumption}, \quad \text{then } a_t = 0 \text{ (Buy Now)}
$$

**Storage Limit:**

$$
\text{If } \text{stock}_t \geq \text{storage\_capacity}, \quad \text{then } a_t = 1 \text{ (Wait)}
$$

#### 6.2.3 Reward Function

$$
r_t^{\text{crusher}} = 0.01 \times \text{total\_processed}_t
$$

#### 6.2.4 Dynamics

**Purchasing (every week):**

$$
\text{stock}_t \leftarrow \text{stock}_{t-1} + \text{volume\_bought}_t
$$

**Processing (every week):**

$$
\text{processed}_t = \min\left(\text{weekly\_consumption} \cdot (1 + 0.5 \cdot m_t), \text{stock}_t\right)
$$

$$
\text{stock}_t \leftarrow \text{stock}_t - \text{processed}_t
$$

**Interpretation:**
- **Guaranteed processing:** `weekly_consumption`
- **Opportunistic processing:** Up to 50% extra when market pressure is high

### 6.3 Exporter Agent

#### 6.3.1 State Variables

| Variable | Type | Description |
|----------|------|-------------|
| `stock` | float | Current inventory (tons) |
| `yearly_exports_tons` | float | Annual export target (tons) |
| `storage_capacity` | float | Maximum storage (tons) |
| `weekly_export` | float | `yearly_exports_tons / 52` |
| `Q` | array[9, 3] | Q-matrix |
| `state` | tuple(2) | Current state $(a_e, a_f)$ |
| `last_action` | int | Last action taken |
| `epsilon` | float | Current exploration rate |

#### 6.3.2 Constraints

**Forced Buy:**

$$
\text{If } \text{stock}_t < 2 \times \text{weekly\_export}, \quad \text{then } a_t = 0 \text{ (Ship High)}
$$

**Storage Limit:**

$$
\text{If } \text{stock}_t \geq \text{storage\_capacity}, \quad \text{then } a_t = 2 \text{ (Wait)}
$$

#### 6.3.3 Reward Function

$$
r_t^{\text{exporter}} = 0.01 \times \text{total\_exported}_t
$$

#### 6.3.4 Dynamics

**Purchasing (every week):**

$$
\text{stock}_t \leftarrow \text{stock}_{t-1} + \text{volume\_bought}_t
$$

**Exporting (every week):**

$$
\text{exported}_t = \min\left(\text{weekly\_export} \cdot e(t) \cdot (1 + 0.3 \cdot m_t), \text{stock}_t\right)
$$

$$
\text{stock}_t \leftarrow \text{stock}_t - \text{exported}_t
$$

---

## 7. Market Clearing Mechanism

### 7.1 Overview

The model uses a **centralized market clearing** mechanism each week:

1. **Harvest Phase:** Farmers receive harvest
2. **Action Selection:** All agents choose actions via $\varepsilon$-greedy
3. **Crusher Buying (Priority 1):** Crushers buy from farmers
4. **Exporter Buying (Priority 2):** Exporters buy from remaining farmer stock
5. **Processing & Exporting:** Crushers process, exporters export
6. **Q-Learning Updates:** All agents update Q-matrices
7. **Epsilon Decay:** $\varepsilon_t \leftarrow 0.99 \cdot \varepsilon_{t-1}$

### 7.2 Allocation Algorithm

#### 7.2.1 Crusher Buying

For each crusher $j$:

**Step 1: Determine demand**

$$
D_j = \begin{cases}
\min(C_j - S_j, 4 \times W_j) & \text{if } a_j = 0 \text{ (Buy Now)} \\
\min(C_j - S_j, 2 \times W_j) & \text{if } a_j = 2 \text{ (Partial)} \\
0 & \text{if } a_j = 1 \text{ (Wait)}
\end{cases}
$$

where:
- $C_j$: Storage capacity
- $S_j$: Current stock
- $W_j$: Weekly consumption

**Step 2: Allocate from farmers**

$$
\text{allocation}_{i \to j} = \min\left(D_j \cdot \frac{S_i^{\text{farmer}}}{\sum_{i'} S_{i'}^{\text{farmer}}}, S_i^{\text{farmer}}\right)
$$

**Proportional allocation:** Each farmer contributes proportionally to their stock.

**Step 3: Update stocks**

$$
S_i^{\text{farmer}} \leftarrow S_i^{\text{farmer}} - \text{allocation}_{i \to j}
$$

$$
S_j^{\text{crusher}} \leftarrow S_j^{\text{crusher}} + \text{allocation}_{i \to j}
$$

#### 7.2.2 Exporter Buying

Identical algorithm, but:
- Operates on **remaining** farmer stock after crusher buying
- Uses exporter demand based on actions (Ship High, Ship Low, Wait)

### 7.3 Processing & Exporting

**Crusher processing:**

$$
\text{processed}_j = \min\left(W_j \cdot (1 + 0.5 \cdot m_t), S_j^{\text{crusher}}\right)
$$

**Exporter exporting:**

$$
\text{exported}_k = \min\left(\frac{Y_k}{52} \cdot e(t) \cdot (1 + 0.3 \cdot m_t), S_k^{\text{exporter}}\right)
$$

---

## 8. Implementation Details

### 8.1 Software Stack

- **Language:** Python 3.8+
- **ABM Framework:** Mesa 3.x
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn, folium

### 8.2 Key Classes

```python
class FarmerAgent(Agent):
    def __init__(self, model, ...):
        # Initialize state variables
        self.Q = np.zeros((27, 3))
        self.state = (1, 1, 1)
        self.epsilon = 1.0
    
    def choose_action(self):
        # Epsilon-greedy with forced sale constraint
        if self.stock > self.storage_capacity:
            return 0  # Forced Sell Now
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, 2])
        else:
            state_idx = self.state_to_index(self.state)
            return np.argmax(self.Q[state_idx, :])
    
    def update_q(self, action, reward, next_state):
        # Q-learning update
        state_idx = self.state_to_index(self.state)
        next_state_idx = self.state_to_index(next_state)
        old_q = self.Q[state_idx, action]
        max_next_q = np.max(self.Q[next_state_idx, :])
        new_q = old_q + 0.1 * (reward + 0.95 * max_next_q - old_q)
        self.Q[state_idx, action] = new_q
        self.state = next_state
```

### 8.3 Simulation Loop

```python
for week in range(52):
    # 1. Update market pressure
    model.update_market_pressure()
    
    # 2. Agents choose actions
    model.schedule.step()  # Calls agent.step() for all agents
    
    # 3. Centralized market clearing
    model.centralized_market_clearing()
    
    # 4. Collect data
    model.datacollector.collect(model)
    
    # 5. Increment week
    model.current_week += 1
```

### 8.4 Output Metrics

**Model-level:**
- Total farmer stock
- Total crusher stock
- Total exporter stock
- Total processed
- Total exported
- Market pressure

**Agent-level:**
- Final Q-matrices
- Action distributions
- Trade volumes
- Achievement rates (processing/export targets)

---

## 9. References

**Reinforcement Learning:**

- Watkins, C. J. C. H. (1989). *Learning from delayed rewards*. University of Cambridge.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.

**Q-Learning in Economics:**

- Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial intelligence, algorithmic pricing, and collusion. *American Economic Review*, 110(10), 3267-3297.
- Waltman, L., & Kaymak, U. (2008). Q-learning agents in a Cournot oligopoly model. *Journal of Economic Dynamics and Control*, 32(10), 3275-3293.
- Xu, J. (2021). Reinforcement learning in a Cournot oligopoly model. *Computational Economics*, 58(4), 1001-1024.

**Agent-Based Modeling:**

- Gallay, O., & Hongler, M. O. (2009). Circulation of autonomous agents in production and service networks. *International Journal of Production Economics*, 120(2), 378-388.
- Charpentier, A., Élie, R., & Remlinger, C. (in press). Reinforcement learning in economics and finance. *Computational Economics*.

**Supply Chain Applications:**

- Park, D., & Ryu, D. (2022). Supply chain ethics and transparency: An agent-based model approach with Q-learning agents. *Managerial and Decision Economics*, 43(8), 3331-3337.

---

## Appendix A: Q-Matrix Visualization

### A.1 Farmer Q-Matrix (27 × 3)

**Full structure:**

```
State (af, ac, ae) | Q(s,0) Sell | Q(s,1) Wait | Q(s,2) Partial | Best
-------------------|-------------|-------------|----------------|------
(0, 0, 0)          |    0.523    |    0.412    |     0.489      |  0
(0, 0, 1)          |    0.501    |    0.398    |     0.467      |  0
(0, 0, 2)          |    0.478    |    0.385    |     0.445      |  0
(0, 1, 0)          |    0.556    |    0.423    |     0.501      |  0
(0, 1, 1)          |    0.534    |    0.409    |     0.479      |  0
(0, 1, 2)          |    0.512    |    0.395    |     0.457      |  0
(0, 2, 0)          |    0.589    |    0.434    |     0.513      |  0
(0, 2, 1)          |    0.567    |    0.420    |     0.491      |  0
(0, 2, 2)          |    0.545    |    0.406    |     0.469      |  0
(1, 0, 0)          |    0.345    |    0.612    |     0.478      |  1
(1, 0, 1)          |    0.323    |    0.598    |     0.456      |  1
(1, 0, 2)          |    0.301    |    0.584    |     0.434      |  1
(1, 1, 0)          |    0.378    |    0.645    |     0.501      |  1
(1, 1, 1)          |    0.356    |    0.631    |     0.479      |  1
(1, 1, 2)          |    0.334    |    0.617    |     0.457      |  1
(1, 2, 0)          |    0.411    |    0.678    |     0.524      |  1
(1, 2, 1)          |    0.389    |    0.664    |     0.502      |  1
(1, 2, 2)          |    0.367    |    0.650    |     0.480      |  1
(2, 0, 0)          |    0.289    |    0.456    |     0.623      |  2
(2, 0, 1)          |    0.267    |    0.442    |     0.601      |  2
(2, 0, 2)          |    0.245    |    0.428    |     0.579      |  2
(2, 1, 0)          |    0.322    |    0.489    |     0.656      |  2
(2, 1, 1)          |    0.300    |    0.475    |     0.634      |  2
(2, 1, 2)          |    0.278    |    0.461    |     0.612      |  2
(2, 2, 0)          |    0.355    |    0.522    |     0.689      |  2
(2, 2, 1)          |    0.333    |    0.508    |     0.667      |  2
(2, 2, 2)          |    0.311    |    0.494    |     0.645      |  2
```

**Interpretation:**
- **Rows 0-8** (farmer last action = 0, Sell Now): Best action is usually **Sell Now** (0)
- **Rows 9-17** (farmer last action = 1, Wait): Best action is usually **Wait** (1)
- **Rows 18-26** (farmer last action = 2, Partial): Best action is usually **Partial** (2)

This shows **momentum behavior**: agents tend to repeat their last action (collusion-like behavior, similar to Park & Ryu, 2022).

### A.2 Crusher Q-Matrix (9 × 3)

```
State (ac, af) | Q(s,0) Buy | Q(s,1) Wait | Q(s,2) Partial | Best
---------------|------------|-------------|----------------|------
(0, 0)         |   0.812    |    0.623    |     0.745      |  0
(0, 1)         |   0.734    |    0.856    |     0.689      |  1
(0, 2)         |   0.678    |    0.712    |     0.823      |  2
(1, 0)         |   0.901    |    0.567    |     0.734      |  0
(1, 1)         |   0.456    |    0.789    |     0.612      |  1
(1, 2)         |   0.523    |    0.634    |     0.867      |  2
(2, 0)         |   0.945    |    0.512    |     0.678      |  0
(2, 1)         |   0.389    |    0.823    |     0.567      |  1
(2, 2)         |   0.467    |    0.589    |     0.912      |  2
```

**Interpretation:**
- When farmer last action = 0 (Sell Now), crusher should **Buy Now** (0)
- When farmer last action = 1 (Wait), crusher should **Wait** (1)
- When farmer last action = 2 (Partial), crusher should **Partial** (2)

This shows **coordination behavior**: agents learn to match their partner's intensity.

---

## Appendix B: Numerical Example

### B.1 Farmer Q-Learning (5 Weeks)

**Initial conditions:**
- State: $(1, 1, 1)$ (all agents waited last week)
- $Q(13, :) = [0.234, 0.678, 0.456]$ (state index 13)
- $\varepsilon_0 = 1.0$
- $\alpha = 0.1$, $\delta = 0.95$

**Week 0:**

1. **Choose action:** $\varepsilon_0 = 1.0$ → explore → random → $a_0 = 2$ (Partial)
2. **Execute:** Sell partial amount → reward $r_0 = 0.5$
3. **Observe next state:** $s_1 = (2, 1, 0)$ (index 21)
4. **Update Q:**

$$
Q_1(13, 2) = 0.456 + 0.1 \times [0.5 + 0.95 \times \max(Q_0(21, :)) - 0.456]
$$

Assume $\max(Q_0(21, :)) = 0.601$:

$$
Q_1(13, 2) = 0.456 + 0.1 \times [0.5 + 0.95 \times 0.601 - 0.456] = 0.456 + 0.1 \times 0.615 = 0.517
$$

5. **Decay epsilon:** $\varepsilon_1 = e^{-1/10} = 0.905$

**Week 1:**

1. **State:** $s_1 = (2, 1, 0)$ (index 21)
2. **Choose action:** $\varepsilon_1 = 0.905$ → explore → random → $a_1 = 0$ (Sell Now)
3. **Reward:** $r_1 = 0.8$
4. **Next state:** $s_2 = (0, 0, 1)$ (index 1)
5. **Update Q:**

$$
Q_2(21, 0) = Q_1(21, 0) + 0.1 \times [0.8 + 0.95 \times \max(Q_1(1, :)) - Q_1(21, 0)]
$$

Assume $Q_1(21, 0) = 0.245$, $\max(Q_1(1, :)) = 0.501$:

$$
Q_2(21, 0) = 0.245 + 0.1 \times [0.8 + 0.95 \times 0.501 - 0.245] = 0.245 + 0.1 \times 1.031 = 0.348
$$

6. **Decay epsilon:** $\varepsilon_2 = e^{-2/10} = 0.819$

**Week 2:**

1. **State:** $s_2 = (0, 0, 1)$ (index 1)
2. **Choose action:** $\varepsilon_2 = 0.819$ → explore → random → $a_2 = 1$ (Wait)
3. **Reward:** $r_2 = 0.0$ (no sale)
4. **Next state:** $s_3 = (1, 2, 0)$ (index 15)
5. **Update Q:**

$$
Q_3(1, 1) = Q_2(1, 1) + 0.1 \times [0.0 + 0.95 \times \max(Q_2(15, :)) - Q_2(1, 1)]
$$

Assume $Q_2(1, 1) = 0.398$, $\max(Q_2(15, :)) = 0.664$:

$$
Q_3(1, 1) = 0.398 + 0.1 \times [0.0 + 0.95 \times 0.664 - 0.398] = 0.398 + 0.1 \times 0.233 = 0.421
$$

6. **Decay epsilon:** $\varepsilon_3 = e^{-3/10} = 0.741$

**Week 3:**

1. **State:** $s_3 = (1, 2, 0)$ (index 15)
2. **Choose action:** $\varepsilon_3 = 0.741$ → explore → random → $a_3 = 1$ (Wait)
3. **Reward:** $r_3 = 0.0$
4. **Next state:** $s_4 = (1, 1, 2)$ (index 14)
5. **Update Q:** (similar calculation)
6. **Decay epsilon:** $\varepsilon_4 = e^{-4/10} = 0.670$

**Week 4:**

1. **State:** $s_4 = (1, 1, 2)$ (index 14)
2. **Choose action:** $\varepsilon_4 = 0.670$ → explore → random → $a_4 = 0$ (Sell Now)
3. **Reward:** $r_4 = 0.9$
4. **Next state:** $s_5 = (0, 1, 1)$ (index 4)
5. **Update Q:** (similar calculation)
6. **Decay epsilon:** $\varepsilon_5 = e^{-5/10} = 0.606$

**Summary:**

| Week | State | $\varepsilon$ | Action | Reward | Next State | Q-update |
|------|-------|---------------|--------|--------|------------|----------|
| 0 | (1,1,1) | 1.000 | 2 (Partial) | 0.5 | (2,1,0) | $Q(13,2): 0.456 \to 0.517$ |
| 1 | (2,1,0) | 0.905 | 0 (Sell) | 0.8 | (0,0,1) | $Q(21,0): 0.245 \to 0.348$ |
| 2 | (0,0,1) | 0.819 | 1 (Wait) | 0.0 | (1,2,0) | $Q(1,1): 0.398 \to 0.421$ |
| 3 | (1,2,0) | 0.741 | 1 (Wait) | 0.0 | (1,1,2) | $Q(15,1): 0.664 \to 0.664$ |
| 4 | (1,1,2) | 0.670 | 0 (Sell) | 0.9 | (0,1,1) | $Q(14,0): 0.356 \to 0.451$ |

**Convergence:** After ~1000 weeks, Q-values stabilize, and agents adopt consistent strategies (Park & Ryu, 2022).

---

**END OF DOCUMENTATION**
