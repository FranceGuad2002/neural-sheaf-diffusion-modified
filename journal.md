### 🗓️ 19 March 2026

---

## 1. Claim Verification  
**Hypothesis:** *Curvature improves sheaf diffusion*

**Observations**
- Diagonal Sheaf Diffusion *(d = 1)* does **not** exhibit the same oversmoothing as Bundle Sheaf Diffusion *(d = 4)*  
- Diagonal sheaves **allow a curvature term**, whereas bundle sheaves do not  
- Stability of the curvature term holds even with **10 layers**

**Key Question**  
> Does curvature flexibility improve sheaf expressivity?

---

## 2. New Insight  
**Curvature induces geometry**

- For \( d = 1 \), curvature on graphs is **highly negative at central nodes**  
- ⇒ The Sheaf Laplacian induces a **hyperbolic geometry** on the graph

**Implications**
- Diagonal sheaves outperform bundle sheaves in \( d = 1 \)  
- Bundle sheaves lack curvature **by construction**  
- Introducing curvature modifies the graph geometry via the Laplacian  

**Working Hypothesis**
> Curvature helps mitigate **oversmoothing / oversquashing**

**Status**
- Preliminary results: ✅ promising  
- Validation: ❗ still needed across datasets  

---

## 3. Action Points

- Test on **more datasets with bottlenecks**  
  - Reference: [Bundle Neural Networks for Message Diffusion on Graphs](https://arxiv.org/pdf/2405.15540)  
- Discuss with **Alessio**  
  - Focus: continuous models + asymptotics  



# 30/03 Journal , action points 

- Luigi: read Forman-Bochner
- Francesco: estimate simple case for oversquashing sheaves
- together: do visualizations and plots with the right notions of curvature Francesco would derive 

# Luigi things to do 
- Take, the graph, take the Laplacian, associate to each edge of the graph the weight w_uv = | \tilde{\Delta}_uv | (where \tilde{\Delta} is indeed the Laplacian you get from our code.)
- so now you have a weight for each edge. Instead, you can assume your nodes to have weight = 1. 
- for each edge, calculate the curvature as shown in FORMULA (11) of the paper https://arxiv.org/pdf/1607.08654 by Melanie Weber, Jurgen Jost et al. 
- plot the curvature of each edge as we did in the initial plots with our first experiments
- you can try also to put w(v) = degree of node v , for each node, if the previous plots are not very informative

# Luigi other things to do 31/03
- Do the visualization, transpose it, with cornell (2 or 3 layers, do both) and save it, we will use it in the slides 
- Plot the histogram of the curvatures for each layer, and also plot the histogram of the topological edge-curvature
- *topological edge curvature*: already done in other jupyters, get the boundary matrix, do B^T B, do fbc there, here you have it 
- plot the histogram with Cora, 3 layers, and save it, no need to plot the cora graph as it is disconnected and visualization is bad, with cora plot only the histogram