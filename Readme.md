# Optimistic Random Exploration

This repository contains a presentation of the paper: ["Optimistic Active Exploration of Dynamical Systems" (Sukhija et al., 2023)](https://arxiv.org/abs/2306.12371) (code available at: [lasgroup/opax](https://github.com/lasgroup/opax)) as as part of the course "Foundations of Reinforcement Learning (263-5255-00L)" at ETH Zurich.

It contains experiments with an additional baseline which we call "Optimistic Random Exploration" (ORX), which replaces the maximization over the hallucination policy $\eta\in[-1,1]^{\dim\mathcal{S}}$ in the OpAx optimal control problem by a random sampling procedure.