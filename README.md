# Cognitive Psychology Models in Python 

## Dependencies
- numpy
- scipy (sometimes)
- matplotlib (sometimes for plotting)

---

## Models
- ALCOVE (Kruschke, 1992)
- Autoencoder
- DIVA (Kurtz, 2007)
- GCM (Nosofsky, from: Pothos & Wills, 2011)
- Multilayer Perceptron/Classifier (MLC)
    - WARP (Kurtz, MLC with exponentional activation function)
    - MLC w/ Momentum
    - MLC w/ Particle Swarm Optimization added to Hidden Layer
    - MLC w/ Self Organizing Map added to Hidden Layer
- Multiple Autoencoders
- Prototype (Minda & Smith, from: Pothos & Wills, 2011)

---

## Overview
- most models include `fit(...)` & `predict(...)` functions when applicable (following industry trends)
- `response(...)` produces probabilities; `predict(...)` produces class predictions

---

## Road Map
- models to add:
    - SUSTAIN
    - COVIS
    - RATIONAL (RMC)

---

## Misc
- feel free to email if there are bugs/issues/problems/etc
    - or flag the issue on github
