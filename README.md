# CogMods: Hand Coded Cognitive Psychology Models in Python 

## Dependencies
- numpy

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
    - ...

- verification needed:
    - GCM & Prototype model seem like they're running correctly, but should be verified
    - ALCOVE needs to be verified; should review Nolan Conaway's implementation, CatLearnR implementation, as well as the original paper
    - DIVA, Autoencoder, MLC are verified against AutoGrad (automatic differentiation tool by HIPS @ Harvard), to the best of my knowledge

---

## Misc
- most of this is written by me (matt wetzel), with heavy influence from (like, sometimes blatantly copying off of :) ) [Nolan Conaway](https://nolanbconaway.github.io/)
    - plus a host of other resources
- feel free to email if there are bugs/issues/problems/etc
    - or flag the issue on github
