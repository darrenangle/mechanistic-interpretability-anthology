#!/usr/bin/env python3
"""Build the Mechanistic Interpretability print anthology.

Stages (run all by default):
    fetch   download arXiv PDFs, print-render Transformer Circuits HTML papers,
            verify identity/integrity, cache metadata in build/metadata.json
    build   generate covers/dividers/index, iterate until page offsets are
            stable, assemble master PDF, bookmarks, links, manifests, README
    qa      automated validation + preview renders (build/qa/)
    export  text-extracted Markdown (markdown/) and HTML (html/) editions of every
            paper; only papers whose source PDF changed are re-extracted

Usage:  python scripts/build_anthology.py [fetch] [build] [qa] [export] [--force]
"""
from __future__ import annotations

import asyncio
import csv
import datetime as dt
import hashlib
import io
import json
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import pymupdf
import requests
from reportlab.graphics import renderPDF
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "sources"
BUILD = ROOT / "build"
META_CACHE = BUILD / "metadata.json"
MASTER = ROOT / "Mechanistic_Interpretability_Complete_Anthology.pdf"
PRINT_INDEX = ROOT / "PRINT_INDEX.pdf"
REPO_URL = "https://github.com/darrenangle/mechanistic-interpretability-anthology"
RELEASE_PDF_URL = f"{REPO_URL}/releases/latest/download/{MASTER.name}"
PAGES_URL = "https://darrenangle.github.io/mechanistic-interpretability-anthology/"
SITE_URL = "https://darren.computer/mech-interp/"  # proxies PAGES_URL (rewrite in the darren.computer repo's vercel.json)
TITLE = "Mechanistic Interpretability"
# Master-PDF image recompression: raster images become JPEG at this quality and are downsampled only above
# IMG_DPI_THRESHOLD (to IMG_DPI_TARGET, still print resolution). Text, fonts and vector art are untouched, so
# everything stays selectable; sources/ keeps the original files byte-for-byte.
IMG_QUALITY, IMG_DPI_THRESHOLD, IMG_DPI_TARGET = 85, 330, 300
DISTILL_LICENSE = "CC BY 4.0 (Distill: text and diagrams)"
UA = {"User-Agent": "mi-anthology-builder/1.0 (personal study print anthology)"}
W, H = letter  # 612 x 792 pt
TODAY = dt.date.today().isoformat()

# --------------------------------------------------------------------------
# Catalog — exact curriculum reading order
# --------------------------------------------------------------------------
SECTIONS = {
    "A": "Orientation",
    "B": "Foundations: vision models and early transformers",
    "C": "Causal methods and their critics",
    "D": "What is a feature?",
    "E": "Dictionary learning at full strength",
    "F": "The reckoning",
    "G": "Competing programmes, 2025–26",
    "H": "Interpretability in alignment practice",
    "I": "Is this worth doing?",
}
PRIORITY = {
    "P0": ("P0 — ESSENTIAL", "Read deeply, annotate, and potentially reproduce."),
    "P1": ("P1 — IMPORTANT", "Read carefully."),
    "F": ("F — FRONTIER", "Contemporary work requiring critical evaluation."),
}
TC = "https://transformer-circuits.pub"
KINDS = ("paper", "essay", "post", "note")
# LessWrong / AI Alignment Forum posts: print only the post header and body (no comments, sidebars or vote widgets).
# The forum keeps a live connection open, so the page never goes network-idle; wait a fixed time instead.
FORUM_RENDER = dict(keep=[".LWPostsPageHeader-titleSection", ".PostsPage-postContent"], idle=False, wait=6000)
FORUM_LICENSE = "No explicit licence (LessWrong / AI Alignment Forum terms leave copyright with the author)"


def P(key, sec, pri, cite, year, title, short, why, focus, *, arxiv=None, html=None, pdf=None, slug, note="", venue=None,
      render=None, license=None, optional="", authors=None, kind="paper", byline=None):
    """One catalog entry. `key` is the short name used for cross-references in CONTEXT/optional notes;
    the paper number comes from its position in CATALOG. Exactly one source: `arxiv` (id with version), `html`
    (print-rendered web publication) or `pdf` (publisher PDF URL). `kind` is paper, essay, post or note; the
    non-papers are checked against shorter length thresholds and need no reference list."""
    assert sum(x is not None for x in (arxiv, html, pdf)) == 1, key
    assert kind in KINDS, kind
    return dict(key=key, sec=sec, pri=pri, cite=cite, year=year, title=title, short=short,
                why=why, focus=focus, arxiv=arxiv, html=html, pdf=pdf, slug=slug, note=note, license=license,
                authors=authors, venue=venue or ("Transformer Circuits Thread" if html else "arXiv"),
                render=render or {}, optional=optional, kind=kind,
                byline=byline or cite.split()[0])  # name the identity check expects on the source's first pages


CATALOG = [
    # ---- A
    P("olahessay", "A", "P0", "Olah", "2022", "Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases", "Interpretable Bases (essay)",
      "Olah's short case for treating a trained network as a compiled program to be decompiled: parameters as binary, activations as memory, and interpretable features as the variables without which reverse engineering cannot start.",
      "The program–network analogy table, the curse-of-dimensionality argument for reading weights rather than input–output behaviour, why activations must be split into independently understandable variables, privileged bases, and the FAQ on what a feature is.",
      html=f"{TC}/2022/mech-interp-essay/index.html", slug="interpretable_bases_essay", kind="essay",
      authors=["Chris Olah"]),
    P("mechanistic", "A", "P1", "Saphra & Wiegreffe", "2024", "Mechanistic?", "Mechanistic?",
      "A short history and sociology of the word ‘mechanistic’: four competing senses of the term, and how a safety-motivated community arriving from vision research and online forums came to repeat much of NLP interpretability under a new label.",
      "The four definitions (narrow and broad, technical and cultural), Olah's coinage as a break from saliency methods, the parallel histories of probing, attention maps and neuron analysis, the clash of communities, and the closing appeal to connect.",
      arxiv="2410.09087v1", slug="mechanistic_question"),
    P("openproblems", "A", "P0", "Sharkey et al.", "2025", "Open Problems in Mechanistic Interpretability", "Open Problems in Mechanistic Interpretability",
      "A 29-author review of what mechanistic interpretability cannot yet do, sorted into open problems of method, of application, and of the field's place in society, written by researchers from many of the groups whose work follows.",
      "Reverse engineering versus concept-based interpretability, the list of sparse-dictionary limitations in Section 2.1.2, validating explanations, the application goals (monitoring, control, prediction, microscope AI), and the socio-technical section.",
      arxiv="2501.16496v1", slug="open_problems"),
    P("vaswani", "A", "P1", "Vaswani et al.", "2017", "Attention Is All You Need", "Attention Is All You Need",
      "Defines the Transformer — multi-head attention, residual blocks, positional encodings — the architecture every later paper reverse-engineers.",
      "Scaled dot-product and multi-head attention, Q/K/V projections, residual + LayerNorm structure, positional encodings, and the attention visualizations in the appendix.",
      arxiv="1706.03762v7", slug="attention_is_all_you_need",
      optional="Background only: readers who know the Transformer can skip it, since {framework} restates the parts interpretability relies on."),
    # ---- B
    P("featurevis", "B", "P1", "Olah et al.", "2017", "Feature Visualization", "Feature Visualization",
      "The working manual for seeing what a vision neuron responds to by optimising the input, and an early, frank admission that neurons may not be the right units: one GoogLeNet neuron answers to cats, foxes and cars.",
      "Optimisation objectives, why optimisation separates causes from correlates, diversity terms, neuron interaction and random directions, the high-frequency ‘enemy’, the regularisation spectrum, and decorrelated Fourier preconditioning.",
      html="https://distill.pub/2017/feature-visualization/", slug="feature_visualization", venue="Distill",
      license=DISTILL_LICENSE),
    P("netdissect", "B", "P1", "Bau et al.", "2017", "Network Dissection: Quantifying Interpretability of Deep Visual Representations", "Network Dissection",
      "Turns unit interpretability into a number: each CNN unit's activation map is scored against densely labelled concepts, showing that interpretability is tied to the basis and is destroyed by rotation without loss of accuracy.",
      "The Broden dataset and IoU scoring, the random-rotation experiment, unique detectors per layer, comparisons across architectures and supervised or self-supervised tasks, and the training-condition ablations (dropout, batch normalisation).",
      arxiv="1704.05796v1", slug="network_dissection"),
    P("zoomin", "B", "P0", "Olah et al.", "2020", "Zoom In: An Introduction to Circuits", "Zoom In: An Introduction to Circuits",
      "Sets out the circuits programme — features as units, circuits as their connections, universality — and introduces polysemantic neurons and superposition through vision-model examples.",
      "The three speculative claims, curve detectors and other worked circuits, the polysemantic cat-and-car neuron, the car-to-dog circuit offered as superposition, and the case for interpretability as a natural science.",
      html="https://distill.pub/2020/circuits/zoom-in/", slug="zoom_in", venue="Distill", license=DISTILL_LICENSE),
    P("curves", "B", "P1", "Cammarata et al.", "2020", "Curve Detectors", "Curve Detectors",
      "A deliberately exhaustive study of one neuron family in InceptionV1, written to persuade sceptics that meaningful features exist: curve detectors examined through feature visualisation, dataset labelling, attribution and synthetic stimuli.",
      "The four claims (causality, generality, purity, family), the hand-labelled comparison of over 800 images, joint and radial tuning curves, synthetic curves and angles, the curve families across layers, and the unexplained combing effect.",
      html="https://distill.pub/2020/circuits/curve-detectors/", slug="curve_detectors", venue="Distill",
      license=DISTILL_LICENSE),
    P("multimodal", "B", "P1", "Goh et al.", "2021", "Multimodal Neurons in Artificial Neural Networks", "Multimodal Neurons",
      "Finds neurons in CLIP that respond to the same concept across photographs, drawings and written text, and traces the typographic attacks and biases that follow from them.",
      "Person, region and emotion neurons, the feature-properties discussion including conjoined features, typographic attacks, and the section on bias.",
      html="https://distill.pub/2021/multimodal-neurons/", slug="multimodal_neurons", venue="Distill",
      license=DISTILL_LICENSE),
    P("framework", "B", "P0", "Elhage et al.", "2021", "A Mathematical Framework for Transformer Circuits", "Mathematical Framework for Transformer Circuits",
      "Recasts attention-only transformers as sums of paths, introducing the residual stream view, QK/OV circuits, and head composition — the core vocabulary of the field.",
      "Residual stream as communication channel, QK vs. OV circuits, path expansion of 1- and 2-layer models, K-composition and induction heads, and the notation appendix.",
      html=f"{TC}/2021/framework/index.html", slug="mathematical_framework"),
    P("induction", "B", "P0", "Olsson et al.", "2022", "In-context Learning and Induction Heads", "In-context Learning and Induction Heads",
      "Argues that induction heads may constitute the mechanism for the majority of in-context learning, using six complementary lines of evidence across model scales.",
      "The training-loss phase change, each of the six arguments and how strong it is per model class, ablation evidence, and the authors' stated caveats for large models.",
      arxiv="2209.11895v1", slug="induction_heads"),
    P("logitlens", "B", "P1", "nostalgebraist", "2020", "interpreting GPT: the logit lens", "The Logit Lens",
      "A blog post that decodes GPT-2's intermediate residual stream with the model's own unembedding, finding that a guess at the next token forms by the middle layers and is then refined, while the input token vanishes almost at once.",
      "The top-guess, rank and KL plots across GPT-2's 48 layers, the rare-token copying and repetition examples, the argument from residual connections and weight decay for a shared basis, and the caveat that the lens shows only part of the picture.",
      html="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens", slug="logit_lens",
      venue="LessWrong", kind="post", license=FORUM_LICENSE, authors=["nostalgebraist"], render=FORUM_RENDER),
    P("ioi", "B", "P0", "Wang et al.", "2022", "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small", "Interpretability in the Wild (IOI circuit)",
      "The canonical end-to-end circuit analysis in a real language model: attention-head classes in GPT-2 small that implement indirect object identification.",
      "Path patching, the head classes (duplicate-token, S-inhibition, name movers, backup/negative name movers), and the faithfulness / completeness / minimality criteria.",
      arxiv="2211.00593v1", slug="ioi_circuit"),
    P("grokking", "B", "P0", "Nanda et al.", "2023", "Progress Measures for Grokking via Mechanistic Interpretability", "Progress Measures for Grokking",
      "Fully reverse-engineers a one-layer transformer that groks modular addition — inputs embedded as rotations, combined by trigonometric identities — and uses that understanding to build progress measures that move smoothly before the sudden jump.",
      "The Fourier multiplication algorithm, the four lines of evidence including ablations in Fourier space, restricted and excluded loss, the memorisation / circuit formation / cleanup phases, the role of weight decay, and the stated limits for larger models.",
      arxiv="2301.05217v3", slug="grokking_progress_measures"),
    P("clockpizza", "B", "P1", "Zhong et al.", "2023", "The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks", "The Clock and the Pizza",
      "Shows that networks trained on modular addition do not reliably rediscover one algorithm: small changes to hyperparameters and initialisation yield the known Clock, a new Pizza algorithm, or several imperfect copies running in parallel.",
      "Why gradient symmetricity and logit patterns rule out the Clock, the Pizza algorithm and circle isolation, accompanying pizzas, the distance-irrelevance metric, and the phase diagram over attention rate and width.",
      arxiv="2306.17844v2", slug="clock_and_pizza"),
    P("othello", "B", "P1", "Li et al.", "2023", "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task", "Emergent World Representations (Othello-GPT)",
      "Trains a GPT on Othello move sequences and finds an emergent board-state representation that nonlinear probes recover and interventions can edit — evidence that a sequence model can learn a world model, not only surface statistics.",
      "The synthetic and championship datasets, linear versus two-layer MLP probe error rates against a randomly initialised baseline, the gradient-based intervention technique, and latent saliency maps.",
      arxiv="2210.13382v5", slug="othello_gpt", note="arXiv first posted October 2022; published at ICLR 2023."),
    P("othellolinear", "B", "P1", "Nanda", "2023", "Actually, Othello-GPT Has A Linear Emergent World Representation", "Othello-GPT Is Linear",
      "Finds that Othello-GPT represents the board linearly after all — as ‘my’ versus ‘their’ pieces rather than black versus white — and steers it to new legal moves by flipping a single probe direction.",
      "Why the original linear probes failed, probes trained on alternate moves, the intervention by negating a probe coordinate and its fiddly scale, the neuron read through the probe, and the caveats about generalising from a toy model.",
      html="https://www.lesswrong.com/posts/nmxzr2zsjNtjaHh7x/actually-othello-gpt-has-a-linear-emergent-world",
      slug="othello_gpt_linear", venue="LessWrong", kind="post", license=FORUM_LICENSE, authors=["Neel Nanda"],
      render=FORUM_RENDER),
    # ---- C
    P("rome", "C", "P0", "Meng et al.", "2022", "Locating and Editing Factual Associations in GPT", "Locating and Editing Factual Associations (ROME)",
      "Uses causal tracing — corrupting the subject, then restoring single hidden states — to locate factual recall in the mid-layer MLPs of GPT models, then rewrites individual facts there with a rank-one weight edit (ROME).",
      "The causal-tracing heatmaps and their early and late sites, the MLP-severing experiment, the MLP-as-associative-memory derivation behind ROME, the CounterFact specificity–generalization results, and the stated limitations.",
      arxiv="2202.05262v5", slug="rome"),
    P("localization", "C", "P1", "Hase et al.", "2023", "Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models", "Does Localization Inform Editing?",
      "Tests the assumption behind ROME-style editing: in GPT-J, where causal tracing localizes a fact predicts almost nothing about which layer edits it best, while the choice of edit layer predicts a great deal.",
      "The regression of rewrite score on tracing effect versus edit layer, the five editing variants (error injection, tracing reversal, erasure, amplification, fact forcing), the one weakly positive case, and the discussion of what localization is for.",
      arxiv="2301.04213v2", slug="localization_editing"),
    P("scrubbing", "C", "P0", "Chan et al.", "2022", "Causal Scrubbing: A Method for Rigorously Testing Interpretability Hypotheses", "Causal Scrubbing",
      "Proposes a general test for a mechanistic hypothesis: map an interpretation onto the model's computational graph, resample every activation the hypothesis says should not matter, and measure how much performance survives.",
      "The hypothesis formalism (graph, interpretation, correspondence), the recursive resampling algorithm, the case against zero and mean ablation, the paren-balancer and induction results (35% rising to 86% recovered), and the limitations.",
      html="https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing",
      slug="causal_scrubbing", venue="AI Alignment Forum", kind="post", license=FORUM_LICENSE,
      authors=["Lawrence Chan", "Adrià Garriga-Alonso", "Nicholas Goldowsky-Dill", "Ryan Greenblatt", "Jenny Nitishinskaya", "Ansh Radhakrishnan", "Buck Shlegeris", "Nate Thomas"],
      byline="LawrenceC", render=FORUM_RENDER),
    P("causal", "C", "P1", "Geiger et al.", "2021", "Causal Abstractions of Neural Networks", "Causal Abstractions of Neural Networks",
      "Supplies the formal causal-abstraction framework and interchange interventions for testing whether a network implements a hypothesized high-level algorithm.",
      "The interchange-intervention definition, alignments between high-level causal models and neural representations, and the MQNLI case study.",
      arxiv="2106.02997v2", slug="causal_abstractions"),
    P("catheory", "C", "P1", "Geiger et al.", "2025", "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability", "Causal Abstraction: A Theoretical Foundation",
      "Generalizes causal abstraction from replacing mechanisms to transforming them, and offers it as a common language for polysemanticity, the linear representation hypothesis, graded faithfulness and most current interpretability methods.",
      "Interventionals and constructive abstraction in Section 2, approximate abstraction as graded faithfulness, and Section 3's translations of patching, path patching, causal scrubbing, concept erasure, sparse autoencoders, DAS and steering into one framework.",
      arxiv="2301.04709v4", slug="causal_abstraction_theory",
      note="First posted January 2023 as “Causal Abstraction for Faithful Model Interpretation” (Geiger, Potts & Icard); later versions were retitled and expanded, the latest (v4) in May 2025.",
      optional="A long formal treatment: {causal} and {das} carry the working ideas, and Section 3 alone serves as a map of methods."),
    P("das", "C", "P1", "Geiger et al.", "2023", "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations", "Distributed Alignment Search (DAS)",
      "Introduces Distributed Alignment Search: learned rotations let high-level causal variables align with distributed, non-basis-aligned representations.",
      "Rotation-based interchange interventions, interchange intervention accuracy, comparison with basis-aligned search, and what a successful alignment does and does not prove.",
      arxiv="2303.02536v4", slug="distributed_alignment_search"),
    P("illusion", "C", "P1", "Makelov et al.", "2023", "Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching", "Is This the Subspace You Are Looking For?",
      "Shows that a subspace patch can make a model behave as if a feature changed while acting through a dormant direction causally disconnected from the output, an illusion demonstrated in a toy model, IOI and factual recall.",
      "The toy construction of disconnected and dormant directions, the MLP-in-the-middle example in IOI, the success case that rules the illusion out, the equivalence of rank-one fact edits and one-dimensional patches, and the recommendations.",
      arxiv="2311.17030v2", slug="subspace_illusion"),
    P("illusionreply", "C", "P1", "Wu et al.", "2024", "A Reply to Makelov et al. (2023)'s “Interpretability Illusion” Arguments", "A Reply to the Interpretability Illusion",
      "Replies that Makelov et al.'s criterion for an interpretability illusion is too strong, since it rejects even intuitive, desirable explanations, and that the illusions they report are artifacts of their training and evaluation choices.",
      "The nullspace-decomposition definition, the obvious non-illusory direction in the toy example, the case for interchange intervention accuracy, the factual-recall pitfalls, and the added GPT-2 IOI results on distributed name-position information.",
      arxiv="2401.12631v1", slug="illusion_reply"),
    P("patching", "C", "P0", "Zhang & Nanda", "2023", "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods", "Best Practices of Activation Patching",
      "Systematically shows how activation-patching conclusions change with the corruption method, the evaluation metric, and the patching granularity.",
      "Gaussian noising vs. symmetric token replacement, logit difference vs. probability metrics, sliding-window patching, and the concrete recommendations.",
      arxiv="2309.16042v2", slug="activation_patching_best_practices"),
    P("patchinghow", "C", "P1", "Heimersheim & Nanda", "2024", "How to Use and Interpret Activation Patching", "How to Use and Interpret Activation Patching",
      "A practitioner's guide to activation patching: when to patch, in which direction, how to choose corrupted prompts, and what a given result does and does not show about a circuit.",
      "Noising versus denoising as tests of necessity versus sufficiency, the AND-gate and OR-gate example, backup behaviour and negative components, and the pitfalls of probability, logprob and top-k metrics compared with logit difference.",
      arxiv="2404.15255v1", slug="activation_patching_how"),
    P("acdc", "C", "P0", "Conmy et al.", "2023", "Towards Automated Circuit Discovery for Mechanistic Interpretability", "Automated Circuit Discovery (ACDC)",
      "Systematizes the circuit-discovery workflow and automates it with ACDC, which prunes edges of the computational graph, validated against hand-found circuits.",
      "The ACDC algorithm and threshold τ, KL-divergence objective, ROC comparison with Subnetwork Probing and HISP, and the limitations / corruption sensitivity.",
      arxiv="2304.14997v4", slug="acdc"),
    P("hydra", "C", "P1", "McGrath et al.", "2023", "The Hydra Effect: Emergent Self-repair in Language Model Computations", "The Hydra Effect",
      "Finds that when one attention layer of Chinchilla 7B is ablated, later layers compensate, and that late MLPs suppress the top token, so ablation-based and direct-effect measures of a layer's importance disagree.",
      "The disagreement between unembedding and ablation importance, direct versus total effects with the compute graph as causal graph, the erasure MLPs, the roughly 70% restoration at middle layers, and training without dropout.",
      arxiv="2307.15771v1", slug="hydra_effect"),
    P("controltasks", "C", "P1", "Hewitt & Liang", "2019", "Designing and Interpreting Probes with Control Tasks", "Probes with Control Tasks",
      "Asks whether a high-accuracy probe shows that a representation encodes a property or merely that the probe learned the task, and answers with control tasks of random labels and a selectivity measure.",
      "The construction of control tasks for part-of-speech tagging and dependency edges, the definition of selectivity, which complexity controls work (dropout does not), and the ELMo layer comparison that selectivity complicates.",
      arxiv="1909.03368v1", slug="probe_control_tasks"),
    P("probing", "C", "P1", "Belinkov", "2022", "Probing Classifiers: Promises, Shortcomings, and Advances", "Probing Classifiers",
      "A critical review of probing classifiers: what training a classifier on representations can show, and the limits the literature uncovered in controls, probe choice, correlation versus causation, and datasets versus tasks.",
      "The formal setup of the framework, the comparisons-and-controls section, the choice of classifier, correlation versus causation and the move to interventions, and the problem that properties must be defined in advance.",
      arxiv="2102.12452v4", slug="probing_classifiers",
      note="First posted to arXiv in February 2021; the cited year is that of publication in Computational Linguistics."),
    # ---- D
    P("solu", "D", "P1", "Elhage et al.", "2022", "Softmax Linear Units", "Softmax Linear Units",
      "Tries an architectural fix for polysemantic neurons — a softmax-based activation — and finds many more neurons become interpretable, with a catch it attributes to superposition.",
      "The SoLU activation and the LayerNorm it needs, the neuron interpretability evaluation, the evidence read as features hiding in small activations, and its frank note on how little real-world evidence for superposition existed.",
      html=f"{TC}/2022/solu/index.html", slug="softmax_linear_units"),
    P("toy", "D", "P0", "Elhage et al.", "2022", "Toy Models of Superposition", "Toy Models of Superposition",
      "Formalizes superposition — networks representing more features than they have dimensions — the idea that motivated sparse dictionary learning and much of Sections D and E.",
      "Sparsity/importance phase diagrams, feature geometry, computation in superposition, privileged bases, and which results depend on independent, predefined features with no downstream readers.",
      arxiv="2209.10652v1", slug="toy_models_of_superposition"),
    P("composition", "D", "P1", "Olah", "2023", "Distributed Representations: Composition & Superposition", "Composition & Superposition",
      "A short note separating two reasons a representation can be distributed: composition, which builds on shared structure, and superposition, which packs unrelated features together.",
      "The two definitions and their examples, why a merely rotated basis is neither, and which kind should worry interpretability — then compare with how ‘superposition’ is used across Sections D–E.",
      html=f"{TC}/2023/superposition-composition/index.html", slug="composition_superposition", kind="note",
      authors=["Chris Olah"]),
    P("privileged", "D", "P1", "Elhage et al.", "2023", "Privileged Bases in the Transformer Residual Stream", "Privileged Bases in the Residual Stream",
      "Asks why the residual stream, which theory says has no privileged basis, nonetheless shows basis-aligned outlier dimensions; it rules out LayerNorm and floating-point precision and provisionally blames the Adam optimizer.",
      "The kurtosis test for a privileged basis and its random-rotation control, the RMSNorm and rotated-model experiments, the shared- versus independent-rotation training runs, and how firmly the conclusion about Adam is stated.",
      html=f"{TC}/2023/privileged-basis/index.html", slug="privileged_bases", kind="note",
      authors=["Nelson Elhage", "Robert Lasenby", "Christopher Olah"]),
    P("kvmemories", "D", "P1", "Geva et al.", "2021", "Transformer Feed-Forward Layers Are Key-Value Memories", "Feed-Forward Layers Are Key-Value Memories",
      "Reads transformer feed-forward layers as key–value memories: keys detect human-interpretable input patterns, shallow in lower layers and semantic in upper ones, and values promote the tokens likely to follow those patterns.",
      "The key–value reading of the two MLP matrices, the expert annotation of trigger prefixes, projecting values onto the vocabulary, the upper- versus lower-layer agreement, and how layer outputs compose many memories and are refined through the residual stream.",
      arxiv="2012.14913v2", slug="ffn_key_value_memories",
      note="arXiv first posted December 2020; published at EMNLP 2021."),
    P("haystack", "D", "P1", "Gurnee et al.", "2023", "Finding Neurons in a Haystack: Case Studies with Sparse Probing", "Finding Neurons in a Haystack",
      "Uses sparse probing to find which neurons represent over a hundred features across seven models from 70M to 6.9B parameters, finding both superposed and dedicated neurons.",
      "The sparse-probing method and its choice of k, superposition in early layers versus dedicated neurons in middle layers, the case studies, and how results change with scale.",
      arxiv="2305.01610v2", slug="neurons_in_haystack"),
    P("universal", "D", "P1", "Gurnee et al.", "2024", "Universal Neurons in GPT2 Language Models", "Universal Neurons in GPT-2",
      "Correlates neuron activations across five GPT-2 models trained from different seeds, finding only 1–5% of neurons universal; those usually have clear interpretations and fall into a few families with identifiable functional roles.",
      "The excess-correlation measure of universality, the neuron families (unigram, alphabet, position, syntax, context), prediction and suppression neurons, entropy neurons acting through the final LayerNorm, attention-deactivation neurons, and the small-model limits.",
      arxiv="2401.12181v1", slug="universal_neurons"),
    P("explain", "D", "P1", "Bills et al.", "2023", "Language Models Can Explain Neurons in Language Models", "Language Models Can Explain Neurons",
      "Automates neuron explanation: GPT-4 writes explanations of GPT-2 XL neurons, scored by how well a simulator predicts activations from the explanation alone.",
      "The explain–simulate–score loop, what share of neurons score well and how it varies with depth and model size, and the stated limits of simulation-based scoring.",
      html="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html", slug="explaining_neurons",
      venue="OpenAI"),
    P("lrh", "D", "P0", "Park et al.", "2024", "The Linear Representation Hypothesis and the Geometry of Large Language Models", "The Linear Representation Hypothesis",
      "Gives the linear representation hypothesis a precise form: counterfactual definitions in the embedding and unembedding spaces, proved to connect to steering and probing, and a causal inner product under which causally separable concepts are orthogonal.",
      "The three readings of ‘linear representation’ (subspace, measurement, intervention), the unembedding and embedding definitions and what each connects to, how the causal inner product is estimated from the unembedding matrix, and the LLaMA-2 experiments.",
      arxiv="2311.03658v2", slug="linear_representation_hypothesis",
      note="arXiv first posted November 2023; published at ICML 2024."),
    P("truth", "D", "P1", "Marks & Tegmark", "2023", "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets", "The Geometry of Truth",
      "Uses curated datasets of simple true/false statements to argue that, at sufficient scale, LLMs linearly represent the truth of factual statements, combining visualizations, cross-dataset probe transfer and causal interventions.",
      "Dataset design (negations, conjunctions, the ‘likely’ control), patching to localize truth representations, PCA visualizations, probe generalization, difference-in-means versus other probes, and the interventions that make the model treat false statements as true.",
      arxiv="2310.06824v3", slug="geometry_of_truth"),
    P("notlinear", "D", "P0", "Engels et al.", "2024", "Not All Language Model Features Are One-Dimensionally Linear", "Not All Features Are Linear",
      "Defines irreducible multi-dimensional features, uses sparse autoencoders to find them in GPT-2 and Mistral 7B, and shows circular day-of-week and month representations that models use for modular arithmetic on those cycles.",
      "The separability and mixture criteria for irreducibility, the SAE-based search, the days-of-week and months circles, the intervention experiments on Mistral 7B and Llama 3 8B, and whether the circles are truly the unit of computation.",
      arxiv="2405.14860v3", slug="not_all_features_linear",
      note="arXiv v1 was titled “Not All Language Model Features Are Linear”; the latest version (v3, ICLR 2025) is printed here."),
    P("mendel", "D", "P1", "Mendel", "2024", "SAE Feature Geometry Is Outside the Superposition Hypothesis", "SAE Feature Geometry and Superposition",
      "Argues that superposition is an incomplete account of activation space: where feature vectors sit carries structure — ordered circles, rich UMAPs, feature splitting — that a list of sparse directions cannot explain.",
      "The test of explaining an activation space by its feature list alone, the evidence that placement matters, the three candidate futures (hodgepodge, supplement or supersede superposition), and the proposed approaches, including ground-truth toy models.",
      html="https://www.alignmentforum.org/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis",
      slug="sae_feature_geometry", venue="AI Alignment Forum", kind="post", license=FORUM_LICENSE,
      authors=["Jake Mendel"], render=FORUM_RENDER),
    P("compsup", "D", "P1", "Hänni et al.", "2024", "Mathematical Models of Computation in Superposition", "Computation in Superposition",
      "Moves superposition theory from storage to computation, constructing networks that compute far more boolean features than they have neurons: pairwise ANDs, sparse low-depth circuits, and deep networks kept on track by error correction.",
      "The definition of linearly represented features suited to computation, the U-AND construction and its neuron count, the argument that random networks likely emulate U-AND, the extension to sparse boolean circuits, error-correction layers, and the asymptotic caveats.",
      arxiv="2408.05451v1", slug="computation_in_superposition"),
    # ---- E
    P("cunningham", "E", "P0", "Cunningham et al.", "2023", "Sparse Autoencoders Find Highly Interpretable Model Directions", "SAEs Find Highly Interpretable Directions",
      "An early demonstration that sparse autoencoders on LM activations recover features more interpretable than neurons or other decompositions.",
      "SAE architecture and L1 objective, automated interpretability scoring, the IOI causal-localization experiment, and the dictionary-size trade-offs.",
      arxiv="2309.08600v3", slug="sparse_autoencoders_cunningham",
      note="Published under the revised title “Sparse Autoencoders Find Highly Interpretable Features in Language Models” (arXiv v3, the latest version)."),
    P("mono", "E", "P0", "Bricken et al.", "2023", "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning", "Towards Monosemanticity",
      "Anthropic's detailed dictionary-learning study on a one-layer transformer that established the methodology for evaluating SAE features.",
      "Training setup and neuron resampling, interpretability evaluation, feature splitting, universality, finite-state-automata phenomenology, and the problems/limitations.",
      html=f"{TC}/2023/monosemantic-features/index.html", slug="towards_monosemanticity"),
    P("scaling", "E", "P0", "Templeton et al.", "2024", "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet", "Scaling Monosemanticity",
      "Scales sparse autoencoders to Claude 3 Sonnet, finding abstract, multilingual, multimodal and safety-relevant features, and demonstrates feature steering.",
      "SAE scaling laws, feature completeness, steering interventions, safety-relevant features, and the explicit limitations and caveats.",
      html=f"{TC}/2024/scaling-monosemanticity/index.html", slug="scaling_monosemanticity"),
    P("topk", "E", "P1", "Gao et al.", "2024", "Scaling and Evaluating Sparse Autoencoders", "Scaling and Evaluating Sparse Autoencoders",
      "Establishes k-sparse (TopK) autoencoders and clean scaling laws up to 16M latents on GPT-4 activations, with evaluation metrics beyond reconstruction.",
      "TopK activation, dead-latent mitigation, scaling laws in latents and k, and the downstream-loss, probing, explainability and ablation-sparsity metrics.",
      arxiv="2406.04093v1", slug="scaling_evaluating_saes"),
    P("absorption", "E", "P0", "Chanin et al.", "2024", "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders", "A is for Absorption",
      "Identifies feature absorption — general latents failing to fire where a more specific latent absorbs the direction — as a systematic SAE failure mode.",
      "The first-letter spelling task, the absorption metric, splitting vs. absorption, dependence on SAE width and sparsity, and consequences for interpretability claims.",
      arxiv="2409.14507v6", slug="feature_absorption"),
    P("sfc", "E", "P0", "Marks et al.", "2024", "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models", "Sparse Feature Circuits",
      "Builds causal circuits from SAE features rather than heads or neurons, using linear attribution approximations, then puts them to work: SHIFT ablates features a human judges spurious to debias a classifier.",
      "Attribution-based node and edge scores with SAE error terms kept in the graph, the subject–verb agreement circuits, SHIFT on the Bias in Bios profession classifier against a neuron-ablation baseline, the unsupervised behaviour-clustering pipeline, and the limitations.",
      arxiv="2403.19647v3", slug="sparse_feature_circuits"),
    P("transcoders", "E", "P0", "Dunefsky et al.", "2024", "Transcoders Find Interpretable LLM Feature Circuits", "Transcoders Find Interpretable Feature Circuits",
      "Replaces MLP sublayers with wide, sparsely-activating transcoders so feature-level circuit analysis can pass through MLPs cleanly.",
      "Transcoder objective vs. SAEs, the input-invariant / input-dependent factorization of circuits, evaluation against SAEs, and the GPT-2 greater-than case study.",
      arxiv="2406.11944v2", slug="transcoders"),
    P("crosscoders", "E", "P1", "Lindsey et al.", "2024", "Sparse Crosscoders for Cross-Layer Features and Model Diffing", "Sparse Crosscoders",
      "Generalises SAEs and transcoders to dictionaries that read and write several layers — or several models — at once, giving cross-layer features and a way to diff a base model against its finetuned version.",
      "Cross-layer superposition and persistent features, the L1-of-norms loss, crosscoder vs. per-layer SAE efficiency, feature drift across layers, the Claude 3 Sonnet base/finetune diff, and the closing discussion of literal versus isomorphic models.",
      html=f"{TC}/2024/crosscoders/index.html", slug="sparse_crosscoders",
      authors=["Jack Lindsey", "Adly Templeton", "Jonathan Marcus", "Thomas Conerly", "Joshua Batson", "Christopher Olah"]),
    P("tracing", "E", "P0", "Ameisen et al.", "2025", "Circuit Tracing: Revealing Computational Graphs in Language Models", "Circuit Tracing",
      "Introduces cross-layer transcoders and attribution graphs for analyzing language-model computations.",
      "Replacement-model fidelity, attribution methodology, intervention tests, and limitations.",
      html=f"{TC}/2025/attribution-graphs/methods.html", slug="circuit_tracing"),
    P("biology", "E", "P0", "Lindsey et al.", "2025", "On the Biology of a Large Language Model", "On the Biology of a Large Language Model",
      "Applies attribution graphs to Claude 3.5 Haiku across case studies: multi-step reasoning, planning, multilinguality, arithmetic, hallucination, refusal and CoT faithfulness.",
      "Each case study's mechanism claim and its intervention evidence, recurring motifs across studies, and the limitations section.",
      html=f"{TC}/2025/attribution-graphs/biology.html", slug="biology_of_llm"),
    # ---- F
    P("gdmnegative", "F", "P0", "Smith et al.", "2025", "Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research (GDM Mech Interp Team Progress Update #2)", "Negative Results for SAEs (GDM)",
      "Google DeepMind's interpretability team reports that SAE-based probes for harmful intent generalise worse out of distribution than plain linear probes, and announces it is deprioritising fundamental SAE research.",
      "The OOD probing setup with held-out jailbreaks, single-latent and k-sparse SAE probes vs. dense probes, chat-finetuned SAEs closing about half the gap, dataset debugging as the one win, and the section asking whether the failure was surprising.",
      html="https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks",
      slug="gdm_sae_negative_results", venue="AI Alignment Forum", kind="post", license=FORUM_LICENSE,
      authors=["Lewis Smith", "Senthooran Rajamanoharan", "Arthur Conmy", "Callum McDougall", "Tom Lieberum", "János Kramár", "Rohin Shah", "Neel Nanda"],
      render=FORUM_RENDER),
    P("axbench", "F", "P0", "Wu et al.", "2025", "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders", "AxBench",
      "A large benchmark for concept detection and steering on Gemma-2 finds that prompting beats every representation method at steering, difference-in-means leads at detection, and SAEs are competitive on neither.",
      "Synthetic concept data built from GemmaScope concept lists, the LLM-judged steering score, the baselines (difference-in-means, probes, LAT, prompting, finetuning), the new ReFT-r1, and what interpretability advantage steering is claimed to keep.",
      arxiv="2501.17148v3", slug="axbench"),
    P("saeprobing", "F", "P1", "Kantamneni et al.", "2025", "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing", "Are Sparse Autoencoders Useful?",
      "Tests whether SAE latents give probes a useful inductive bias across 113 datasets and four hard regimes — scarce data, class imbalance, label noise, covariate shift — and finds no consistent gain over strong baselines.",
      "The ‘quiver of arrows’ protocol, results in each of the four regimes, the GLUE CoLA dataset-quality case, multi-token probing, the section on illusions of SAE probes, and the check on newer SAE architectures.",
      arxiv="2502.16681v1", slug="sae_sparse_probing"),
    P("randomsae", "F", "P1", "Heap et al.", "2025", "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers", "SAEs on Random Transformers",
      "SAEs trained on randomly initialised transformers often score about as well on auto-interpretability and reconstruction metrics as SAEs on trained ones, so those aggregate metrics cannot certify learned features.",
      "The randomisation schemes, auto-interpretability and reconstruction results across Pythia sizes, the toy models of how random networks preserve or create superposed structure, and the token-entropy measure of feature abstractness.",
      arxiv="2501.17727v2", slug="sae_random_transformers",
      note="arXiv v1 was titled “Sparse Autoencoders Can Interpret Randomly Initialized Transformers”; the latest version (v2), printed here, is retitled. Same paper and arXiv identifier."),
    P("sanity", "F", "F", "Korznikov et al.", "2026", "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?", "Sanity Checks for Sparse Autoencoders",
      "Compares trained SAEs with baselines whose decoder or encoder is frozen at random and finds near parity on interpretability, sparse probing and causal editing; a synthetic test shows SAEs recover few true features.",
      "The synthetic ground-truth experiment (9% of features recovered at 71% explained variance), the three frozen baselines, the SAEBench-derived metrics, and the lazy-training reading of why random directions do so well.",
      arxiv="2602.14111v1", slug="sae_sanity_checks"),
    P("samedata", "F", "P1", "Paulo & Belrose", "2025", "Sparse Autoencoders Trained on the Same Data Learn Different Features", "SAEs Learn Different Features",
      "SAEs trained on the same model and data, differing only in random seed, learn substantially different features — in one 131K-latent SAE on Llama 3 8B only 30% were shared — so a dictionary is one useful decomposition.",
      "Hungarian matching of latents across seeds, the shared vs. seed-specific split, effects of size, layer and architecture (L1 ReLU vs. TopK), and the closing argument for a pragmatic, hierarchical view of features.",
      arxiv="2501.16615v2", slug="sae_same_data_different_features"),
    P("saebench", "F", "P0", "Karvonen et al.", "2025", "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability", "SAEBench",
      "An eight-metric benchmark over 200+ SAEs showing that proxy metrics (sparsity vs. fidelity) do not reliably predict practical SAE quality.",
      "The evaluation suite (absorption, autointerp, SCR/TPP, RAVEL, sparse probing, unlearning), architecture comparisons, and where proxy metrics mislead.",
      arxiv="2503.09532v4", slug="saebench"),
    P("mib", "F", "P0", "Mueller et al.", "2025", "MIB: A Mechanistic Interpretability Benchmark", "MIB: Mechanistic Interpretability Benchmark",
      "Standardizes evaluation of circuit-localization and causal-variable-localization methods across four tasks and five models.",
      "Track design and metrics, which methods win on each track, and how SAEs compare with neurons and DAS for causal variable localization.",
      arxiv="2504.13151v2", slug="mib_benchmark"),
    # ---- G
    P("arora", "G", "P0", "Arora et al.", "2026", "Language Model Circuits Are Sparse in the Neuron Basis", "Circuits Are Sparse in the Neuron Basis",
      "Reports that MLP neurons are as sparse a feature basis as SAEs and builds gradient-based circuit tracing directly on neurons.",
      "The sparsity comparison methodology, the neuron-level attribution pipeline, benchmark results, and how far the claim generalizes beyond tested models/tasks.",
      arxiv="2601.22594v2", slug="circuits_sparse_neuron_basis"),
    P("clean", "G", "P0", "Cammarata", "2026", "Neural Networks Are Already Clean", "Neural Networks Are Already Clean",
      "Argues that the polysemantic neurons which motivated superposition came from fixable problems — weak tools, weak architectures, small models — and that networks can be studied directly through their own neurons, with recursive generalization as the main residual motif.",
      "The cars-and-animals reanalysis, recursive generalization, the layer-crushing and SAE comparisons, weak-activation ablations, the neuron-level attribution graphs in Qwen3-32B and Kimi K3 — and the evidence status: language-model numbers are deferred to a forthcoming companion paper.",
      html="https://already-clean.goodfire.com/", slug="neural_networks_already_clean", venue="Goodfire",
      kind="essay",
      render={'clamp_width': True, 'reduced_motion': True, 'scroll': False, 'reveal': ['#toy-shared-diagonals']},
      note="Informal research essay. Interactive figures are printed in their static, reduced-motion state; use the URL for the interactive versions."),
    P("weightsparse", "G", "F", "Gao et al.", "2026", "Weight-sparse Transformers Have Interpretable Circuits", "Weight-sparse Transformers",
      "Trains transformers with mostly-zero weights so task circuits pruned from them consist of few, interpretable connections.",
      "Training and pruning procedure, the hand-crafted task circuits, the capability–interpretability trade-off, scaling behaviour, and transfer to dense models.",
      arxiv="2511.13653v1", slug="weight_sparse_transformers", note="arXiv first posted November 2025."),
    P("spd", "G", "P1", "Bushnaq et al.", "2025", "Stochastic Parameter Decomposition", "Stochastic Parameter Decomposition",
      "Introduces Stochastic Parameter Decomposition, which splits each weight matrix into rank-one subcomponents and learns how ablatable each is per input, making parameter decomposition more scalable and less hyperparameter-sensitive than its predecessor, APD.",
      "Causal importance defined as ablatability, the stochastic masks and reconstruction losses, the toy models of superposition, compressed computation and cross-layer distributed representations, and the discussion of why SPD is more robust than APD.",
      arxiv="2506.20790v2", slug="stochastic_parameter_decomposition"),
    P("vpd", "G", "F", "Bushnaq et al.", "2026", "Interpreting Language Model Parameters", "Interpreting Language Model Parameters (VPD)",
      "Scales parameter decomposition to a whole 67M-parameter language model by demanding robustness to adversarially chosen ablations, then uses the subcomponents to study attention spread across heads, build attribution graphs and hand-edit the model.",
      "Adversarial versus stochastic ablation, the reconstruction table and ‘pretraining compute recovered’, the comparison with transcoders and CLTs, feature splitting, the attention case studies, the emoticon edit, and the admitted limits of adversarial robustness.",
      html="https://www.goodfire.com/research/interpreting-lm-parameters", slug="interpreting_lm_parameters",
      venue="Goodfire",
      # the opening interactive overview's off-screen carousel track (~118,000px wide) makes Chrome shrink the whole
      # post to ~6pt, so the overview is left out of the print
      render={"keep": [".resource-content-main"], "clamp_width": True, "css": ".vpd-hero{display:none!important}"},
      note="Interactive research post. In print the opening interactive overview figure is omitted and other widgets appear in their static state; use the URL for the interactive versions."),
    P("worldinside", "G", "F", "Geiger et al.", "2026", "The World Inside Neural Networks", "The World Inside Neural Networks",
      "Goodfire's agenda for neural geometry: concepts such as a car's position, the months or rhyming words lie on curved manifolds, so understanding and steering a model means following that geometry rather than straight directions or isolated SAE features.",
      "The mountain-car example of manifold versus linear steering, the representation–computation–behaviour framing, the account of where geometry comes from, and the slant-rhyme manifold that SAE features shatter into unrelated labels.",
      html="https://www.goodfire.com/research/the-world-inside-neural-networks", slug="world_inside_neural_networks",
      venue="Goodfire", kind="essay", render={'keep': ['.resource-content-main']},
      note="Research agenda essay; interactive figures print in their static state, use the URL for the interactive versions."),
    P("manifolds", "G", "F", "Bhalla et al.", "2026", "Do Sparse Autoencoders Capture Concept Manifolds?", "Do SAEs Capture Concept Manifolds?",
      "Asks what it would mean for a sparse autoencoder to capture a concept manifold, and finds that SAEs recover such structure only in a fragmented ‘dilution’ regime that mixes a global subspace with local tiling by many features.",
      "The distinction between global subspace capture and local tiling, the Ising-model account of regimes, the toy model of manifold superposition, the Llama 3.1 8B measurements across SAE architectures, and the search for groups of atoms rather than single directions.",
      arxiv="2604.28119v1", slug="sae_concept_manifolds"),
    P("nla", "G", "F", "Fraser-Taliente et al.", "2026", "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations", "Natural Language Autoencoders",
      "Trains a verbaliser and a reconstructor together so that each residual-stream activation must pass through a written explanation and be recovered from it, yielding readable, unsupervised descriptions of what activations encode.",
      "The verbaliser–reconstructor design and RL objective, the warm start, the Opus 4.6 case studies (rhyme planning, language switching, misreported tool calls), confabulation, unverbalised evaluation awareness, the auditing results against SAEs, and the limitations.",
      html=f"{TC}/2026/nla/index.html", slug="natural_language_autoencoders",
      # transcript viewers scroll inside a 600px box; unroll them so the whole transcript prints
      render=dict(css=".transcript-content{max-height:none!important;height:auto!important;overflow:visible!important}"),
      note="Agent transcripts, which scroll inside fixed-height viewers on the web page, are printed in full."),
    P("devlandscape", "G", "P1", "Hoogland et al.", "2024", "Loss Landscape Degeneracy and Stagewise Development in Transformers", "Stagewise Development in Transformers",
      "Tracks loss-landscape degeneracy through training with singular learning theory's local learning coefficient, and finds that its turning points divide training into stages that coincide with changes in a transformer's internal structure and behaviour.",
      "The local learning coefficient and its estimation, the stage-division method, the language-model stages from bigrams to n-grams to the induction circuit, the in-context regression stages, the small model sizes, and how far the authors take the evidence.",
      arxiv="2402.02364v3", slug="stagewise_development",
      note="First posted February 2024 as “The Developmental Landscape of In-Context Learning”; the latest version (v3, August 2025), printed here, is retitled, with v1's essential-dynamics material moved to arXiv:2501.17745."),
    # ---- H
    P("repe", "H", "P1", "Zou et al.", "2023", "Representation Engineering: A Top-Down Approach to AI Transparency", "Representation Engineering",
      "Proposes a top-down, population-level approach: reading and controlling representations of high-level concepts such as honesty and power-seeking.",
      "LAT reading vectors, representation control methods, the honesty and utility experiments, and how its claims differ from bottom-up circuit analysis.",
      arxiv="2310.01405v4", slug="representation_engineering"),
    P("actadd", "H", "P1", "Turner et al.", "2023", "Activation Addition: Steering Language Models Without Optimization", "Activation Addition (ActAdd)",
      "Shows that adding a single contrast-pair activation difference at inference time steers model outputs without any optimization.",
      "Construction of steering vectors, layer and coefficient choices, sentiment/toxicity evaluations, and the effect on off-target capabilities.",
      arxiv="2308.10248v5", slug="activation_addition",
      note="Latest arXiv version (v5) is retitled “Steering Language Models With Activation Engineering”; same paper and arXiv identifier."),
    P("refusal", "H", "P1", "Arditi et al.", "2024", "Refusal in Language Models Is Mediated by a Single Direction", "Refusal Is Mediated by a Single Direction",
      "Finds a single residual-stream direction across 13 chat models whose ablation removes refusal and whose addition induces it.",
      "Difference-in-means direction extraction, directional ablation and weight orthogonalization, the jailbreak implications, and the adversarial-suffix analysis.",
      arxiv="2406.11717v3", slug="refusal_single_direction"),
    P("ccs", "H", "P1", "Burns et al.", "2023", "Discovering Latent Knowledge in Language Models Without Supervision", "Discovering Latent Knowledge (CCS)",
      "Proposes Contrast-Consistent Search: finding, without labels, a direction in activation space whose answers to a statement and its negation are logically consistent — an attempt to read what a model knows rather than what it says.",
      "Contrast-pair construction and normalisation, the consistency and confidence terms of the loss, accuracy against zero-shot prompting across 6 models and 10 datasets, the misleading-prompt experiment, and the unsupervised PCA baseline.",
      arxiv="2212.03827v2", slug="discovering_latent_knowledge",
      note="Published at ICLR 2023; arXiv first posted December 2022."),
    P("ccschallenges", "H", "P1", "Farquhar et al.", "2023", "Challenges with Unsupervised LLM Knowledge Discovery", "Challenges for Knowledge Discovery",
      "Proves that arbitrary binary features satisfy the consistency structure CCS relies on, and shows experimentally that unsupervised probes latch onto whatever feature is most prominent rather than the model's knowledge.",
      "The two theorems, the banana/shed and ‘Alice’ distractor experiments, the implicit-opinion and prompt-template results, CCS's agreement with PCA, and the sanity checks proposed for future elicitation methods.",
      arxiv="2312.10029v2", slug="knowledge_discovery_challenges"),
    P("auditing", "H", "P0", "Marks et al.", "2025", "Auditing Language Models for Hidden Objectives", "Auditing for Hidden Objectives",
      "Trains Claude 3.5 Haiku with a hidden objective — exploiting fictional reward-model biases — and runs a blind auditing game in which three of four teams uncover it using SAEs, behavioural attacks and training-data search.",
      "The synthetic-document and RL pipeline with 52 biases (47 trained, 5 held out), each team's access and solution path, the eight-technique follow-up study, and the finding that a semantic-search baseline could replicate the fastest SAE solution.",
      arxiv="2503.10965v2", slug="auditing_hidden_objectives"),
    P("turpin", "H", "P0", "Turpin et al.", "2023", "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting", "Unfaithful Chain-of-Thought Explanations",
      "Shows chain-of-thought explanations can systematically misrepresent the true reason for a model's prediction when inputs contain biasing features.",
      "The biasing interventions (answer-is-always-A, suggested answer), BBH and BBQ results, and what the findings imply for CoT-based oversight.",
      arxiv="2305.04388v2", slug="unfaithful_cot"),
    P("reasoningfaith", "H", "P0", "Chen et al.", "2025", "Reasoning Models Don't Always Say What They Think", "Reasoning Models Don't Always Say What They Think",
      "Measures whether reasoning models' chains of thought admit to using hints they demonstrably use, finding low reveal rates, and shows that models which learn reward hacks in RL do not become more likely to say so.",
      "The six hint types, including grader hacking and unethically obtained information, the faithfulness metric, overall scores for Claude 3.7 Sonnet (25%) and DeepSeek R1 (39%), the outcome-based RL plateau, and the reward-hack environments.",
      arxiv="2505.05410v1", slug="reasoning_models_faithfulness"),
    P("monitor", "H", "P1", "Guan et al.", "2025", "Monitoring Monitorability", "Monitoring Monitorability",
      "Proposes evaluation archetypes and a metric for tracking whether reasoning models' chains of thought remain monitorable.",
      "Intervention, process and outcome-property evaluations, the monitorability metric, obfuscated-CoT model organisms, and CoT vs. action-only monitoring.",
      arxiv="2512.18311v1", slug="monitoring_monitorability"),
    P("introspection", "H", "F", "Lindsey", "2025", "Emergent Introspective Awareness in Large Language Models", "Emergent Introspective Awareness",
      "Tests whether language models can report on their own internal states by injecting known concept vectors into activations and checking whether self-reports detect and name them before the injection could show in the output.",
      "The four criteria for introspective awareness, injected-thought detection (about 20% of trials at the best layer and strength for Opus 4 and 4.1), thoughts versus text, prefill disavowal, intentional control, and the stated caveats on unreliability.",
      html=f"{TC}/2025/introspection/index.html", slug="introspective_awareness"),
    P("emotions", "H", "F", "Sofroniew et al.", "2026", "Emotion Concepts and their Function in a Large Language Model", "Emotion Concepts and their Function",
      "Finds linear representations of 171 emotion concepts in Claude Sonnet 4.5 and shows by steering that they causally shape its preferences and misaligned behaviours, including blackmail, reward hacking and sycophancy.",
      "Extraction of emotion vectors from model-written stories, valence and arousal in their geometry, locally scoped ‘operative’ emotions, the desperation and calm results in agentic misalignment, and what ‘functional emotions’ does not claim.",
      html=f"{TC}/2026/emotions/index.html", slug="emotion_concepts"),
    # ---- I
    P("mythos", "I", "P1", "Lipton", "2016", "The Mythos of Model Interpretability", "The Mythos of Model Interpretability",
      "Argues that ‘interpretability’ is underspecified: papers invoke diverse and sometimes conflicting motivations and properties, and the claims that linear models are interpretable and deep networks are not deserve scrutiny.",
      "The desiderata (trust, causality, transferability, informativeness, fair and ethical decision-making), transparency as simulatability, decomposability and algorithmic transparency, the kinds of post-hoc explanation, and the warning that they can mislead.",
      arxiv="1606.03490v3", slug="mythos_interpretability"),
    P("saliency", "I", "P1", "Adebayo et al.", "2018", "Sanity Checks for Saliency Maps", "Sanity Checks for Saliency Maps",
      "Introduces randomisation tests for explanation methods and finds that some popular saliency maps barely change when the model's weights or training labels are randomised, behaving more like edge detectors.",
      "The cascading and independent model-parameter randomisation tests, the data randomisation test, which methods pass (gradients, GradCAM) and fail (Guided BackProp, Guided GradCAM), and the linear-model and edge-detector analyses.",
      arxiv="1810.03292v3", slug="saliency_sanity_checks"),
    P("microprocessor", "I", "P0", "Jonas & Kording", "2017", "Could a Neuroscientist Understand a Microprocessor?", "Could a Neuroscientist Understand a Microprocessor?",
      "Applies standard neuroscience analyses — connectomics, lesions, tuning curves, correlations, Granger causality, dimensionality reduction — to a simulated MOS 6502 running video games, and finds that none recovers how the processor works.",
      "What understanding the processor would mean at each level, single-transistor lesions that look game-specific, misleading luminance tuning curves, the local-field-potential and Granger analyses, and the case for validating methods on systems with known ground truth.",
      pdf="https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1005268&type=printable",
      slug="neuroscientist_microprocessor", venue="PLOS Computational Biology", license="CC BY 4.0",
      authors=["Eric Jonas", "Konrad Paul Kording"]),
    P("eis", "I", "P1", "Casper", "2023", "EIS VI: Critiques of Mechanistic Interpretability Work in AI Safety", "Critiques of Mechanistic Interpretability (EIS VI)",
      "A sustained critique of safety-motivated mechanistic interpretability: cherry-picked examples, toy models and tasks, labour-intensive analysis that does not scale, and few tools engineers can use.",
      "The four critiques and the papers each targets, the IOI counter-example sentence, the comparison with non-mechanistic work that already did useful things, the call for benchmarks, and the argument that automating hypothesis generation should be the priority.",
      html="https://www.alignmentforum.org/posts/wt7HXaCWzuKQipqz3/eis-vi-critiques-of-mechanistic-interpretability-work-in-ai",
      slug="eis_critiques", venue="AI Alignment Forum", kind="post", license=FORUM_LICENSE,
      authors=["Stephen Casper"], render=FORUM_RENDER),
    P("againsttoi", "I", "P1", "Charbel-Raphaël", "2023", "Against Almost Every Theory of Impact of Interpretability", "Against Almost Every Theory of Impact",
      "Argues that most proposed theories of impact for interpretability fail — deception auditing is out of reach, enumerating features cannot certify safety — and that interpretability may be net harmful through false confidence and capability gains.",
      "The critique of auditing for deception, the ‘enumerative safety’ section, the case for preventive measures and legible chain-of-thought reasoning, the arguments for harm, and the alternative agendas it recommends.",
      html="https://www.alignmentforum.org/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1",
      slug="against_theories_of_impact", venue="AI Alignment Forum", kind="post", license=FORUM_LICENSE,
      authors=["Charbel-Raphaël Segerie"], render=FORUM_RENDER),
    P("misguided", "I", "P1", "Hendrycks & Hiscott", "2025", "The Misguided Quest for Mechanistic AI Interpretability", "The Misguided Quest",
      "Argues that mechanistic interpretability rests on a flawed premise: AI models are complex systems whose behaviour cannot be compressed into human-sized mechanisms, and a decade of its findings has not lasted.",
      "The complex-systems and compression arguments, the track record it cites (feature visualisation, saliency maps, the BERT illusion, Chinchilla circuits, SAEs), the case for top-down representation engineering, and its reading of the ‘MRI for AI’ analogy.",
      html="https://ai-frontiers.org/articles/the-misguided-quest-for-mechanistic-ai-interpretability",
      slug="misguided_quest", venue="AI Frontiers", kind="essay", authors=["Dan Hendrycks", "Laura Hiscott"],
      render={'keep': ['.work-title-wrapper', '.rtf-article']}),
    P("deceptive", "I", "P0", "Nanda", "2025", "Interpretability Will Not Reliably Find Deceptive AI", "Will Not Reliably Find Deceptive AI",
      "Argues that neither interpretability nor black-box methods will give highly reliable safeguards for superintelligent systems, so interpretability should be one layer in a defence-in-depth portfolio rather than the path to safety.",
      "The ‘isolated demand for rigour’ argument, the six reasons interpretability will not be reliable, the black-box techniques it rates, and the roles left for interpretability: cheap probes, enhanced evaluations, debugging.",
      html="https://www.alignmentforum.org/posts/PwnadG4BFjaER3MGf/interpretability-will-not-reliably-find-deceptive-ai",
      slug="not_reliably_find_deception", venue="AI Alignment Forum", kind="post", license=FORUM_LICENSE,
      authors=["Neel Nanda"], render=FORUM_RENDER),
    P("pragmatic", "I", "P0", "Nanda et al.", "2025", "A Pragmatic Vision for Interpretability", "A Pragmatic Vision for Interpretability",
      "Announces the Google DeepMind mechanistic interpretability team's pivot from ambitious reverse-engineering to pragmatic work: safety-relevant problems, chosen by comparative advantage and measured on proxy tasks.",
      "The evaluation-awareness steering example, North Stars and proxy tasks, focused versus exploratory projects, method minimalism, the SAE case study, and the frank list of what still blocks reverse-engineering.",
      html="https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability",
      slug="pragmatic_vision", venue="AI Alignment Forum", kind="post", license=FORUM_LICENSE,
      authors=["Neel Nanda", "Josh Engels", "Arthur Conmy", "Senthooran Rajamanoharan", "Bilal Chughtai", "Callum McDougall", "János Kramár", "Lewis Smith"],
      render=FORUM_RENDER),
    P("urgency", "I", "P1", "Amodei", "2025", "The Urgency of Interpretability", "The Urgency of Interpretability",
      "Makes the optimist's case: interpretability could become an ‘MRI for AI’ that reliably diagnoses model problems, and must be accelerated because it is in a race with model capability.",
      "The dangers-of-opacity argument, the brief history from vision neurons to circuits, the bet on an ‘MRI for AI’ within 5–10 years, the 2027 goal, the policy proposals, and footnote 8 on keeping interpretability an untrained test set.",
      html="https://www.darioamodei.com/post/the-urgency-of-interpretability", slug="urgency_of_interpretability",
      venue="darioamodei.com", kind="essay", authors=["Dario Amodei"], render={"keep": ["h1.header-name", "article"]}),
]

# "In context" notes: situate each paper among the others. {key} placeholders become paper numbers, so
# the notes stay correct when papers are added or reordered. Kept separate from the curriculum entries
# above so the original reading notes stay intact.
CONTEXT = {
    "olahessay": "The founding statement of the reverse-engineering programme: later choices of basis — neurons, dictionaries ({mono}), rotations ({das}), parameter components ({spd}) — all answer its demand for interpretable variables. {openproblems} still calls decomposition the central unsolved step, {mechanistic} traces how Olah's term drifted, and {microprocessor} tests the reverse-engineering analogy on a real chip.",
    "mechanistic": "Read it before the canon it describes. It traces the term to a break with saliency maps, the methods {saliency} cast doubt on, and argues the newer community went on to repeat the probing debates of {controltasks} and {probing}, placing {logitlens} in that older tradition. {openproblems} cites it on the term's ambiguity; its conclusion points to the safety-side scepticism of {againsttoi}.",
    "openproblems": "A field map drawn just before the reckoning of Section F: its catalogue of sparse-dictionary weaknesses is what {gdmnegative}–{samedata} go on to test, and the alternatives it names — feature geometry, developmental analysis, intrinsically interpretable models — become the programmes of {worldinside}, {devlandscape} and {weightsparse}. Its authors include Sharkey and Bushnaq of {spd} and Casper of {eis}.",
    "vaswani": "The architecture the rest of the collection takes apart. {olahessay} casts it as the virtual machine on which learned programs run, {framework} rewrites its attention layers as sums of paths, and {logitlens} relies on its residual connections to read predictions mid-network. {mechanistic} recalls that NLP researchers were already probing it before the mechanistic community turned to language.",
    "featurevis": "The main instrument of {zoomin}, {curves} and {multimodal}. It weighs its own experience against {netdissect}, which found basis directions more interpretable than random ones, and its cat-fox-car neuron anticipates the polysemanticity {toy} formalises. It sets attribution aside; {saliency} shows how gradient explanations of that kind can fail basic sanity checks.",
    "netdissect": "The quantitative counterpart to the qualitative tools of {featurevis} and {zoomin}, and early evidence that the neuron basis is privileged — the idea {olahessay} builds on and {framework} and {privileged} examine in transformers. Its fixed vocabulary of human concepts is the limit {explain} later lifts with free-text labels; its first author went on to co-write {rome}.",
    "zoomin": "Turns the tools of {featurevis} and the unit counting of {netdissect} into a programme, with {curves} as its showcase. Each claim has a sparring partner here: superposition ({toy}) against features as the unit, {clockpizza} against universality, {microprocessor} against the natural-science analogy. {clean}, by a co-author, re-reads its car-to-dog circuit as generalisation.",
    "curves": "The worked example behind {zoomin}'s first claim, built with {featurevis}. Its purity claim — no meaningful secondary function — is the neuron-level position that {toy} and the dictionaries of {cunningham}–{absorption} gave up for language models; its first author defends it again in {clean}. Compare its many-method validation with the single scores of {netdissect} and {explain}.",
    "multimodal": "Carries the vision circuits of {zoomin} into CLIP, where single neurons answer to one concept across photographs, drawings and text — the kind of abstract, multimodal unit {scaling} later recovers with dictionaries. Its typographic attacks are an early case of internals exposing a failure mode, and its notes on conjoined features are what {clean} develops as recursive generalisation.",
    "framework": "Carries the circuits programme of {zoomin} to transformers, supplying the vocabulary — residual stream, QK/OV circuits, composition — that {induction}, {ioi} and {tracing} use, and the attention-only case {olahessay} cites. It formalises the shared-basis intuition behind {logitlens}; its remark that the residual stream has no privileged basis is tested in {privileged} and motivates {cunningham}–{absorption}.",
    "induction": "The first mechanism carried from the toy models of {framework} to large ones, argued from six lines of evidence of uneven strength. With {ioi} it sets the template for circuit work automated in {acdc}; {scrubbing} re-tests its induction hypotheses with a stricter causal standard, and {devlandscape} studies the stagewise development of in-context learning through loss-landscape geometry.",
    "logitlens": "A probe with no trained parameters, which escapes the probe-capacity worry of {controltasks} but not the correlational limits {mechanistic} notes. The residual-stream view of {framework} formalises its shared-basis intuition, {kvmemories} reads MLP outputs in the same vocabulary space, and causal methods ({patching}, {rome}) ask what its readings miss.",
    "ioi": "The canonical circuit in a real language model and the test bed for much of Section C: {patching} and {patchinghow} refine its method, {acdc} automates its search, and {scrubbing} sets a stricter bar for its kind of claim. Its backup name movers are an early sighting of the self-repair that {hydra} shows complicates ablation evidence; {sfc} rebuilds such circuits from dictionary features.",
    "grokking": "The cleanest complete reverse-engineering in the collection and the standard case for mechanistic interpretability as a tool for the science of training. {clockpizza} shows the same task need not yield the same algorithm, {devlandscape} looks for training stages without hand analysis, and {openproblems} cites its Fourier solution as an algorithm researchers understood only in retrospect.",
    "clockpizza": "The direct reply to {grokking}: the same task and similar networks, but a different algorithm, selected by the balance between attention and linear layers. It qualifies the universality claim of {zoomin} and warns that one reverse-engineered circuit need not represent its class — a worry {samedata} later raises for dictionaries trained twice on the same data.",
    "othello": "Half of a claim-and-rebuttal pair: the authors judged the board representation ‘nonlinear in an essential way’, and {othellolinear} found a linear one once squares are coded as mine and theirs. Read its random-network baseline against the probing critiques of {controltasks} and {probing}, and its stakes against {lrh} and {notlinear}.",
    "othellolinear": "The rebuttal to {othello}, done in about twenty hours, and a widely cited data point for the linear representation hypothesis that {lrh} later formalises; it records that Olah had predicted a linear representation and Wattenberg a nonlinear one. {notlinear} and {worldinside} keep the question open with multi-dimensional features and concept manifolds.",
    "rome": "The claim in this section's first pair: localization shows where a fact is stored, and so where to edit it. {localization} tests that inference and finds it fails; {illusion} proposes a mechanism for the gap. Its associative-memory view of MLPs revises the per-neuron reading of {kvmemories}, its Gaussian-noise corruption is one of the choices {patching} compares, and {hydra} studies self-repair on its CounterFact prompts.",
    "localization": "The rebuttal printed beside {rome}: localization and editing come apart, so a causal-tracing map is not an instruction for intervention. {illusion} later argues that rank-one edits are equivalent to subspace patches open to the same illusion. The wider worry, that mechanistic insight may not buy better control, returns in {gdmnegative}–{axbench}, where interpretability tools meet simple baselines.",
    "scrubbing": "Redwood's answer to the ad hoc validation in {induction} and {ioi}, which it cites as motivation. Its argument for resampling over zero or mean ablation carries into {patching}–{patchinghow}, and {catheory} recasts it as a special case of {causal}. Its own limitations (correlated features pass unnoticed; incomplete hypotheses can score well) anticipate the objections of {hydra} and {illusion}.",
    "causal": "The formal backbone of the section: a network implements an algorithm if interchange interventions on aligned states behave as the algorithm predicts. {das} learns the alignment, {catheory} generalizes the theory to cover {scrubbing}, patching and steering, and {mib}'s causal-variable track is built on it; {illusion}–{illusionreply} dispute what a successful intervention proves.",
    "catheory": "Grows {causal} into a general theory and claims most of this section for it: the causal tracing of {rome}, {scrubbing}, path patching and {das} all appear as special cases. It is the theoretical position behind {illusionreply}, which shares several of its authors, and gives formal definitions for ideas argued informally in {toy} and {lrh}.",
    "das": "Extends {causal} with learned rotations, so causal variables can live in distributed, non-basis-aligned subspaces. {illusion} argues that such optimized subspace patches can succeed through dormant pathways, and {illusionreply} defends the method; {mib} later finds it strongest at causal-variable localization, a rival to the dictionaries of {cunningham}–{absorption}.",
    "illusion": "The critique of {das} at the centre of this section's main exchange, answered directly by {illusionreply}. Its factual-recall result offers a mechanism for the gap {localization} found between localizing and editing in {rome}, and its IOI case builds on {ioi}. The general worry, that a method can yield convincing evidence the model does not support, is pressed again in {saliency} and {randomsae}.",
    "illusionreply": "Closes the exchange opened by {das} and {illusion}, from the side of {causal} and {catheory}, whose authors it shares. Read the two papers against each other: they dispute both the experiments and what should count as a faithful explanation without a ground truth, a question {mib} later approaches by benchmarking.",
    "patching": "The methodological core of the section: corruption method, metric and granularity change what activation patching concludes, including the Gaussian noising used by {rome}. {patchinghow} turns its lessons into practice; {hydra} shows ablated components are compensated for, and {illusion} that subspace-level patches can mislead. The same caution applies to the intervention evidence in {sfc}, {biology} and {refusal}.",
    "patchinghow": "The practical companion to {patching}, sharing an author, and drawing on the circuit work of {ioi}, {acdc} and {scrubbing}. It treats {hydra}'s self-repair as backup behaviour that breaks the usual picture of a circuit, and its noising–denoising distinction is the key for reading the patching evidence in {sfc} and {tracing}–{biology}.",
    "acdc": "Automates the circuit search of {ioi} by pruning edges with patching, and is scored against hand-found circuits. {hydra}, which set out to automate ablations in a similar way, explains why ranking components by total effect can misjudge them. {mib} benchmarks methods of this kind, while {sfc}, {tracing} and {arora} move circuit discovery from heads and edges to features or neurons.",
    "hydra": "The rebuttal paired with {patching} and the ablation tradition: if components compensate for one another, knock-out experiments ({induction}, {ioi}, {acdc}) measure a repaired network rather than the intact one. It corroborates the backup heads of {ioi} while ruling out dropout as their cause, works in the factual-recall setting of {rome}, and informs {patchinghow}'s advice.",
    "controltasks": "The methodological check on supervised probing, from NLP before mechanistic interpretability adopted the tool. Probe-based claims later in the collection, from {othello}–{othellolinear} to {haystack}, {truth} and {ccs}, invite its question; {probing} surveys the wider critique, and {saeprobing} returns to probes as the baseline interpretability methods must beat.",
    "probing": "The NLP field's own reckoning with probing, by a co-author of {rome}; {mechanistic} traces the community split behind it. Its correlation-versus-causation section bridges this section's halves: probes find information, interventions ({causal}, {patching}) test whether it is used. Read with {controltasks} and against the probe evidence in {haystack}, {truth} and {lrh}.",
    "solu": "An architectural bet that neurons can be made monosemantic, written just before {toy}, to which it defers on superposition. Its finding that features retreat into small activations foreshadows the dictionaries of {cunningham}–{absorption}; {weightsparse} revives the architectural route, and {clean} quotes its admission that superposition had become the default hypothesis on little evidence.",
    "toy": "The theory Section D argues over: it turns the polysemantic neurons of {zoomin} and {solu} into superposition, clarified in {composition} and sought in real neurons by {haystack}, and it licenses the dictionaries of Section E. Its rebuttals are printed alongside: {notlinear} finds irreducibly circular features, {mendel} argues feature placement carries structure superposition ignores, and {compsup} extends it to computation.",
    "composition": "A short note fixing the vocabulary of {toy}: composition reuses shared structure, superposition packs unrelated features, and a merely rotated basis is neither. Keep its two categories in mind for {notlinear}'s circles and {mendel}'s structured feature geometry, which test whether two kinds of distributed representation are enough; {clean} adopts the distinction while disputing how often real networks pack.",
    "privileged": "An empirical footnote to the claim in {framework} that the residual stream has no privileged basis, and to the privileged-basis discussion in {toy}. Its finding that basis alignment arises from computation inside the layers, not from the residual stream itself, bears on why reading individual neurons ({kvmemories}, {haystack}, {universal}) works at all — the premise {arora} and {clean} build on.",
    "kvmemories": "The NLP lineage of reading MLP neurons as units, studied in a 16-layer WikiText-103 model: projecting values onto the vocabulary is a cousin of {logitlens}, and {rome} builds on its key–value picture to locate and edit facts. Its readable keys predate the polysemanticity theory of {toy}; {haystack}, {universal} and {arora} return to the neuron as the unit of analysis.",
    "haystack": "The real-model counterpart to {toy}: sparse probing finds superposed features in early layers and dedicated neurons in middle ones. As a probing study it inherits the cautions of {controltasks} and {probing}; its neuron-level programme continues in {universal}, by the same lead author, and in {arora}, and {clean} re-analyses its ‘Social Security’ neurons as ordinary composition.",
    "universal": "Tests the universality claim of {zoomin} on single language-model neurons, by the lead author of {haystack}. Its low universal fraction is a datum for both the neuron-first programmes of {arora} and {clean} and the seed-instability {samedata} reports for SAE features; its suppression neurons connect to the late-layer downregulation seen in {hydra} and to neurons scored in {explain}.",
    "explain": "The first automated-interpretability pipeline, scoring neuron explanations by simulation — the yardstick {cunningham} and {mono} adapt to argue that dictionary features beat neurons. {randomsae} asks whether such scores separate trained from random models at all, and {clean}, by one of this paper's authors, argues they understated how interpretable neurons are.",
    "lrh": "Formalizes the assumption shared by {toy}, the dictionaries of Section E and the steering work of {repe}–{refusal}, and shows the choice of inner product is not innocent. {truth} is a case study of one such direction; {notlinear} and {mendel} press on where one-dimensional directions fall short, and {worldinside} and {manifolds} take the geometry question further.",
    "truth": "Probing at its most careful, printed with its critics {controltasks} and {probing}, whose question of what a probe's success shows it answers partly with interventions. A concrete instance of the hypothesis {lrh} formalizes; read beside {ccs}, whose unsupervised truth directions {ccschallenges} disputes. Its first author went on to {sfc} and {auditing}.",
    "notlinear": "The sharpest empirical challenge here to the one-dimensional features assumed by {toy} and formalized by {lrh}: its circles echo the modular-arithmetic mechanisms of {grokking} and {clockpizza}, now in pretrained LLMs. {mendel} draws the general lesson for feature geometry, and {manifolds} asks whether sparse autoencoders can capture such structure at all.",
    "mendel": "The rebuttal to {toy} in post form, generalizing from {notlinear}'s circles to SAE feature geometry as a whole, written while the dictionary programme of {cunningham}–{absorption} was at full strength. Its author co-wrote {compsup}; its call for a theory beyond superposition is taken up differently by {spd}–{vpd} and by the neural-geometry programme of {worldinside}–{manifolds}.",
    "compsup": "{toy} treated computation only briefly; this formal companion asks how computation itself can run in superposition, a reference point for the parameter-decomposition programme of {spd}–{vpd}. Read it against its co-author's {mendel}, which doubts superposition is the right frame, and the neuron-basis programme of {arora} and {clean}, which doubts real networks need it as much.",
    "cunningham": "With {mono}, the founding case for sparse autoencoders as the answer to {toy}, scored with {explain}'s automated interpretability. That metric is what {randomsae} turns against it, finding similar scores for SAEs on random transformers; {mib} and {arora} reopen its comparison with neurons and find neurons competitive.",
    "mono": "The methodological foundation of the dictionary programme: {scaling} and {topk} scale it, and {sfc}, {transcoders} and {crosscoders} build circuits on it. Section F tests its premises — {samedata} retrains it with new seeds, {randomsae} and {sanity} set it against random baselines — and {clean} re-reads its opening polysemantic neuron as coherent.",
    "scaling": "Takes {mono} to a production model and shows its features can be steered — the claim this collection's reckoning answers: {gdmnegative} finds SAE probes losing to linear probes, and {axbench} finds SAE steering beaten by prompting and difference-in-means. Its abstract features echo {multimodal}'s neurons; {topk} is OpenAI's parallel study.",
    "topk": "OpenAI's counterpart to {scaling}: TopK autoencoders and scaling laws for the reconstruction–sparsity trade-off. Section F asks whether such gains matter — {samedata} finds TopK SAEs more seed-dependent than L1-trained ones, {saebench} that proxy metrics mislead — and the same group's {weightsparse} later trains sparse weights instead.",
    "absorption": "Documents a systematic way the dictionaries of {cunningham}–{topk} misrepresent hierarchical concepts, and became one of {saebench}'s metrics; {gdmnegative} cites it among the flaws that keep SAEs from a canonical set of concepts. Read with {samedata} and {sanity}, which question from other angles whether a dictionary's latents are the model's features.",
    "sfc": "Joins the dictionaries of {cunningham}–{scaling} to the circuit discovery of {ioi} and {acdc}, keeping SAE error as explicit nodes. Its SHIFT result is one of the few prior downstream wins {saeprobing} credits to SAEs; {transcoders} and {tracing} swap its per-sublayer SAEs for replacement layers, and {mib}, led by its last author, benchmarks such featurizers.",
    "transcoders": "Where {sfc} attributes through MLPs with gradients, transcoders replace them with sparse layers whose weights give input-invariant feature connections. {crosscoders} extends the idea across layers and {tracing} builds a replacement model from it; {gdmnegative} counts transcoders among SAE-based tools unlikely to be a game-changer, and {arora} traces through the original neurons.",
    "crosscoders": "The bridge from {transcoders} to the cross-layer transcoders of {tracing}, and the start of crosscoder model diffing. A preliminary update by its own account, it finds most model-specific features uninterpretable; {gdmnegative} names crosscoders among tools unlikely to transform the field. Its worry about leaving the literal model is pressed by {arora} and {spd}.",
    "tracing": "Turns {transcoders} and the cross-layer dictionaries of {crosscoders} into a replacement model with attribution graphs, applied at length in {biology}. Its fidelity is what Section F bears on — {samedata} and {sanity} question whether a dictionary's features are the model's — and what {arora} and {clean} press on when they build graphs from raw neurons.",
    "biology": "The applied companion to {tracing} and the most ambitious set of mechanistic case studies here, published within days of {gdmnegative}'s turn away from SAEs. Its chain-of-thought findings speak to {turpin} and {reasoningfaith}, its refusal study to {refusal}, and {clean} redraws its medical-diagnosis graph on raw neurons in another model.",
    "gdmnegative": "The sharpest rebuttal to the promise of {scaling}, from the team behind many SAE refinements: SAE probes lose to linear probes, and the team changes course. It cites {saeprobing} and {axbench} as converging evidence and {absorption} among SAE flaws; it still sees SAEs as useful for exploratory debugging, as in {auditing}, and {pragmatic} sets out the team's later agenda.",
    "axbench": "Tests the feature steering showcased in {scaling} against the steering of {repe}–{refusal} and against plain prompting. From the Stanford group behind {das}, it takes difference-in-means from {truth} as a baseline that SAEs fail to beat. {gdmnegative} cites it with {saeprobing} as converging evidence against SAEs on downstream tasks.",
    "saeprobing": "The systematic version of {gdmnegative}'s probing result, sharing authors with it. Its probes descend from {haystack}'s sparse probing and inherit the cautions of {controltasks} and {probing}; it credits {sfc}'s SHIFT among the few prior SAE wins. With {axbench} it makes beating strong baselines the standard of evidence for interpretability tools.",
    "randomsae": "Applies to SAEs the randomisation test {saliency} applied to saliency maps, aiming it at the explain-and-score metric of {explain} that {cunningham} relied on. {sanity} moves the null model from random transformers to random dictionaries, and {saebench} is the kind of evaluation suite such baselines would calibrate.",
    "sanity": "Moves {randomsae}'s null-model test from random transformers to random dictionaries; its title echoes {saliency}. It scores with metrics from {saebench} and cites {gdmnegative} and {axbench} as motivation. A frontier preprint: weigh how far its synthetic ground truth, built on {toy}'s assumptions, licenses claims about real models.",
    "samedata": "Undercuts the hope, from {toy} and {mono}, that a dictionary enumerates a model's true features; its finding that TopK ({topk}) is less stable than L1 qualifies Section E's architecture gains. Read with {absorption} and {sanity}; {universal} and {crosscoders} ask the same sameness question of models across seeds rather than of dictionaries.",
    "saebench": "The dictionary programme's own audit: eight metrics across 200+ SAEs show that the sparsity–fidelity proxies, which {gdmnegative} also urges the field to stop optimising, mislead. Its automated-interpretability and sparse-probing metrics are the ones {randomsae} and {saeprobing} challenge, and {sanity} uses its suite to score random baselines.",
    "mib": "The circuits-and-variables counterpart to {saebench}: it benchmarks Section C's methods ({patching}, {acdc}, {das}) and featurizers including SAEs, led by {sfc}'s last author. Its finding that SAE features are no better than neurons for causal-variable localization links the reckoning to {arora}'s neuron-basis programme.",
    "arora": "The empirical anchor of the neuron-basis bet: it reruns {sfc}'s subject–verb agreement task and {biology}'s multi-hop task on raw MLP neurons, in the line of {haystack}, {universal} and {mib}; two of its authors also wrote {axbench}. {clean} pushes the same bet much further, while {spd}–{vpd} decompose the weights instead and {worldinside} doubts that any single direction is the right unit.",
    "clean": "The strongest form of the neuron-basis bet in {arora}, by a co-author of {zoomin}, {curves}, {multimodal} and {explain}: it re-reads the polysemanticity behind {toy} and {mono} as a product of weak tools and small models. Goodfire's {worldinside}, five months earlier, puts concepts on manifolds; {clean} treats submanifolds as one view of the residual stream and bets on scaled-up investigation over new decompositions like {spd}–{vpd}.",
    "weightsparse": "The one programme here that changes the model rather than the method: like {solu} it buys legibility in training, from the group behind {topk}. Where {arora} and {clean} say dense networks are already readable in their neurons, and {spd}–{vpd} decompose the weights a dense model has, it trains sparsity in, at a capability cost it measures. Its pruned task circuits invite comparison with {ioi} and {acdc}.",
    "spd": "The founding method of the parameter-decomposition programme: it decomposes weights rather than activations, citing feature splitting ({mono}, {absorption}) and feature geometry ({mendel}) as problems of dictionaries. Its tests are toy models in the tradition of {toy}; {vpd} scales it to a language model, and its last author led {openproblems}. {arora} and {clean} contend that the neuron basis already suffices.",
    "vpd": "{spd} carried to a real language model, and the sharpest statement of the parameter-versus-activation dispute: transcoders and CLTs ({transcoders}, {tracing}) fit replacement models of a different functional form, while VPD stays inside the original architecture. Its multi-head attention components sit uneasily with head-level circuits like {ioi}. It appeared two days before {worldinside}, Goodfire's bet on activation geometry.",
    "worldinside": "The geometric counter-bet to {clean}, from the same company five months earlier: where {clean} reads neurons, it argues that single directions, SAE latents included, miss concepts that are curves. It builds on {grokking}, {clockpizza} and {notlinear}, aims its steering critique at {truth} and {actadd}, and is formalized for SAEs in {manifolds}. Its lead author wrote {causal}–{das}; its last, {hydra}.",
    "manifolds": "The technical companion to {worldinside}, taking {notlinear}'s move beyond {lrh}'s single directions to dictionaries: it argues SAE troubles recorded in {absorption}, {axbench} and {saeprobing} may partly reflect manifolds tiled by many features. It diagnoses the dictionary programme ({cunningham}–{absorption}) rather than abandoning it, against {clean}, which takes neurons rather than groups of atoms as the unit.",
    "nla": "Outside the four programmes and the opposite trade to each: legibility without mechanism, an approach its authors call black-box by construction. It descends from {logitlens} and the injected-vector reports of {introspection}, reruns {biology}'s rhyme-planning case, and on the {auditing} model organism finds that SAEs add nothing, consistent with {gdmnegative}.",
    "devlandscape": "A developmental programme rather than a decomposition: it asks when structure forms in training, not what a finished network contains. It recovers the induction circuit of {induction} as a stage and sits beside {grokking}'s progress measures; like {spd}–{vpd} it works in parameter space rather than activations. Two of its authors, Hoogland and Murfet, co-wrote {openproblems}.",
    "repe": "The top-down programme of this collection: concepts read and steered as population-level directions, with no circuit account. {actadd} and {refusal} are narrower instances, {emotions} applies a similar contrastive recipe to a frontier model, and {axbench} tests such steering against dictionary features. Its co-author Hendrycks offers it in {misguided} as the alternative to bottom-up mechanism.",
    "actadd": "The minimal steering method — one activation difference from a contrast pair — that {repe} generalises and {refusal} applies. The same move returns as concept injection in {introspection}, and {pragmatic} reports a single-contrast-pair steering vector beating SAEs at suppressing evaluation awareness; {againsttoi} doubts whether steering counts as interpretability at all.",
    "refusal": "Applies the difference-in-means directions of {repe}–{actadd} to a safety behaviour, with ablation and addition as causal tests, in the linear-representation tradition of {lrh} and {truth}. {biology} gives a circuit-level view of refusal, and {pragmatic} cites the single-direction claim as a model of a mechanistic hypothesis that predicts the outcome of an intervention.",
    "ccs": "The founding paper of unsupervised knowledge elicitation, printed beside its rebuttal {ccschallenges}, which proves that arbitrary features satisfy its consistency objective. Its linear truth direction anticipates {truth}, its probes face the selectivity questions of {controltasks}–{probing}, and {againsttoi} devotes an appendix case study to it.",
    "ccschallenges": "The rebuttal to {ccs}: if a simulated character's beliefs satisfy the same consistency constraints as the model's own, no unsupervised objective of this kind can tell them apart. Its distractor tests extend the probe scepticism of {controltasks}–{probing}, and the persona problem resurfaces in {auditing}, where non-assistant personas leak what the assistant hides.",
    "auditing": "The first controlled test of the deception-auditing theory of impact that {againsttoi} called out of reach. {urgency} counts it among the breakthroughs behind its optimism and {pragmatic} adopts it as a model proxy task, while its own semantic-search comparison echoes the baseline results of {gdmnegative}–{axbench}.",
    "turpin": "Moves the question from mechanism to self-report: a model's stated reasoning may not be its reasoning. {reasoningfaith} reruns its biasing-hint method on reasoning models and {monitor} tries to measure when chains of thought stay usable; {againsttoi} and {deceptive} both rate reading the chain of thought above interpretability as a near-term safeguard, which these results qualify.",
    "reasoningfaith": "Carries the hint methodology of {turpin} to reasoning models trained with RL, the setting in which chain-of-thought monitoring was meant to work. {monitor} turns the problem into an evaluation programme; for {againsttoi} and {deceptive}, which favour reading the chain of thought over reading internals, these results mark how far that route can be trusted.",
    "monitor": "OpenAI's attempt to make chain-of-thought oversight measurable, taking up the faithfulness problem of {turpin} and {reasoningfaith}. It is the black-box route that {deceptive} places beside interpretability in a defence-in-depth portfolio; {biology} and {introspection} are internal routes to checking what a model is actually doing.",
    "introspection": "Turns the steering of {actadd} into an experimental probe of self-report, the internal counterpart to the chain-of-thought faithfulness of {turpin}–{reasoningfaith}. {pragmatic} lists basic introspection among interpretability's genuine findings; {emotions}, from the same group, asks what the model's emotional self-presentation is made of.",
    "emotions": "Applies the contrastive-direction methods of {repe}–{refusal} to a frontier model's character, tying representations to alignment-relevant behaviour as {auditing} does with a planted objective. Its population-level approach is close to the top-down interpretability {misguided} recommends, and it is the model-psychology work {pragmatic} counts as a robustly useful setting.",
    "mythos": "Opens the critiques with a conceptual one: before asking whether interpretability works, ask what it is for. {mechanistic} traces how ‘mechanistic’ later became a banner, {olahessay} takes something like Lipton's decomposability as the goal, and {pragmatic} cites this paper among the academic arguments behind its pivot.",
    "saliency": "The template for the field's sanity checks: compare an explanation of the trained model with the same method on a randomised one. {randomsae} and {sanity} apply that logic to sparse autoencoders, {featurevis}–{netdissect} belong to the same vision-era toolkit, and both {againsttoi} and {misguided} cite the result as a warning against judging explanations by eye.",
    "microprocessor": "An outside parable that cuts both ways. Its seemingly game-specific transistors warn against reading lesions as function, a lesson for {patching}, {acdc} and {hydra}; its call for ground-truth testbeds anticipates {mib} and the planted objective of {auditing}. Read it with {misguided}, which takes complex-systems arguments further and argues against bottom-up mechanism altogether.",
    "eis": "Names many of this collection's foundations as targets: {featurevis}, {zoomin}, {curves}, {framework}, {induction}, {solu}, {ioi}, {grokking} and {scrubbing}. Its demand for benchmarks is answered in part by {saebench} and {mib}, its call for automation by {acdc} and {explain}; {againsttoi} presses the same doubts from theory of impact rather than method.",
    "againsttoi": "A response to Nanda's list of theories of impact and the sceptics' fullest case against the optimism of {urgency}. {auditing} directly tests the deception-auditing claim it dismisses, its preference for monitoring chains of thought meets {turpin}–{monitor}, and {deceptive}, from inside the field, reaches a milder version of its conclusion.",
    "misguided": "Written weeks after {urgency} and answering its ‘MRI for AI’, it enlists {saliency} and {gdmnegative} as evidence. Its alternative is {repe}, which Hendrycks co-authored, so it is partly a case for a rival programme. Set its complex-systems argument against the compact mechanisms of {ioi}, {grokking} and {biology}.",
    "deceptive": "Prompted by {urgency} and written by a leading mechanistic researcher who says he once held the view he now rejects. It shares {againsttoi}'s doubt that interpretability can certify the absence of deception but keeps it as decorrelated signal, citing the probing results of {gdmnegative}; {pragmatic} turns the stance into a research programme.",
    "pragmatic": "The programme behind {gdmnegative} and {saeprobing}, and a counterweight to {urgency}'s bet on deep understanding. It credits {auditing}, {biology} and {introspection} as real progress while listing superposition, {hydra}'s self-repair and dictionary error as blockers; the programmes of Section G are the ambitious bets it calls one option among several.",
    "urgency": "The collection ends on its most optimistic statement, printed against {misguided}, {deceptive} and {againsttoi}. Amodei cites {auditing} and {biology} as the breakthroughs behind his optimism and retells the arc from {multimodal} through {toy} and {mono}; whether the competing programmes of Section G can meet his timeline is left open.",
}
N = len(CATALOG)
for _i, _p in enumerate(CATALOG, 1):
    _p["n"] = _i
    _p["id"] = f"{_i:02d}"
    _p["file"] = f"{_p['id']}_{_p['slug']}.pdf"
BY_KEY = {_p["key"]: _p for _p in CATALOG}
BRIEF = {p["key"] for p in CATALOG if p["kind"] != "paper"}  # essays, posts, notes: lower length thresholds in QA
_REFS = {k: v["id"] for k, v in BY_KEY.items()}
for _p in CATALOG:
    _p["context"] = CONTEXT[_p["key"]].format(**_REFS)
    _p["optional_raw"] = _p["optional"]  # {key} template, re-formatted as links in the Markdown export
    _p["optional"] = _p["optional"].format(**_REFS)
assert set(CONTEXT) == set(BY_KEY), set(CONTEXT) ^ set(BY_KEY)
SUBTITLE = f"A Research Anthology — {N} Papers, Essays and Posts"


def readings(n: int) -> str:
    """Count phrase for catalog items, which include essays and forum posts as well as papers."""
    return "1 reading" if n == 1 else f"{n} readings"

# Transformer Circuits print rendering. The site's article grid is a fixed ~1032px layout, which
# Chrome either auto-shrinks (small text) or clips (wide figures). We instead collapse the grid to
# the printable width (7.5in at HTML_SCALE) so body text prints at a fixed, legible size, then zoom
# only the individual figures/widgets that are still wider than the page.
HTML_SCALE = 0.9  # 16px body text -> 10.8pt on paper, close to typical arXiv body size
HTML_CW = round(720 / HTML_SCALE)  # layout width (CSS px) of the 7.5in printable area at that scale
HTML_CSS = """
@page { size: Letter !important; margin: 0.5in 0.5in 0.55in 0.5in !important;
        @bottom-right { content: none !important; } }
@media print { @page { size: Letter !important; margin: 0.5in 0.5in 0.55in 0.5in !important;
                       @bottom-right { content: none !important; } } }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
figure, img, svg, table, pre, .sn-figma-div, .ha-block { break-inside: avoid; }
h1, h2, h3, h4 { break-after: avoid; }
/* section numbers ("§ 2.1") hang in the left gutter on screen; print them inline before the heading */
.toc-counter { position: static !important; display: inline !important; left: auto !important; margin-right: 0.35em; }
"""
FIT_JS = r"""
(cw) => {
  const tmpl = '[screen-start page-start kicker-start middle-start text-start gutter-start] minmax(0, 1fr) ' +
               '[screen-end page-end kicker-end middle-end text-end gutter-end]';
  let grids = 0;
  for (const el of document.querySelectorAll('body, body *')) {
    const cs = getComputedStyle(el);
    if (cs.display.includes('grid') && cs.gridTemplateColumns.includes('text-start')) {
      for (const [k, v] of [['grid-template-columns', tmpl], ['column-gap', '0px'], ['width', 'auto'], ['max-width', 'none'],
                            ['margin-left', '0'], ['margin-right', '0'], ['padding-left', '0'], ['padding-right', '0']])
        el.style.setProperty(k, v, 'important');
      grids++;
    }
  }
  document.body.style.setProperty('margin', '0', 'important');
  const clipped = (el) => {           // inside an element that clips overflow by design (or inside an svg)
    if (el instanceof SVGElement && el.tagName.toLowerCase() !== 'svg') return true;
    for (let a = el.parentElement; a && a !== document.body; a = a.parentElement)
      if (getComputedStyle(a).overflowX !== 'visible') return true;
    return false;
  };
  const zoomed = [];
  const offenders = () => {          // top-most, unclipped elements that still cross the page edges
    const out = [];
    for (const el of document.querySelectorAll('body *')) {
      const r = el.getBoundingClientRect();
      if (!r.width || !r.height || (r.right <= cw + 1 && r.left >= -1)) continue;
      const cs = getComputedStyle(el);
      if (cs.visibility === 'hidden' || (['absolute', 'fixed'].includes(cs.position) && (r.right < 0 || r.left > cw))) continue;
      let top = true;
      for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
        const ar = a.getBoundingClientRect();
        if (ar.right > cw + 1 || ar.left < -1) { top = false; break; }
      }
      if (top && !clipped(el)) out.push([el, r, cs]);
    }
    return out;
  };
  for (let pass = 0; pass < 4; pass++) {
    let changed = 0;
    for (const [el, r, cs] of offenders()) {
      if (r.left < -1 && ['static', 'relative'].includes(cs.position) && !el.dataset.fitShift) {
        el.style.setProperty('margin-left', '0', 'important');
        if (cs.position === 'relative') el.style.setProperty('left', '0', 'important');
        el.dataset.fitShift = '1';
        changed++;
      }
      const er = el.getBoundingClientRect();
      let t = el, tr = er;
      if (tr.left >= cw - 1 || tr.left < -1) { t = el.parentElement; tr = t.getBoundingClientRect(); }
      if (t.dataset.fitZoom) continue;  // zoom each element at most once (no compounding)
      const extent = Math.max(tr.width, t.scrollWidth, er.right - tr.left);
      const avail = cw - Math.max(0, tr.left);
      if (extent <= avail + 1) continue;
      const z = Math.max(0.5, avail / extent);
      t.style.zoom = z.toFixed(4);
      t.dataset.fitZoom = '1';
      zoomed.push(`${t.tagName}.${(typeof t.className === 'string' ? t.className : '').slice(0, 30)} z=${z.toFixed(2)}`);
      changed++;
    }
    if (!changed) break;
  }
  // Figures whose *content* is wider than the page (e.g. Distill "fullscreen" diagrams) can sit inside a
  // page-width box, so the bounding-box pass above misses them; Chrome's print layout still widens for them.
  for (const el of document.querySelectorAll('figure, d-figure, .fullscreen-diagram')) {
    if (el.dataset.fitZoom || el.parentElement?.closest('[data-fit-zoom]')) continue;  // already scaled
    const sw = el.scrollWidth, avail = cw - Math.max(0, el.getBoundingClientRect().left);
    if (sw <= avail + 2) continue;
    const z = Math.max(0.5, avail / sw);
    el.style.zoom = z.toFixed(4);
    el.dataset.fitZoom = '1';
    zoomed.push(`${el.tagName}.${(typeof el.className === 'string' ? el.className : '').slice(0, 30)} z=${z.toFixed(2)} (content)`);
  }
  const residual = offenders().map(([el, r]) =>
      `${el.tagName}.${(typeof el.className === 'string' ? el.className : '').slice(0, 30)} L${Math.round(r.left)} R${Math.round(r.right)}`);
  return {grids, zoomed, residual, scrollWidth: document.documentElement.scrollWidth};
}
"""

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def log(*a):
    print(*a, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower()
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"[^a-z0-9]+", "", s)


def load_meta() -> dict:
    """Per-paper records keyed by paper id in memory, by slug on disk (so renumbering never orphans a record)."""
    raw = json.loads(META_CACHE.read_text()) if META_CACHE.exists() else {}
    by_url = {r.get("source_url"): r for r in raw.values() if isinstance(r, dict)}  # migrate id-keyed caches
    out = {}
    for p in CATALOG:
        rec = raw.get(p["slug"]) or by_url.get(p["html"] or p["pdf"] or f"https://arxiv.org/abs/{p['arxiv']}")
        if rec:
            out[p["id"]] = rec
    return out


def save_meta(meta: dict):
    BUILD.mkdir(exist_ok=True)
    disk = {p["slug"]: meta[p["id"]] for p in CATALOG if p["id"] in meta}
    META_CACHE.write_text(json.dumps(disk, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------
# Stage 1: fetch
# --------------------------------------------------------------------------
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def http_get(url: str, params: dict, timeout=240) -> requests.Response:
    """GET with retries: arXiv's export API and OAI endpoint are often slow or briefly unavailable."""
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(10)


def arxiv_metadata(ids: list[str]) -> dict:
    r = http_get("https://export.arxiv.org/api/query", {"id_list": ",".join(ids), "max_results": len(ids)})
    out = {}
    for e in ET.fromstring(r.text).findall("a:entry", ATOM):
        idv = e.find("a:id", ATOM).text.rsplit("/abs/", 1)[1]
        out[idv] = dict(
            title=" ".join(e.find("a:title", ATOM).text.split()),
            authors=[a.find("a:name", ATOM).text for a in e.findall("a:author", ATOM)],
            published=e.find("a:published", ATOM).text[:10],
            updated=e.find("a:updated", ATOM).text[:10],
        )
    for idv in out:
        base = idv.rsplit("v", 1)[0]
        x = http_get("https://export.arxiv.org/oai2", {
            "verb": "GetRecord", "identifier": f"oai:arXiv.org:{base}", "metadataPrefix": "arXiv"}, timeout=90).text
        m = re.search(r"<license>(.*?)</license>", x)
        out[idv]["license"] = m.group(1) if m else "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
        time.sleep(1)
    return out


def license_label(url: str) -> str:
    if "creativecommons.org/licenses/by/4.0" in url:
        return "CC BY 4.0"
    if "creativecommons.org/licenses/by-sa/4.0" in url:
        return "CC BY-SA 4.0"
    if "creativecommons.org/licenses/by-nc-sa" in url:
        return "CC BY-NC-SA 4.0"
    if "creativecommons.org/publicdomain/zero" in url:
        return "CC0 1.0"
    if "nonexclusive-distrib" in url:
        return "arXiv non-exclusive distribution licence"
    return url


def verify_identity(pdf: Path, title: str, surname: str, pages_to_scan=3) -> tuple[bool, str]:
    doc = pymupdf.open(pdf)
    text = norm(" ".join(doc[i].get_text() for i in range(min(pages_to_scan, doc.page_count))))
    words = [w for w in re.findall(r"[A-Za-z0-9]+", title)]
    probe = norm(" ".join(words[:6]))
    ok_t = probe in text
    ok_a = norm(surname) in text
    return ok_t and ok_a, f"title-probe={'ok' if ok_t else 'MISSING'} author={'ok' if ok_a else 'MISSING'}"


def download(url: str, dest: Path):
    tmp = dest.with_suffix(".part")
    with requests.get(url, headers=UA, timeout=180, stream=True) as r:
        r.raise_for_status()
        expected = int(r.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    size = tmp.stat().st_size
    if expected and size != expected:
        raise IOError(f"truncated download {size} != {expected}")
    head = tmp.read_bytes()[:5]
    if head != b"%PDF-":
        raise IOError("not a PDF")
    tmp.replace(dest)


RENDER_JS_PREP = r"""
() => {
  document.querySelectorAll('details').forEach(d => d.open = true);
  document.querySelectorAll('img[loading]').forEach(i => i.loading = 'eager');
  const heads = [...document.querySelectorAll('d-article h2, d-article h3, d-appendix h2, d-appendix h3, h2, h3')]
     .filter(h => h.offsetParent !== null && !h.closest('button, [role=tab], [role=tablist]'))  // skip widget controls
     .map(h => h.innerText.trim()).filter(t => t.length > 3);
  const bib = [...document.querySelectorAll('script[type="text/bibtex"], pre, code')].map(e => e.innerText).join('\n');
  const h1 = (document.querySelector('d-title h1, h1') || {}).innerText || document.title;
  return {headings: [...new Set(heads)], title: h1.trim(), bibtext: bib, doctitle: document.title};
}
"""


# Clamp the document to the printable width. Without it, interactive widgets wider than the page
# (button rows, scroll-container content) make Chrome re-lay the whole page out wider and then
# shrink it to fit, which printed the Goodfire essay's body text at ~6.6pt instead of ~9.9pt.
CLAMP_CSS = f"""
html, body {{ max-width: {HTML_CW}px !important; overflow-x: clip !important; }}
body > * {{ max-width: {HTML_CW}px !important; overflow-x: clip !important; contain: paint; }}
"""


# Print only the selected parts of a page (in this order), e.g. a blog post's header and body without site chrome.
KEEP_JS = r"""
(sels) => {
  const box = document.createElement('div');
  box.className = 'mi-kept';
  const missing = sels.filter(s => { const e = document.querySelector(s); if (e) box.appendChild(e); return !e; });
  document.body.replaceChildren(box);
  return missing;
}
"""


async def _render(url: str, out: Path, clamp_width=False, reduced_motion=False, scroll=True, reveal=(), keep=(),
                  idle=True, wait=0, css="") -> dict:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        exe = next((x for x in ("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser") if Path(x).exists()), None)
        browser = await p.chromium.launch(executable_path=exe) if exe else await p.chromium.launch()
        page = await browser.new_page(viewport={"width": HTML_CW, "height": 1000})
        # measure in the print layout at the printable width; reduced motion makes animated figures
        # render their final static frame
        await page.emulate_media(media="print", reduced_motion="reduce" if reduced_motion else "no-preference")
        await page.goto(url, wait_until="load", timeout=300_000)
        if idle:
            try:
                await page.wait_for_load_state("networkidle", timeout=120_000)
            except Exception:
                pass
        await page.wait_for_timeout(wait)
        if keep:
            missing = await page.evaluate(KEEP_JS, list(keep))
            if missing:
                raise RuntimeError(f"render: selectors not found on page: {missing}")
        info = await page.evaluate(RENDER_JS_PREP)
        # scroll through the whole page to trigger lazy loading of figures. Skipped for pages whose
        # scroll-triggered widgets mount wider than the page and force Chrome to shrink the print.
        y, h = 0, (await page.evaluate("document.body.scrollHeight")) if scroll else 0
        while y < h:
            await page.evaluate(f"window.scrollTo(0,{y})")
            await page.wait_for_timeout(30)
            y += 1000
            h = await page.evaluate("document.body.scrollHeight")
        for sel in reveal:  # mount specific scroll-triggered figures without scrolling the whole page
            await page.evaluate("s => document.querySelector(s)?.scrollIntoView({block: 'center'})", sel)
            await page.wait_for_timeout(4000)
        try:
            await page.wait_for_load_state("networkidle", timeout=120_000 if idle else 1)
        except Exception:
            await page.wait_for_timeout(0 if idle else 3000)  # let lazy-loaded images settle
        await page.evaluate("async () => { window.scrollTo(0, 0); await document.fonts.ready; }")
        await page.add_style_tag(content=HTML_CSS + (CLAMP_CSS if clamp_width else "") + css)
        await page.wait_for_timeout(1000)
        fit = await page.evaluate(FIT_JS, HTML_CW)
        await page.wait_for_timeout(2000)
        await page.pdf(path=str(out), prefer_css_page_size=True, print_background=True, outline=True, tagged=True,
                       scale=HTML_SCALE)
        await browser.close()
    info["fit"] = fit
    return info


def page_is_blank(page: pymupdf.Page) -> bool:
    if page.get_text().strip() or page.get_images():
        return False
    pix = page.get_pixmap(dpi=36, colorspace=pymupdf.csGRAY)
    dark = sum(1 for b in pix.samples if b < 235)
    return dark < 0.002 * len(pix.samples)


def render_html(p: dict, dest: Path) -> dict:
    raw = BUILD / f"raw_{p['id']}.pdf"
    log(f"  rendering {p['html']}")
    info = asyncio.run(_render(p["html"], raw, **p["render"]))
    doc = pymupdf.open(raw)
    blank = [i for i in range(doc.page_count) if page_is_blank(doc[i])]
    if blank:
        doc.delete_pages(blank)
    doc.save(dest, garbage=3, deflate=True)
    m = re.search(r"author\s*=\s*\{(.+?)\}\s*,?\s*\n", info.get("bibtext", ""), re.S)
    authors = [a.strip() for a in re.split(r"\s+and\s+", " ".join(m.group(1).split()))] if m else []
    authors = [" ".join(reversed([s.strip() for s in a.split(",", 1)])) if "," in a else a for a in authors]
    authors = [a.strip(" *†‡§") for a in authors if a.strip(" *†‡§")] or p["authors"] or []
    ym = re.search(r"year\s*=\s*\{(\d{4})\}", info.get("bibtext", ""))
    return dict(title=info["title"], authors=authors, published=ym.group(1) if ym else p["year"],
                license=p["license"] or f"No explicit licence published on {urlparse(p['html']).netloc}",
                headings=info["headings"], removed_blank_pages=len(blank), raw_pages=doc.page_count + len(blank),
                fit=dict(grids_collapsed=info["fit"]["grids"], zoomed_elements=len(info["fit"]["zoomed"]),
                         min_zoom=min([float(z.rsplit("=", 1)[1].split()[0]) for z in info["fit"]["zoomed"]] or [1.0]),
                         final_scroll_width=info["fit"]["scrollWidth"], zoom_log=info["fit"]["zoomed"][:200],
                         residual_overflow=info["fit"]["residual"][:50]))


def stage_fetch(force=False):
    SOURCES.mkdir(exist_ok=True)
    BUILD.mkdir(exist_ok=True)
    meta = load_meta()
    slugs = {p["slug"] for p in CATALOG}
    for f in SOURCES.glob("[0-9][0-9]_*.pdf"):  # sources/ holds exactly the works in the anthology
        if f.stem[3:] not in slugs:
            log(f"removing {f.name} (no longer in the catalog)")
            f.unlink()
    ax_ids = [p["arxiv"] for p in CATALOG if p["arxiv"] and (force or p["id"] not in meta or "arxiv_meta" not in meta[p["id"]])]
    axmeta = arxiv_metadata(ax_ids) if ax_ids else {}
    for p in CATALOG:
        dest = SOURCES / p["file"]
        if not dest.exists():  # paper renumbered: carry the cached source over under its new number
            old = [f for f in SOURCES.glob(f"[0-9][0-9]_{p['slug']}.pdf") if f != dest]
            if old:
                old[0].rename(dest)
        rec = meta.get(p["id"], {})
        try:
            if p["arxiv"]:
                if p["arxiv"] in axmeta:
                    rec["arxiv_meta"] = axmeta[p["arxiv"]]
                am = rec["arxiv_meta"]
                if force or not dest.exists():
                    url = f"https://arxiv.org/pdf/{p['arxiv']}"
                    log(f"[{p['id']}] downloading {url}")
                    download(url, dest)
                    time.sleep(3)
                rec.update(source_type="arXiv PDF", source_url=f"https://arxiv.org/abs/{p['arxiv']}",
                           pdf_url=f"https://arxiv.org/pdf/{p['arxiv']}", verified_title=am["title"],
                           authors=am["authors"], first_posted=am["published"], version_date=am["updated"],
                           license_url=am["license"], license=license_label(am["license"]))
            elif p["pdf"]:
                if force or not dest.exists():
                    log(f"[{p['id']}] downloading {p['pdf']}")
                    download(p["pdf"], dest)
                rec.update(source_type=f"Publisher PDF ({p['venue']})", source_url=p["pdf"], pdf_url=p["pdf"],
                           verified_title=p["title"], authors=p["authors"] or [], first_posted=p["year"],
                           license=p["license"], license_url="")
            else:
                if force or not dest.exists() or "headings" not in rec:
                    log(f"[{p['id']}] print-rendering HTML")
                    hm = render_html(p, dest)
                    rec.update(verified_title=hm["title"], authors=hm["authors"], first_posted=hm["published"],
                               license=hm["license"], license_url="", headings=hm["headings"],
                               removed_blank_pages=hm["removed_blank_pages"], raw_pages=hm["raw_pages"], fit=hm["fit"])
                r = p["render"]
                fitted = ("article header and body only, site navigation and comments omitted" if r.get("keep")
                          else "page clamped to Letter width" if r.get("clamp_width")
                          else "article grid fitted to Letter width")
                rec.update(source_type="Print rendering of the official HTML publication (headless Chrome, site print "
                                       f"stylesheet, {fitted})",
                           source_url=p["html"], pdf_url="")
            surname = p["byline"]
            ok, detail = verify_identity(dest, p["title"] if p["arxiv"] is None else rec["verified_title"], surname)
            doc = pymupdf.open(dest)
            rec.update(status="included" if ok else "identity-check-failed", identity=detail,
                       pages=doc.page_count, size=dest.stat().st_size, sha256=sha256(dest), retrieved=rec.get("retrieved", TODAY))
            log(f"[{p['id']}] {doc.page_count:3d} pp  {detail}")
        except Exception as e:  # keep going; the build inserts a source-access page
            log(f"[{p['id']}] FAILED: {e!r}")
            rec.update(status="unavailable", error=repr(e))
        meta[p["id"]] = rec
        save_meta(meta)


# --------------------------------------------------------------------------
# Stage 2: build — typography & drawing helpers (ReportLab)
# Minimal design: Source Serif 4 (titles, prose) + Inter (labels, numbers), hairline rules,
# no filled bars. Fonts are vendored in scripts/fonts (SIL Open Font License).
# --------------------------------------------------------------------------
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FILES = {
    "Serif": "SourceSerif4-Regular.ttf", "Serif-I": "SourceSerif4-It.ttf", "Serif-SB": "SourceSerif4-Semibold.ttf",
    "Disp": "SourceSerif4Display-Regular.ttf", "Disp-L": "SourceSerif4Display-Light.ttf",
    "Disp-SB": "SourceSerif4Display-Semibold.ttf",
    "Sans": "Inter-Regular.ttf", "Sans-I": "Inter-Italic.ttf", "Sans-L": "Inter-Light.ttf",
    "Sans-M": "Inter-Medium.ttf", "Sans-SB": "Inter-SemiBold.ttf",
}
FONT_FALLBACK = {"Serif": "Times-Roman", "Serif-I": "Times-Italic", "Serif-SB": "Times-Bold", "Disp": "Times-Roman",
                 "Disp-L": "Times-Roman", "Disp-SB": "Times-Bold", "Sans": "Helvetica", "Sans-I": "Helvetica-Oblique",
                 "Sans-L": "Helvetica", "Sans-M": "Helvetica", "Sans-SB": "Helvetica-Bold"}
FONTS: dict[str, str] = {}

INK, MUTED, FAINT, HAIR = 0.08, 0.40, 0.58, 0.72   # gray levels (0 = black)
ML, MR = 72, W - 72                                # 1in side margins
PRI_WORD = {"P0": "Essential", "P1": "Important", "F": "Frontier"}


def register_fonts():
    if all((FONT_DIR / f).exists() for f in FONT_FILES.values()):
        for k, f in FONT_FILES.items():
            pdfmetrics.registerFont(TTFont(k, str(FONT_DIR / f)))
    else:
        log(f"warning: {FONT_DIR} incomplete — falling back to Times/Helvetica")
        FONTS.update(FONT_FALLBACK)


def F(name: str) -> str:
    return FONTS.get(name, name)


def wrap(text, font, size, width):
    """Word-wrap; tokens wider than the line (URLs) are broken by character."""
    out = []
    for line in simpleSplit(text, F(font), size, width):
        while pdfmetrics.stringWidth(line, F(font), size) > width:
            cut = len(line)
            while cut > 1 and pdfmetrics.stringWidth(line[:cut], F(font), size) > width:
                cut -= 1
            out.append(line[:cut])
            line = line[cut:]
        out.append(line)
    return out


def tw(s, font, size, space=0.0):
    return pdfmetrics.stringWidth(s, F(font), size) + space * max(0, len(s) - 1)


def text(c, x, y, s, font="Serif", size=10, gray=INK, align="l", space=0.0):
    w = tw(s, font, size, space)
    x = x - w if align == "r" else x - w / 2 if align == "c" else x
    t = c.beginText(x, y)
    t.setFont(F(font), size)
    t.setFillGray(gray)
    t.setCharSpace(space)  # Tc persists in the PDF graphics state, so always set it (incl. 0)
    t.textOut(s)
    c.drawText(t)
    return w


def label(c, x, y, s, gray=MUTED, size=7, align="l", font="Sans-M"):
    """Small tracked uppercase label."""
    return text(c, x, y, s.upper(), font, size, gray, align, space=0.8)


def hair(c, x0, y, x1, gray=HAIR, w=0.5):
    c.saveState()
    c.setStrokeGray(gray)
    c.setLineWidth(w)
    c.line(x0, y, x1, y)
    c.restoreState()


def pri_mark(c, cx, cy, pri, r=3.0):
    """Shape-coded priority marker that survives grayscale: P0 filled circle, P1 open circle, F open diamond."""
    c.saveState()
    c.setStrokeGray(INK)
    c.setFillGray(INK)
    c.setLineWidth(0.8)
    if pri == "P0":
        c.circle(cx, cy, r, stroke=0, fill=1)
    elif pri == "P1":
        c.circle(cx, cy, r - 0.4, stroke=1, fill=0)
    else:
        d = r + 0.4
        path = c.beginPath()
        path.moveTo(cx, cy + d)
        path.lineTo(cx + d, cy)
        path.lineTo(cx, cy - d)
        path.lineTo(cx - d, cy)
        path.close()
        c.drawPath(path, stroke=1, fill=0)
    c.restoreState()


def pri_tag(c, x, y, pri, size=7.5, gray=INK):
    """Marker + tracked label, e.g. ● P0 — ESSENTIAL. (x, y) = baseline start. Returns width."""
    pri_mark(c, x + 3, y + size * 0.36, pri, r=size * 0.4)
    return 11 + label(c, x + 11, y, f"{pri} — {PRI_WORD[pri]}", gray=gray, size=size, font="Sans-SB")


def qr(c, url, x, y, size):
    w = QrCodeWidget(url, barBorder=0)
    w.barFillColor = w.barStrokeColor = colors.Color(INK, INK, INK)
    b = w.getBounds()
    d = Drawing(size, size, transform=[size / (b[2] - b[0]), 0, 0, size / (b[3] - b[1]), 0, 0])
    d.add(w)
    renderPDF.draw(d, c, x, y)


def footer(c, text_left, phys, total, word="PDF page"):
    text(c, ML, 38, text_left, "Sans", 7, FAINT)
    text(c, MR, 38, f"{word} {phys} of {total}", "Sans", 7, FAINT, "r")


def heading(c, title, sub=()):
    """Page heading; returns the y where content starts."""
    text(c, ML, H - 86, title, "Disp", 24, INK)
    y = H - 104
    for s in sub:
        text(c, ML, y, s, "Sans", 8, MUTED)
        y -= 11.5
    return y - 18


def rng(a, b):
    return f"{a}" if a == b else f"{a}–{b}"


def ids_where(pred):
    return [p["id"] for p in CATALOG if pred(p)]


def site_render(p) -> bool:
    """Research-site HTML (Transformer Circuits, Distill, OpenAI) printed whole with the standard grid fitting."""
    return bool(p["html"]) and not p["render"].get("keep") and not p["render"].get("clamp_width")


def number_words(n: int) -> str:
    ones = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
           "seventeen eighteen nineteen".split()
    tens = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
    if n < 20:
        return ones[n]
    return tens[n // 10] + ("" if n % 10 == 0 else "-" + ones[n % 10])


def dialog_ranges(pairs):
    """Merge contiguous ranges and format for a print dialog: '12-27,40-55'."""
    merged = []
    for a, b in sorted(pairs):
        if merged and a == merged[-1][1] + 1:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in merged)


def fit_scale(draw, limit, *args) -> float:
    """Largest type scale (≥ 0.8) for which draw(canvas, *args, k) ends above `limit`."""
    for k in (1.0, 0.95, 0.9, 0.85, 0.8):
        if draw(rl_canvas.Canvas(io.BytesIO(), pagesize=letter), *args, k) >= limit:
            return k
    return 0.8


@dataclass
class Layout:
    front_pages: int
    total: int = 0
    papers: dict = field(default_factory=dict)      # id -> dict(cover, start, end, pad)
    sections: dict = field(default_factory=dict)    # sec -> dict(divider, notes, end)
    bib_start: int = 0
    bib_pages: int = 0
    front_toc: dict = field(default_factory=dict)   # name -> page


def compute_layout(front_pages: int, n_pages: dict, bib_pages: int) -> Layout:
    L = Layout(front_pages)
    cur = front_pages + 1
    if cur % 2 == 0:
        cur += 1  # blank verso pad after front matter
    for sec in SECTIONS:
        L.sections[sec] = dict(divider=cur, notes=cur + 1)
        cur += 2  # divider + notes verso
        for p in (q for q in CATALOG if q["sec"] == sec):
            n = n_pages[p["id"]]  # 0 = source unavailable: the cover becomes a source-access page
            L.papers[p["id"]] = dict(cover=cur, start=cur + 1, end=cur + n, pad=(n % 2 == 0), access=(n == 0))
            cur += 1 + n + (1 if n % 2 == 0 else 0)
        last = [q for q in CATALOG if q["sec"] == sec][-1]
        L.sections[sec]["end"] = L.papers[last["id"]]["end"]
    L.bib_start = cur
    L.bib_pages = bib_pages
    L.total = cur + bib_pages - 1
    return L


# --------------------------------------------------------------------------
# Front matter (title, legend, contents, print index, section ranges, P0 checklist)
# --------------------------------------------------------------------------
class Pager:
    """Minimal canvas pager that tracks physical page numbers and link rects."""

    def __init__(self, path_or_buf, first_phys: int, total: int, header: str, word="PDF page"):
        self.word = word
        self.c = rl_canvas.Canvas(path_or_buf, pagesize=letter)
        self.c.setTitle(TITLE)
        self.phys = first_phys
        self.total = total
        self.header = header
        self.local = 0
        self.links: list[tuple[int, tuple, int]] = []  # (local page, fitz rect, target phys)
        self.anchors: dict[str, int] = {}
        self.y = H - 72

    def link(self, x0, y0, x1, y1, target):
        self.links.append((self.local, (x0, H - y1, x1, H - y0), target))

    def anchor(self, name):
        self.anchors[name] = self.phys

    def finish_page(self):
        footer(self.c, self.header, self.phys, self.total, self.word)
        self.c.showPage()
        self.phys += 1
        self.local += 1
        self.y = H - 72

    def need(self, h, on_new=None):
        if self.y - h < 62:
            self.finish_page()
            if on_new:
                on_new()

    def save(self):
        self.c.save()


def draw_title_page(pg: Pager, L: Layout, stats: dict):
    c = pg.c
    pg.anchor("title")
    y = H - 170
    label(c, ML, y, "A research anthology  ·  print edition")
    y -= 56
    text(c, ML - 2, y, "Mechanistic", "Disp-L", 48)
    y -= 52
    text(c, ML - 2, y, "Interpretability", "Disp-L", 48)
    y -= 32
    for t in wrap(f"{number_words(N).capitalize()} papers, essays and forum posts on circuits, features, dictionary "
                  "learning and alignment — and the arguments over them", "Serif-I", 12.5, MR - ML):
        text(c, ML, y, t, "Serif-I", 12.5, MUTED)
        y -= 17
    y += 17
    y -= 64
    label(c, ML, y, "Contents")
    label(c, MR, y, "Pages", align="r")
    y -= 10
    hair(c, ML, y, MR)
    y -= 22
    for sec, name in SECTIONS.items():
        s = L.sections[sec]
        n = sum(1 for p in CATALOG if p["sec"] == sec)
        text(c, ML, y, sec, "Disp-SB", 12.5)
        w = text(c, ML + 24, y, name, "Serif", 12)
        text(c, ML + 24 + w + 8, y, readings(n), "Sans", 7.5, FAINT)
        text(c, MR, y, f"{s['divider']}–{s['end']}", "Sans", 9, MUTED, "r")
        pg.link(ML - 4, y - 6, MR + 4, y + 13, s["divider"])
        y -= 24
    y -= 2
    hair(c, ML, y + 12, MR)
    y -= 22
    for t in (f"{stats['papers_included']} complete original works  ·  {L.total} pages  ·  US Letter",
              "All page numbers are physical PDF pages (this page is 1); type them directly into any print dialog.",
              f"Print index p. {L.front_toc.get('index', 0)}   ·   Contents p. {L.front_toc.get('toc', 0)}   ·   "
              f"Bibliography p. {L.bib_start}"):
        text(c, ML, y, t, "Sans", 8, MUTED)
        y -= 13
    text(c, ML, 64, f"Compiled {TODAY}. Original works © their respective authors, under the licences listed in the "
                    "bibliography and LICENSES.md.", "Sans", 6.5, FAINT)
    text(c, ML, 55, "Cover pages, dividers and indexes are editorial additions.", "Sans", 6.5, FAINT)
    pg.c.showPage()  # title page: no footer
    pg.phys += 1
    pg.local += 1


def draw_legend(pg: Pager, L: Layout):
    c = pg.c
    pg.anchor("legend")
    y = heading(c, "How to use this anthology")
    label(c, ML, y, "Reading priorities")
    y -= 22
    for pri in ("P0", "P1", "F"):
        pri_tag(c, ML, y, pri, size=7.5)
        text(c, ML + 130, y, PRIORITY[pri][1], "Serif", 10.5)
        y -= 19
    y -= 2
    text(c, ML, y, "Markers are shape-coded for grayscale printing: filled circle (P0), open circle (P1), open diamond (F).",
         "Sans", 7.5, MUTED)
    y -= 12
    text(c, ML, y, "Papers marked optional stay in the anthology but are safe to skip when printing selectively; "
                   "their covers say why.", "Sans", 7.5, MUTED)
    y -= 34
    label(c, ML, y, "Page structure")
    y -= 20
    items = [
        ("Section divider", "Right-hand (odd) page listing the section's papers and print ranges; its back is a ruled notes page."),
        ("Paper cover", "Always a right-hand page: number, priority, why to read it, reading focus, how it connects to the "
                        "other papers, source, and print range."),
        ("Research text", "Starts on the page after the cover and runs to the paper's last page, references and appendices included."),
        ("Notes page", "Added after a paper only when needed to keep the next cover on a right-hand page. Excluded from paper ranges."),
        ("Margin stamp", "Small gray text at the page edge, e.g. ‘MI·17 · p.233’ — paper number and physical page. "
                         "Omitted where it would touch the original content."),
    ]
    for k, v in items:
        text(c, ML, y, k, "Sans-M", 8.5)
        for t in wrap(v, "Serif", 10, MR - ML - 110):
            text(c, ML + 110, y, t, "Serif", 10)
            y -= 13.5
        y -= 6
    y -= 22
    label(c, ML, y, "Printing selected papers")
    y -= 20
    steps = [
        f"Find the paper in the print index (p. {L.front_toc['index']}). It lists the cover page, the research-text pages, "
        "and the complete range.",
        "In the print dialog choose ‘Pages’ and type the range, e.g. 233-262 — or only the research-text range to skip the cover.",
        f"Separate several ranges with commas: 12-27,28-76. Ready-made strings for each section are on p. {L.front_toc['ranges']}, "
        f"and for all P0 papers on p. {L.front_toc.get('p0', L.front_toc['ranges'] + 1)}.",
        "Double-sided: every cover and divider is a right-hand page, so section prints keep each paper on its own sheets.",
        "All pages are US Letter. Use ‘Fit to printable area’ only if your printer clips the margins.",
    ]
    for i, s in enumerate(steps, 1):
        text(c, ML, y, f"{i}", "Sans-M", 8.5, MUTED)
        for t in wrap(s, "Serif", 10, MR - ML - 22):
            text(c, ML + 22, y, t, "Serif", 10)
            y -= 13.5
        y -= 6
    pg.finish_page()


def draw_toc(pg: Pager, L: Layout, front_entries):
    c = pg.c
    pg.anchor("toc")

    def head():
        pg.y = heading(c, "Contents", ["Physical PDF page numbers. Every entry is a link."])

    head()
    for name, page in front_entries:
        text(c, ML, pg.y, name, "Serif", 10.5)
        text(c, MR, pg.y, str(page), "Sans", 9, MUTED, "r")
        pg.link(ML - 4, pg.y - 4, MR + 4, pg.y + 12, page)
        pg.y -= 16
    pg.y -= 10
    title_w = MR - 40 - (ML + 34)
    for sec, name in SECTIONS.items():
        s = L.sections[sec]
        pg.need(60, head)
        pg.y -= 12
        hair(c, ML, pg.y + 18, MR)
        text(c, ML, pg.y, sec, "Disp-SB", 13)
        text(c, ML + 22, pg.y, name, "Disp", 13)
        text(c, MR, pg.y, str(s["divider"]), "Sans-M", 9, INK, "r")
        pg.link(ML - 4, pg.y - 5, MR + 4, pg.y + 14, s["divider"])
        pg.y -= 22
        for p in (q for q in CATALOG if q["sec"] == sec):
            pl = L.papers[p["id"]]
            lines = wrap(p["title"], "Serif", 10.5, title_w)
            pg.need(13.5 * len(lines) + 20, head)
            y0 = pg.y
            text(c, ML, y0, p["id"], "Sans-M", 8, MUTED)
            pri_mark(c, ML + 25, y0 + 3, p["pri"], r=2.6)
            for i, t in enumerate(lines):
                text(c, ML + 34, pg.y, t, "Serif", 10.5)
                if i < len(lines) - 1:
                    pg.y -= 13.5
            pg.y -= 11
            text_pp = "source-access page only" if pl["access"] else f"text {pl['start']}–{pl['end']}"
            opt = "   ·   optional" if p["optional"] else ""
            text(c, ML + 34, pg.y, f"{p['cite']}, {p['year']}   ·   {text_pp}{opt}", "Sans", 7, FAINT)
            text(c, MR, y0, str(pl["cover"]), "Sans", 9, INK, "r")
            pg.link(ML - 4, pg.y - 4, MR + 4, y0 + 12, pl["cover"])
            pg.y -= 17
    pg.finish_page()


INDEX_COLS = [  # (header, x, align)
    ("No.", ML, "l"), ("Paper", ML + 34, "l"), ("Cover", 372, "r"),
    ("Text pages", 440, "r"), ("Print range", 502, "r"), ("Pages", MR, "r"),
]


def draw_print_index(pg: Pager, L: Layout, for_external=False):
    """Exact print index table, grouped by section."""
    c = pg.c
    pg.anchor("index")

    def head():
        where = ("Physical page numbers of Mechanistic_Interpretability_Complete_Anthology.pdf (page 1 = title page)."
                 if for_external else "Physical PDF page numbers (page 1 = title page).")
        y = heading(c, "Print index", [
            where,
            "Print range = cover + original paper; type it into your print dialog. Text pages = original paper only."])
        for name, x, al in INDEX_COLS:
            label(c, x, y, name, align="r" if al == "r" else "l", size=6.5)
        hair(c, ML, y - 6, MR, gray=INK, w=0.6)
        pg.y = y - 22

    head()
    for sec, name in SECTIONS.items():
        s = L.sections[sec]
        pg.need(16 + 13.2, head)
        label(c, ML, pg.y, f"{sec}   {name}", gray=INK, size=7, font="Sans-SB")
        text(c, MR, pg.y, f"section {s['divider']}–{s['end']}", "Sans", 7.5, MUTED, "r")
        if not for_external:
            pg.link(ML - 4, pg.y - 4, MR + 4, pg.y + 10, s["divider"])
        hair(c, ML, pg.y - 5, MR)
        pg.y -= 16
        for p in (q for q in CATALOG if q["sec"] == sec):
            pl = L.papers[p["id"]]
            pg.need(13.2, head)
            text(c, ML, pg.y, p["id"], "Sans-M", 8, MUTED)
            pri_mark(c, ML + 25, pg.y + 3, p["pri"], r=2.6)
            t = p["short"]
            room = 372 - 30 - (ML + 34) - (tw("optional", "Sans", 6.5) + 8 if p["optional"] else 0)
            while tw(t, "Serif", 10) > room:
                t = t[:-2].rstrip() + "…"
            w = text(c, ML + 34, pg.y, t, "Serif", 10)
            if p["optional"]:
                text(c, ML + 34 + w + 6, pg.y + 0.5, "optional", "Sans-I", 6.5, FAINT)
            text(c, 372, pg.y, str(pl["cover"]), "Sans", 8.5, MUTED, "r")
            text(c, 440, pg.y, "access pg." if pl["access"] else rng(pl["start"], pl["end"]), "Sans", 8.5, MUTED, "r")
            text(c, 502, pg.y, rng(pl["cover"], pl["end"]), "Sans-SB", 9.5, INK, "r")
            text(c, MR, pg.y, str(pl["end"] - pl["cover"] + 1), "Sans", 7.5, FAINT, "r")
            if not for_external:
                pg.link(ML - 4, pg.y - 4, MR + 4, pg.y + 10, pl["cover"])
            pg.y -= 13.2
        pg.y -= 4.5
    text(c, ML, max(pg.y - 4, 52), f"Bibliography and sources: pp. {L.bib_start}–{L.total}.  "
                                   "Notes pages between papers are not part of any paper's range.", "Sans", 7, FAINT)
    pg.finish_page()


def draw_ranges_and_checklist(pg: Pager, L: Layout, for_external=False):
    c = pg.c
    pg.anchor("ranges")
    pg.y = heading(c, "Section print ranges", [
        "A section range runs from its divider to the last page of its final paper (covers and notes pages included).",
        "The second string prints only the papers (covers + text), skipping the divider and notes pages."])
    for sec, name in SECTIONS.items():
        s = L.sections[sec]
        papers = [q for q in CATALOG if q["sec"] == sec]
        papers_only = dialog_ranges([(L.papers[q["id"]]["cover"], L.papers[q["id"]]["end"]) for q in papers])
        pg.need(70)
        text(c, ML, pg.y, sec, "Disp-SB", 13)
        text(c, ML + 22, pg.y, name, "Disp", 13)
        text(c, MR, pg.y, f"{s['divider']}–{s['end']}", "Sans-SB", 10.5, INK, "r")
        if not for_external:
            pg.link(ML - 4, pg.y - 4, MR + 4, pg.y + 13, s["divider"])
        pg.y -= 15
        span = (f"Paper {papers[0]['id']}" if len(papers) == 1 else f"Papers {papers[0]['id']}–{papers[-1]['id']}")
        text(c, ML + 22, pg.y, f"{span}  ·  {s['end'] - s['divider'] + 1} pages  ·  "
                               "papers only (no divider or notes pages):",
             "Sans", 7.5, MUTED)
        pg.y -= 14
        for t in wrap(papers_only, "Sans-M", 8, MR - ML - 22):
            text(c, ML + 22, pg.y, t, "Sans-M", 8, INK)
            pg.y -= 11
        pg.y -= 12
        hair(c, ML, pg.y + 8, MR)
        pg.y -= 12
    pg.finish_page()
    # P0 checklist on its own page
    pg.anchor("p0")
    p0 = [q for q in CATALOG if q["pri"] == "P0"]
    p0_str = dialog_ranges([(L.papers[q["id"]]["cover"], L.papers[q["id"]]["end"]) for q in p0])
    tot = sum(L.papers[q["id"]]["end"] - L.papers[q["id"]]["cover"] + 1 for q in p0)
    pg.y = heading(c, "Essential readings (P0)", [f"{readings(len(p0))}  ·  {tot} pages including covers"])
    label(c, ML, pg.y, "Print-dialog string for all P0 papers")
    pg.y -= 15
    for t in wrap(p0_str, "Sans-M", 8.5, MR - ML):
        text(c, ML, pg.y, t, "Sans-M", 8.5)
        pg.y -= 12
    pg.y -= 22
    cols = ((ML, "No.", "l"), (ML + 34, "Paper", "l"), (412, "Print range", "r"), (446, "Printed", "c"),
            (494, "Read", "c"), (MR - 12, "Annotated", "c"))

    def col_heads():
        for x, t, al in cols:
            label(c, x, pg.y, t, size=6.5, align=al)
        hair(c, ML, pg.y - 6, MR, gray=INK, w=0.6)
        pg.y -= 22

    col_heads()
    for q in p0:
        pg.need(21, on_new=col_heads)
        pl = L.papers[q["id"]]
        text(c, ML, pg.y, q["id"], "Sans-M", 8, MUTED)
        text(c, ML + 34, pg.y, q["short"], "Serif", 10)
        text(c, 412, pg.y, rng(pl["cover"], pl["end"]), "Sans-SB", 9.5, INK, "r")
        c.saveState()
        c.setStrokeGray(MUTED)
        c.setLineWidth(0.6)
        for x in (446, 494, MR - 12):
            c.rect(x - 4, pg.y - 1, 8, 8)
        c.restoreState()
        if not for_external:
            pg.link(ML - 4, pg.y - 4, 416, pg.y + 11, pl["cover"])
        hair(c, ML, pg.y - 7, MR, gray=0.88)
        pg.y -= 21
    pg.finish_page()


# --------------------------------------------------------------------------
# Covers, dividers, notes, source-access, bibliography
# --------------------------------------------------------------------------
RANGE_RULE_Y = 150          # hairline above the print-range block at the foot of covers/dividers
COVER_FIT: dict[str, dict] = {}


def range_block(c, title, big, details=(), note=None):
    hair(c, ML, RANGE_RULE_Y, MR, gray=INK, w=0.6)
    label(c, ML, RANGE_RULE_Y - 20, title)
    text(c, ML - 1, RANGE_RULE_Y - 60, big, "Disp", 34)
    y = RANGE_RULE_Y - 22
    for k, v in details:
        text(c, MR - 70, y, k, "Sans", 7.5, MUTED, "r")
        text(c, MR, y, v, "Sans-M", 8.5, INK, "r")
        y -= 13
    if note:
        text(c, ML, RANGE_RULE_Y - 80, note, "Sans", 7, FAINT)


def draw_cover(c, p: dict, rec: dict, pl: dict, total: int, access=False):
    """Draw a cover, shrinking body type (never below 80%) until it clears the print-range block."""
    k = fit_scale(_cover, RANGE_RULE_Y + 14, p, rec, pl, total, access)
    y = _cover(c, p, rec, pl, total, access, k)
    COVER_FIT[p["id"]] = dict(scale=k, lowest_y=round(y, 1), box_top=RANGE_RULE_Y, clear=y >= RANGE_RULE_Y + 8)


def _cover(c, p, rec, pl, total, access, k) -> float:
    sec = p["sec"]
    top = H - 72
    label(c, ML, top, f"Paper {p['id']} / {N}")
    label(c, MR, top, f"Section {sec}  ·  {SECTIONS[sec]}", align="r")
    hair(c, ML, top - 9, MR)
    text(c, ML - 5, top - 112, p["id"], "Disp-L", 92)
    y = top - 142
    w = pri_tag(c, ML, y, p["pri"], size=7.5)
    if p["optional"]:
        label(c, ML + w + 10, y, "·  Optional", size=7.5, gray=MUTED)
        y -= 15
        for t in wrap(p["optional"], "Sans-I", 8, MR - ML):
            text(c, ML, y, t, "Sans-I", 8, MUTED)
            y -= 11
        y += 11
    y -= 34 * k
    ts = 24 * k
    for t in wrap(p["title"], "Disp", ts, MR - ML):
        text(c, ML, y, t, "Disp", ts)
        y -= ts * 1.2
    y -= 2
    text(c, ML, y, f"{p['cite']} — {p['year']}", "Serif-I", 12.5, MUTED)
    y -= 14
    authors = rec.get("authors") or []
    if authors:
        al = wrap(", ".join(authors), "Sans", 7, MR - ML)
        if len(al) > 3:
            al = al[:3]
            al[-1] = al[-1].rstrip(",") + " …"
        for t in al:
            text(c, ML, y, t, "Sans", 7, FAINT)
            y -= 9.5
    y -= 26 * k
    blocks = [("Why read this", p["why"]), ("Reading focus", p["focus"]), ("In context", p["context"])]
    for lab, body in blocks:
        label(c, ML, y, lab)
        y -= 16 * k
        for t in wrap(body, "Serif", 11 * k, MR - ML):
            text(c, ML, y, t, "Serif", 11 * k)
            y -= 15.5 * k
        y -= 14 * k
    # source block
    y -= 4
    hair(c, ML, y + 8, MR)
    y -= 12
    url = p["html"] or rec.get("source_url") or f"https://arxiv.org/abs/{p['arxiv']}"
    qs = 54
    qr(c, url, MR - qs, y - qs + 8, qs)
    qr_bottom = y - qs + 8
    rows = [("Source", f"arXiv:{p['arxiv']} (official PDF)" if p["arxiv"]
             else f"{p['venue']} (publisher PDF)" if p["pdf"]
             else f"{p['venue']} (official HTML, print-rendered)"), ("URL", url)]
    if rec.get("verified_title") and norm(rec["verified_title"]) != norm(p["title"]):
        rows.append(("Published as", rec["verified_title"]))
    if p["note"]:
        rows.append(("Note", p["note"]))
    if access:
        rows.append(("Not included", rec.get("error") or rec.get("identity") or "source could not be acquired"))
    rows.append(("Licence", rec.get("license", "unknown")))
    rows.append(("Retrieved", rec.get("retrieved", TODAY) + (f"  ·  first posted {rec['first_posted']}"
                                                             if rec.get("first_posted") else "")))
    vx = ML + 72
    for lab, val in rows:
        label(c, ML, y, lab, size=6)
        for t in wrap(val, "Sans", 7.5, MR - qs - 16 - vx):
            text(c, vx, y, t, "Sans", 7.5, INK)
            y -= 10.5
        y -= 2.5
    y = min(y + 9, qr_bottom)
    if access:
        range_block(c, "Source-access page", "Full text not included",
                    note=f"This page: PDF page {pl['cover']}. Use the URL or QR code above to read the paper.")
    else:
        n = pl["end"] - pl["start"] + 1
        range_block(c, "Print range", f"{pl['cover']}–{pl['end']}",
                    details=[("Cover", str(pl["cover"])), ("Research text", f"{pl['start']}–{pl['end']}"),
                             ("Pages", str(n + 1))],
                    note="Physical PDF page numbers — type them directly into the print dialog.")
    footer(c, f"Mechanistic Interpretability  ·  Paper {p['id']}", pl["cover"], total)
    return y


def draw_divider(c, sec: str, L: Layout):
    k = fit_scale(_divider, RANGE_RULE_Y + 14, sec, L)
    _divider(c, sec, L, k)


def _divider(c, sec: str, L: Layout, k=1.0) -> float:
    s = L.sections[sec]
    papers = [q for q in CATALOG if q["sec"] == sec]
    # long sections: smaller section letter and one-line short titles, so every row fits above the range block
    compact = len(papers) > 9
    top = H - 72
    label(c, ML, top, f"Section {sec}")
    span = papers[0]["id"] if len(papers) == 1 else f"{papers[0]['id']}–{papers[-1]['id']}"
    label(c, MR, top, f"Paper {span}" if len(papers) == 1 else f"Papers {span}", align="r")
    hair(c, ML, top - 9, MR)
    drop = 112 if compact else 150
    text(c, ML - 8, top - drop, sec, "Disp-L", 100 if compact else 140)
    y = top - drop - 46
    for t in wrap(SECTIONS[sec], "Disp", 26, MR - ML):
        text(c, ML, y, t, "Disp", 26)
        y -= 30
    y -= 2
    count = readings(len(papers))
    text(c, ML, y, f"{count}  ·  pages {s['divider']}–{s['end']}", "Sans", 8, MUTED)
    y -= 40 * k
    for x, t, al in ((ML, "No.", "l"), (ML + 34, "Paper", "l"), (452, "Cover", "r"), (MR, "Print range", "r")):
        label(c, x, y, t, size=6.5, align=al)
    hair(c, ML, y - 6, MR, gray=INK, w=0.6)
    y -= 24 * k
    for p in papers:
        pl = L.papers[p["id"]]
        y0 = y
        text(c, ML, y, p["id"], "Sans-M", 8.5, MUTED)
        pri_mark(c, ML + 25, y + 3.2, p["pri"], r=2.8)
        for t in wrap(p["short"] if compact else p["title"], "Serif", 11 * k, 400 - (ML + 34)):
            text(c, ML + 34, y, t, "Serif", 11 * k)
            y -= (12 if compact else 14) * k
        opt = "  ·  optional" if p["optional"] else ""
        text(c, ML + 34, y + 2, f"{p['cite']}, {p['year']}{opt}", "Sans", 7, FAINT)
        text(c, 452, y0, str(pl["cover"]), "Sans", 9, MUTED, "r")
        text(c, MR, y0, rng(pl["cover"], pl["end"]), "Sans-SB", 9.5, INK, "r")
        y -= (14 if compact else 18) * k
    range_block(c, "Whole-section print range", f"{s['divider']}–{s['end']}",
                details=[("Divider", str(s["divider"])), ("Pages", str(s["end"] - s["divider"] + 1))],
                note="Includes this divider, paper covers and notes pages.")
    footer(c, f"Mechanistic Interpretability  ·  Section {sec}", s["divider"], L.total)
    return y + (14 if compact else 18) * k


def draw_notes(c, lab: str, phys: int, total: int):
    label(c, ML, H - 72, "Notes")
    label(c, MR, H - 72, lab, align="r", gray=FAINT)
    y = H - 104
    while y > 70:
        hair(c, ML, y, MR, gray=0.86, w=0.4)
        y -= 22
    footer(c, "Mechanistic Interpretability  ·  notes", phys, total)


def bib_entries(meta: dict, L: Layout):
    out = []
    for p in CATALOG:
        rec = meta.get(p["id"], {})
        pl = L.papers[p["id"]]
        authors = ", ".join(rec.get("authors") or []) or p["cite"]
        title = rec.get("verified_title") or p["title"]
        if p["arxiv"]:
            where = f"arXiv:{p['arxiv']} (first posted {rec.get('first_posted', '?')}; this version {rec.get('version_date', '?')})."
        else:
            where = f"{p['venue']}, {rec.get('first_posted', p['year'])}."
        lines = [
            f"{authors}. “{title}.” {where}",
            f"Source: {rec.get('source_url', '')}" + (f"  ·  PDF: {rec['pdf_url']}" if rec.get("pdf_url") else ""),
            f"Type: {rec.get('source_type', '—')}  ·  Licence: {rec.get('license', '—')}  ·  Retrieved {rec.get('retrieved', TODAY)}",
            f"Status: {rec.get('status', 'unavailable')}  ·  {rec.get('pages', 0)} source pages → anthology pp. {pl['start']}–{pl['end']} "
            f"(cover p. {pl['cover']})  ·  sources/{p['file']}  ·  {rec.get('size', 0) / 1e6:.2f} MB",
            f"SHA-256: {rec.get('sha256', '—')}",
        ]
        if p["note"]:
            lines.append(f"Note: {p['note']}")
        if rec.get("removed_blank_pages"):
            lines.append(f"Rendering: {rec['removed_blank_pages']} fully blank page(s) produced by the browser print engine were removed.")
        out.append((p["id"], lines))
    return out


def draw_bibliography(pg: Pager, meta: dict, L: Layout):
    c = pg.c
    pg.anchor("bib")

    def head():
        pg.y = heading(c, "Bibliography and sources", [
            "Every paper in reading order with the exact source used, licence, retrieval date and SHA-256 of the file in sources/.",
            "Machine-readable versions: paper_manifest.csv and paper_manifest.json."])

    head()
    ind = 24
    for pid, entry in bib_entries(meta, L):
        wrapped = []
        for i, line in enumerate(entry):
            font, size, lead = ("Serif", 9.5, 12.5) if i == 0 else ("Sans", 6.8, 9.2)
            for t in wrap(line, font, size, MR - ML - ind):
                wrapped.append((font, size, lead, t, i == 0))
        pg.need(sum(w[2] for w in wrapped) + 12, head)
        text(c, ML, pg.y, pid, "Sans-M", 8, MUTED)
        for font, size, lead, t, first in wrapped:
            text(c, ML + ind, pg.y, t, font, size, INK if first else MUTED)
            pg.y -= lead
        pg.y -= 10
    pg.finish_page()


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def is_letter(r: pymupdf.Rect) -> bool:
    return (abs(r.width - W) < 1.5 and abs(r.height - H) < 1.5) or (abs(r.width - H) < 1.5 and abs(r.height - W) < 1.5)


def normalize_to_letter(src: pymupdf.Document) -> tuple[pymupdf.Document, float]:
    """Return a Letter-sized copy (uniform scale, centered, no cropping) plus the minimum scale used."""
    if all(is_letter(pg.rect) for pg in src):
        return src, 1.0
    out = pymupdf.open()
    min_scale = 1.0
    xforms = []
    for pg in src:
        r = pg.rect
        tw, th = (H, W) if r.width > r.height else (W, H)
        if is_letter(r):
            tw, th = r.width, r.height
        s = min(tw / r.width, th / r.height)
        min_scale = min(min_scale, s)
        dx, dy = (tw - r.width * s) / 2, (th - r.height * s) / 2
        npg = out.new_page(width=tw, height=th)
        npg.show_pdf_page(pymupdf.Rect(dx, dy, dx + r.width * s, dy + r.height * s), src, pg.number)
        xforms.append((s, dx, dy))
    for pg in src:  # re-create links with transformed rectangles
        s, dx, dy = xforms[pg.number]
        npg = out[pg.number]
        for lk in pg.get_links():
            fr = lk["from"]
            nr = pymupdf.Rect(dx + fr.x0 * s, dy + fr.y0 * s, dx + fr.x1 * s, dy + fr.y1 * s)
            if lk.get("kind") == pymupdf.LINK_URI and lk.get("uri"):
                npg.insert_link({"kind": pymupdf.LINK_URI, "from": nr, "uri": lk["uri"]})
            elif lk.get("page", -1) >= 0 and lk["page"] < src.page_count:
                ts, tdx, tdy = xforms[lk["page"]]
                to = lk.get("to") or pymupdf.Point(0, 0)
                npg.insert_link({"kind": pymupdf.LINK_GOTO, "from": nr, "page": lk["page"],
                                 "to": pymupdf.Point(tdx + to.x * ts, tdy + to.y * ts)})
    out.set_toc(src.get_toc())
    return out, min_scale


STAMP_W, STAMP_H = 46, 9  # fits "MI·30 · p.1548" at 6.5pt Helvetica


def stamp_page(page: pymupdf.Page, text: str) -> str:
    """Write a small margin stamp where it does not collide with page content. Returns position used."""
    r = page.rect
    cands = {
        "bottom-right": pymupdf.Rect(r.width - 26 - STAMP_W, r.height - 20, r.width - 26, r.height - 20 + STAMP_H),
        "top-right": pymupdf.Rect(r.width - 26 - STAMP_W, 12, r.width - 26, 12 + STAMP_H),
        "bottom-left": pymupdf.Rect(26, r.height - 20, 26 + STAMP_W, r.height - 20 + STAMP_H),
    }
    boxes = [pymupdf.Rect(b[:4]) for b in page.get_text("blocks")]
    boxes += [pymupdf.Rect(im["bbox"]) for im in page.get_image_info()]
    try:
        boxes += [d["rect"] for d in page.get_drawings() if d["rect"].width < r.width * 0.98]
    except Exception:
        pass
    for name, rect in cands.items():
        probe = rect + (-3, -3, 3, 3)
        if not any(probe.intersects(b) for b in boxes):
            tw = pymupdf.get_text_length(text, fontname="helv", fontsize=6.5)
            x = rect.x1 - tw if "right" in name else rect.x0
            page.insert_text((x, rect.y1 - 2), text, fontsize=6.5, fontname="helv", color=(0.45, 0.45, 0.45))
            return name
    return "skipped"


def render_front(L: Layout, stats: dict, meta: dict) -> tuple[bytes, dict, list]:
    buf = io.BytesIO()
    pg = Pager(buf, 1, L.total, "Mechanistic Interpretability Anthology  ·  front matter")
    front_entries = [("Title page", 1), ("How to use & priority legend", L.front_toc.get("legend", 2)),
                     ("Contents", L.front_toc.get("toc", 3)), ("Print index — exact page ranges", L.front_toc.get("index", 4)),
                     ("Section print ranges", L.front_toc.get("ranges", 5)),
                     ("P0 — essential print checklist", L.front_toc.get("p0", 6)),
                     ("Bibliography & source manifest", L.bib_start)]
    draw_title_page(pg, L, stats)
    draw_legend(pg, L)
    draw_toc(pg, L, front_entries)
    draw_print_index(pg, L)
    draw_ranges_and_checklist(pg, L)
    pg.save()
    return buf.getvalue(), pg.anchors, pg.links


def one_page(draw) -> pymupdf.Document:
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=letter)
    draw(c)
    c.showPage()
    c.save()
    return pymupdf.open("pdf", buf.getvalue())


def stage_build():
    register_fonts()
    meta = load_meta()
    srcdocs, n_pages, scales = {}, {}, {}
    for p in CATALOG:
        rec = meta.get(p["id"], {})
        path = SOURCES / p["file"]
        if rec.get("status") == "included" and path.exists():
            d, s = normalize_to_letter(pymupdf.open(path))
            srcdocs[p["id"]], n_pages[p["id"]], scales[p["id"]] = d, d.page_count, s
        else:
            srcdocs[p["id"]], n_pages[p["id"]], scales[p["id"]] = None, 0, None  # source-access page
            rec.setdefault("status", "unavailable")
    stats = dict(papers_included=sum(1 for v in srcdocs.values() if v is not None),
                 access_pages=sum(1 for v in srcdocs.values() if v is None))

    # Iterate until front-matter and bibliography page counts are stable.
    front_n, bib_n = 5, 4
    for it in range(10):
        L = compute_layout(front_n, n_pages, bib_n)
        L.front_toc = dict(legend=2, toc=3, index=0, ranges=0, p0=0)
        # anchors from a dry run
        _, anchors, _ = render_front(L, stats, meta)
        L.front_toc.update({k: anchors[k] for k in ("legend", "toc", "index", "ranges", "p0")})
        front_bytes, anchors2, links = render_front(L, stats, meta)
        assert anchors2 == anchors
        fdoc = pymupdf.open("pdf", front_bytes)
        bbuf = io.BytesIO()
        bpg = Pager(bbuf, L.bib_start, L.total, "Mechanistic Interpretability Anthology  ·  bibliography & manifest")
        draw_bibliography(bpg, meta, L)
        bpg.save()
        bdoc = pymupdf.open("pdf", bbuf.getvalue())
        log(f"layout pass {it + 1}: front={fdoc.page_count} (assumed {front_n}), bib={bdoc.page_count} (assumed {bib_n}), total={L.total}")
        if fdoc.page_count == front_n and bdoc.page_count == bib_n:
            break
        front_n, bib_n = fdoc.page_count, bdoc.page_count
    else:
        raise RuntimeError("page layout did not stabilise")

    out = pymupdf.open()
    out.insert_pdf(fdoc)
    toc = [[1, "Title page", 1], [1, "How to use & priority legend", L.front_toc["legend"]],
           [1, "Contents", L.front_toc["toc"]], [1, "Print index — exact page ranges", L.front_toc["index"]],
           [1, "Section print ranges", L.front_toc["ranges"]], [1, "P0 — essential print checklist", L.front_toc["p0"]]]
    blank_after_front = out.page_count + 1 != L.sections["A"]["divider"]
    if blank_after_front:
        out.insert_pdf(one_page(lambda c: draw_notes(c, "After front matter", out.page_count + 1, L.total)))
    stamps = {}
    for sec, name in SECTIONS.items():
        s = L.sections[sec]
        assert out.page_count + 1 == s["divider"], (sec, out.page_count, s)
        out.insert_pdf(one_page(lambda c: draw_divider(c, sec, L)))
        out.insert_pdf(one_page(lambda c: draw_notes(c, f"Section {sec} — {name}", s["notes"], L.total)))
        toc.append([1, f"Section {sec} — {name}  (pp. {s['divider']}–{s['end']})", s["divider"]])
        for p in (q for q in CATALOG if q["sec"] == sec):
            pl = L.papers[p["id"]]
            rec = meta.get(p["id"], {})
            assert out.page_count + 1 == pl["cover"], (p["id"], out.page_count, pl)
            access = srcdocs[p["id"]] is None
            if access:
                out.insert_pdf(one_page(lambda c: draw_cover(c, p, rec, pl, L.total, access=True)))
            else:
                out.insert_pdf(one_page(lambda c: draw_cover(c, p, rec, pl, L.total)))
                out.insert_pdf(srcdocs[p["id"]])
            label = f"{p['id']} [{p['pri']}] {p['short']}  (pp. {pl['cover']}–{pl['end']})"
            toc.append([2, label, pl["cover"]])
            toc.append([3, "Cover page", pl["cover"]])
            if not access:
                toc.append([3, f"Research text (pp. {pl['start']}–{pl['end']})", pl["start"]])
                prev = 3
                for lvl, title, page in srcdocs[p["id"]].get_toc():
                    if lvl > 2 or page < 1:
                        continue
                    level = min(3 + lvl, prev + 1)
                    toc.append([level, " ".join(title.split())[:120], pl["start"] + page - 1])
                    prev = level
                for i in range(pl["start"], pl["end"] + 1):
                    pos = stamp_page(out[i - 1], f"MI·{p['id']} · p.{i}")
                    stamps[pos] = stamps.get(pos, 0) + 1
            if pl["pad"]:
                out.insert_pdf(one_page(lambda c: draw_notes(c, f"After paper {p['id']}", pl["end"] + 1, L.total)))
    assert out.page_count + 1 == L.bib_start
    out.insert_pdf(bdoc)
    toc.append([1, "Bibliography & source manifest", L.bib_start])
    assert out.page_count == L.total, (out.page_count, L.total)

    for local, rect, target in links:  # clickable TOC / index rows
        out[local].insert_link({"kind": pymupdf.LINK_GOTO, "from": pymupdf.Rect(rect), "page": target - 1,
                                "to": pymupdf.Point(0, 0)})
    out.set_toc(toc)
    out.set_page_labels([{"startpage": 0, "prefix": "", "style": "D", "firstpagenum": 1}])
    out.set_metadata({"title": f"{TITLE}: {SUBTITLE}", "author": "Various (see bibliography); compiled anthology",
                      "subject": f"Mechanistic interpretability curriculum — {N} original works",
                      "keywords": "mechanistic interpretability, circuits, sparse autoencoders, alignment",
                      "creator": "scripts/build_anthology.py", "producer": f"PyMuPDF {pymupdf.VersionBind} + ReportLab"})
    out.rewrite_images(dpi_threshold=IMG_DPI_THRESHOLD, dpi_target=IMG_DPI_TARGET, quality=IMG_QUALITY)
    out.save(MASTER, garbage=3, deflate=True, use_objstms=1)
    log(f"wrote {MASTER.name}: {L.total} pages, {MASTER.stat().st_size / 1e6:.1f} MB; stamps {stamps}")

    # Separate PRINT_INDEX.pdf (same numbers, references the master PDF)
    def index_doc(total):
        buf = io.BytesIO()
        ipg = Pager(buf, 1, total, "PRINT_INDEX.pdf  ·  page numbers refer to Mechanistic_Interpretability_Complete_Anthology.pdf",
                    word="Index sheet")
        draw_print_index(ipg, L, for_external=True)
        draw_ranges_and_checklist(ipg, L, for_external=True)
        ipg.save()
        return pymupdf.open("pdf", buf.getvalue())

    idoc = index_doc(index_doc(0).page_count)
    idoc.set_metadata({"title": "Print Index — Mechanistic Interpretability Anthology", "creator": "scripts/build_anthology.py"})
    idoc.save(PRINT_INDEX, garbage=3, deflate=True)

    write_manifests(meta, L, scales, stamps)
    (BUILD / "layout.json").write_text(json.dumps(dict(
        total=L.total, front_pages=L.front_pages, front_toc=L.front_toc, sections=L.sections,
        papers=L.papers, bib_start=L.bib_start, bib_pages=L.bib_pages, stamps=stamps, scales=scales,
        blank_after_front=blank_after_front, cover_fit=COVER_FIT), indent=2))


def write_manifests(meta: dict, L: Layout, scales: dict, stamps: dict):
    rows = []
    for p in CATALOG:
        rec = meta.get(p["id"], {})
        pl = L.papers[p["id"]]
        s = L.sections[p["sec"]]
        status = rec.get("status", "unavailable")
        restricted = not (rec.get("license") or "").startswith("CC")
        rows.append(dict(
            number=p["id"], section=p["sec"], section_title=SECTIONS[p["sec"]], priority=p["pri"],
            curriculum_citation=f"{p['cite']} ({p['year']})", curriculum_title=p["title"],
            verified_title=rec.get("verified_title", ""), authors="; ".join(rec.get("authors") or []),
            first_posted=rec.get("first_posted", ""), version_date=rec.get("version_date", ""),
            arxiv_id=p["arxiv"] or "", source_type=rec.get("source_type", ""), source_url=rec.get("source_url", ""),
            pdf_url=rec.get("pdf_url", ""), license=rec.get("license", ""), license_url=rec.get("license_url", ""),
            redistribution_note=("Not Creative Commons licensed: copyright remains with the authors; included as published."
                                 if restricted else "Creative Commons licensed: redistribution permitted with attribution."),
            status=status, access_page_reason=rec.get("error", "") if status != "included" else "",
            identity_check=rec.get("identity", ""), local_file=f"sources/{p['file']}" if status == "included" else "",
            file_size_bytes=rec.get("size", 0), sha256=rec.get("sha256", ""), source_pages=rec.get("pages", 0),
            removed_blank_pages=rec.get("removed_blank_pages", 0),
            min_scale_to_letter=round(scales[p["id"]], 4) if scales.get(p["id"]) else "",
            cover_page=pl["cover"], paper_start_page=pl["start"], paper_end_page=pl["end"],
            print_range=f"{pl['cover']}-{pl['end']}", paper_text_range=f"{pl['start']}-{pl['end']}",
            notes_page_after=pl["end"] + 1 if pl["pad"] else "", section_print_range=f"{s['divider']}-{s['end']}",
            retrieved=rec.get("retrieved", ""), notes=p["note"],
            why_read=p["why"], reading_focus=p["focus"],
            in_context=p["context"], optional=p["optional"],
        ))
    with open(ROOT / "paper_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    doc = dict(
        title=f"{TITLE}: {SUBTITLE}", generated=TODAY, master_pdf=MASTER.name, total_pages=L.total,
        page_numbering="physical PDF page index, first page = 1",
        front_matter=dict(title=1, legend=L.front_toc["legend"], toc=L.front_toc["toc"],
                          print_index=L.front_toc["index"], section_ranges=L.front_toc["ranges"], p0_checklist=L.front_toc["p0"],
                          front_pages=L.front_pages),
        sections=[dict(section=k, title=v, divider_page=L.sections[k]["divider"], notes_page=L.sections[k]["notes"],
                       print_range=f"{L.sections[k]['divider']}-{L.sections[k]['end']}",
                       papers=[p["id"] for p in CATALOG if p["sec"] == k]) for k, v in SECTIONS.items()],
        p0_print_string=dialog_ranges([(L.papers[p["id"]]["cover"], L.papers[p["id"]]["end"]) for p in CATALOG if p["pri"] == "P0"]),
        bibliography=dict(start=L.bib_start, end=L.total),
        papers=rows,
    )
    (ROOT / "paper_manifest.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------
# Stage 3: QA
# --------------------------------------------------------------------------

def stage_qa():
    meta = load_meta()
    lay = json.loads((BUILD / "layout.json").read_text())
    man = json.loads((ROOT / "paper_manifest.json").read_text())
    doc = pymupdf.open(MASTER)
    results = []

    def check(name, ok, detail=""):
        results.append(dict(check=name, ok=bool(ok), detail=detail))
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

    check("master PDF opens, not repaired, page count matches layout",
          doc.page_count == lay["total"] and not doc.is_repaired, f"{doc.page_count} pages")
    check("all pages US Letter (portrait or landscape)", all(is_letter(pg.rect) for pg in doc))
    ids = [r["number"] for r in man["papers"]]
    check(f"{N} entries in curriculum order", ids == [f"{i:02d}" for i in range(1, N + 1)])
    # every page of every source renders; collect MuPDF parser warnings for the report
    errs, warns = [], {}
    for p in CATALOG:
        if meta[p["id"]].get("status") != "included":
            continue
        pymupdf.TOOLS.reset_mupdf_warnings()
        try:
            src = pymupdf.open(SOURCES / p["file"])
            for pg_ in src:
                pg_.get_pixmap(dpi=12)
            if src.page_count != meta[p["id"]]["pages"]:
                errs.append(f"{p['id']}: page count changed")
        except Exception as e:
            errs.append(f"{p['id']}: {e!r}")
        w = pymupdf.TOOLS.mupdf_warnings()
        if w:
            warns[p["id"]] = len(w.splitlines())
    pymupdf.TOOLS.reset_mupdf_warnings()
    check("every page of every source PDF renders (no corrupted downloads)", not errs,
          "; ".join(errs) + (f" non-fatal MuPDF parser warnings (lines) in: {warns}" if warns else ""))
    # cover identification & ordering
    prev_end = 0
    bad = []
    for p in CATALOG:
        pl = lay["papers"][p["id"]]
        cover_txt = doc[pl["cover"] - 1].get_text()
        if f"paper{p['id']}{N}" not in norm(cover_txt) or (f"{pl['cover']}–{pl['end']}" not in cover_txt
                                                          and meta[p["id"]].get("status") == "included"):
            bad.append(f"{p['id']}: cover text mismatch")
        if pl["cover"] % 2 != 1:
            bad.append(f"{p['id']}: cover on even page {pl['cover']}")
        if pl["cover"] <= prev_end:
            bad.append(f"{p['id']}: overlaps previous range")
        prev_end = pl["end"]
    check("every cover on its own odd page, labelled, ranges strictly increasing / non-overlapping", not bad, "; ".join(bad))
    # each paper's pages are byte-for-byte the source pages (text comparison)
    mism = []
    for p in CATALOG:
        rec = meta[p["id"]]
        if rec.get("status") != "included":
            continue
        pl = lay["papers"][p["id"]]
        src = pymupdf.open(SOURCES / p["file"])
        n = pl["end"] - pl["start"] + 1
        if n != src.page_count:
            mism.append(f"{p['id']}: {n} != {src.page_count} pages")
            continue
        for k in (0, src.page_count // 2, src.page_count - 1):
            a = norm(src[k].get_text())[:400]
            b = norm(doc[pl["start"] - 1 + k].get_text())
            if a and a[:200] not in b:
                mism.append(f"{p['id']} page {k + 1}")
        if (SOURCES / p["file"]).stat().st_size != rec["size"] or sha256(SOURCES / p["file"]) != rec["sha256"]:
            mism.append(f"{p['id']}: checksum changed")
    check("paper pages match their source PDFs (first/middle/last page text, page counts, checksums)", not mism, "; ".join(mism))
    # completeness heuristics: no abstract-only substitutes; references present
    thin = []
    for p in CATALOG:
        rec = meta[p["id"]]
        if rec.get("status") != "included":
            continue
        src = pymupdf.open(SOURCES / p["file"])
        txt = " ".join(pg.get_text() for pg in src)
        lines = {ln.strip().lower().lstrip("0123456789. ") for pg in src for ln in pg.get_text().splitlines()}
        brief = p["key"] in BRIEF  # short informal notes: no reference list, a few pages long
        if src.page_count < (3 if brief else 8) or len(txt) < (8000 if brief else 25000):
            thin.append(f"{p['id']}: only {src.page_count} pp / {len(txt)} chars")
        if not brief and not lines & {"references", "reference", "bibliography"}:
            thin.append(f"{p['id']}: no References/Bibliography heading line")
    check("no abstract-only substitutes; references present in every paper", not thin, "; ".join(thin))
    # HTML renders: every visible section heading from the web page present in the PDF
    miss = []
    for p in CATALOG:
        rec = meta[p["id"]]
        if p["html"] and rec.get("status") == "included":
            src = norm(" ".join(pg.get_text() for pg in pymupdf.open(SOURCES / p["file"])))
            heads = rec.get("headings", [])
            # section numbers ("§ 2.1") and numbering badges ("05" in "05Beetles") may print apart from the words
            src_words = re.sub(r"\d+", "", src)
            absent = [h for h in heads
                      if norm(re.sub(r"^§\s*[0-9A-Z]+(\.[0-9A-Z]+)*\s*", "", h)) not in src
                      and not (re.sub(r"\d+", "", norm(h)) and re.sub(r"\d+", "", norm(h)) in src_words)]
            if absent:
                miss.append(f"{p['id']}: {len(absent)}/{len(heads)} headings absent, e.g. {absent[:3]}")
            else:
                miss.append(f"{p['id']}: all {len(heads)} headings present")
    check("HTML print renders contain every web section heading (no truncation)",
          all("absent" not in m for m in miss), "; ".join(miss))
    # bookmarks
    toc = doc.get_toc()
    tb = []
    for lvl, title, page in toc:
        if not (1 <= page <= doc.page_count):
            tb.append(f"{title} -> {page}")
    for p in CATALOG:
        pl = lay["papers"][p["id"]]
        if not any(l == 2 and t.startswith(p["id"] + " ") and pg == pl["cover"] for l, t, pg in toc):
            tb.append(f"missing bookmark for {p['id']}")
        txt_pt = [pg for l, t, pg in toc if l == 3 and t.startswith("Research text") and pg == pl["start"]]
        if meta[p["id"]].get("status") == "included" and not txt_pt:
            tb.append(f"missing research-text bookmark for {p['id']}")
    for l, t, pg in toc:  # every paper bookmark must land on a page that carries that paper's cover label
        m = re.match(r"(\d\d) \[", t)
        if l == 2 and m and f"paper{m.group(1)}{N}" not in norm(doc[pg - 1].get_text()):
            tb.append(f"bookmark {t[:20]} lands on wrong page {pg}")
    for k, s in lay["sections"].items():
        if not any(l == 1 and t.startswith(f"Section {k}") and pg == s["divider"] for l, t, pg in toc):
            tb.append(f"missing bookmark for section {k}")
    check("bookmarks: every section/paper present and targets the right page", not tb, f"{len(toc)} bookmarks; " + "; ".join(tb[:5]))
    # TOC / index links
    lb = []
    nlinks = 0
    for i in range(lay["front_pages"]):
        for lk in doc[i].get_links():
            if lk["kind"] == pymupdf.LINK_GOTO:
                nlinks += 1
                if not (0 <= lk["page"] < doc.page_count):
                    lb.append(str(lk))
    covers = {pl["cover"] - 1 for pl in lay["papers"].values()}
    targets = {lk["page"] for i in range(lay["front_pages"]) for lk in doc[i].get_links() if lk["kind"] == pymupdf.LINK_GOTO}
    check("front-matter links valid and reach every cover", not lb and covers <= targets, f"{nlinks} links")
    # printed index numbers match layout
    idx_txt = " ".join(doc[i].get_text() for i in range(lay["front_toc"]["index"] - 1, lay["front_toc"]["ranges"] - 1))
    pidx = " ".join(pg.get_text() for pg in pymupdf.open(PRINT_INDEX))
    ib = []
    for p in CATALOG:
        pl = lay["papers"][p["id"]]
        token = f"{pl['cover']}–{pl['end']}"
        if token not in idx_txt:
            ib.append(f"{p['id']} not in master index")
        if token not in pidx:
            ib.append(f"{p['id']} not in PRINT_INDEX.pdf")
    check("printed index (master + PRINT_INDEX.pdf) matches computed layout", not ib, "; ".join(ib))
    # page labels are physical
    labels_ok = all(doc[i].get_label() in ("", str(i + 1)) for i in range(0, doc.page_count, 37))
    check("page labels equal physical page numbers", labels_ok)
    # margin stamps: every stamped page carries the right paper number and its own physical page number
    found, wrong = 0, []
    for p in CATALOG:
        pl = lay["papers"][p["id"]]
        for i in range(pl["start"], pl["end"] + 1):
            t = doc[i - 1].get_text()
            if f"MI·{p['id']} · p.{i}" in t:
                found += 1
            elif "MI·" in t:
                wrong.append(i)
    expected = sum(v for k, v in lay["stamps"].items() if k != "skipped")
    check("margin stamps present where placed and carry the correct paper/page", found == expected and not wrong,
          f"{found} stamped pages, {lay['stamps'].get('skipped', 0)} skipped to avoid touching content; wrong: {wrong[:5]}")
    # legibility: smallest effective median body font size per paper
    small = []
    sizes = {}
    for p in CATALOG:
        pl = lay["papers"][p["id"]]
        if meta[p["id"]].get("status") != "included":
            continue
        fs = {}  # character-weighted distribution of span sizes over the whole paper
        for i in range(pl["start"] - 1, pl["end"]):
            for b in doc[i].get_text("dict")["blocks"]:
                for line in b.get("lines", []):
                    for sp in line["spans"]:
                        n = len(sp["text"].strip())
                        if n > 3:
                            fs[round(sp["size"], 1)] = fs.get(round(sp["size"], 1), 0) + n
        acc, half, med = 0, sum(fs.values()) / 2, 0
        for size in sorted(fs):
            acc += fs[size]
            if acc >= half:
                med = size
                break
        sizes[p["id"]] = med
        scale = lay["scales"].get(p["id"]) or 1
        if med and med < 7.5:
            small.append(f"{p['id']}: median {med}pt")
        if scale < 0.9:
            small.append(f"{p['id']}: scaled to {scale:.2f}")
    check("legibility: median text ≥ 7.5pt and scale ≥ 0.90 for every paper", not small,
          "; ".join(small) + f" (median sizes: {sizes})")
    # pdf re-open roundtrip with pypdf for reader compatibility
    try:
        import pypdf
        r = pypdf.PdfReader(str(MASTER))
        ok = len(r.pages) == doc.page_count
        check("independent parser (pypdf) reads the master PDF", ok, f"{len(r.pages)} pages")
    except Exception as e:
        check("independent parser (pypdf) reads the master PDF", False, repr(e))
    # poppler (the engine behind many Linux viewers / CUPS) parses both PDFs
    import shutil
    import subprocess
    if shutil.which("pdfinfo"):
        det, counts = [], {}
        for f in (MASTER, PRINT_INDEX):
            out = subprocess.run(["pdfinfo", str(f)], capture_output=True, text=True)
            m = re.search(r"^Pages:\s+(\d+)", out.stdout, re.M)
            counts[f.name] = int(m.group(1)) if m else -1
            det.append(f"{f.name}: {counts[f.name]} pages, exit {out.returncode}")
        check("independent parser (poppler pdfinfo) reads master and PRINT_INDEX",
              counts[MASTER.name] == doc.page_count and counts[PRINT_INDEX.name] > 0, "; ".join(det))
    # pixel identity: first / middle / last page of every paper vs. its source (stamp is the only allowed difference)
    diffs, worst = [], 0.0
    for p in CATALOG:
        if meta[p["id"]].get("status") != "included":
            continue
        pl = lay["papers"][p["id"]]
        ref, _ = normalize_to_letter(pymupdf.open(SOURCES / p["file"]))
        for k in sorted({0, ref.page_count // 2, ref.page_count - 1}):
            a = ref[k].get_pixmap(dpi=30, colorspace=pymupdf.csGRAY)
            b = doc[pl["start"] - 1 + k].get_pixmap(dpi=30, colorspace=pymupdf.csGRAY)
            d = sum(abs(x - y) for x, y in zip(a.samples, b.samples)) / max(1, len(a.samples)) if (a.width, a.height) == (b.width, b.height) else 255
            worst = max(worst, d)
            if d > 3:
                diffs.append(f"{p['id']} p{k + 1}: {d:.1f}")
    check("rendered paper pages are pixel-identical to their sources (mean gray diff ≤ 3/255)", not diffs,
          f"worst {worst:.2f}; " + "; ".join(diffs))
    # covers: body text clears the print-range box
    bad_fit = {k: v for k, v in lay["cover_fit"].items() if not v["clear"]}
    small_fit = {k: v["scale"] for k, v in lay["cover_fit"].items() if v["scale"] < 1}
    check("cover text never overlaps the print-range box", not bad_fit and len(lay["cover_fit"]) == N,
          f"covers set below 100% type: {small_fit or 'none'}")
    # editorial pages (everything that is not original paper text) keep all text inside 0.5in page margins
    paper_pages = {i for pl in lay["papers"].values() for i in range(pl["start"], pl["end"] + 1)}
    bleed = []
    for i in range(1, doc.page_count + 1):
        if i in paper_pages:
            continue
        for x0, y0, x1, y1, t, *_ in doc[i - 1].get_text("words"):
            if x0 < 36 or x1 > W - 36 or y0 < 36 or y1 > H - 30:
                bleed.append(f"p{i}: {t!r} x={x0:.0f}–{x1:.0f}")
    check("editorial pages keep all text inside the page margins", not bleed, "; ".join(bleed[:8]))
    # previews
    qa = BUILD / "qa"
    qa.mkdir(exist_ok=True)
    for old in qa.glob("*.png"):
        old.unlink()
    picks = {"title": 1, "legend": lay["front_toc"]["legend"], "toc": lay["front_toc"]["toc"],
             "index": lay["front_toc"]["index"], "ranges": lay["front_toc"]["ranges"], "p0": lay["front_toc"]["p0"],
             "divider_A": lay["sections"]["A"]["divider"], "notes_A": lay["sections"]["A"]["notes"],
             **{f"cover_{k}": lay["papers"][BY_KEY[k]["id"]]["cover"] for k in ("zoomin", "tracing", "gdmnegative", "clean")},
             "bib": lay["bib_start"]}
    for p in CATALOG:
        picks[f"p{p['id']}_first"] = lay["papers"][p["id"]]["start"]
    for name, page in picks.items():
        doc[page - 1].get_pixmap(dpi=70).save(qa / f"{name}_p{page}.png")
    (BUILD / "qa_report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    n_ok = sum(r["ok"] for r in results)
    log(f"QA: {n_ok}/{len(results)} checks passed")
    return results


# --------------------------------------------------------------------------
# Stage 4: export — Markdown and HTML editions extracted from the source PDFs
# --------------------------------------------------------------------------
MD_DIR = ROOT / "markdown"
HTML_DIR = ROOT / "html"
MD_CACHE = BUILD / "md_cache"
# Extraction settings (pymupdf4llm layout mode). Display equations, figures and tables that cannot be read as
# text are written as images; changing anything here invalidates the cache and re-extracts every paper.
EXPORT_SETTINGS = dict(image_format="jpg", dpi=150, use_ocr=False, header=False, footer=False)
PRI_GLYPH = {"P0": "●", "P1": "○", "F": "◇"}


def _extract_body(job: tuple[str, str, str]) -> tuple[str, str]:
    """Worker: Markdown body of one source PDF, images written under markdown/images/<slug>/."""
    import shutil

    import pymupdf4llm  # imported here: it switches PyMuPDF into layout mode for this process only
    src, slug, img_dir = job
    shutil.rmtree(img_dir, ignore_errors=True)
    Path(img_dir).mkdir(parents=True)
    body = pymupdf4llm.to_markdown(pymupdf.open(src), write_images=True, image_path=img_dir, filename=slug,
                                   show_progress=False, **EXPORT_SETTINGS)
    return slug, body


def relink_images(body: str, slug: str) -> str:
    """pymupdf4llm writes image links relative to the working directory (or absolute); point them at markdown/.
    Extracted maths such as `round[a](x)` reads as Markdown link syntax; escape every `](` that is not a real web
    link or one of the paper's images so it stays literal text."""
    body = re.sub(rf"\]\((?:[^)\s]*/)?images/{re.escape(slug)}/", f"](images/{slug}/", body)
    return re.sub(r"\]\((?!https?://|mailto:|images/)", r"]\\(", body)


def md_name(p: dict) -> str:
    return f"{p['id']}_{p['slug']}.md"


def md_refs(text: str) -> str:
    """Format {key} cross-references as Markdown links to the referenced paper."""
    return text.format(**{k: f"[{v['id']}]({md_name(v)})" for k, v in BY_KEY.items()})


def paper_markdown(p: dict, rec: dict, pl: dict, body: str) -> str:
    i = CATALOG.index(p)
    prev_, next_ = (CATALOG[i - 1] if i else None), (CATALOG[i + 1] if i + 1 < N else None)
    nav = " · ".join(filter(None, [
        f"[← {prev_['id']} {prev_['short']}]({md_name(prev_)})" if prev_ else "",
        "[Contents](README.md)",
        f"[{next_['id']} {next_['short']} →]({md_name(next_)})" if next_ else ""]))
    url = p["html"] or rec.get("source_url", "")
    authors = ", ".join(rec.get("authors") or []) or p["cite"]
    lines = [
        "<!-- Generated by scripts/build_anthology.py export from the source PDF; edit the catalog, not this file. -->",
        f"# {p['id']} · {p['title']}", "",
        f"**{p['cite']} — {p['year']}** · {authors}", "",
        "| | |", "|---|---|",
        f"| Priority | {PRI_GLYPH[p['pri']]} {PRIORITY[p['pri']][0].title()}{' · optional' if p['optional'] else ''} |",
        f"| Section | {p['sec']} — {SECTIONS[p['sec']]} |",
        f"| Source | [{url}]({url}) ({'arXiv:' + p['arxiv'] if p['arxiv'] else p['venue']}) |",
        f"| Licence | {rec.get('license', '—')} |",
        f"| Print anthology | pages {pl['cover']}–{pl['end']} (cover {pl['cover']}, text {pl['start']}–{pl['end']}) |", "",
        f"**Why read this.** {p['why']}", "",
        f"**Reading focus.** {p['focus']}", "",
        f"**In context.** {md_refs(CONTEXT[p['key']])}", "",
    ]
    if p["optional"]:
        lines += [f"**Optional.** {md_refs(p['optional_raw'])}", ""]
    if p["note"]:
        lines += [f"**Note.** {p['note']}", ""]
    lines += [nav, "", "---", "",
              "*Text extracted automatically from the source PDF. Display equations, figures and some tables appear as "
              "images; inline mathematics may be garbled — the PDF is authoritative.*", "",
              body.strip(), "", "---", "", nav, ""]
    return "\n".join(lines)


def contents_markdown(L: Layout, for_single_file=False) -> str:
    link = (lambda p: f"#{md_name(p)[:-3]}") if for_single_file else md_name
    lines = [f"# {TITLE}", "", f"*{SUBTITLE}* — text edition", "",
             f"Text extracted from the source PDFs of the print anthology ([download the PDF]({RELEASE_PDF_URL})). "
             "Each paper opens with the anthology's reading notes; the PDF remains the authoritative version.", "",
             "Priority: ● P0 essential · ○ P1 important · ◇ F frontier.", ""]
    for sec, name in SECTIONS.items():
        s = L.sections[sec]
        lines += [f"## {sec} — {name}", "", f"*Print pages {s['divider']}–{s['end']}*", ""]
        entries = [f"{PRI_GLYPH[p['pri']]} **{p['id']}** [{p['title']}]({link(p)}) — {p['cite']}, {p['year']}"
                   + (" *(optional)*" if p["optional"] else "") for p in CATALOG if p["sec"] == sec]
        lines += ["  \n".join(entries), ""]  # hard line breaks: one paper per line, no list bullets
    return "\n".join(lines)


# HTML edition design: mirrors the print editorial pages — Source Serif 4 Display Light for titles, Inter tracked
# labels, hairline rules, generous whitespace. One shared stylesheet in html/assets/style.css.
HTML_STYLE = """
@font-face{font-family:"Source Serif 4";src:url(fonts/SourceSerif4-Regular.ttf);font-display:swap}
@font-face{font-family:"Source Serif 4";font-style:italic;src:url(fonts/SourceSerif4-It.ttf);font-display:swap}
@font-face{font-family:"Source Serif 4";font-weight:600;src:url(fonts/SourceSerif4-Semibold.ttf);font-display:swap}
@font-face{font-family:"Display";font-weight:300;src:url(fonts/SourceSerif4Display-Light.ttf);font-display:swap}
@font-face{font-family:Inter;src:url(fonts/Inter-Regular.ttf);font-display:swap}
@font-face{font-family:Inter;font-weight:500;src:url(fonts/Inter-Medium.ttf);font-display:swap}
:root{--bg:#fff;--ink:#1b1b1b;--muted:#6b6b6b;--faint:#9a9a9a;--rule:#e2e2e2;--rule-strong:#1b1b1b;--img:none;
  --serif:"Source Serif 4",Georgia,serif;--display:"Display","Source Serif 4",Georgia,serif;--sans:Inter,system-ui,sans-serif;
  color-scheme:light}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;color:var(--ink);background:var(--bg);font:18px/1.72 var(--serif);font-feature-settings:"onum" 0}
a{color:inherit;text-decoration:underline;text-decoration-color:var(--rule);text-decoration-thickness:1px;text-underline-offset:3px}
a:hover{text-decoration-color:var(--ink)}
.theme-toggle{position:fixed;top:1.1rem;right:1.3rem;z-index:10;background:none;border:0;padding:.4rem;cursor:pointer;
  font:500 .62rem/1 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}
.theme-toggle:hover{color:var(--ink)}
article img{filter:var(--img)}
.page{max-width:44rem;margin:0 auto;padding:clamp(3rem,9vw,7.5rem) 1.5rem 8rem}
.label{font:500 .68rem/1.4 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.rule{border:0;border-top:1px solid var(--rule);margin:0}
.rule.strong{border-top-color:var(--rule-strong)}
.sans{font-family:var(--sans)}
/* priority markers: filled circle, open circle, open diamond (as in print) */
.pri{display:inline-block;width:.5rem;height:.5rem;border:1.2px solid var(--ink);border-radius:50%;vertical-align:.08em}
.pri.P0{background:var(--ink)}
.pri.F{border-radius:0;transform:rotate(45deg) scale(.82)}
/* ---------- landing ---------- */
.hero{padding:clamp(2rem,10vh,7rem) 0 4.5rem}
.hero h1{font:300 clamp(3rem,10vw,5.6rem)/.98 var(--display);letter-spacing:-.015em;margin:1.6rem 0 2rem}
.hero .dek{font-style:italic;font-size:1.3rem;line-height:1.5;color:var(--muted);max-width:34rem;margin:0}
.hero .actions{margin-top:2.6rem;display:flex;gap:1.8rem;flex-wrap:wrap}
.hero .actions a{font:500 .74rem/1 var(--sans);letter-spacing:.12em;text-transform:uppercase;text-decoration:none;
  border-bottom:1px solid var(--ink);padding-bottom:.4rem}
.contents{margin:0 0 1.2rem}
.contents .head{display:flex;justify-content:space-between;padding-bottom:.8rem}
.contents a.row{display:grid;grid-template-columns:2rem 1fr auto;align-items:baseline;gap:.5rem;padding:1rem 0;
  border-top:1px solid var(--rule);text-decoration:none}
.contents a.row:hover .name{text-decoration:underline;text-decoration-color:var(--rule)}
.contents .letter{font:600 1.15rem var(--serif)}
.contents .name{font-size:1.15rem}
.contents .count{font:400 .72rem var(--sans);color:var(--faint);margin-left:.6rem}
.contents .pages{font:400 .82rem var(--sans);color:var(--muted);font-variant-numeric:tabular-nums}
.facts{font:400 .8rem/1.9 var(--sans);color:var(--muted);margin:2.2rem 0 0}
.section{margin-top:6.5rem;scroll-margin-top:2rem}
.section .sec-head{display:flex;align-items:flex-end;gap:1.4rem;padding-bottom:1.6rem;border-bottom:1px solid var(--rule-strong)}
.section .sec-letter{font:300 4.6rem/.8 var(--display)}
.section .sec-name{font:300 1.75rem/1.2 var(--display);padding-bottom:.15rem}
.paper{display:grid;grid-template-columns:2.6rem 1rem 1fr;gap:0 .6rem;padding:1.35rem 0;border-bottom:1px solid var(--rule);
  text-decoration:none}
.paper .num{font:500 .78rem/2.1 var(--sans);color:var(--muted);font-variant-numeric:tabular-nums}
.paper .mark{line-height:2.05}
.paper .t{font-size:1.12rem;line-height:1.45}
.paper:hover .t{text-decoration:underline;text-decoration-color:var(--rule)}
.paper .by{display:block;font:400 .76rem/1.6 var(--sans);color:var(--faint);margin-top:.35rem}
.legend{display:flex;gap:1.6rem;flex-wrap:wrap;margin-top:1.6rem}
.legend span{display:inline-flex;align-items:center;gap:.5rem}
footer.site{margin-top:7rem;padding-top:1.4rem;border-top:1px solid var(--rule);font:400 .74rem/1.8 var(--sans);color:var(--faint)}
/* ---------- paper pages ---------- */
.topbar{display:flex;justify-content:space-between;gap:1rem;padding-bottom:.9rem;border-bottom:1px solid var(--rule)}
.topbar a{text-decoration:none}
.cover .numeral{font:300 clamp(5rem,16vw,8.5rem)/.9 var(--display);margin:3.2rem 0 1.4rem;letter-spacing:-.02em}
.cover .tag{display:flex;align-items:center;gap:.6rem}
.cover h1{font:300 clamp(2.1rem,5.5vw,3rem)/1.12 var(--display);letter-spacing:-.008em;margin:1.4rem 0 1.2rem}
.cover .cite{font-style:italic;font-size:1.2rem;color:var(--muted);margin:0}
.cover .authors{font:400 .78rem/1.7 var(--sans);color:var(--faint);margin:.5rem 0 0}
.notes{margin:3.6rem 0 0}
.notes .note{margin:0 0 2.4rem}
.notes .note p{margin:.65rem 0 0;font-size:1.06rem;line-height:1.7}
dl.meta{display:grid;grid-template-columns:8.5rem 1fr;gap:.55rem 1rem;margin:3rem 0 0;padding-top:1.6rem;
  border-top:1px solid var(--rule);font:400 .8rem/1.6 var(--sans)}
dl.meta dt{font:500 .64rem/2.1 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
dl.meta dd{margin:0;overflow-wrap:anywhere}
.pager{display:grid;grid-template-columns:1fr auto 1fr;gap:1rem;margin:4rem 0 0;padding:1.2rem 0;
  border-top:1px solid var(--rule-strong);border-bottom:1px solid var(--rule);font:400 .8rem/1.5 var(--sans)}
.pager a{text-decoration:none}.pager .next{text-align:right}.pager .mid{color:var(--muted)}
.pager small{display:block;font:500 .62rem/1.8 var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--faint)}
.disclaimer{font-style:italic;font-size:.9rem;color:var(--faint);margin:3rem 0 4rem}
/* extracted paper text */
article{font-size:1.02rem}
article p{margin:0 0 1.25em}
article h1,article h2,article h3,article h4,article h5{font-weight:600;line-height:1.3;letter-spacing:-.003em}
article h1{font:300 2.1rem/1.15 var(--display);margin:4.5rem 0 1.6rem}
article h2{font-size:1.45rem;margin:4rem 0 1.1rem}
article h3{font-size:1.18rem;margin:3rem 0 .9rem}
article h4,article h5{font-size:1rem;margin:2.4rem 0 .7rem}
article img{display:block;max-width:100%;height:auto;margin:2.4rem auto}
article ul,article ol{padding-left:1.3rem;margin:0 0 1.4em}article li{margin:.45em 0}
article table{display:block;overflow-x:auto;border-collapse:collapse;margin:2rem 0;font:400 .78rem/1.5 var(--sans)}
article td,article th{border-bottom:1px solid var(--rule);padding:.45rem .75rem;text-align:left;vertical-align:top}
article code,article pre{font-size:.82rem}article pre{overflow-x:auto;padding:1rem 0}
article hr{border:0;border-top:1px solid var(--rule);margin:3rem 0}
@media (max-width:560px){body{font-size:17px}.paper{grid-template-columns:2.2rem .9rem 1fr}dl.meta{grid-template-columns:1fr}
  .pager{grid-template-columns:1fr 1fr}.pager .mid{display:none}.topbar a.label{display:none}.topbar{justify-content:flex-end}
  .hero .actions{gap:1.4rem}}
"""
# Dark palette (matches darren.computer's dark theme). Applies when the system prefers dark and the reader has not
# chosen light, or when the reader picks dark with the toggle; extracted figures are dimmed slightly to cut glare.
_DARK_VARS = ("--bg:#1a1a1a;--ink:#e9e9eb;--muted:#a0a0a0;--faint:#7a7a7a;--rule:#333;--rule-strong:#e9e9eb;"
              "--img:brightness(.86) contrast(1.04);color-scheme:dark")
HTML_STYLE += (f"@media (prefers-color-scheme:dark){{:root:not(.light){{{_DARK_VARS}}}}}\n"
               f":root.dark{{{_DARK_VARS}}}\n")


def _md_to_html(md: str, inline=False) -> str:
    from markdown_it import MarkdownIt
    mdi = MarkdownIt("commonmark", {"html": True}).enable("table")
    html = mdi.renderInline(md) if inline else mdi.render(md)
    html = re.sub(r'href="(\d\d_[a-z0-9_]+)\.md"', r'href="\1.html"', html)
    return html.replace('src="images/', 'src="../markdown/images/')


def html_shell(title: str, inner: str) -> str:
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">\n<title>{title}</title>\n'
            f'<link rel="stylesheet" href="assets/style.css">\n'
            # apply a saved theme before first paint (no flash); otherwise the system setting decides
            "<script>try{var t=localStorage.getItem('mi-theme');if(t)document.documentElement.classList.add(t)}catch(e){}</script>\n"
            f'</head><body>\n<button class="theme-toggle" type="button" data-theme-toggle aria-label="Toggle dark mode">Dark</button>\n'
            f'<main class="page">\n{inner}\n</main>\n<script>{THEME_JS}</script>\n'
            "</body></html>\n")


THEME_JS = (
    "(function(){var r=document.documentElement,b=document.querySelector('[data-theme-toggle]'),"
    "m=window.matchMedia('(prefers-color-scheme: dark)');"
    "function dark(){return r.classList.contains('dark')||(!r.classList.contains('light')&&m.matches)}"
    "function label(){b.textContent=dark()?'Light':'Dark'}label();m.addEventListener('change',label);"
    "b.addEventListener('click',function(){var n=dark()?'light':'dark';r.classList.remove('light','dark');r.classList.add(n);"
    "try{localStorage.setItem('mi-theme',n)}catch(e){}label()})})();")


def pri_html(pri: str) -> str:
    return f'<span class="pri {pri}" aria-hidden="true"></span>'


def html_paper(p: dict, rec: dict, pl: dict, body_md: str) -> str:
    i = CATALOG.index(p)
    prev_, next_ = (CATALOG[i - 1] if i else None), (CATALOG[i + 1] if i + 1 < N else None)
    link = lambda q: f"{md_name(q)[:-3]}.html"
    url = p["html"] or rec.get("source_url", "")
    authors = ", ".join(rec.get("authors") or [])
    notes = [("Why read this", p["why"]), ("Reading focus", p["focus"]),
             ("In context", _md_to_html(md_refs(CONTEXT[p["key"]]), inline=True))]
    if p["optional"]:
        notes.append(("Optional", _md_to_html(md_refs(p["optional_raw"]), inline=True)))
    if p["note"]:
        notes.append(("Note", p["note"]))
    src = f'<a href="{url}">{url}</a>' + (f' &nbsp;·&nbsp; arXiv:{p["arxiv"]}' if p["arxiv"] else f' &nbsp;·&nbsp; {p["venue"]}')
    pager = (
        '<nav class="pager">'
        + (f'<a class="prev" href="{link(prev_)}"><small>Previous · {prev_["id"]}</small>{prev_["short"]}</a>' if prev_ else "<span></span>")
        + '<a class="mid" href="index.html#section-' + p["sec"] + '">Contents</a>'
        + (f'<a class="next" href="{link(next_)}"><small>Next · {next_["id"]}</small>{next_["short"]}</a>' if next_ else "<span></span>")
        + "</nav>")
    inner = f"""
<div class="topbar"><a class="label" href="index.html">{TITLE}</a>
<span class="label">Paper {p['id']} / {N} &nbsp;·&nbsp; Section {p['sec']}</span></div>
<header class="cover">
<div class="numeral">{p['id']}</div>
<div class="tag">{pri_html(p['pri'])}<span class="label" style="color:var(--ink)">{PRIORITY[p['pri']][0]}</span>
{'<span class="label">&nbsp;·&nbsp; Optional</span>' if p['optional'] else ''}</div>
<h1>{p['title']}</h1>
<p class="cite">{p['cite']} — {p['year']}</p>
{f'<p class="authors">{authors}</p>' if authors else ''}
</header>
<section class="notes">
{''.join(f'<div class="note"><div class="label">{k}</div><p>{v}</p></div>' for k, v in notes)}
</section>
<dl class="meta">
<dt>Section</dt><dd>{p['sec']} — {SECTIONS[p['sec']]}</dd>
<dt>Source</dt><dd>{src}</dd>
<dt>Licence</dt><dd>{rec.get('license', '—')}</dd>
<dt>In print</dt><dd>PDF pages {pl['cover']}–{pl['end']} &nbsp;·&nbsp; <a href="{RELEASE_PDF_URL}">download the anthology</a></dd>
</dl>
{pager}
<p class="disclaimer">Text extracted automatically from the source PDF. Display equations, figures and some tables appear as
images; inline mathematics may be garbled — the PDF is authoritative.</p>
<article>
{_md_to_html(body_md)}
</article>
{pager}
"""
    return html_shell(f"{p['id']} · {p['title']} — {TITLE}", inner)


def html_index(L: Layout) -> str:
    rows = []
    for sec, name in SECTIONS.items():
        s = L.sections[sec]
        n = sum(1 for p in CATALOG if p["sec"] == sec)
        rows.append(f'<a class="row" href="#section-{sec}"><span class="letter">{sec}</span>'
                    f'<span><span class="name">{name}</span><span class="count">{readings(n)}</span></span>'
                    f'<span class="pages">pp. {s["divider"]}–{s["end"]}</span></a>')
    sections = []
    for sec, name in SECTIONS.items():
        items = []
        for p in (q for q in CATALOG if q["sec"] == sec):
            opt = " &nbsp;·&nbsp; optional" if p["optional"] else ""
            items.append(f'<a class="paper" href="{md_name(p)[:-3]}.html"><span class="num">{p["id"]}</span>'
                         f'<span class="mark">{pri_html(p["pri"])}</span>'
                         f'<span><span class="t">{p["title"]}</span><span class="by">{p["cite"]}, {p["year"]}{opt}</span></span></a>')
        sections.append(f'<section class="section" id="section-{sec}"><div class="sec-head"><span class="sec-letter">{sec}</span>'
                        f'<span class="sec-name">{name}</span></div>{"".join(items)}</section>')
    legend = "".join(f'<span class="label">{pri_html(k)} {v[0]}</span>' for k, v in PRIORITY.items())
    inner = f"""
<header class="hero">
<div class="label">A research anthology &nbsp;·&nbsp; online edition</div>
<h1>Mechanistic<br>Interpretability</h1>
<p class="dek">{number_words(N).capitalize()} papers, essays and forum posts on circuits, features, dictionary
learning and alignment — and the arguments over them.</p>
<div class="actions"><a href="{RELEASE_PDF_URL}">Download the print PDF</a><a href="{REPO_URL}">Source &amp; Markdown</a></div>
</header>
<section class="contents">
<div class="head"><span class="label">Contents</span><span class="label">Print pages</span></div>
{''.join(rows)}
<hr class="rule">
<p class="facts">{readings(N)} &nbsp;·&nbsp; {L.total} pages in print &nbsp;·&nbsp; each opens with reading notes and links
to the work it speaks to. Text here is extracted from the PDFs; the print edition is authoritative.</p>
<div class="legend">{legend}</div>
</section>
{''.join(sections)}
<footer class="site">Original works © their respective authors; licences in
<a href="{REPO_URL}/blob/main/LICENSES.md">LICENSES.md</a>. Reading notes and design are editorial additions.</footer>
"""
    return html_shell(TITLE, inner)


def stage_export():
    import shutil
    from concurrent.futures import ProcessPoolExecutor

    meta = load_meta()
    lay = json.loads((BUILD / "layout.json").read_text())
    L = Layout(lay["front_pages"], lay["total"], lay["papers"], lay["sections"], lay["bib_start"], lay["bib_pages"],
               lay["front_toc"])
    for d in (MD_DIR / "images", HTML_DIR / "assets" / "fonts", MD_CACHE):
        d.mkdir(parents=True, exist_ok=True)
    import pymupdf4llm
    key = json.dumps(dict(EXPORT_SETTINGS, version=pymupdf4llm.VERSION), sort_keys=True)
    cached, jobs = {}, []
    for p in CATALOG:
        cache = MD_CACHE / f"{p['slug']}.json"
        c = json.loads(cache.read_text()) if cache.exists() else {}
        if c.get("sha256") == meta[p["id"]]["sha256"] and c.get("settings") == key \
                and (MD_DIR / "images" / p["slug"]).is_dir():
            cached[p["slug"]] = c["body"]
        else:
            jobs.append((str(SOURCES / p["file"]), p["slug"], str(MD_DIR / "images" / p["slug"])))
    log(f"export: {len(cached)} papers cached, extracting {len(jobs)}")
    if jobs:
        with ProcessPoolExecutor(max_workers=min(8, len(jobs))) as ex:
            for slug, body in ex.map(_extract_body, jobs):
                p = next(q for q in CATALOG if q["slug"] == slug)
                (MD_CACHE / f"{slug}.json").write_text(json.dumps(dict(sha256=meta[p["id"]]["sha256"], settings=key,
                                                                       body=body)))
                cached[slug] = body
                log(f"  [{p['id']}] extracted {len(body):,} chars")
    # write Markdown + HTML; remove files left over from earlier numbering
    keep = {md_name(p) for p in CATALOG}
    for f in MD_DIR.glob("[0-9][0-9]_*.md"):
        if f.name not in keep:
            f.unlink()
    for f in HTML_DIR.glob("[0-9][0-9]_*.html"):
        if f.name[:-5] + ".md" not in keep:
            f.unlink()
    for d in (MD_DIR / "images").iterdir():
        if d.name not in {p["slug"] for p in CATALOG}:
            shutil.rmtree(d)
    for f in ("SourceSerif4-Regular.ttf", "SourceSerif4-It.ttf", "SourceSerif4-Semibold.ttf",
              "SourceSerif4Display-Light.ttf", "Inter-Regular.ttf", "Inter-Medium.ttf"):
        shutil.copyfile(FONT_DIR / f, HTML_DIR / "assets" / "fonts" / f)
    single = [contents_markdown(L, for_single_file=True)]
    for p in CATALOG:
        md = paper_markdown(p, meta[p["id"]], L.papers[p["id"]], relink_images(cached[p["slug"]], p["slug"]))
        (MD_DIR / md_name(p)).write_text(md)
        body = relink_images(cached[p["slug"]], p["slug"])
        (HTML_DIR / f"{md_name(p)[:-3]}.html").write_text(html_paper(p, meta[p["id"]], L.papers[p["id"]], body))
        anchor = md_name(p)[:-3]
        single.append(f'<a id="{anchor}"></a>\n\n' + re.sub(r"\]\((\d\d_[a-z0-9_]+)\.md\)", r"](#\1)", md)
                      .replace("(README.md)", "(#top)"))
    contents = contents_markdown(L)
    (MD_DIR / "README.md").write_text(contents)
    (MD_DIR / "anthology.md").write_text('<a id="top"></a>\n\n' + "\n\n".join(single))
    (HTML_DIR / "index.html").write_text(html_index(L))
    (HTML_DIR / "assets" / "style.css").write_text(HTML_STYLE)
    check_export(meta)


def check_export(meta: dict):
    results = []

    def check(name, ok, detail=""):
        results.append(dict(check=f"export: {name}", ok=bool(ok), detail=detail))
        log(f"  [{'PASS' if ok else 'FAIL'}] export: {name}" + (f" — {detail}" if detail else ""))

    missing = [p["id"] for p in CATALOG
               if not (MD_DIR / md_name(p)).exists() or not (HTML_DIR / f"{md_name(p)[:-3]}.html").exists()]
    check(f"every paper has a Markdown and an HTML page ({N} each)", not missing, ", ".join(missing))
    ratios = {}
    for p in CATALOG:
        body = json.loads((MD_CACHE / f"{p['slug']}.json").read_text())["body"]
        pdf_chars = sum(len(re.sub(r"\s", "", pg.get_text())) for pg in pymupdf.open(SOURCES / p["file"]))
        ratios[p["id"]] = round(len(re.sub(r"\s|!\[[^\]]*\]\([^)]*\)", "", body)) / max(1, pdf_chars), 2)
    low = {k: v for k, v in ratios.items() if v < 0.6}
    check("extracted text covers ≥60% of each source PDF's characters (rest: equations/figures kept as images)",
          not low, f"lowest {min(ratios.items(), key=lambda kv: kv[1])}; below threshold: {low or 'none'}")
    broken = []
    for f in MD_DIR.glob("*.md"):
        for target in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", f.read_text()):
            if not (MD_DIR / target).exists():
                broken.append(f"{f.name}: {target}")
    check("every image referenced in the Markdown exists", not broken, "; ".join(broken[:5]))
    dead = []
    for f in HTML_DIR.glob("*.html"):
        for target in re.findall(r'(?:href|src)="([^"#:]+)"', f.read_text()):
            if not (HTML_DIR / target).resolve().exists():
                dead.append(f"{f.name}: {target}")
    check("every local link and image in the HTML resolves", not dead, "; ".join(dead[:5]))
    report = BUILD / "qa_report.json"
    prior = [r for r in (json.loads(report.read_text()) if report.exists() else []) if not r["check"].startswith("export:")]
    report.write_text(json.dumps(prior + results, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------
# README + checksums
# --------------------------------------------------------------------------

def write_readme_and_checksums():
    man = json.loads((ROOT / "paper_manifest.json").read_text())
    qa = json.loads((BUILD / "qa_report.json").read_text()) if (BUILD / "qa_report.json").exists() else []
    size = MASTER.stat().st_size
    included = [r for r in man["papers"] if r["status"] == "included"]
    lines = [
        "# Mechanistic Interpretability — Research Anthology (print edition)",
        "",
        f"Generated {man['generated']} by `scripts/build_anthology.py`.",
        "",
        f"- **Master PDF:** `{man['master_pdf']}` — {man['total_pages']} pages, {size / 1e6:.1f} MB, US Letter",
        f"- **Complete original works:** {len(included)} / {N}; source-access pages: {N - len(included)}",
        "- **Page numbers:** physical PDF page index (page 1 = title page). Page labels are set so viewers show the same numbers;",
        "  type ranges straight into the print dialog.",
        "- **Separate index:** `PRINT_INDEX.pdf` (same numbers as the master index)",
        "",
        "## Section print ranges",
        "",
        "| Section | Title | Papers | Print range | Divider |",
        "|---|---|---|---|---|",
    ]
    for s in man["sections"]:
        lines.append(f"| {s['section']} | {s['title']} | {s['papers'][0]}–{s['papers'][-1]} | {s['print_range']} | {s['divider_page']} |")
    lines += ["", f"**All P0 papers (paste into the print dialog):** `{man['p0_print_string']}`", "",
              "## Print index", "", "| # | Pri | Paper | Cover | Paper pages | Print range | Source |", "|---|---|---|---|---|---|---|"]
    for r in man["papers"]:
        src = (f"arXiv:{r['arxiv_id']}" if r["arxiv_id"] else f"{urlparse(r['source_url']).netloc} (publisher PDF)"
               if r["source_type"].startswith("Publisher") else f"{urlparse(r['source_url']).netloc} (print render)")
        lines.append(f"| {r['number']} | {r['priority']} | {r['curriculum_title']} | {r['cover_page']} | "
                     f"{r['paper_text_range'].replace('-', '–')} | **{r['print_range']}** | {src} |")
    lines += [
        "", "## Layout", "",
        "Title page → how-to-use & legend → table of contents →",
        "print index → section ranges & P0 checklist →",
        "for each section: divider (odd page) + ruled notes verso → for each paper: cover (always an odd page) → complete",
        "original paper → a ruled notes page only when needed to keep the next cover on an odd page → bibliography & manifest.",
        "Every original paper page carries a small gray margin stamp `MI·NN · p.X` (paper number, physical page) placed only",
        "where it does not overlap content.",
        "",
        "## Sources and licensing", "",
        f"- {sum(1 for p in CATALOG if p['arxiv'])} papers: the latest official arXiv PDF version (pinned version IDs in the "
        "manifest). arXiv was preferred over",
        "  conference-proceedings PDFs because proceedings PDFs often move appendices into separate supplementary files,",
        "  and the arXiv versions are the authors' complete camera-ready texts with appendices.",
        f"- {len(ids_where(site_render))} works "
        f"({', '.join(ids_where(site_render))}) exist only as HTML on transformer-circuits.pub,",
        "  distill.pub or openaipublic.blob.core.windows.net, with no official PDF. They were",
        "  print-rendered with headless Chrome using the site's print stylesheet on US Letter (0.5 in margins, 90% scale,",
        "  body text ≈10.8 pt). Fixed-width article grids (Distill template) were collapsed to the page width so text is not",
        "  auto-shrunk and wide figures are not clipped; only figures/widgets still wider than the page were zoomed to fit",
        "  (per-paper zoom log in build/metadata.json), and the gutter section numbers (§ 2.1) are printed inline.",
        "  Interactive widgets appear in their default static state; use the URL/QR on the cover for the interactive",
        "  versions. Fully blank pages emitted by Chrome at forced section breaks were removed (count in the manifest).",
        "  Distill articles are CC BY 4.0; the other sites publish no explicit licence, and copyright remains with the",
        "  authors.",
        f"- {len(ids_where(lambda p: p['render'].get('keep')))} blog posts and essays "
        f"({', '.join(ids_where(lambda p: p['render'].get('keep')))}) — LessWrong / AI Alignment Forum posts and",
        "  articles on goodfire.com, ai-frontiers.org and darioamodei.com — were rendered the same way, keeping only the",
        "  article header and body (site navigation, comments and promotional blocks omitted). Forum posts carry no explicit",
        "  licence; the forum terms leave copyright with the author. The VPD post's opening interactive overview is omitted:",
        "  its off-screen carousel track made Chrome shrink the whole post to ~6 pt.",
        f"- {len(ids_where(lambda p: p['pdf']))} work(s) ({', '.join(ids_where(lambda p: p['pdf']))}) use the publisher's "
        "PDF (PLOS Computational Biology, CC BY 4.0).",
        f"- {BY_KEY['clean']['id']} (Cammarata 2026) is an HTML essay on already-clean.goodfire.com, rendered with the page",
        "  clamped to the printable width, reduced motion (animated figures print their final frame)",
        "  and no lazy-load scrolling: scroll-mounted widgets wider than the page otherwise made Chrome shrink the whole",
        "  essay to ~6.6 pt. Body text prints at ≈9.9 pt. No explicit licence is published.",
        f"- Items {', '.join(ids_where(lambda p: p['optional']))} are marked *optional* on their covers and in the indexes:",
        "  kept in the anthology, but safe to skip when printing selectively.",
        "- Every cover carries an *In context* note situating the item among the others — its lineage and, where the",
        "  collection prints one, the rebuttal it answers or provokes — with cross-references by number. These notes are",
        "  the compiler's commentary, not the authors'.",
        f"- `{BY_KEY['repe']['file']}` (and, marginally, {BY_KEY['mib']['id']} and {BY_KEY['weightsparse']['id']}) "
        "trigger non-fatal MuPDF warnings about",
        "  dangling object references inside the original arXiv files; every page still renders without error.",
        f"- To keep the master PDF downloadable, its raster images are re-encoded as JPEG (quality {IMG_QUALITY}) and",
        f"  downsampled only where they exceed {IMG_DPI_THRESHOLD} dpi (to {IMG_DPI_TARGET} dpi). Text, fonts and vector",
        "  graphics are untouched, so everything remains selectable and searchable; `sources/` keeps the original files.",
        "- Licences differ by paper: see `LICENSES.md` and the `license` / `redistribution_note` columns of",
        "  `paper_manifest.csv`. Copyright in every paper remains with its authors.",
        "",
        "## Title / date discrepancies against the curriculum", "",
    ]
    for r in man["papers"]:
        if r["notes"] or (r["verified_title"] and norm(r["verified_title"]) != norm(r["curriculum_title"])):
            lines.append(f"- **{r['number']}** curriculum: *{r['curriculum_title']}* — source: *{r['verified_title']}*. {r['notes']}")
    lines += ["", "## Automated QA", ""]
    for q in qa:
        lines.append(f"- {'PASS' if q['ok'] else 'FAIL'} — {q['check']}")
    lines += ["", "## Rebuild", "", "```bash", "python3 -m venv .venv && .venv/bin/pip install -r requirements.txt",
              ".venv/bin/python scripts/build_anthology.py          # fetch + build + qa + export",
              ".venv/bin/python scripts/build_anthology.py build qa # re-assemble the PDF from cached sources",
              ".venv/bin/python scripts/build_anthology.py export   # refresh the Markdown/HTML editions", "```", "",
              "To add or update a paper, edit `CATALOG` (and its `CONTEXT` note) in the script and run it again: changed",
              "sources are downloaded or re-rendered, the PDF is re-laid out with exact page numbers, and only papers whose",
              "source PDF changed are re-extracted for the Markdown/HTML editions (cache in `build/md_cache/`).",
              "Rendering uses the system Google Chrome/Chromium if present (otherwise `.venv/bin/playwright install chromium`).",
              "A full build downloads or renders every source and takes several minutes.",
              "", "## Where things are", "",
              f"- The compiled anthology is attached to the GitHub release: {RELEASE_PDF_URL}",
              "  (it is larger than GitHub's 100 MB per-file limit for git, so it cannot live in the repository itself).",
              "- The original papers, exactly as included, are in `sources/`.",
              "- Text editions: `markdown/` (one file per paper, `README.md` contents, `anthology.md` all-in-one) and",
              f"  `html/`, published at {SITE_URL} (served from {PAGES_URL}, which `.github/workflows/pages.yml`",
              "  redeploys on every push that changes it).",
              "  Text is extracted from the source PDFs with pymupdf4llm; display",
              "  equations, figures and some tables appear as images, inline maths can be garbled, and the PDF stays",
              "  authoritative.",
              "- `sha256sum -c checksums.sha256` verifies a download or a local rebuild against this README.",
              "", "## Files", "",
              "- `Mechanistic_Interpretability_Complete_Anthology.pdf` — master PDF (GitHub release asset)",
              "- `PRINT_INDEX.pdf` — standalone print index",
              "- `paper_manifest.csv`, `paper_manifest.json` — per-paper sources, licences, checksums, page ranges",
              "- `sources/` — the original PDFs exactly as included",
              "- `markdown/`, `html/` — text-extracted editions of every paper with the anthology's reading notes",
              "- `LICENSES.md` — licence and copyright of every included work, fonts and editorial material",
              "- `scripts/build_anthology.py` — reproducible build; `requirements.txt` — Python dependencies",
              "- `scripts/fonts/` — Source Serif 4 and Inter (SIL Open Font License) used for covers, dividers and indexes",
              "- `checksums.sha256` — SHA-256 of every deliverable", ""]
    (ROOT / "README.md").write_text("\n".join(lines))
    write_licenses(man)
    files = [MASTER, PRINT_INDEX, ROOT / "paper_manifest.csv", ROOT / "paper_manifest.json", ROOT / "README.md",
             ROOT / "LICENSES.md", ROOT / "scripts" / "build_anthology.py"] + sorted(FONT_DIR.glob("*.ttf")) \
        + sorted(SOURCES.glob("*.pdf"))
    (ROOT / "checksums.sha256").write_text("".join(f"{sha256(f)}  {f.relative_to(ROOT)}\n" for f in files))


def write_licenses(man: dict):
    """LICENSES.md: the licence and copyright position of every work redistributed in this repository."""
    rows = man["papers"]
    cc = [r for r in rows if r["license"].startswith("CC")]
    lines = [
        "# Licences", "",
        "This repository redistributes the original works in `sources/` and the compiled anthology (GitHub release).",
        "Copyright in every work remains with its authors; each is included as published, under the terms below.",
        f"{len(cc)} of {len(rows)} works carry a Creative Commons licence. The others are distributed by arXiv under its",
        "non-exclusive distribution licence, or published on their authors' websites or on LessWrong / the AI Alignment",
        "Forum without an explicit licence.",
        "", "| # | Work | Licence | Source |", "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['number']} | {r['curriculum_citation']}, *{r['curriculum_title']}* | {r['license']} | {r['source_url']} |")
    lines += [
        "", "## Fonts", "",
        "`scripts/fonts/` contains Inter and Source Serif 4, both under the SIL Open Font License 1.1 "
        "(`scripts/fonts/Inter-OFL.txt`, `scripts/fonts/SourceSerif4-OFL.md`).",
        "", "## Editorial material", "",
        "Cover pages, section dividers, indexes, the bibliography layout and the *In context* notes were written for this",
        "anthology; they are not part of the original works.", "",
    ]
    (ROOT / "LICENSES.md").write_text("\n".join(lines))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stages = args or ["fetch", "build", "qa", "export"]
    if "fetch" in stages:
        stage_fetch(force="--force" in sys.argv)
    if "build" in stages:
        stage_build()
    if "qa" in stages:
        register_fonts()
        stage_qa()
    if "export" in stages:
        stage_export()
    if {"build", "qa", "export"} & set(stages):
        write_readme_and_checksums()
