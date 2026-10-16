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
    "A": "Transformer foundations and circuits",
    "B": "Causal methods and circuit discovery",
    "C": "Superposition and sparse autoencoders",
    "D": "Feature circuits and large-scale interpretability",
    "E": "The 2026 frontier",
    "F": "Alignment, steering, and reasoning",
    "G": "Conclusion",
}
PRIORITY = {
    "P0": ("P0 — ESSENTIAL", "Read deeply, annotate, and potentially reproduce."),
    "P1": ("P1 — IMPORTANT", "Read carefully."),
    "F": ("F — FRONTIER", "Contemporary work requiring critical evaluation."),
}
TC = "https://transformer-circuits.pub"


def P(key, sec, pri, cite, year, title, short, why, focus, *, arxiv=None, html=None, slug, note="", venue=None,
      render=None, license=None, optional="", authors=None):
    """One catalog entry. `key` is the short name used for cross-references in CONTEXT/optional notes;
    the paper number comes from its position in CATALOG."""
    return dict(key=key, sec=sec, pri=pri, cite=cite, year=year, title=title, short=short,
                why=why, focus=focus, arxiv=arxiv, html=html, slug=slug, note=note, license=license, authors=authors,
                venue=venue or ("Transformer Circuits Thread" if html else "arXiv"), render=render or {},
                optional=optional)


CATALOG = [
    # ---- A
    P("vaswani", "A", "P0", "Vaswani et al.", "2017", "Attention Is All You Need", "Attention Is All You Need",
      "Defines the Transformer — multi-head attention, residual blocks, positional encodings — the architecture every later paper reverse-engineers.",
      "Scaled dot-product and multi-head attention, Q/K/V projections, residual + LayerNorm structure, positional encodings, and the attention visualizations in the appendix.",
      arxiv="1706.03762v7", slug="attention_is_all_you_need"),
    P("zoomin", "A", "P0", "Olah et al.", "2020", "Zoom In: An Introduction to Circuits", "Zoom In: An Introduction to Circuits",
      "Sets out the circuits programme — features as units, circuits as their connections, universality — and introduces polysemantic neurons and superposition through vision-model examples.",
      "The three speculative claims, curve detectors and other worked circuits, the polysemantic cat-and-car neuron, the car-to-dog circuit offered as superposition, and the case for interpretability as a natural science.",
      html="https://distill.pub/2020/circuits/zoom-in/", slug="zoom_in", venue="Distill", license=DISTILL_LICENSE),
    P("multimodal", "A", "P1", "Goh et al.", "2021", "Multimodal Neurons in Artificial Neural Networks", "Multimodal Neurons",
      "Finds neurons in CLIP that respond to the same concept across photographs, drawings and written text, and traces the typographic attacks and biases that follow from them.",
      "Person, region and emotion neurons, the feature-properties discussion including conjoined features, typographic attacks, and the section on bias.",
      html="https://distill.pub/2021/multimodal-neurons/", slug="multimodal_neurons", venue="Distill", license=DISTILL_LICENSE),
    P("framework", "A", "P0", "Elhage et al.", "2021", "A Mathematical Framework for Transformer Circuits", "Mathematical Framework for Transformer Circuits",
      "Recasts attention-only transformers as sums of paths, introducing the residual stream view, QK/OV circuits, and head composition — the core vocabulary of the field.",
      "Residual stream as communication channel, QK vs. OV circuits, path expansion of 1- and 2-layer models, K-composition and induction heads, and the notation appendix.",
      html=f"{TC}/2021/framework/index.html", slug="mathematical_framework"),
    P("induction", "A", "P0", "Olsson et al.", "2022", "In-context Learning and Induction Heads", "In-context Learning and Induction Heads",
      "Argues that induction heads may constitute the mechanism for the majority of in-context learning, using six complementary lines of evidence across model scales.",
      "The training-loss phase change, each of the six arguments and how strong it is per model class, ablation evidence, and the authors' stated caveats for large models.",
      arxiv="2209.11895v1", slug="induction_heads"),
    P("solu", "A", "P1", "Elhage et al.", "2022", "Softmax Linear Units", "Softmax Linear Units",
      "Tries an architectural fix for polysemantic neurons — a softmax-based activation — and finds many more neurons become interpretable, with a catch it attributes to superposition.",
      "The SoLU activation and the LayerNorm it needs, the neuron interpretability evaluation, the evidence read as features hiding in small activations, and its frank note on how little real-world evidence for superposition existed.",
      html=f"{TC}/2022/solu/index.html", slug="softmax_linear_units"),
    P("toy", "A", "P0", "Elhage et al.", "2022", "Toy Models of Superposition", "Toy Models of Superposition",
      "Formalizes superposition — networks representing more features than they have dimensions — the idea that motivated sparse dictionary learning and much of Sections C and D.",
      "Sparsity/importance phase diagrams, feature geometry, computation in superposition, privileged bases, and which results depend on independent, predefined features with no downstream readers.",
      arxiv="2209.10652v1", slug="toy_models_of_superposition"),
    P("ioi", "A", "P0", "Wang et al.", "2022", "Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small",
      "Interpretability in the Wild (IOI circuit)",
      "The canonical end-to-end circuit analysis in a real language model: attention-head classes in GPT-2 small that implement indirect object identification.",
      "Path patching, the head classes (duplicate-token, S-inhibition, name movers, backup/negative name movers), and the faithfulness / completeness / minimality criteria.",
      arxiv="2211.00593v1", slug="ioi_circuit"),
    # ---- B
    P("patching", "B", "P0", "Zhang & Nanda", "2023", "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods",
      "Best Practices of Activation Patching",
      "Systematically shows how activation-patching conclusions change with the corruption method, the evaluation metric, and the patching granularity.",
      "Gaussian noising vs. symmetric token replacement, logit difference vs. probability metrics, sliding-window patching, and the concrete recommendations.",
      arxiv="2309.16042v2", slug="activation_patching_best_practices"),
    P("acdc", "B", "P0", "Conmy et al.", "2023", "Towards Automated Circuit Discovery for Mechanistic Interpretability",
      "Automated Circuit Discovery (ACDC)",
      "Systematizes the circuit-discovery workflow and automates it with ACDC, which prunes edges of the computational graph, validated against hand-found circuits.",
      "The ACDC algorithm and threshold τ, KL-divergence objective, ROC comparison with Subnetwork Probing and HISP, and the limitations / corruption sensitivity.",
      arxiv="2304.14997v4", slug="acdc"),
    P("causal", "B", "P1", "Geiger et al.", "2021", "Causal Abstractions of Neural Networks", "Causal Abstractions of Neural Networks",
      "Supplies the formal causal-abstraction framework and interchange interventions for testing whether a network implements a hypothesized high-level algorithm.",
      "The interchange-intervention definition, alignments between high-level causal models and neural representations, and the MQNLI case study.",
      arxiv="2106.02997v2", slug="causal_abstractions"),
    P("das", "B", "P1", "Geiger et al.", "2023", "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations",
      "Distributed Alignment Search (DAS)",
      "Introduces Distributed Alignment Search: learned rotations let high-level causal variables align with distributed, non-basis-aligned representations.",
      "Rotation-based interchange interventions, interchange intervention accuracy, comparison with basis-aligned search, and what a successful alignment does and does not prove.",
      arxiv="2303.02536v4", slug="distributed_alignment_search"),
    # ---- C
    P("composition", "C", "P1", "Olah", "2023", "Distributed Representations: Composition & Superposition", "Composition & Superposition",
      "A short note separating two reasons a representation can be distributed: composition, which builds on shared structure, and superposition, which packs unrelated features together.",
      "The two definitions and their examples, why a merely rotated basis is neither, and which kind should worry interpretability — then compare with how ‘superposition’ is used across Sections C–D.",
      html=f"{TC}/2023/superposition-composition/index.html", slug="composition_superposition",
      authors=["Chris Olah"]),  # informal note: no BibTeX block to read authors from
    P("explain", "C", "P1", "Bills et al.", "2023", "Language Models Can Explain Neurons in Language Models", "Language Models Can Explain Neurons",
      "Automates neuron explanation: GPT-4 writes explanations of GPT-2 XL neurons, scored by how well a simulator predicts activations from the explanation alone.",
      "The explain–simulate–score loop, what share of neurons score well and how it varies with depth and model size, and the stated limits of simulation-based scoring.",
      html="https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html", slug="explaining_neurons", venue="OpenAI"),
    P("haystack", "C", "P1", "Gurnee et al.", "2023", "Finding Neurons in a Haystack: Case Studies with Sparse Probing", "Finding Neurons in a Haystack",
      "Uses sparse probing to find which neurons represent over a hundred features across seven models from 70M to 6.9B parameters, finding both superposed and dedicated neurons.",
      "The sparse-probing method and its choice of k, superposition in early layers versus dedicated neurons in middle layers, the case studies, and how results change with scale.",
      arxiv="2305.01610v2", slug="neurons_in_haystack"),
    P("cunningham", "C", "P0", "Cunningham et al.", "2023", "Sparse Autoencoders Find Highly Interpretable Model Directions",
      "SAEs Find Highly Interpretable Directions",
      "An early demonstration that sparse autoencoders on LM activations recover features more interpretable than neurons or other decompositions.",
      "SAE architecture and L1 objective, automated interpretability scoring, the IOI causal-localization experiment, and the dictionary-size trade-offs.",
      arxiv="2309.08600v3", slug="sparse_autoencoders_cunningham",
      note="Published under the revised title “Sparse Autoencoders Find Highly Interpretable Features in Language Models” (arXiv v3, the latest version)."),
    P("mono", "C", "P0", "Bricken et al.", "2023", "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning",
      "Towards Monosemanticity",
      "Anthropic's detailed dictionary-learning study on a one-layer transformer that established the methodology for evaluating SAE features.",
      "Training setup and neuron resampling, interpretability evaluation, feature splitting, universality, finite-state-automata phenomenology, and the problems/limitations.",
      html=f"{TC}/2023/monosemantic-features/index.html", slug="towards_monosemanticity"),
    P("scaling", "C", "P0", "Templeton et al.", "2024", "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet",
      "Scaling Monosemanticity",
      "Scales sparse autoencoders to Claude 3 Sonnet, finding abstract, multilingual, multimodal and safety-relevant features, and demonstrates feature steering.",
      "SAE scaling laws, feature completeness, steering interventions, safety-relevant features, and the explicit limitations and caveats.",
      html=f"{TC}/2024/scaling-monosemanticity/index.html", slug="scaling_monosemanticity"),
    P("topk", "C", "P1", "Gao et al.", "2024", "Scaling and Evaluating Sparse Autoencoders", "Scaling and Evaluating Sparse Autoencoders",
      "Establishes k-sparse (TopK) autoencoders and clean scaling laws up to 16M latents on GPT-4 activations, with evaluation metrics beyond reconstruction.",
      "TopK activation, dead-latent mitigation, scaling laws in latents and k, and the downstream-loss, probing, explainability and ablation-sparsity metrics.",
      arxiv="2406.04093v1", slug="scaling_evaluating_saes"),
    P("jumprelu", "C", "P1", "Rajamanoharan et al.", "2024", "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders",
      "JumpReLU Sparse Autoencoders",
      "Introduces JumpReLU SAEs trained directly on an L0 penalty with straight-through estimators, improving reconstruction fidelity at fixed sparsity.",
      "The JumpReLU activation and thresholds, pseudo-derivatives via straight-through estimation, and comparisons with Gated and TopK SAEs on Gemma 2 9B.",
      arxiv="2407.14435v3", slug="jumprelu_saes",
      optional="An architectural refinement on the frontier of {topk}; worth printing if you train sparse autoencoders yourself."),
    P("absorption", "C", "P0", "Chanin et al.", "2024", "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders",
      "A is for Absorption",
      "Identifies feature absorption — general latents failing to fire where a more specific latent absorbs the direction — as a systematic SAE failure mode.",
      "The first-letter spelling task, the absorption metric, splitting vs. absorption, dependence on SAE width and sparsity, and consequences for interpretability claims.",
      arxiv="2409.14507v6", slug="feature_absorption"),
    # ---- D
    P("transcoders", "D", "P0", "Dunefsky et al.", "2024", "Transcoders Find Interpretable LLM Feature Circuits", "Transcoders Find Interpretable Feature Circuits",
      "Replaces MLP sublayers with wide, sparsely-activating transcoders so feature-level circuit analysis can pass through MLPs cleanly.",
      "Transcoder objective vs. SAEs, the input-invariant / input-dependent factorization of circuits, evaluation against SAEs, and the GPT-2 greater-than case study.",
      arxiv="2406.11944v2", slug="transcoders"),
    P("tracing", "D", "P0", "Ameisen et al.", "2025", "Circuit Tracing: Revealing Computational Graphs in Language Models", "Circuit Tracing",
      "Introduces cross-layer transcoders and attribution graphs for analyzing language-model computations.",
      "Replacement-model fidelity, attribution methodology, intervention tests, and limitations.",
      html=f"{TC}/2025/attribution-graphs/methods.html", slug="circuit_tracing"),
    P("biology", "D", "P0", "Lindsey et al.", "2025", "On the Biology of a Large Language Model", "On the Biology of a Large Language Model",
      "Applies attribution graphs to Claude 3.5 Haiku across case studies: multi-step reasoning, planning, multilinguality, arithmetic, hallucination, refusal and CoT faithfulness.",
      "Each case study's mechanism claim and its intervention evidence, recurring motifs across studies, and the limitations section.",
      html=f"{TC}/2025/attribution-graphs/biology.html", slug="biology_of_llm"),
    P("saebench", "D", "P0", "Karvonen et al.", "2025", "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability",
      "SAEBench",
      "An eight-metric benchmark over 200+ SAEs showing that proxy metrics (sparsity vs. fidelity) do not reliably predict practical SAE quality.",
      "The evaluation suite (absorption, autointerp, SCR/TPP, RAVEL, sparse probing, unlearning), architecture comparisons, and where proxy metrics mislead.",
      arxiv="2503.09532v4", slug="saebench"),
    P("mib", "D", "P0", "Mueller et al.", "2025", "MIB: A Mechanistic Interpretability Benchmark", "MIB: Mechanistic Interpretability Benchmark",
      "Standardizes evaluation of circuit-localization and causal-variable-localization methods across four tasks and five models.",
      "Track design and metrics, which methods win on each track, and how SAEs compare with neurons and DAS for causal variable localization.",
      arxiv="2504.13151v2", slug="mib_benchmark"),
    # ---- E
    P("consistency", "E", "F", "Song et al.", "2026", "Mechanistic Interpretability Should Prioritize Feature Consistency in Sparse Autoencoders",
      "Prioritize Feature Consistency in SAEs",
      "Position paper arguing SAE features should converge across independent training runs, and proposing PW-MCC to measure that consistency.",
      "The PW-MCC metric, the synthetic ground-truth validation, architecture effects on consistency — and whether consistency is the right target at all.",
      arxiv="2505.20254v1", slug="sae_feature_consistency",
      note="arXiv title: “Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs”; first posted May 2025."),
    P("arora", "E", "P0", "Arora et al.", "2026", "Language Model Circuits Are Sparse in the Neuron Basis", "Circuits Are Sparse in the Neuron Basis",
      "Reports that MLP neurons are as sparse a feature basis as SAEs and builds gradient-based circuit tracing directly on neurons.",
      "The sparsity comparison methodology, the neuron-level attribution pipeline, benchmark results, and how far the claim generalizes beyond tested models/tasks.",
      arxiv="2601.22594v2", slug="circuits_sparse_neuron_basis"),
    P("weightsparse", "E", "F", "Gao et al.", "2026", "Weight-sparse Transformers Have Interpretable Circuits", "Weight-sparse Transformers",
      "Trains transformers with mostly-zero weights so task circuits pruned from them consist of few, interpretable connections.",
      "Training and pruning procedure, the hand-crafted task circuits, the capability–interpretability trade-off, scaling behaviour, and transfer to dense models.",
      arxiv="2511.13653v1", slug="weight_sparse_transformers", note="arXiv first posted November 2025."),
    # ---- F
    P("repe", "F", "P1", "Zou et al.", "2023", "Representation Engineering: A Top-Down Approach to AI Transparency", "Representation Engineering",
      "Proposes a top-down, population-level approach: reading and controlling representations of high-level concepts such as honesty and power-seeking.",
      "LAT reading vectors, representation control methods, the honesty and utility experiments, and how its claims differ from bottom-up circuit analysis.",
      arxiv="2310.01405v4", slug="representation_engineering"),
    P("actadd", "F", "P1", "Turner et al.", "2023", "Activation Addition: Steering Language Models Without Optimization", "Activation Addition (ActAdd)",
      "Shows that adding a single contrast-pair activation difference at inference time steers model outputs without any optimization.",
      "Construction of steering vectors, layer and coefficient choices, sentiment/toxicity evaluations, and the effect on off-target capabilities.",
      arxiv="2308.10248v5", slug="activation_addition",
      note="Latest arXiv version (v5) is retitled “Steering Language Models With Activation Engineering”; same paper and arXiv identifier."),
    P("refusal", "F", "P1", "Arditi et al.", "2024", "Refusal in Language Models Is Mediated by a Single Direction", "Refusal Is Mediated by a Single Direction",
      "Finds a single residual-stream direction across 13 chat models whose ablation removes refusal and whose addition induces it.",
      "Difference-in-means direction extraction, directional ablation and weight orthogonalization, the jailbreak implications, and the adversarial-suffix analysis.",
      arxiv="2406.11717v3", slug="refusal_single_direction"),
    P("turpin", "F", "P0", "Turpin et al.", "2023", "Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting",
      "Unfaithful Chain-of-Thought Explanations",
      "Shows chain-of-thought explanations can systematically misrepresent the true reason for a model's prediction when inputs contain biasing features.",
      "The biasing interventions (answer-is-always-A, suggested answer), BBH and BBQ results, and what the findings imply for CoT-based oversight.",
      arxiv="2305.04388v2", slug="unfaithful_cot"),
    P("monitor", "F", "P1", "Guan et al.", "2025", "Monitoring Monitorability", "Monitoring Monitorability",
      "Proposes evaluation archetypes and a metric for tracking whether reasoning models' chains of thought remain monitorable.",
      "Intervention, process and outcome-property evaluations, the monitorability metric, obfuscated-CoT model organisms, and CoT vs. action-only monitoring.",
      arxiv="2512.18311v1", slug="monitoring_monitorability"),
    P("cotsae", "F", "F", "Chen et al.", "2026", "How Does Chain of Thought Think? Mechanistic Interpretability of Chain-of-Thought Reasoning with Sparse Autoencoding",
      "How Does Chain of Thought Think?",
      "A feature-level causal study of CoT faithfulness combining SAEs and activation patching on Pythia models solving GSM8K.",
      "Feature-swap patching between CoT and no-CoT runs, the reported scale threshold (70M vs 2.8B), sparsity/interpretability effects — and the small-model caveats.",
      arxiv="2507.22928v1", slug="how_does_cot_think", note="arXiv first posted July 2025.",
      optional="A small-model study; {turpin}, {monitor} and {biology} cover chain-of-thought faithfulness more directly."),
    P("survey", "F", "P1", "Zhang et al.", "2026", "Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models",
      "Locate, Steer, and Improve (survey)",
      "A survey organizing mechanistic interpretability as an actionable pipeline: localize, steer, and improve alignment, capability and efficiency.",
      "The taxonomy of interpretable objects, localization vs. steering methods, the application sections, and the open problems — use as a map back to the primary literature.",
      arxiv="2601.14004v4", slug="locate_steer_improve",
      optional="About a hundred pages of reference material — better consulted than printed."),
    # ---- G
    P("clean", "G", "P0", "Cammarata", "2026", "Neural Networks Are Already Clean", "Neural Networks Are Already Clean",
      "Argues that the polysemantic neurons which motivated superposition came from fixable problems — weak tools, weak architectures, small models — and that networks can be studied directly through their own neurons, with recursive generalization as the main residual motif.",
      "The cars-and-animals reanalysis, recursive generalization, the layer-crushing and SAE comparisons, weak-activation ablations, the neuron-level attribution graphs in Qwen3-32B and Kimi K3 — and the evidence status: language-model numbers are deferred to a forthcoming companion paper.",
      html="https://already-clean.goodfire.com/", slug="neural_networks_already_clean", venue="Goodfire",
      note="Informal research essay. Interactive figures are printed in their static, reduced-motion state; use the URL for the interactive versions.",
      render=dict(clamp_width=True, reduced_motion=True, scroll=False, reveal=["#toy-shared-diagonals"])),
]

# "In context" notes: situate each paper among the others. {key} placeholders become paper numbers, so
# the notes stay correct when papers are added or reordered. Kept separate from the curriculum entries
# above so the original reading notes stay intact.
CONTEXT = {
    "vaswani": "The object of nearly everything that follows: {framework} and {induction} recast its attention layers as readable circuits, and {solu} and {weightsparse} modify its blocks to make them easier to read. {clean} argues its residual connections and normalization are part of why modern networks are legible at all.",
    "zoomin": "The root of the collection's two lines. Its circuits programme leads through {framework} and {induction} to {ioi}; its car-to-dog circuit is the first published example of superposition, formalized in {toy} — and re-read as ordinary generalization in {clean}, by one of this paper's authors.",
    "multimodal": "Carries the vision circuits of {zoomin} into CLIP and finds neurons organized around abstract concepts — a precursor to the abstract features {scaling} finds with dictionaries. Its subsection on conjoined features is the idea {clean} develops as recursive generalization.",
    "framework": "Translates the circuits programme of {zoomin} to transformers, supplying the vocabulary — residual stream, QK/OV circuits, composition — that {induction}, {ioi} and {tracing} build on. Its observation that the residual stream has no privileged basis is a starting point for the dictionary work of {composition}–{absorption}.",
    "induction": "The first mechanism carried from toy models ({framework}) to large ones, found on the model's own components. With {ioi} it sets the template for circuit work later automated in {acdc} and benchmarked in {mib}; its large-model evidence is correlational, which is where {patching} and {causal} matter.",
    "solu": "An architectural answer to the polysemantic neurons of {zoomin}, written just before {toy}, which it says supersedes its discussion of superposition. The architectural route returns in {weightsparse}, and {clean} quotes its admission that superposition had become the default hypothesis with little evidence.",
    "toy": "The pivot of the collection: it turns the polysemantic neurons of {zoomin} and {solu} into a theory, clarified in {composition} and sought in real models by {haystack}. It motivates the dictionaries of {cunningham}–{absorption} and the replacement models of {transcoders}–{tracing}; {consistency}–{weightsparse} and {clean} challenge it from different sides.",
    "ioi": "The canonical circuit in a real language model, built from attention heads with path patching. {patching} refines the patching it relies on, {acdc} automates the search, and {causal}–{das} formalize what it means for a circuit to implement an algorithm; its faithfulness criteria anticipate the metrics of {mib}.",
    "patching": "Methodological glue for Section B: it shows how corruption and metric choices change the conclusions drawn in {ioi} and {acdc}. The same caution applies to the ablation and steering evidence in {biology}, {arora}, {refusal} and {clean}.",
    "acdc": "Automates the hand-built circuit search of {ioi} and is evaluated against it. {mib} benchmarks methods of this kind, while {transcoders}–{tracing} and {arora} move circuit discovery from heads and edges to features or neurons.",
    "causal": "The formal backbone for asking whether a circuit ({ioi}, {acdc}) actually implements a hypothesized algorithm. {das} extends it to distributed representations, and {mib}'s causal-variable track is built on it.",
    "das": "Extends {causal} with learned rotations, so causal variables need not align with neurons — an alternative to the dictionaries of {cunningham}–{absorption}. {mib} finds it strongest at causal-variable localization, while {arora} and {clean} argue the neuron basis is more usable than this line assumed.",
    "composition": "A short clarification of {toy}'s vocabulary, separating representations that compose from those that pack unrelated features, and noting that a merely rotated basis is neither. The dictionaries that follow ({cunningham}–{absorption}) inherit its framing; {clean} adopts its distinction while disputing how often real networks pack.",
    "explain": "The first automated-interpretability pipeline; its explain-and-simulate scoring is the yardstick {cunningham} and {mono} adapt to show dictionary features beat neurons. {saebench} keeps an automated-interpretability metric, and {clean}, by one of this paper's authors, argues simulation scores understated how interpretable neurons are.",
    "haystack": "The real-model counterpart to {toy}: it looks for superposition in actual neurons, finding packed features in early layers and dedicated neurons in middle ones. Its neuron-level probing is a forerunner of {arora}, and {clean} re-analyses its ‘Social Security’ neurons as ordinary composition.",
    "cunningham": "Published alongside {mono}, the open-source case for sparse autoencoders as the answer to {toy}, scored with the method of {explain}. Its comparison against neurons is revisited by {mib} and {arora}, which find neurons competitive, and by {clean}, which argues neuron baselines of the time were depressed by tooling.",
    "mono": "The methodological foundation of the dictionary programme: {scaling} scales it, {topk}–{jumprelu} refine the architecture, and {absorption} and {consistency} document its failure modes. Its opening example of a polysemantic neuron, like {zoomin}'s cat-and-car neuron, is re-read in {clean} as coherent.",
    "scaling": "Takes {mono} to a production model, finding abstract, multimodal features reminiscent of {multimodal}'s neurons and showing they can be steered — a link to {repe}–{refusal}. Its methods lead to the cross-layer transcoders of {tracing}–{biology}; {topk} is the parallel study from OpenAI.",
    "topk": "OpenAI's counterpart to {scaling}, introducing TopK autoencoders and clean scaling laws. {jumprelu} proposes a competing activation, {saebench} tests whether such proxy gains carry over in practice, and the same group's {weightsparse} later pursues legibility by training sparse weights instead.",
    "jumprelu": "Competes with {topk} on the reconstruction–sparsity frontier. {absorption} and {saebench} ask whether gains on that frontier produce more faithful features, and {consistency} asks whether they produce the same features twice.",
    "absorption": "Documents a systematic way the autoencoders of {cunningham}–{jumprelu} misrepresent hierarchical concepts, and becomes one of {saebench}'s metrics. Read with {consistency} and {clean}, which offer different accounts of where such dictionary artifacts come from.",
    "transcoders": "Bridges the dictionaries of {cunningham}–{jumprelu} and the circuits of {ioi} and {acdc} by replacing MLPs with sparse layers; {tracing} generalizes the idea to cross-layer transcoders. {arora} and {clean} instead trace circuits through the original MLP neurons.",
    "tracing": "Combines the dictionary line ({mono}–{transcoders}) with circuit analysis ({ioi}, {acdc}) into attribution graphs over a replacement model, applied at length in {biology}. Its replacement-model fidelity is what {arora} and {clean} press on when they build graphs from raw neurons.",
    "biology": "The applied companion to {tracing} and the most ambitious set of mechanistic case studies here. Its chain-of-thought findings speak to {turpin}–{cotsae}, its refusal study to {refusal}, and {clean} reproduces its medical-diagnosis graph on raw neurons in another model.",
    "saebench": "Turns the architectures of {topk}–{jumprelu} and the failure modes of {absorption} into a shared evaluation, finding that proxy metrics mislead — the backdrop for {consistency}'s call for new targets. {mib} is its counterpart for circuits and causal variables.",
    "mib": "The benchmark for Section B's methods ({patching}–{das}) and for featurizers, comparing SAEs, DAS ({das}) and raw neurons. Its finding that SAE features are no better than neurons connects directly to {arora} and {clean}.",
    "consistency": "A critique of the dictionary programme ({cunningham}–{jumprelu}) from inside it: if features change across training seeds, which are real? Read with {absorption} and {saebench}; {clean} argues the instability belongs to dictionaries rather than to models.",
    "arora": "Revisits the neuron-versus-SAE comparison behind {cunningham} and finds neurons just as sparse, then traces circuits through them as {transcoders}–{tracing} do through replacement models. With {haystack} and {mib} it anchors the neuron-basis line that {clean} develops at length.",
    "weightsparse": "Answers the problem posed by {toy} by changing training rather than adding a dictionary — extending the architectural route of {solu}, from the group behind {topk}. Its pruned circuits invite comparison with {ioi} and {acdc}; {clean} argues ordinary dense architectures already provide much of this legibility.",
    "repe": "The top-down counterpart to the bottom-up work of Sections A–D: it reads and steers population-level directions. {actadd} and {refusal} are narrower instances of the same idea, and {scaling} shows dictionary features can be steered too.",
    "actadd": "The minimal steering method — a single difference vector — that {repe} generalizes and {refusal} applies to refusal. Compare with feature steering in {scaling} and single-neuron steering in {clean}.",
    "refusal": "Applies the direction-finding of {repe}–{actadd} to a safety behavior, with clean causal tests. {biology}'s refusal case study gives a circuit-level view of the same behavior.",
    "turpin": "Shifts from mechanism to faithfulness: what a model says about its reasoning may not reflect it. {monitor} builds an oversight programme around chain of thought, {cotsae} looks for CoT features mechanistically, and {biology} shows unfaithful reasoning at the circuit level.",
    "monitor": "Takes the problem of {turpin} to frontier scale, treating chain-of-thought monitoring as an oversight tool that has to be measured. The mechanistic work of {biology} and {cotsae} is the complementary route to checking what a model is doing.",
    "cotsae": "Connects {turpin}'s faithfulness question to the autoencoders of {cunningham}–{jumprelu} and the patching of {patching}, on small Pythia models. Read its scale threshold alongside {clean}'s argument that small models understate legibility.",
    "survey": "A map back across the collection: its localize–steer–improve frame spans the circuit work of Sections A–B, the dictionaries of C–D and the steering of F. Use it to find what this anthology leaves out.",
    "clean": "Written from inside the tradition it criticizes — its author co-wrote {zoomin}, {multimodal} and {explain} — it re-reads the evidence behind {zoomin}, {toy} and {mono}, sides with the findings of {mib} and {arora}, and redraws graphs in the style of {tracing}–{biology} on raw neurons.",
}
N = len(CATALOG)
for _i, _p in enumerate(CATALOG, 1):
    _p["n"] = _i
    _p["id"] = f"{_i:02d}"
    _p["file"] = f"{_p['id']}_{_p['slug']}.pdf"
BY_KEY = {_p["key"]: _p for _p in CATALOG}
BRIEF = {"composition"}  # informal notes checked against lower length thresholds in QA
_REFS = {k: v["id"] for k, v in BY_KEY.items()}
for _p in CATALOG:
    _p["context"] = CONTEXT[_p["key"]].format(**_REFS)
    _p["optional_raw"] = _p["optional"]  # {key} template, re-formatted as links in the Markdown export
    _p["optional"] = _p["optional"].format(**_REFS)
assert set(CONTEXT) == set(BY_KEY), set(CONTEXT) ^ set(BY_KEY)
SUBTITLE = f"A Research Anthology — {N - 1} Original Papers and a Conclusion"

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
        rec = raw.get(p["slug"]) or by_url.get(p["html"] or f"https://arxiv.org/abs/{p['arxiv']}")
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


def arxiv_metadata(ids: list[str]) -> dict:
    r = requests.get("https://export.arxiv.org/api/query",
                     params={"id_list": ",".join(ids), "max_results": len(ids)}, headers=UA, timeout=60)
    r.raise_for_status()
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
        x = requests.get("https://export.arxiv.org/oai2", headers=UA, timeout=60, params={
            "verb": "GetRecord", "identifier": f"oai:arXiv.org:{base}", "metadataPrefix": "arXiv"}).text
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


async def _render(url: str, out: Path, clamp_width=False, reduced_motion=False, scroll=True, reveal=()) -> dict:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        exe = next((x for x in ("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser") if Path(x).exists()), None)
        browser = await p.chromium.launch(executable_path=exe) if exe else await p.chromium.launch()
        page = await browser.new_page(viewport={"width": HTML_CW, "height": 1000})
        # measure in the print layout at the printable width; reduced motion makes animated figures
        # render their final static frame
        await page.emulate_media(media="print", reduced_motion="reduce" if reduced_motion else "no-preference")
        await page.goto(url, wait_until="load", timeout=300_000)
        try:
            await page.wait_for_load_state("networkidle", timeout=120_000)
        except Exception:
            pass
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
            await page.wait_for_load_state("networkidle", timeout=120_000)
        except Exception:
            pass
        await page.evaluate("async () => { window.scrollTo(0, 0); await document.fonts.ready; }")
        await page.add_style_tag(content=HTML_CSS + (CLAMP_CSS if clamp_width else ""))
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
            else:
                if force or not dest.exists() or "headings" not in rec:
                    log(f"[{p['id']}] print-rendering HTML")
                    hm = render_html(p, dest)
                    rec.update(verified_title=hm["title"], authors=hm["authors"], first_posted=hm["published"],
                               license=hm["license"], license_url="", headings=hm["headings"],
                               removed_blank_pages=hm["removed_blank_pages"], raw_pages=hm["raw_pages"], fit=hm["fit"])
                fitted = ("page clamped to Letter width" if p["render"].get("clamp_width")
                          else "article grid fitted to Letter width")
                rec.update(source_type="Print rendering of the official HTML publication (headless Chrome, site print "
                                       f"stylesheet, {fitted})",
                           source_url=p["html"], pdf_url="")
            surname = p["cite"].split()[0]
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
    for t in wrap(f"{number_words(N - 1).capitalize()} original papers on circuits, superposition, sparse autoencoders "
                  "and alignment — and a concluding essay", "Serif-I", 12.5, MR - ML):
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
        text(c, ML + 24 + w + 8, y, "1 paper" if n == 1 else f"{n} papers", "Sans", 7.5, FAINT)
        text(c, MR, y, f"{s['divider']}–{s['end']}", "Sans", 9, MUTED, "r")
        pg.link(ML - 4, y - 6, MR + 4, y + 13, s["divider"])
        y -= 24
    y -= 2
    hair(c, ML, y + 12, MR)
    y -= 22
    for t in (f"{stats['papers_included']} complete original papers  ·  {L.total} pages  ·  US Letter",
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
    pg.y = heading(c, "Essential papers (P0)", [f"{len(p0)} papers  ·  {tot} pages including covers"])
    label(c, ML, pg.y, "Print-dialog string for all P0 papers")
    pg.y -= 15
    for t in wrap(p0_str, "Sans-M", 8.5, MR - ML):
        text(c, ML, pg.y, t, "Sans-M", 8.5)
        pg.y -= 12
    pg.y -= 22
    cols = ((ML, "No.", "l"), (ML + 34, "Paper", "l"), (412, "Print range", "r"), (446, "Printed", "c"),
            (494, "Read", "c"), (MR - 12, "Annotated", "c"))
    for x, t, al in cols:
        label(c, x, pg.y, t, size=6.5, align=al)
    hair(c, ML, pg.y - 6, MR, gray=INK, w=0.6)
    pg.y -= 22
    for q in p0:
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
    top = H - 72
    label(c, ML, top, f"Section {sec}")
    span = papers[0]["id"] if len(papers) == 1 else f"{papers[0]['id']}–{papers[-1]['id']}"
    label(c, MR, top, f"Paper {span}" if len(papers) == 1 else f"Papers {span}", align="r")
    hair(c, ML, top - 9, MR)
    text(c, ML - 8, top - 150, sec, "Disp-L", 140)
    y = top - 196
    for t in wrap(SECTIONS[sec], "Disp", 26, MR - ML):
        text(c, ML, y, t, "Disp", 26)
        y -= 30
    y -= 2
    count = "1 paper" if len(papers) == 1 else f"{len(papers)} papers"
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
        for t in wrap(p["title"], "Serif", 11 * k, 400 - (ML + 34)):
            text(c, ML + 34, y, t, "Serif", 11 * k)
            y -= 14 * k
        opt = "  ·  optional" if p["optional"] else ""
        text(c, ML + 34, y + 2, f"{p['cite']}, {p['year']}{opt}", "Sans", 7, FAINT)
        text(c, 452, y0, str(pl["cover"]), "Sans", 9, MUTED, "r")
        text(c, MR, y0, rng(pl["cover"], pl["end"]), "Sans-SB", 9.5, INK, "r")
        y -= 18 * k
    range_block(c, "Whole-section print range", f"{s['divider']}–{s['end']}",
                details=[("Divider", str(s["divider"])), ("Pages", str(s["end"] - s["divider"] + 1))],
                note="Includes this divider, paper covers and notes pages.")
    footer(c, f"Mechanistic Interpretability  ·  Section {sec}", s["divider"], L.total)
    return y + 18 * k


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
             **{f"cover_{k}": lay["papers"][BY_KEY[k]["id"]]["cover"] for k in ("zoomin", "tracing", "jumprelu", "clean")},
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
    """pymupdf4llm writes image links relative to the working directory (or absolute); point them at markdown/."""
    return re.sub(rf"\]\((?:[^)\s]*/)?images/{re.escape(slug)}/", f"](images/{slug}/", body)


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


HTML_CSS_PAGE = """
@font-face{font-family:"Source Serif 4";src:url(assets/fonts/SourceSerif4-Regular.ttf)}
@font-face{font-family:"Source Serif 4";font-style:italic;src:url(assets/fonts/SourceSerif4-It.ttf)}
@font-face{font-family:"Source Serif 4";font-weight:600;src:url(assets/fonts/SourceSerif4-Semibold.ttf)}
@font-face{font-family:"Source Serif 4 Display";font-weight:300;src:url(assets/fonts/SourceSerif4Display-Light.ttf)}
@font-face{font-family:Inter;src:url(assets/fonts/Inter-Regular.ttf)}
@font-face{font-family:Inter;font-weight:500;src:url(assets/fonts/Inter-Medium.ttf)}
:root{--ink:#161616;--muted:#666;--rule:#d8d8d8}
body{max-width:46rem;margin:3rem auto 6rem;padding:0 1.25rem;font:17px/1.62 "Source Serif 4",Georgia,serif;color:var(--ink)}
h1{font:300 2.3rem/1.15 "Source Serif 4 Display","Source Serif 4",serif;margin:0 0 1rem}
h2,h3,h4{font-weight:600;line-height:1.3;margin:2rem 0 .6rem}
a{color:inherit;text-decoration-color:var(--rule);text-underline-offset:2px}
nav,.meta,table,figcaption,code{font-family:Inter,system-ui,sans-serif}
nav{font-size:.78rem;color:var(--muted);margin:1.5rem 0}
hr{border:0;border-top:1px solid var(--rule);margin:2.5rem 0}
img{max-width:100%;height:auto;display:block;margin:1rem auto}
table{border-collapse:collapse;font-size:.8rem;display:block;overflow-x:auto;margin:1rem 0}
td,th{border-bottom:1px solid var(--rule);padding:.3rem .6rem;text-align:left;vertical-align:top}
code,pre{font-size:.82rem}pre{overflow-x:auto}
ul{padding-left:1.2rem}li{margin:.25rem 0}
"""


def html_page(title: str, body_md: str) -> str:
    from markdown_it import MarkdownIt
    html = MarkdownIt("commonmark", {"html": True}).enable("table").render(body_md)
    html = re.sub(r'href="(\d\d_[a-z0-9_]+)\.md', r'href="\1.html', html).replace('href="README.md"', 'href="index.html"')
    html = html.replace('src="images/', 'src="../markdown/images/')
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
            f"<style>{HTML_CSS_PAGE}</style></head><body>\n{html}</body></html>\n")


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
        (HTML_DIR / f"{md_name(p)[:-3]}.html").write_text(html_page(f"{p['id']} · {p['title']}", md))
        anchor = md_name(p)[:-3]
        single.append(f'<a id="{anchor}"></a>\n\n' + re.sub(r"\]\((\d\d_[a-z0-9_]+)\.md\)", r"](#\1)", md)
                      .replace("(README.md)", "(#top)"))
    contents = contents_markdown(L)
    (MD_DIR / "README.md").write_text(contents)
    (MD_DIR / "anthology.md").write_text('<a id="top"></a>\n\n' + "\n\n".join(single))
    (HTML_DIR / "index.html").write_text(html_page(TITLE, contents))
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
        src = f"arXiv:{r['arxiv_id']}" if r["arxiv_id"] else "transformer-circuits.pub (print render)"
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
        f"- {len(ids_where(lambda p: p['html'] and not p['render']))} papers "
        f"({', '.join(ids_where(lambda p: p['html'] and not p['render']))}) exist only as HTML on transformer-circuits.pub,",
        "  distill.pub or openaipublic.blob.core.windows.net, with no official PDF. They were",
        "  print-rendered with headless Chrome using the site's print stylesheet on US Letter (0.5 in margins, 90% scale,",
        "  body text ≈10.8 pt). Fixed-width article grids (Distill template) were collapsed to the page width so text is not",
        "  auto-shrunk and wide figures are not clipped; only figures/widgets still wider than the page were zoomed to fit",
        "  (per-paper zoom log in build/metadata.json), and the gutter section numbers (§ 2.1) are printed inline.",
        "  Interactive widgets appear in their default static state; use the URL/QR on the cover for the interactive",
        "  versions. Fully blank pages emitted by Chrome at forced section breaks were removed (count in the manifest).",
        "  Distill articles are CC BY 4.0; the other sites publish no explicit licence, and copyright remains with the",
        "  authors.",
        f"- Paper {BY_KEY['clean']['id']} (Cammarata 2026, the conclusion) is an HTML essay on already-clean.goodfire.com, "
        "rendered the same",
        "  way but with the page clamped to the printable width, reduced motion (animated figures print their final frame)",
        "  and no lazy-load scrolling: scroll-mounted widgets wider than the page otherwise made Chrome shrink the whole",
        "  essay to ~6.6 pt. Body text prints at ≈9.9 pt. No explicit licence is published.",
        f"- Papers {', '.join(ids_where(lambda p: p['optional']))} are marked *optional* on their covers and in the indexes:",
        "  kept in the anthology, but safe to skip when printing selectively.",
        "- Every cover carries an *In context* note situating the paper among the others in the collection, with",
        "  cross-references by paper number. These notes are the compiler's commentary, not the authors'.",
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
              f"  `html/`, published at {PAGES_URL} by `.github/workflows/pages.yml` on every push that changes it.",
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
        "This repository redistributes the original papers in `sources/` and the compiled anthology (GitHub release).",
        "Copyright in every paper remains with its authors; each is included as published, under the terms below.",
        f"{len(cc)} of {len(rows)} works carry a Creative Commons licence. The others are distributed by arXiv under its",
        "non-exclusive distribution licence, or published on their authors' websites without an explicit licence.",
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
