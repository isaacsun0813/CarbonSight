# Scheduling research references

Original PDFs downloaded from the linked primary sources on November 18, 2026.
The papers retain their authors' copyright and publication terms; CarbonSight's
MIT license does not apply to them. PDFs are unchanged.

| Paper | Local PDF | Primary source |
|---|---|---|
| Zongheng Yang et al., **SkyPilot: An Intercloud Broker for Sky Computing**, NSDI 2023 | [PDF](skypilot-nsdi23.pdf) | [USENIX](https://www.usenix.org/conference/nsdi23/presentation/yang-zongheng) |
| Zhanghao Wu et al., **Can't Be Late: Optimizing Spot Instance Savings under Deadlines**, NSDI 2024 | [PDF](cant-be-late-nsdi24.pdf) | [USENIX](https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao) |
| Zhifei Li et al., **SkyNomad: On Using Multi-Region Spot Instances to Minimize AI Batch Job Cost**, arXiv:2601.06520v1, 2026 | [PDF](skynomad-2026.pdf) | [arXiv](https://arxiv.org/abs/2601.06520v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

## Runtime and deadlines

**SkyPilot §4.3:** a runtime estimator is optional and supplied by the user.
Benchmarking can inform it. Without it, the paper chooses by hourly resource price.

**Can't Be Late §§3, 5.2, 5.6:** Uniform Progress schedules work against a deadline
using a runtime estimate. Section 5.6 explicitly handles inaccurate estimates:
keep executing when the estimate is exhausted, and use on-demand after a later
Spot interruption. Underestimation weakens the original deadline guarantee.
Its greedy policy (§3.4) is a baseline, not its proposed Uniform Progress policy.

**SkyNomad §§4.1, 4.4–4.7:** assumes runtime can be estimated beforehand. Its
prediction model estimates Spot capacity lifetime, which is different from the
training job's runtime. Pratik Mishra's affiliation is AMD.

## CarbonSight behavior

Runtime is optional for ordinary advice and launches. Without an estimate, results
compare one hour of execution and are labeled accordingly; job-total ledger
estimates are skipped. Deadline and delayed-start planning require an explicit
estimate. Reaching that estimate does not complete a real job: observed SkyPilot
success does. The deadline monitor still exits at the original deadline and leaves
an unfinished cluster running with a warning; this is narrower than the continuing
execution policy described in Can't Be Late §5.6.
