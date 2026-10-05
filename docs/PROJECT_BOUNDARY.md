# Boundary between the descriptor and the research

The descriptor answers: what is the resource, how was it built, what does each variable mean, and how well is it validated? The research answers: does admissible evidence retrieval improve decision quality and institutional accountability, under which policies and resource constraints?

| Artifact or claim | Descriptor owner | Research owner |
| --- | --- | --- |
| Full raw acquisition, protocol/event mapping and processed Atlas | Authoritative construction and source lineage | Import a fixed release; record the selected inputs |
| All variables, missingness, coverage and full lifecycle validation | Authoritative dictionary and technical validation | State experimental features, relevant coverage limits and label caveats |
| Hugging Face dataset card, inventory, Croissant and preservation | Existing descriptor release plan | Check that the pinned research inputs resolve; report missing assets |
| Benchmark cohort, split IDs and model-visible fields | Cite reusable record definitions | Own selection rules, exclusions, labels, splits and admissibility |
| Model prompts, inference, action costs and guard behavior | No duplicate experiment pipeline | Own implementation, calibration, controls and run records |
| Economic or AI conclusions | Measurement checks only | Own research questions, analyses, uncertainty and limits |
| Manuscript data sections | Full descriptor treatment | Concise resource citation plus a self-contained experiment description |

Do not copy the full Atlas ETL, full dictionary or lifecycle appendix into this code project. Do not move the experiment cohort, action policy, admissible information, baselines or limitations out of the research simply because the descriptor exists.

If an upstream semantic error could change labels or features, freeze the affected experiment, notify the descriptor team under its existing instructions, pin the corrected release, and regenerate affected results. A downstream code wrapper cannot repair a data provenance gap.

The research camera-ready will cite the actual public descriptor arXiv version. Both manuscripts must describe the same released inputs and identify overlap honestly. Public assets link to public papers and released data; private manuscript access is handled separately.

Development changes stay in the personal fork. Organization-repository changes require a separate instruction from the corresponding author. A later anonymous WWW artifact must be prepared separately: this named repository and its history disclose authorship.
