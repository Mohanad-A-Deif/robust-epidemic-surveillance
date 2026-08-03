# Reproducibility and Execution Report

## Environment checks
- Python: 3.13.5
- Initial interpreter check: `2 + 2 = 4`
- Core test suite after revisions: `6 passed`
- RKI package integration test: `1 passed`
- Original quick end-to-end software smoke test: exit code 0; 54 method-scenario executions; 22 PNG figures; 17 LaTeX tables. These smoke outputs were not used as manuscript evidence.

## Real-data provenance
- Source: RKI StopptCOVID symptom-onset expected-case CSV.
- Verified MD5: `e59953809e7a8050bb40045c6172ee30`.
- Source data rows: 29,712.
- Processed window: 2020-02-20 to 2021-02-18, 365 days.
- States: Baden-Württemberg, Bayern, Hessen, Niedersachsen, Nordrhein-Westfalen, Rheinland-Pfalz.
- Latent state: `log1p(daily incidence per 100,000 population)`.
- Population reference date: 2020-12-31; Destatis GENESIS table 12411-0017.

## Leakage protocol
- Training: 219 days.
- Validation: 73 days.
- Locked test: 73 days.
- Validation seeds: five independent corruption seeds.
- Final semi-real seeds: 1101-1120.
- Test results were not used to select hyperparameters, stopping criteria, graph threshold, or model configuration.
- Locked configuration file: `final_results/protocol/locked_test_protocol.json`.

## Selected validation configuration
- Candidate: `C06_sigma025`.
- Validation RMSE: 0.4750 ± 0.1738.
- Default validation RMSE: 0.4835 ± 0.2206.
- Selected `beta=0.12`, `gamma=0.08`, `sigma_w=0.25`, Huber `kappa=1.5`, `lambda_w=lambda_g=0.015`.
- The validation gain was small and was not described as a major improvement.

## Locked-test findings
### Retrospective reconstruction
- Proposed revised RMSE: 0.3197 ± 0.0515; Pearson r: 0.8182.
- No-delay ablation mean RMSE: 0.3007.
- Original proposed ablation mean RMSE: 0.3097.
- Fixed geographic graph mean RMSE: 0.3230.
- Contrasts among these close methods did not remain significant after Holm correction.

### Causal rolling nowcasting
- Kalman/RTS internal mean RMSE: 0.8683.
- Delay-aware state-space internal mean RMSE: 1.0343.
- Proposed revised mean RMSE: 1.0857; Pearson r: 0.8695; MAE: 1.0661.
- The proposed model tracked temporal shape but showed material level/scale bias and high runtime.

### Robustness
- Proposed RMSE increased from 0.2999 at 0% missingness to 0.6147 at 50% missingness.
- It increased from 0.3105 under light delay to 0.3440 under severe delay.
- It increased from 0.2124 at 0% outliers to 0.6570 at 30% outliers.
- Delay backprojection was best when no outliers were injected. At positive outlier levels, Proposed revised beat No-robustness but No-delay retained lower mean RMSE. Proposed revised was best at 50% missingness and under delay-prior mismatch.

### Synthetic graph recovery
- Support F1: approximately 0.280-0.308.
- AUROC: approximately 0.463-0.536.
- Normalized Frobenius error: approximately 0.985-1.048.
- Edge-weight correlation: near zero.
- State RMSE remained approximately 0.067-0.070.
- Conclusion: state recovery did not imply identifiable directed edges.

### Model mismatch
- Proposed method was strong under nonstationary delays and close to best under abrupt interventions.
- Delay-aware state space was clearly better under nonlinear epidemic dynamics.
- Negative results were retained.

## Integrity checks
- No baseline was removed because it outperformed the proposed method.
- No test-based tuning was performed.
- RKI geographic adjacency was never called ground-truth transmission.
- True graph support was not used as a candidate mask in synthetic recovery.
- Real RKI trajectories, semi-real transport corruption, and synthetic graph experiments are labeled separately.
- No real-world or 6G deployment-readiness claim is made.
