# Training History Retention Shortlist

- Keep: `3`
- Review: `20`
- Delete-recommended: `6`
- Potential reclaim if delete-recommended is removed: `273.37 MB`

## Keep

- `confirm_vn100_fnb_committee_20260408_235445_r01` | reason: top committee result | best_test=0.004762 | best_val=-0.002380 | committee_test=0.049321 | stable_median=0.051640 | size=0.03MB
- `biaspush_signmag_narrow_rawmag_20260409_111710` | reason: strong standalone result with acceptable val/test gap | best_test=0.034253 | best_val=0.021013 | committee_test= | stable_median= | size=0.05MB
- `overnight_fnb_w5_mag_base_20260409_100007` | reason: strong standalone result with acceptable val/test gap | best_test=0.034253 | best_val=0.021013 | committee_test= | stable_median= | size=0.03MB

## Review

- `biaspush_signmag_sector_rawmag_20260409_111710` | reason: high test score but unstable val/test gap | best_test=0.056165 | best_val=-0.005822 | gap=0.061987 | size=0.06MB
- `residual_fnb_vn100_context_plain_r01` | reason: high test score but unstable val/test gap | best_test=0.050485 | best_val=0.021013 | gap=0.029472 | size=0.02MB
- `residual_fnb_vn100_context_signmag_r01` | reason: high test score but unstable val/test gap | best_test=0.050485 | best_val=0.021013 | gap=0.029472 | size=0.02MB
- `biaspush_signmag_sector_rawmag_highercap_20260409_111710` | reason: high test score but unstable val/test gap | best_test=0.040047 | best_val=0.006263 | gap=0.033784 | size=0.03MB
- `mini_bat_ong_san_g01_return_w20_patched_v1` | reason: high test score but unstable val/test gap | best_test=0.032545 | best_val=-0.004724 | gap=0.037269 | size=0.03MB
- `overnight_fnb_w7_mag_20260409_101741` | reason: high test score but unstable val/test gap | best_test=0.031770 | best_val=0.003101 | gap=0.028669 | size=0.03MB
- `biaspush_signmag_sector_base_20260409_111710` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.029913 | best_val=-0.007213 | gap=0.037126 | size=0.06MB
- `overnight_fnb_w5_mag_higher_units_20260409_101741` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.024127 | best_val=-0.010692 | gap=0.034818 | size=0.03MB
- `biaspush_signmag_narrow_rawmag_nomagw_20260409_111710` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.018636 | best_val=0.007239 | gap=0.011397 | size=0.03MB
- `overnight_fnb_w5_nomag_20260409_101741` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.017934 | best_val=-0.008298 | gap=0.026232 | size=0.03MB
- `overnight_fnb_w5_mag_tighter_lr_20260409_101741` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.017842 | best_val=0.000426 | gap=0.017416 | size=0.03MB
- `overnight_fnb_w10_mag_20260409_101741` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.017312 | best_val=0.004818 | gap=0.012495 | size=0.03MB
- `collapse_fix_sharp_v1_20260409` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.013740 | best_val=-0.045803 | gap=0.059543 | size=0.02MB
- `collapse_fix_sharp_v2_20260409` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.012039 | best_val=-0.013812 | gap=0.025851 | size=0.02MB
- `overnight_shared_vn100_w60_u64_32_relscore_20260408_173407` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.007579 | best_val=-0.002482 | gap=0.010061 | size=0.04MB
- `overnight_wholemarket_shared_vn100_w60_20260409_010816` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.005859 | best_val=-0.000832 | gap=0.006691 | size=0.04MB
- `overnight_shared_vn100_w20_u64_32_relscore_20260408_173407` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.004762 | best_val=-0.002380 | gap=0.007142 | size=0.03MB
- `overnight_wholemarket_shared_vn100_w20_20260409_010816` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.004762 | best_val=-0.002380 | gap=0.007142 | size=0.03MB
- `overnight_shared_vn30_w20_u64_32_relscore_20260408_173407` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.003350 | best_val=0.007008 | gap=0.003658 | size=0.03MB
- `overnight_shared_vn30_w60_u64_32_relscore_20260408_173407` | reason: not clearly strong enough to keep and not large enough to force-delete | best_test=0.000624 | best_val=0.003881 | gap=0.003257 | size=0.04MB

## Delete-Recommended

- `biaspush_signmag_narrow_base_20260409_111710` | reason: duplicate config+metrics; keep canonical run only | duplicate_of=overnight_fnb_w5_mag_base_20260409_100007 | best_test=0.034253 | size=0.05MB
- `overnight_fnb_w5_mag_base_20260409_101741` | reason: duplicate config+metrics; keep canonical run only | duplicate_of=overnight_fnb_w5_mag_base_20260409_100007 | best_test=0.034253 | size=0.06MB
- `20260409_144422_fnb10_expert` | reason: heavy run with weak standalone score | best_test=0.011196 | size=82.23MB
- `20260409_144422_bds15_expert` | reason: heavy run with weak standalone score | best_test=0.004408 | size=102.81MB
- `shared_vn30_return_w20_relscore_20260408_170355` | reason: duplicate config+metrics; keep canonical run only | duplicate_of=overnight_shared_vn30_w20_u64_32_relscore_20260408_173407 | best_test=0.003350 | size=0.03MB
- `20260409_144422_bank12_expert` | reason: heavy run with weak standalone score | best_test=-0.004871 | size=88.19MB
