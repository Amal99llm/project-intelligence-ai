# Critical context trace — before fix

Captured against the local 136-project database on 2026-07-18, before the
fix in this change. Wrappers observed `followup_gate`, field detection,
contract understanding, entity resolution, and the committed session state.

```text
TURN 1: اعطني ملخص مشروع الباحث
active_project=None
field_alias=None; contract_request=None
calls=followup_gate fires=False -> project_entity_resolver matched PJ-GP-HRSDSR
decision=project_summary; updates_active=PJ-GP-HRSDSR
committed_active=PJ-GP-HRSDSR

TURN 2: كم قيمة العقد؟
active_project=PJ-GP-HRSDSR
field_alias=total_contract_value
contract_request=ContractRequest(metrics=(total_contract_value,), operation=get)
calls=contract-priority path; no project resolver call
decision=project_kpi; updates_active=PJ-GP-HRSDSR
answer=قيمة العقد الإجمالية بعد التعديلات 1.12 مليار ريال.
committed_active=PJ-GP-HRSDSR

TURN 3: الباخث
active_project=PJ-GP-HRSDSR
field_alias=None; contract_request=None
calls=followup_gate fires=False -> project_entity_resolver confirmation PJ-GP-HRSDSR
decision=project_summary; answer=هل تقصد مشروع «الباحث الاجتماعي الثاني»؟
committed_active=PJ-GP-HRSDSR

TURN 4: اي
active_project=PJ-GP-HRSDSR
field_alias=None; contract_request=None
calls=pending-selection path; project resolver not called
decision=project_summary; updates_active=PJ-GP-HRSDSR
committed_active=PJ-GP-HRSDSR

TURN 5: كم العقد الأساسي؟
active_project=PJ-GP-HRSDSR
field_alias=None
contract_request=ContractRequest(metrics=(contract_value,), operation=get)
calls=project_entity_resolver confirmation PJ-GP-HAJRGR
decision=project_kpi
answer=هل تقصد مشروع «الاستقبال والترحيب والتوجيه والتوديع ... -تسهيل»؟
committed_active=PJ-GP-HRSDSR
```

## Proven cause

Hypothesis 1 is not reproducible on the current code: `كم قيمة العقد؟` is
recognized and preserves the active project. Hypothesis 2 is proven. The
short base-contract wording was not returned by the field registry, although
contract understanding recognized it. The contract branch then treated the
remaining field wording as a project phrase and called fuzzy entity resolution
before honoring the active project. Both the incomplete field alias expansion
and the contract branch's resolution priority contributed to turn 5.

`اي` is not an accidental partial match: normalization maps it to a value
explicitly present in `_YES_NORM`.
