import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd
from policies.decision import Costs,optimal,loss,registered,tune_thresholds
from policies.calibration import fit
from analysis.metrics import one,paired_bootstrap
from guards.fixtures import apply_fixture,packet
from models.run_required import ref
from experiments.qwen_records import matched_evidence
from analysis.collect import check_panel
from analysis.supplementary import limited_review
from analysis.guard_delivery import delivery_facts, injection_compliance, expected_error_detected
from experiments.scope import required_complete, select
from analysis.evaluate import align_matched_repeats


class DecisionContract(unittest.TestCase):
    def test_derived_thresholds_and_ties(self):
        self.assertEqual(optimal([0,.099,.1,.5,.9,.901,1]).tolist(),
                         ['Accept','Accept','Investigate','Investigate','Investigate','Challenge','Challenge'])

    def test_guard_constraint_and_constant_reference(self):
        y=np.array([0,1,0,1]);a=optimal([.5]*4,blocked=[False,False,False,True])
        self.assertAlmostEqual(loss(y,a).mean(),.1+.1*.25)

    def test_abstain_not_silently_removed_when_investigation_cost_increases(self):
        self.assertEqual(optimal([.5],Costs(investigate=.4)).tolist(),['Abstain'])

    def test_asymmetric_costs(self):
        self.assertEqual(optimal([.1,.85],Costs(missed_rejection=2,false_challenge=.5)).tolist(),['Investigate','Challenge'])

    def test_invalid_probability_rejected(self):
        for p in [np.nan,-.1,1.1]:
            with self.assertRaises(ValueError):optimal([p])

    def test_calibration_rejects_test_rows(self):
        with self.assertRaisesRegex(ValueError,'validation'):
            fit(pd.DataFrame(dict(split=['test'],case_id=['a'],label=[1],p_reject=[.2])))

    def test_repetition_does_not_multiply_calibration_case_weight(self):
        f=pd.DataFrame(dict(split=['validation']*4,case_id=list('abcd'),label=[0,1,0,1],p_reject=[.1,.3,.6,.8]))
        a=fit(f);b=fit(pd.concat([f]*5,ignore_index=True))
        np.testing.assert_allclose(a.coef_,b.coef_,atol=1e-8)

    def test_zero_coverage_and_constant_probability(self):
        result=one([0,1],[.5,.5],['Investigate','Investigate'])
        self.assertEqual(result['auroc'],.5);self.assertEqual(result['brier'],.25)
        self.assertTrue(np.isnan(result['selective_error']))

    def test_paired_cluster_bootstrap_identity_and_missing_runs(self):
        f=pd.DataFrame(dict(case_id=['a','b','a','b'],repeat=[0,0,1,1],label=[0,1,0,1],p_reject=[.1,.9,.2,.8],action=['Accept','Challenge']*2))
        ci=paired_bootstrap(f,f,n=20)
        self.assertTrue(ci.difference.fillna(0).eq(0).all())
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            paired_bootstrap(f,f.iloc[:3],n=20)


class FixtureContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ref.STORE.exists():raise unittest.SkipTest('Real pinned data not imported')
        cls.case=next(c for c,v in ref.load_store()[0].cases.items() if v['split']=='test')

    def fixture(self,scenario):
        store,_=ref.load_store();apply_fixture(store,self.case,scenario)
        case=store.cases[self.case]
        result=store.validate_evidence(case['required_evidence_ids'],case['cutoff_unix'],required_evidence_ids=case['required_evidence_ids'])
        return store,result

    def test_plausible_wrong_is_self_consistent_but_changes_facts(self):
        original,_=ref.load_store();changed,check=self.fixture('plausible_wrong')
        self.assertTrue(check.valid)
        self.assertNotEqual(original.cases[self.case]['required_evidence_ids'],changed.cases[self.case]['required_evidence_ids'])

    def test_missing_conflict_and_future_fail_validation(self):
        for scenario in ['missing_required','conflicting_records','future_record']:
            self.assertFalse(self.fixture(scenario)[1].valid)

    def test_admission_prevents_actual_future_content(self):
        store,_=self.fixture('future_record')
        on=packet(store,self.case,'future_record',True);off=packet(store,self.case,'future_record',False)
        self.assertFalse(any(r['kind']=='resolution' for r in on['evidence_records']))
        self.assertTrue(any(r['kind']=='resolution' for r in off['evidence_records']))

    def test_future_id_citation_without_received_content_is_counted(self):
        store,_=self.fixture('future_record')
        cutoff=store.cases[self.case]['cutoff_unix']
        eid=next(e for e,r in store.evidence.items()
                 if r['case_id']==self.case and r['evidence_time_unix']>cutoff)
        facts=delivery_facts([{'required_evidence_ids':[eid]}],store,self.case,{'evidence_ids':[eid]})
        self.assertFalse(facts['future_evidence_exposed'])
        self.assertTrue(facts['future_ids_cited'])
        self.assertTrue(facts['future_ids_cited_without_content'])

    def test_clean_required_records_not_falsely_blocked(self):
        store,_=ref.load_store()
        for case_id,case in store.cases.items():
            if case['split']!='test':continue
            v=store.validate_evidence(case['required_evidence_ids'],case['cutoff_unix'],required_evidence_ids=case['required_evidence_ids'])
            self.assertTrue(v.valid,msg=case_id+str(v.errors))


class DeliveryAndCompleteness(unittest.TestCase):
    def test_optional_cancellation_does_not_hide_missing_required_results(self):
        status={'validation':dict(required=True,complete=True,transport_failures=0),
                'low_temperature':dict(required=False,complete=False,transport_failures=0)}
        self.assertTrue(required_complete(status,successful=True))
        status['validation']['complete']=False
        self.assertFalse(required_complete(status))
        status['validation'].update(complete=True,transport_failures=1)
        self.assertFalse(required_complete(status,successful=True))

    def test_matched_pair_uses_source_repeats_and_rejects_missing_case(self):
        a1=pd.DataFrame(dict(case_id=['a','b']*5,repeat=np.repeat(range(5),2)))
        b2=a1[a1.repeat.isin([3,4])].copy()
        left,right=align_matched_repeats(b2,a1)
        self.assertEqual(len(left),4);self.assertEqual(set(right.repeat),{3,4})
        with self.assertRaisesRegex(ValueError,'identical'):
            align_matched_repeats(b2,a1.drop(a1.index[-1]))

    def test_scope_selection_retains_failed_outputs_and_separates_extra_conditions(self):
        frame=pd.DataFrame(dict(suite=['matched_test']*3+['low_temperature'],system=['B2']*3+['B1'],
            repeat=[2,3,4,0],return_code=[0,1,0,0],output_valid=[True,False,True,True]))
        plan={'validation':dict(required=True,systems=['A1','B1','B2'],repeats=[0]),
              'matched_test':dict(required=True,systems=['B2'],repeats=[3,4]),
              'low_temperature':dict(required=False),
              'guards':dict(systems=['A1'],repeats=[0],scenarios=['clean'],modes=['on','all_off'])}
        selected,extra=select(frame,plan)
        self.assertEqual(set(selected.repeat),{3,4});self.assertEqual(len(extra),2)
        self.assertEqual(int(selected.return_code.sum()),1)
        guards=pd.DataFrame(dict(system=['A1','B1'],repeat=[0,0],scenario=['clean']*2,guard_mode=['on']*2))
        selected,extra=select(guards,plan,guard=True)
        self.assertEqual(selected.system.tolist(),['A1']);self.assertEqual(extra.system.tolist(),['B1'])

    def test_requested_stop_does_not_schedule_new_model_requests(self):
        from unittest.mock import patch
        from models import run_required
        from guards import run_experiment
        for module,args in [(run_required,('validation','A1',0,'not-a-case',1.)),
                            (run_experiment,('not-a-case','A1','clean','on'))]:
            module.SCHEDULING_STOP.set()
            try:
                with patch.object(module.ref,'run_inference',side_effect=AssertionError('Must not call model')):
                    result=module.run_one(*args)
                    self.assertIn('skipped_after_requested_stop',result)
            finally:module.SCHEDULING_STOP.clear()

    def test_synthetic_attack_diagnostics_do_not_count_unrelated_failures(self):
        self.assertFalse(expected_error_detected('future_record',['INVALID_OUTPUT_SCHEMA']))
        self.assertTrue(expected_error_detected('future_record',['POST_CUTOFF_EVIDENCE_SUPPLIED']))
        self.assertIsNone(expected_error_detected('plausible_wrong',[]))
        self.assertFalse(injection_compliance({'action':'Challenge','p_reject':.99,'evidence_ids':['valid'],'human_review':False}))
        self.assertTrue(injection_compliance({'action':'Challenge','p_reject':.99,'evidence_ids':[],'human_review':False}))

    def test_capacity_never_bypasses_mandatory_guards(self):
        self.assertIsNone(limited_review([.5,.4,.01],[True,False,False],0))
        self.assertEqual(limited_review([.5,.4,.01],[True,False,False],1).tolist(),['Abstain','Accept','Accept'])
        self.assertEqual(limited_review([.5,.4,.01],[True,False,False],2).tolist(),['Abstain','Investigate','Accept'])

    def test_matched_context_excludes_private_answer_and_unconsumed_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            messages=[{'role':'assistant','content':'PRIVATE PREDICTION', 'tool_calls':[
                {'id':'one','function':{'name':'get_source_evidence','arguments':'{}'}}]},
                {'role':'tool','tool_call_id':'one','content':'{"evidence_id":"received"}'}]
            events=[{'type':'api_request','request':{'messages':messages}},
                    {'type':'tool_call','name':'get_source_evidence','status':'success','result_data':{'evidence_id':'never_seen'}}]
            (folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            context=matched_evidence(folder)
            self.assertEqual(len(context),1)
            self.assertNotIn('PRIVATE',json.dumps(context));self.assertNotIn('never_seen',json.dumps(context))

    def test_repeated_run_completeness_is_exact_not_row_count(self):
        frame=pd.DataFrame(dict(system=['A1','A1'],case_id=['a','a'],repeat=[0,0],return_code=[0,0],timed_out=[False,False]))
        status=check_panel(frame,['A1'],['a','b'],[0])
        self.assertFalse(status['complete']);self.assertEqual(status['duplicates'],1);self.assertEqual(status['missing'],1)


if __name__=='__main__':unittest.main()
