"""Replicated evaluation of the complete local multi-model score renderer.

Training is performed on three isolated source corpora. Evaluation scores
and source identities are fixed before training. Family ablations and
wrong-label controls isolate the effects of conditioning and each model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, replace
import json
import statistics

from research.instrument_note_data import build_corpus as short_corpus, training_schedule as short_schedule
from research.instrument_harmonic_model import initialize_model
from research.instrument_harmonic_study import StudyPlan as ShortPlan, train_model
from research.instrument_long_data import build_corpus as long_corpus, training_schedule as long_schedule
from research.instrument_long_model import initialize_windowed
from research.instrument_long_study import StudyPlan as LongPlan, train as train_long
from research.percussion_data import build_corpus as drum_corpus, schedule as drum_schedule
from research.percussion_model import initialize as initialize_drums
from research.percussion_study import StudyPlan as DrumPlan, train as train_drums
from research.multimodel_music_routes import FAMILIES, Exclusions, route_score, describe_routes
from research.multimodel_music_render import Models, RenderPlan, compare_score
from research.hybrid_music_study import ScoreCase
from sonumio.arrangement import compose_arrangement, arrangement_digest

PROTOCOL = "sonumio-multimodel-score-study-v1"
CASES = (ScoreCase("cinematic","minor",60,144,211,.05),
         ScoreCase("rhythmic","major",55,160,223,.15),
         ScoreCase("calm","minor",63,120,227))
MAX_RENDER_SAMPLES = 350_000_000


@dataclass(frozen=True, slots=True)
class StudyPlan:
    short: ShortPlan = ShortPlan()
    sustained: LongPlan = LongPlan()
    percussion: DrumPlan = DrumPlan()
    cases: tuple[ScoreCase,...] = CASES
    excitation_replica: int = 701

    def __post_init__(self):
        if (not isinstance(self.short,ShortPlan) or not isinstance(self.sustained,LongPlan)
                or not isinstance(self.percussion,DrumPlan)):
            raise ValueError("Expected independent research training plans")
        if not self.short.model_seeds == self.sustained.seeds == self.percussion.seeds:
            raise ValueError("Model replicate seeds must match across families")
        if (type(self.cases) is not tuple or not 1 <= len(self.cases) <= 6
                or any(not isinstance(c,ScoreCase) for c in self.cases)
                or len(set(self.cases)) != len(self.cases)):
            raise ValueError("Invalid multi-model score case panel")
        if type(self.excitation_replica) is not int or not 0 <= self.excitation_replica < 2**32:
            raise ValueError("Invalid evaluation excitation replica")
        # Baseline, initialization, trained, three family ablations,
        # one pitched-label and one drum-label intervention per replicate.
        samples = sum(describe_routes(route_score(compose_arrangement(c.to_plan())))["event_samples"]
                      for c in self.cases)
        if samples * 2 * (1 + len(self.short.model_seeds)*7) > MAX_RENDER_SAMPLES:
            raise ValueError("Multi-model score study exceeds render budget")


@dataclass(frozen=True, slots=True)
class Sources:
    short: object
    sustained: object
    percussion: object
    exclusions: Exclusions
    schedules: tuple
    schedule_hashes: tuple

    @classmethod
    def build(cls,plan):
        a = short_corpus(plan.short.grid)
        b = long_corpus(plan.sustained.grid)
        c = drum_corpus(plan.percussion.grid)
        aa, ah = short_schedule(a.panel("train"), updates=plan.short.updates,
            batch_size=plan.short.batch_size, seed=plan.short.sampler_seed)
        bb, bh = long_schedule(b.get("train"), updates=plan.sustained.updates,
            batch_size=plan.sustained.batch_size, seed=plan.sustained.sampler_seed)
        cc, ch = drum_schedule(c.get("train"), updates=plan.percussion.updates,
            batch_size=plan.percussion.batch_size, seed=plan.percussion.sampler_seed)
        exclusions = Exclusions(frozenset(x.spec.identity for x in a.panel("train")),
            frozenset(x.note.identity for x in b.get("train")),
            frozenset(x.spec.seed for x in c.get("train")))
        return cls(a,b,c,exclusions,(aa,bb,cc),(ah,bh,ch))

    def receipt(self):
        return {
            "short":self.short.report(),"sustained":self.sustained.receipt(),
            "percussion":self.percussion.report(),
            "schedules":dict(zip(FAMILIES,self.schedule_hashes,strict=True)),
            "excluded_short_controls":len(self.exclusions.short),
            "excluded_sustained_controls":len(self.exclusions.sustained),
            "excluded_drum_seeds":sorted(self.exclusions.drum_seeds),
        }


def panel(plan,sources):
    items = []
    identities = set()
    used = dict.fromkeys(FAMILIES,0)
    for case in plan.cases:
        if case.seed in sources.exclusions.drum_seeds:
            raise ValueError("Evaluation score seed occurs in percussion training")
        score = compose_arrangement(case.to_plan())
        digest = arrangement_digest(score)
        if digest in identities:
            raise ValueError("Duplicate multi-model evaluation score")
        identities.add(digest)
        coverage = describe_routes(route_score(score,sources.exclusions))
        for family in FAMILIES:
            used[family] += coverage["events_by_family"][family]
        items.append((case,score,coverage))
    if any(n == 0 for n in used.values()):
        raise ValueError("Evaluation score panel omits a learned family")
    return tuple(items)


def initialize(plan,seed):
    return Models(initialize_model(plan.short.width,seed),
                  initialize_windowed(plan.sustained.width,seed),
                  initialize_drums(plan.percussion.width,seed))


def train_all(models,sources,plan):
    a,b,c = sources.schedules
    return {
        "short":train_model(models.short,sources.short,plan.short,a),
        "sustained":train_long(models.sustained,sources.sustained,plan.sustained,b),
        "percussion":train_drums(models.percussion,sources.percussion,plan.percussion,c),
    }


def _compare(score,arm,plan,sources,models=None,**kwargs):
    return compare_score(score,RenderPlan(arm=arm,replica=plan.excitation_replica,**kwargs),
                         models=models,exclusions=sources.exclusions)


def _family_metric(report,family):
    macro = report["routing"]["note_diagnostics"][family]["track_macro"]
    if macro is None:
        return None
    return macro["objective" if family == "percussion" else "relative_wave_mse"]


def _pair_metric(control,trained,family):
    a,b = _family_metric(control,family),_family_metric(trained,family)
    if (a is None) != (b is None):
        raise ValueError("Paired multi-model family coverage differs")
    return None if a is None else a-b


def _verify_same(reference,*reports):
    for report in reports:
        if (report["reference_wav_sha256"] != reference["reference_wav_sha256"]
                or report["routing"]["score_sha256"] != reference["routing"]["score_sha256"]
                or report["routing"]["eligible"] != reference["routing"]["eligible"]
                or report["routing"]["replica"] != reference["routing"]["replica"]):
            raise ValueError("Multi-model paired source changed")


def summarize(generic,initial,trained,ablations,wrong_pitch,wrong_drum):
    _verify_same(generic,initial,trained,*ablations.values(),wrong_pitch,wrong_drum)
    full = trained["routing"]
    for report in (generic,initial,wrong_pitch,wrong_drum):
        if report["routing"]["substituted_events"] != full["substituted_events"]:
            raise ValueError("Full multi-model controls changed eligible events")
    for family,report in ablations.items():
        expected = {name:(0 if name == family else count)
                    for name,count in full["substituted_events"].items()}
        if report["routing"]["substituted_events"] != expected:
            raise ValueError("Ablation did not isolate one model family")
    for report in (*ablations.values(),wrong_pitch,wrong_drum):
        if report["routing"]["model_sha256"] != full["model_sha256"]:
            raise ValueError("Multi-model intervention changed trained weights")
    if wrong_pitch["routing"]["pitched_offset"] != 1 or wrong_pitch["routing"]["drum_offset"] != 0:
        raise ValueError("Invalid pitched label intervention")
    if wrong_drum["routing"]["drum_offset"] != 1 or wrong_drum["routing"]["pitched_offset"] != 0:
        raise ValueError("Invalid drum label intervention")
    for family in FAMILIES:
        unchanged = wrong_pitch if family == "percussion" else wrong_drum
        if _family_metric(unchanged,family) != _family_metric(trained,family):
            raise ValueError("Wrong-label intervention affected unrelated family")
    metrics = {}
    for family in FAMILIES:
        metrics[family] = {
            "initial_minus_trained":_pair_metric(initial,trained,family),
            "generic_minus_trained":_pair_metric(generic,trained,family),
            "wrong_minus_matched":_pair_metric(wrong_drum if family=="percussion" else wrong_pitch,trained,family),
        }
    target = trained["mix_features"]["objective"]
    return {"family_note_errors":metrics,
            "mix_feature_error":target,
            "initial_minus_trained":initial["mix_features"]["objective"]-target,
            "generic_minus_trained":generic["mix_features"]["objective"]-target,
            # Removing a family restores exact reference notes. A negative
            # result is expected when procedural substitution reduces error.
            "ablation_minus_all_learned":{name:report["mix_features"]["objective"]-target
                                         for name,report in ablations.items()}}


def _aggregate(replicates):
    result = {}
    for name in FAMILIES:
        result[name] = {}
        for key in ("initial_minus_trained","generic_minus_trained","wrong_minus_matched"):
            values = []
            for replicate in replicates:
                observations = [row["summary"]["family_note_errors"][name][key] for row in replicate["cases"]]
                observations = [x for x in observations if x is not None]
                if not observations:
                    raise ValueError("Empty multi-model family replicate")
                values.append(statistics.fmean(observations))
            result[name][key] = {"mean":statistics.fmean(values),"min":min(values),"max":max(values),
                                 "positive_replicates":sum(x>0 for x in values)}
    return result


def run_study(plan=None):
    plan = StudyPlan() if plan is None else plan
    if not isinstance(plan,StudyPlan):
        raise ValueError("Expected multi-model study plan")
    sources = Sources.build(plan)
    cases = panel(plan,sources)
    generic = tuple(_compare(score,"generic",plan,sources) for _,score,_ in cases)
    replicates = []
    for seed in plan.short.model_seeds:
        models = initialize(plan,seed)
        initial = tuple(_compare(score,"learned",plan,sources,models) for _,score,_ in cases)
        training = train_all(models,sources,plan)
        results = []
        for index,(case,score,coverage) in enumerate(cases):
            trained = _compare(score,"learned",plan,sources,models)
            ablations = {family:_compare(score,"learned",plan,sources,models,
                                         families=tuple(f for f in FAMILIES if f!=family))
                         for family in FAMILIES}
            wrong_pitch = _compare(score,"learned",plan,sources,models,pitched_offset=1)
            wrong_drum = _compare(score,"learned",plan,sources,models,drum_offset=1)
            summary = summarize(generic[index],initial[index],trained,ablations,wrong_pitch,wrong_drum)
            results.append({"case":asdict(case),"coverage":coverage,"initial":initial[index],
                            "trained":trained,"ablations":ablations,"wrong_pitch":wrong_pitch,
                            "wrong_drum":wrong_drum,"summary":summary})
        replicates.append({"seed":seed,"training":training,"cases":results})
    return {"protocol":PROTOCOL,"plan":asdict(plan),"sources":sources.receipt(),
            "generic_controls":generic,"replicates":replicates,
            "family_summary":_aggregate(replicates),
            "evaluation_used_for_training":False,"model_weights_published":False,
            "external_recordings_used":False,"perceptual_quality_established":False,
            "generated_audio_written":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true",help="Use smaller optimizer budgets for a smoke experiment")
    args=parser.parse_args(argv)
    plan=StudyPlan()
    if args.quick:
        plan=replace(plan,short=replace(plan.short,updates=12),
                     sustained=replace(plan.sustained,updates=12),
                     percussion=replace(plan.percussion,updates=12))
    try:
        print(json.dumps(run_study(plan),sort_keys=True,allow_nan=False))
    except (ValueError,RuntimeError) as exc:
        parser.exit(2,f"Multi-model score study failed: {exc}\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
