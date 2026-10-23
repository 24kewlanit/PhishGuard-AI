"""
test_ml_models.py
=================
Tests for the machine-learning part of Email / PDF detection.

Run:   python -m unittest test_ml_models -v

What is checked
  * the shipped email_model.pkl / pdf_model.pkl load and match the code
  * the ML can only RAISE concern: it can never hide a rule finding, never turn
    an INCONCLUSIVE analysis into SAFE, and on its own it can only reach SUSPICIOUS
  * the real trained models do not create false alarms on the legitimate samples
    and do not weaken any phishing/malicious sample
  * the training-data generator is deterministic and the training pipeline is reproducible
  * a quick leave-one-family-out guard (train without a scam type, test on it)

IMPORTANT: the models are trained on SYNTHETIC data. Passing these tests does not
mean real-world accuracy - see evaluate_models.py / model_evaluation.json.
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import email_pdf_fixtures as F
import evaluate_models as E
import make_training_data as M
import pdf_analyzer
from email_analyzer import normalize_text, parse_eml_bytes, parse_email_text
from email_model import ML_MIN_PROB, SEVERITY, load_email_model, predict_email
from pdf_analyzer import analyze_pdf
from pdf_model import PDF_FEATURE_NAMES, PDF_ML_MIN_PROB, load_pdf_model

EMAIL_MODEL = load_email_model()
PDF_MODEL = load_pdf_model()
BASE = os.path.dirname(os.path.abspath(__file__))


class FakeEmailModel:
    """Returns fixed class probabilities (order: phishing, safe, spam)."""
    classes_ = ["phishing", "safe", "spam"]

    def __init__(self, phishing=0.0, safe=1.0, spam=0.0):
        self.p = [[phishing, safe, spam]]

    def predict_proba(self, X):
        return self.p


class FakePdfModel:
    classes_ = ["malicious", "safe", "suspicious"]

    def __init__(self, malicious=0.0, safe=1.0, suspicious=0.0):
        self.p = [[malicious, safe, suspicious]]

    def predict_proba(self, X):
        return self.p


def pdf_with(model, data):
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.write(fd, data)
    os.close(fd)
    try:
        with mock.patch.object(pdf_analyzer, "_ML_MODEL", model):
            return analyze_pdf(path)
    finally:
        os.remove(path)


CLEAN = parse_eml_bytes(F.EML_NORMAL.encode())
NO_SENDER = parse_email_text("Please review the quarterly report when you can, thank you very much.")
SUSP = parse_eml_bytes(F.EML_REPLYTO_MISMATCH.encode())          # rules say SUSPICIOUS (25-59)
PHISH = parse_eml_bytes(F.EML_PHISHING_PASSWORD.encode())        # rules say PHISHING


class ShippedModels(unittest.TestCase):

    def test_model_files_load_and_match_the_code(self):
        self.assertIsNotNone(EMAIL_MODEL, "email_model.pkl missing - run: python train_email_model.py")
        self.assertIsNotNone(PDF_MODEL, "pdf_model.pkl missing - run: python train_pdf_model.py --csv dataset/pdf_features.csv")
        self.assertEqual(sorted(EMAIL_MODEL.classes_), ["phishing", "safe", "spam"])
        self.assertEqual(sorted(PDF_MODEL.classes_), ["malicious", "safe", "suspicious"])
        self.assertEqual(PDF_MODEL.n_features_in_, len(PDF_FEATURE_NAMES))
        self.assertEqual(list(PDF_MODEL.feature_names_in_), PDF_FEATURE_NAMES)

    def test_datasets_exist_and_are_labelled(self):
        e = pd.read_csv(os.path.join(BASE, "dataset", "email_dataset.csv"))
        self.assertEqual(set(e["label"]), {"safe", "spam", "phishing"})
        self.assertGreater(len(e), 5000)
        p = pd.read_csv(os.path.join(BASE, "dataset", "pdf_features.csv"))
        self.assertEqual(set(p["label"]), {"safe", "suspicious", "malicious"})
        self.assertTrue(set(PDF_FEATURE_NAMES) <= set(p.columns))

    def test_models_give_valid_probabilities(self):
        x = parse_eml_bytes(F.EML_PHISHING_PASSWORD.encode())["model_text"]
        pr = EMAIL_MODEL.predict_proba([x])[0]
        self.assertAlmostEqual(float(pr.sum()), 1.0, places=5)

    def test_models_learned_something_on_in_distribution_data(self):
        # sanity only - this is NOT real-world accuracy
        a = predict_email(parse_eml_bytes(F.EML_PHISHING_PASSWORD.encode()), EMAIL_MODEL)
        self.assertTrue(a["ml_used"])
        self.assertGreater(a["probabilities"]["phishing"], a["probabilities"]["safe"])
        b = predict_email(CLEAN, EMAIL_MODEL)
        self.assertGreater(b["probabilities"]["safe"], b["probabilities"]["phishing"])


class MlCanOnlyRaiseConcern(unittest.TestCase):

    def test_ml_alone_reaches_suspicious_never_phishing(self):
        r = predict_email(CLEAN, FakeEmailModel(phishing=0.99, safe=0.005, spam=0.005))
        self.assertEqual(r["classification"], "SUSPICIOUS")
        self.assertIn("only the machine-learning model", " ".join(r["reasons"]).lower())
        self.assertEqual(r["risk_score"], 30)

    def test_ml_agreeing_with_suspicious_rules_promotes_to_phishing(self):
        self.assertEqual(predict_email(SUSP, None)["classification"], "SUSPICIOUS")
        r = predict_email(SUSP, FakeEmailModel(phishing=0.95, safe=0.04, spam=0.01))
        self.assertEqual(r["classification"], "PHISHING")
        self.assertGreaterEqual(r["risk_score"], 60)

    def test_low_confidence_ml_changes_nothing(self):
        for p in (0.5, 0.7, ML_MIN_PROB - 0.01):
            r = predict_email(CLEAN, FakeEmailModel(phishing=p, safe=1 - p, spam=0.0))
            self.assertEqual(r["classification"], "SAFE", p)

    def test_confident_spam_only_raises_safe(self):
        r = predict_email(CLEAN, FakeEmailModel(phishing=0.02, safe=0.03, spam=0.95))
        self.assertEqual(r["classification"], "SPAM")

    def test_ml_never_lowers_a_rule_finding(self):
        safe_model = FakeEmailModel(phishing=0.0, safe=1.0, spam=0.0)
        for parsed in (PHISH, SUSP, parse_eml_bytes(F.EML_ATTACHMENT_EXE.encode()),
                       parse_eml_bytes(F.EML_IMPERSONATION.encode())):
            rules = predict_email(parsed, None)["classification"]
            with_ml = predict_email(parsed, safe_model)["classification"]
            self.assertGreaterEqual(SEVERITY[with_ml], SEVERITY[rules], (rules, with_ml))

    def test_ml_never_turns_inconclusive_into_safe(self):
        self.assertEqual(predict_email(NO_SENDER, None)["classification"], "INCONCLUSIVE")
        r = predict_email(NO_SENDER, FakeEmailModel(phishing=0.0, safe=1.0, spam=0.0))
        self.assertEqual(r["classification"], "INCONCLUSIVE")
        self.assertIsNone(r["risk_score"])

    def test_unparseable_input_skips_the_model(self):
        bad = parse_email_text("\x00\x01\x02\x03\x04\x05\x06\x07\x08 " * 20)
        self.assertFalse(bad["parse_ok"])
        r = predict_email(bad, FakeEmailModel(phishing=0.0, safe=1.0, spam=0.0))
        self.assertEqual(r["classification"], "INCONCLUSIVE")
        self.assertFalse(r["ml_used"])

    def test_model_disagreement_is_disclosed_not_used_to_downgrade(self):
        r = predict_email(PHISH, FakeEmailModel(phishing=0.0, safe=1.0, spam=0.0))
        self.assertEqual(r["classification"], "PHISHING")
        self.assertIn("did not flag", " ".join(r["reasons"]))

    # ---- PDF ----
    def test_pdf_ml_alone_only_reaches_suspicious(self):
        r = pdf_with(FakePdfModel(malicious=0.99, safe=0.01, suspicious=0.0), F.text_pdf(F.NORMAL_TEXT))
        self.assertEqual(r["classification"], "SUSPICIOUS")
        self.assertIn("only the machine-learning model", " ".join(r["reasons"]).lower())

    def test_pdf_low_confidence_ml_changes_nothing(self):
        r = pdf_with(FakePdfModel(malicious=0.5, safe=0.1, suspicious=0.4), F.text_pdf(F.NORMAL_TEXT))
        self.assertEqual(r["classification"], "SAFE")
        self.assertLess(PDF_ML_MIN_PROB, 1.0)

    def test_pdf_ml_never_lowers_or_hides(self):
        safe_model = FakePdfModel(malicious=0.0, safe=1.0, suspicious=0.0)
        r = pdf_with(safe_model, F.text_pdf(F.PHISHING_TEXT, "flate"))
        self.assertEqual(r["classification"], "POTENTIALLY MALICIOUS")
        r = pdf_with(safe_model, F.objstm_javascript_pdf())
        self.assertIn(r["classification"], {"SUSPICIOUS", "POTENTIALLY MALICIOUS"})

    def test_pdf_ml_never_turns_inconclusive_into_safe(self):
        safe_model = FakePdfModel(malicious=0.0, safe=1.0, suspicious=0.0)
        for data in (F.image_only_pdf(), F.encrypted_pdf(F.NORMAL_TEXT), b"%PDF-1.4\ngarbage \x00\x01"):
            r = pdf_with(safe_model, data)
            self.assertEqual(r["classification"], "INCONCLUSIVE")
            self.assertIsNone(r["risk_score"])


class RealModelsOnIndependentSamples(unittest.TestCase):
    """Uses the actual shipped models on samples that are NOT produced by make_training_data.py."""

    def test_email_no_false_alarms_on_legitimate_samples(self):
        wrong = []
        for i, t in enumerate(E.INDEP_LEGIT):
            r = predict_email(parse_eml_bytes(t.encode()), EMAIL_MODEL)
            if r["classification"] != "SAFE":
                wrong.append((i, r["classification"], r["reasons"][:1]))
        self.assertEqual(wrong, [])

    def test_email_no_phishing_sample_is_missed(self):
        missed = []
        for i, t in enumerate(E.INDEP_PHISH):
            r = predict_email(parse_eml_bytes(t.encode()), EMAIL_MODEL)
            if r["classification"] not in ("SUSPICIOUS", "PHISHING"):
                missed.append((i, r["classification"]))
        self.assertEqual(missed, [])

    def test_ml_adds_to_rules_never_subtracts_on_every_sample(self):
        for t in E.INDEP_LEGIT + E.INDEP_PHISH + E.INDEP_SPAM:
            parsed = parse_eml_bytes(t.encode())
            rules = predict_email(parsed, None)["classification"]
            both = predict_email(parsed, EMAIL_MODEL)["classification"]
            self.assertGreaterEqual(SEVERITY[both], SEVERITY[rules])

    def test_pdf_real_model_on_independent_fixtures(self):
        safe, bad = E.indep_pdfs()
        for data in safe:
            self.assertEqual(pdf_with(PDF_MODEL, data)["classification"], "SAFE")
        for _label, data in bad:
            self.assertNotEqual(pdf_with(PDF_MODEL, data)["classification"], "SAFE")

    def test_ml_does_not_change_inconclusive_results(self):
        for data in (F.image_only_pdf(), F.encrypted_pdf(F.NORMAL_TEXT), F.corrupt_flate_pdf()):
            self.assertEqual(pdf_with(PDF_MODEL, data)["classification"], "INCONCLUSIVE")


class TrainingPipeline(unittest.TestCase):

    def test_generator_is_deterministic_and_complete(self):
        a = M.make_email_rows(3, 3, 3, seed=5)
        b = M.make_email_rows(3, 3, 3, seed=5)
        self.assertEqual(a, b)
        self.assertEqual({r["label"] for r in a}, {"safe", "spam", "phishing"})
        self.assertEqual(len({r["family"] for r in a}), len(M.SAFE_FAMILIES) + len(M.SPAM_FAMILIES) + len(M.PHISH_FAMILIES))
        self.assertNotEqual(a, M.make_email_rows(3, 3, 3, seed=6))

    def test_generated_text_has_the_same_format_the_app_feeds_the_model(self):
        for row in M.make_email_rows(1, 1, 1, seed=2):
            self.assertTrue(row["text"].startswith("sender: "))
            self.assertIn(" subject: ", row["text"])
            self.assertIn(" body: ", row["text"])
            self.assertEqual(row["text"], row["text"].lower())

    def test_retraining_from_scratch_works_and_predicts(self):
        rows = M.make_email_rows(40, 30, 30, seed=3)
        df = pd.DataFrame(rows)
        pipe = E.email_pipeline().fit(df["text"].map(normalize_text), df["label"])
        probe = parse_eml_bytes(F.EML_PHISHING_PASSWORD.encode())["model_text"]
        self.assertEqual(pipe.predict([probe])[0], "phishing")

    def test_leave_one_family_out_guard(self):
        """Train WITHOUT a scam type, test on it. Loose bounds - guards against the pipeline silently breaking."""
        df = pd.read_csv(os.path.join(BASE, "dataset", "email_dataset.csv"))
        df["x"] = df["text"].map(normalize_text)
        for fam, floor, kind in (("phishing/credentials", 0.90, "recall"), ("phishing/bank_kyc", 0.90, "recall"),
                                 ("safe/workplace", 0.95, "specificity")):
            train, test = df[df.family != fam], df[df.family == fam]
            pred = E.email_pipeline().fit(train.x, train.label).predict(test.x)
            score = float((pred == "safe").mean()) if kind == "specificity" else float((pred != "safe").mean())
            self.assertGreaterEqual(score, floor, f"{fam} {kind}={score:.3f}")

    def test_pdf_features_are_consistent_with_the_model(self):
        feats = pdf_analyzer.extract_pdf_findings
        from pdf_model import findings_to_features
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(F.text_pdf(F.PHISHING_TEXT, "flate"))
        try:
            f = findings_to_features(feats(fh.name))
        finally:
            os.remove(fh.name)
        self.assertEqual(set(f), set(PDF_FEATURE_NAMES))
        self.assertGreater(f["cred_count"], 0)
        self.assertGreater(f["urgent_count"], 0)
        self.assertGreater(f["payment_risk_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
