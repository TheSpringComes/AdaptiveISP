"""CPU regression checks; no FiveK files or LPIPS download required.

Run: python -m unittest debug.regression.test_human_v31_audit -v
"""
import contextlib
import io
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from controller.adaptiveisp import AdaptiveISPController
from controller.adaptiveisp.human_reward import HumanReward, StepwiseHumanReward
from engine.trainer_human import HumanTrainer
from engine.util import load_config
from isp.registry import build_operator
from pipeline import PipelineExecutor
from pipeline.action import ISPAction
from search import SearchSpace
from search.priors.action_mask import build_from_config


def fake_quality(image, target, **kwargs):
    q = image.mean((1, 2, 3)).reshape(-1, 1)
    return q, dict(quality=q, ssim=q, lpips=1 - q, lab_ab=2 * (1 - q))


class HumanAuditTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.names = ['whitebalance', 'n_awb', 'sharpen', 'denoise', 'saturation', 'ccm']
        self.ops = {n: build_operator(n) for n in self.names}
        self.runtime = PipelineExecutor(self.ops, self.names)

    def test_config_priors_and_family_ablation(self):
        cfg = load_config('configs/adaptiveisp_human_v31_fixed.yaml')
        # The audit only requires that the setting is explicit and configurable;
        # experiments may choose a small ablation value.
        self.assertIn('stop_bonus_beta', cfg)
        prior = build_from_config(cfg.action_mask, self.names)
        space = SearchSpace(self.ops, self.names, [prior])
        state = self.runtime.initial_state(torch.zeros(2, 3, 16, 16))
        state.op_usage[0, 0] = 1
        state.op_usage[1, 1] = 1
        mask = space.valid_actions(state).op_mask
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[0, 1])  # no family is configured in the base
        self.assertTrue(mask[1, 0])
        self.assertFalse(mask[1, 1])
        state.op_usage[0, 2] = 1
        state.op_usage[1, 4] = 1
        mask = space.valid_actions(state).op_mask
        self.assertFalse(mask[0, 3])  # sharpen -> forbid denoise
        self.assertTrue(mask[1, 3])
        self.assertFalse(mask[1, [0, 1, 5]].any())
        family = build_from_config({'group_budget': {'enabled': True, 'groups': [
            {'name': 'awb', 'ops': ['whitebalance', 'n_awb'], 'max_select': 1}
        ]}}, self.names)
        mask = SearchSpace(self.ops, self.names, [prior, family]).valid_actions(state).op_mask
        self.assertFalse(mask[:, :2].any())

    @patch('controller.adaptiveisp.human_reward.quality_score', side_effect=fake_quality)
    def test_stop_gating_defaults_and_batch_sign(self, _):
        baseline = torch.full((2, 3, 16, 16), .5)
        before = self.runtime.initial_state(torch.stack([
            torch.full((3, 16, 16), .7), torch.full((3, 16, 16), .29)]))
        before.step[:] = 3
        action = ISPAction(torch.zeros(2, dtype=torch.long), torch.zeros(2, 1),
                           torch.ones(2, dtype=torch.bool))
        after = self.runtime.step(before, action)
        q_before, _ = fake_quality(before.image, baseline)
        kwargs = dict(image_initial=baseline, target=baseline, state_before=before,
                      action=action, state_after=after, q_initial=torch.full((2, 1), .5),
                      q_before=q_before)
        reward = StepwiseHumanReward(n_ops=6, max_steps=8, quality_scale=5)
        r, bd, _ = reward.compute(**kwargs)
        self.assertTrue(torch.equal(r, torch.zeros_like(r)))
        self.assertTrue(torch.equal(bd.task_delta, torch.zeros_like(r)))
        reward.beta = 1
        r, bd, _ = reward.compute(**kwargs)
        self.assertGreater(r[0].item(), 0)
        self.assertEqual(r[1].item(), 0)  # below baseline never earns bonus
        task_total = 5 * (q_before - .5)
        self.assertLess(task_total.mean().item(), 0)
        self.assertGreater((task_total + r).mean().item(), 0)  # old batch sign reversal
        before.step[:] = 7
        _, bd, _ = reward.compute(**kwargs)
        self.assertEqual(bd.stop_bonus.sum().item(), 0)  # forced STOP gets no bonus

    def test_masked_pdf_entropy_and_minimum_stop_train_eval(self):
        ctrl = AdaptiveISPController(self.ops, self.names, obs_hw=64, mid_channels=8,
                                     feature_dim=32, fc1_size=16, dropout_keep_prob=1,
                                     min_rollout_length=3, max_steps=8)
        state = self.runtime.initial_state(torch.zeros(2, 3, 64, 64))
        state.step[:] = torch.tensor([2, 3])
        c = SearchSpace(self.ops, self.names).valid_actions(state)
        c.op_mask[:, 1:] = False
        for training in (True, False):
            ctrl.train(training)
            out = ctrl.act(state, c)
            torch.testing.assert_close(out.pdf.sum(1), torch.ones(2))
            self.assertEqual(out.pdf[0, -1].item(), 0)
            self.assertGreater(out.pdf[1, -1].item(), 0)
            self.assertFalse(out.pdf[:, 1:-1].any())
            logp, _, ent = ctrl.evaluate(state, c, out.action.op_indices, out.action.is_stop)
            torch.testing.assert_close(logp, out.log_prob.flatten())
            torch.testing.assert_close(ent, out.entropy.flatten())
            self.assertLessEqual(ent[1].item(), math.log(2) + 1e-6)
        win = dict(entropy_sum=0., entropy_max_sum=0., entropy_norm_sum=0., entropy_count=0)
        HumanTrainer._accumulate_entropy(win, out, state, 8)
        self.assertEqual(win['entropy_count'], 2)
        self.assertAlmostEqual(win['entropy_max_sum'], math.log(2), places=6)
        self.assertTrue(math.isfinite(win['entropy_norm_sum']))
        state.stopped[0] = True
        state.step[1] = 7
        HumanTrainer._accumulate_entropy(win, out, state, 8)
        self.assertEqual(win['entropy_count'], 2)

    def test_terminal_entropy_gap(self):
        before = self.runtime.initial_state(torch.zeros(2, 3, 16, 16))
        action = ISPAction(torch.zeros(2, dtype=torch.long), torch.zeros(2, 1),
                           torch.zeros(2, dtype=torch.bool))
        reward = HumanReward(n_ops=6, max_steps=8)
        ent_max = torch.tensor([0., math.log(2)])
        _, bd, _ = reward.compute(before.image, before.image, before, action, before,
                                  entropy=ent_max, entropy_max=ent_max)
        torch.testing.assert_close(bd.entropy_penalty, torch.zeros(2, 1))

    def test_paired_validation_weights_and_uneven_batches(self):
        trainer = HumanTrainer.__new__(HumanTrainer)
        trainer.cfg = SimpleNamespace(test_steps=3)
        trainer.device = torch.device('cpu')
        trainer.runtime = self.runtime
        trainer.search_space = SearchSpace(self.ops, self.names)
        trainer.writer = SimpleNamespace(add_scalar=lambda *a, **k: None)
        trainer.task_model = SimpleNamespace(lambda_ssim=1., lambda_lpips=.5, lambda_lab_ab=.1,
            compute_metrics=lambda image, target: fake_quality(image, target)[1])
        calls = []
        def front(image, metadata):
            calls.append(metadata['camera_id'].tolist())
            return image + .1
        trainer.front_isp = front
        # A deterministic one-op tail, followed by learned STOP. This tests
        # validation aggregation without depending on network initialization.
        class Controller(torch.nn.Module):
            def act(self, state, constraint):
                return SimpleNamespace(action=ISPAction(torch.zeros(state.batch_size, dtype=torch.long),
                    torch.ones(state.batch_size, 1), state.step >= 1),
                    pdf=torch.nn.functional.one_hot(torch.zeros(state.batch_size, dtype=torch.long), 7).float())
        trainer.controller = Controller()
        original_step = trainer.runtime.step
        def step(state, action):
            # Execute an identity STOP / fixed +.05 test operator.
            after = original_step(state, ISPAction(action.op_indices, action.params,
                                                   torch.ones_like(action.is_stop)))
            after.image = state.image + (~action.is_stop).view(-1, 1, 1, 1) * .05
            after.stopped = action.is_stop
            return after
        trainer.runtime.step = step
        trainer.val_loader = [(torch.full((b, 3, 16, 16), x), torch.zeros(b, 3, 16, 16),
                               torch.arange(b)) for b, x in [(2, .2), (1, .5)]]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            metrics = trainer._run_val(0)
        self.assertEqual(len(calls), 2)  # once per batch, same baseline used for both branches
        self.assertAlmostEqual(metrics['val/front/quality'], .4, places=6)
        self.assertAlmostEqual(metrics['val/quality'], .45, places=6)
        self.assertAlmostEqual(metrics['val/delta/quality'], .05, places=6)
        self.assertEqual(metrics['val/improved_q'], 3)
        self.assertEqual(metrics['val/degraded_q'], 0)
        self.assertEqual(metrics['val/mean_length'], 1)
        self.assertEqual(metrics['val/pct_learned_stop'], 1)
        self.assertIn('Front only', output.getvalue())
        self.assertTrue(trainer.controller.training)
        terms = trainer._q_terms(dict(ssim=torch.tensor(.8), lpips=torch.tensor(.2), lab_ab=torch.tensor(7.)))
        self.assertAlmostEqual(terms['ssim'].item(), .8, places=6)
        self.assertAlmostEqual(terms['lpips'].item(), -.1, places=6)
        self.assertAlmostEqual(terms['lab_ab'].item(), -.7, places=6)


if __name__ == '__main__':
    unittest.main()
