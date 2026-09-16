"""Probability replay and full-quality gradient regressions (CPU, no downloads)."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from controller.adaptiveisp import AdaptiveISPController
from engine.trainer_human import HumanTrainer
from isp.registry import build_operator
from pipeline import PipelineExecutor
from search import SearchSpace


class PolicyConsistencyTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        self.names = ['exposure', 'gamma']
        self.ops = {n: build_operator(n) for n in self.names}
        self.runtime = PipelineExecutor(self.ops, self.names)
        self.space = SearchSpace(self.ops, self.names)
        self.controller = AdaptiveISPController(
            self.ops, self.names, obs_hw=64, mid_channels=8,
            feature_dim=32, fc1_size=16, dropout_keep_prob=.5,
            deterministic_features=True, min_rollout_length=1, max_steps=4,
        )
        self.state = self.runtime.initial_state(torch.rand(3, 3, 64, 64))

    def test_no_update_probability_replay_is_batch_independent(self):
        c = self.controller.train()
        self.assertTrue(c.training)  # sample during training
        self.assertFalse(c.select_features.training)
        self.assertFalse(c.param_features.training)
        self.assertFalse(c.value_net.training)
        before = {n: b.clone() for n, b in c.named_buffers()}
        out = c.act(self.state, self.space.valid_actions(self.state))
        # PPO reshuffles states into a different minibatch composition.
        solo = self.runtime.initial_state(self.state.image[1:2])
        lp, _, _ = c.evaluate(solo, self.space.valid_actions(solo),
                             out.action.op_indices[1:2], out.action.is_stop[1:2])
        torch.testing.assert_close(lp, out.log_prob[1:2].flatten(), atol=1e-6, rtol=1e-6)
        for n, b in c.named_buffers():
            torch.testing.assert_close(b, before[n])
        (-out.log_prob.mean() + out.action.params.mean()).backward()
        self.assertGreater(c.select_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(c.param_heads['exposure'][-1].weight.grad.abs().sum().item(), 0)
        c.eval()
        evaluation = c.act(self.state, self.space.valid_actions(self.state))
        torch.testing.assert_close(out.pdf, evaluation.pdf)

    def test_forced_stop_and_external_stop_mask(self):
        self.state.step[:] = torch.tensor([0, 1, 3])
        constraint = self.space.valid_actions(self.state)
        constraint.stop_allowed[1] = False
        out = self.controller.act(self.state, constraint)
        self.assertEqual(out.pdf[0, -1].item(), 0)
        self.assertEqual(out.pdf[1, -1].item(), 0)
        self.assertTrue(out.action.is_stop[2])
        self.assertEqual(out.pdf[2, -1].item(), 1)
        self.assertEqual(out.entropy[2].item(), 0)
        lp, _, _ = self.controller.evaluate(self.state, constraint,
                                           out.action.op_indices, out.action.is_stop)
        torch.testing.assert_close(lp, out.log_prob.flatten())
        self.assertEqual(lp[2].item(), 0)
        constraint.op_mask[0] = False
        with self.assertRaisesRegex(ValueError, 'No valid action'):
            self.controller.act(self.state, constraint)

    def test_routing_distinguishes_entropy_from_input_dependence(self):
        same = HumanTrainer._routing_stats(torch.full((4, 2), .5))
        self.assertEqual(same['input_js'], 0)
        self.assertEqual(same['dominant_share'], 1)
        different = HumanTrainer._routing_stats(torch.eye(2))
        self.assertAlmostEqual(different['input_js'], .693147, places=5)
        self.assertEqual(different['unique_argmax'], 2)
        self.assertEqual(different['dominant_share'], .5)

    def test_policy_summary_excludes_forced_and_padded_stops(self):
        from controller.adaptiveisp.reward import RewardBreakdown
        from engine.base_trainer import BaseTrainer
        self.state.step[:] = torch.tensor([1, 3, 2])
        self.state.stopped[2] = True
        out = self.controller.act(self.state, self.space.valid_actions(self.state))
        zeros = torch.zeros(3, 1)
        breakdown = RewardBreakdown(task_delta=zeros, overflow_penalty=zeros,
            entropy_penalty=zeros, usage_penalty=zeros, early_stop_penalty=zeros,
            runtime_penalty=zeros, total=zeros)
        after = self.runtime.step(self.state, out.action)
        win = BaseTrainer._make_window(n_ops=2)
        BaseTrainer._accumulate_window(None, win, breakdown, out.entropy, out,
            after, 2, 4, policy_active=torch.tensor([True, False, False]))
        self.assertEqual(win['pdf_n'], 1)
        self.assertEqual(win['argmax_seen'], 1)
        self.assertEqual(win['n_stop'], int(out.action.is_stop[0]))
        torch.testing.assert_close(torch.from_numpy(win['pdf_sum']).float(), out.pdf[0])

    def test_full_quality_value_and_gradient(self):
        t = HumanTrainer.__new__(HumanTrainer)
        t.task_model = SimpleNamespace(lambda_ssim=1., lambda_lpips=.5,
                                       lambda_lab_ab=.1, lpips_net='vgg')
        t.reward_fn = SimpleNamespace(alpha=5.)
        image = torch.tensor([.2, .4]).reshape(2, 1, 1, 1).requires_grad_()
        q_initial = torch.tensor([[.1], [.3]], requires_grad=True)
        def ssim(x, target):
            return x.flatten(1).mean(1, keepdim=True)
        def lpips(x, target, net, grad=False):
            v = x.square().flatten(1).mean(1, keepdim=True)
            return v if grad else v.detach()
        def ab(x, target):
            return 3 * x.flatten(1).mean(1, keepdim=True)
        with patch('tasks.human_quality.metrics.ssim_batch', ssim), \
             patch('tasks.human_quality.metrics.lpips_batch', lpips), \
             patch('tasks.human_quality.metrics.lab_ab_loss', ab):
            loss = t._parameter_quality_loss(image, torch.zeros_like(image), q_initial)
        expected = -5 * (image.flatten() - .5 * image.flatten().square()
                         - .1 * 3 * image.flatten() - q_initial.detach().flatten()).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(image.grad.flatten(), -5 / 2 * (.7 - image.detach().flatten()))
        self.assertIsNone(q_initial.grad)


if __name__ == '__main__':
    unittest.main()
