"""Module for the tianshou agent."""
from typing import Any, Callable, Optional, Tuple, Type, Union

import numpy as np
import torch
from numba import njit
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy, PPOPolicy, DQNPolicy, DiscreteSACPolicy


@njit
def _gae_return(
    v_s: np.ndarray,
    v_s_: np.ndarray,
    rew: np.ndarray,
    end_flag: np.ndarray,
    dt: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    returns = np.zeros(rew.shape)
    delta = rew + v_s_ * (gamma**dt) - v_s
    discount = (1.0 - end_flag) * ((gamma**dt) * gae_lambda)
    gae = 0.0
    for i in range(len(rew) - 1, -1, -1):
        gae = delta[i] + discount[i] * gae
        returns[i] = gae
    return returns



@njit
def _nstep_return(
    rew: np.ndarray,
    end_flag: np.ndarray,
    target_q: np.ndarray,
    indices: np.ndarray,
    dt: np.ndarray,
    gamma: float,
    n_step: int,
) -> np.ndarray:
    #if n_step == 1:
    indices = indices[0]
    bsz = target_q.shape[0]
    target_q = rew[indices].reshape(bsz, 1) + (1.0- end_flag[indices].reshape(bsz, 1)) * (gamma ** dt[indices].reshape(bsz, 1)) * target_q.reshape(bsz, 1)
    return target_q
    # else:
    #     # n_rewards = np.zeros_like(rew[indices])
    #     # max_dts = np.zeros_like(dt[indices])
    #     # mask = np.ones_like(rew[indices]) #TODO
    #     # for t in range(max_seq_length):
    #     #     n_rewards[:, t, :] = ((mask * rew)[:, t:t + n_step, :] * (gamma ** dt)[:,
    #     #                                                                     t:t + n_step,
    #     #                                                                     :]).sum(dim=1)
    #     #     max_dts[:, t, :] = np.sum(dt[:, t:t + n_step, :], dim=1)

    #     # steps = mask.flip(1).cumsum(dim=1).flip(1).clamp_max(n_step).long()

    #     # indices = np.linspace(0,
    #     #                          max_seq_length - 1,
    #     #                          steps=max_seq_length).long()

    #     # n_targets_terminated = np.take(target_q * (1 - end_flag),
    #     #                                     dim=1,
    #     #                                     index=steps + indices - 1)
    #     # targets = n_rewards + (gamma ** max_dts) * n_targets_terminated

    #     # return targets
    #     #TODO dt
    #     gamma_buffer = np.ones(n_step + 1)
    #     for i in range(1, n_step + 1):
    #         gamma_buffer[i] = gamma_buffer[i - 1] * gamma
    #     target_shape = target_q.shape
    #     bsz = target_shape[0]
    #     # change target_q to 2d array
    #     target_q = target_q.reshape(bsz, -1)
    #     returns = np.zeros(target_q.shape)
    #     gammas = np.full(indices[0].shape, n_step)
    #     for n in range(n_step - 1, -1, -1):
    #         now = indices[n]
    #         gammas[end_flag[now] > 0] = n + 1
    #         returns[end_flag[now] > 0] = 0.0
    #         returns = rew[now].reshape(bsz, 1) + gamma * returns
    #     target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
    #     return target_q.reshape(target_shape)

class SemiDQNPolicy(DQNPolicy):
    @staticmethod
    def compute_nstep_return(
        batch: Batch,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> Batch:
        r"""Compute n-step return for Q-learning targets.
        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})
        where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
        :math:`d_t` is the done flag of step :math:`t`.
        :param Batch batch: a data batch, which is equal to buffer[indice].
        :param ReplayBuffer buffer: the data buffer.
        :param function target_q_fn: a function which compute target Q value
            of "obs_next" given data buffer and wanted indices.
        :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param int n_step: the number of estimation step, should be an int greater
            than 0. Default to 1.
        :param bool rew_norm: normalize the reward to Normal(0, 1), Default to False.
        :return: a Batch. The result will be stored in batch.returns as a
            torch.Tensor with the same shape as target_q_fn's return tensor.
        """
        assert not rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        rew = buffer.rew
        dt = buffer.info["dt"]
        bsz = len(indice)
        indices = [indice]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indice',
        # and are truncated at the end of each episode
        terminal = indices[-1]
        with torch.no_grad():
            target_q_torch = target_q_fn(buffer, terminal)  # (bsz, ?)
        target_q = to_numpy(target_q_torch.reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        target_q = _nstep_return(rew, end_flag, target_q, indices, dt, gamma, n_step)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch
    
    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        i64 = np.array([[0, 1]], dtype=np.int64)
        i642 = np.array([0, 1], dtype=np.int64)
        _gae_return(f64, f64, f64, b, i642, 0.1, 0.1)
        _gae_return(f32, f32, f64, b, i642, 0.1, 0.1)
        _nstep_return(f64, b, f32.reshape(-1, 1), i64, i642 ,0.1, 1)


class   SemiPPOPolicy(PPOPolicy):
    """Semi-markov PPO."""
    
    
    def __init__(self, actor: torch.nn.Module, critic: torch.nn.Module, optim: torch.optim.Optimizer, dist_fn: Type[torch.distributions.Distribution], eps_clip: float = 0.2, dual_clip: Optional[float] = None, value_clip: bool = False, advantage_normalization: bool = True, recompute_advantage: bool = False,  **kwargs: Any) -> None:
        super().__init__(actor, critic, optim, dist_fn, eps_clip, dual_clip, value_clip, advantage_normalization, recompute_advantage, **kwargs)

    @staticmethod
    def compute_episodic_return(
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray,
        v_s_: Optional[Union[np.ndarray, torch.Tensor]] = None,
        v_s: Optional[Union[np.ndarray, torch.Tensor]] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute returns over given batch.

        Use Implementation of Generalized Advantage Estimator (arXiv:1506.02438)
        to calculate q/advantage value of given batch.

        :param batch: a data batch which contains several episodes of data in
            sequential order. Mind that the end of each finished episode of batch
            should be marked by done flag, unfinished (or collecting) episodes will be
            recognized by buffer.unfinished_index().
        :param indices: tell batch's location in buffer, batch is equal
            to buffer[indices].
        :param v_s_: the value function of all next states :math:`V(s')`.
        :param gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param gae_lambda: the parameter for Generalized Advantage Estimation,
            should be in [0, 1]. Default to 0.95.
        :param buffer: the buffer from which the batch originates
        :param v_s: the value function of all states.

        :return: two numpy arrays (returns, advantage) with each shape (bsz, ).
        """
        rew = batch.rew
        dt = batch.info["dt"]
        if len(dt.shape) > 1:
            dt = dt[:, -1]  # this is because infos get stacked, we only want the last time stamp
        if v_s_ is None:
            assert np.isclose(gae_lambda, 1.0)
            v_s_ = np.zeros_like(rew)
        else:
            v_s_ = to_numpy(v_s_.flatten())
            v_s_ = v_s_ * BasePolicy.value_mask(buffer, indices)
        v_s = np.roll(v_s_, 1) if v_s is None else to_numpy(v_s.flatten())

        end_flag = np.logical_or(batch.terminated, batch.truncated)
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        advantage = _gae_return(v_s, v_s_, rew, end_flag, dt, gamma, gae_lambda)
        returns = advantage + v_s
        # normalization varies from each policy, so we don't do it here
        return returns, advantage
    #TODO
    
    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        # with torch.no_grad():
        #     batch.logp_old = self(batch).dist.log_prob(batch.act)
        #TODO
        old_log_prob = []
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                old_log_prob.append(self(minibatch).dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)    
        
        return batch

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        i64 = np.array([[0, 1]], dtype=np.int64)
        i642 = np.array([0, 1], dtype=np.int64)
        _gae_return(f64, f64, f64, b, i642, 0.1, 0.1)
        _gae_return(f32, f32, f64, b, i642, 0.1, 0.1)
        _nstep_return(f64, b, f32.reshape(-1, 1), i64, i642 ,0.1, 1)

class SemiDiscreteSACPolicy(DiscreteSACPolicy):
    @staticmethod
    def compute_nstep_return(
        batch: Batch,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> Batch:
        r"""Compute n-step return for Q-learning targets.
        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})
        where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
        :math:`d_t` is the done flag of step :math:`t`.
        :param Batch batch: a data batch, which is equal to buffer[indice].
        :param ReplayBuffer buffer: the data buffer.
        :param function target_q_fn: a function which compute target Q value
            of "obs_next" given data buffer and wanted indices.
        :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param int n_step: the number of estimation step, should be an int greater
            than 0. Default to 1.
        :param bool rew_norm: normalize the reward to Normal(0, 1), Default to False.
        :return: a Batch. The result will be stored in batch.returns as a
            torch.Tensor with the same shape as target_q_fn's return tensor.
        """
        assert not rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        rew = buffer.rew
        dt = buffer.info["dt"]
        bsz = len(indice)
        indices = [indice]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indice',
        # and are truncated at the end of each episode
        terminal = indices[-1]
        with torch.no_grad():
            target_q_torch = target_q_fn(buffer, terminal)  # (bsz, ?)
        target_q = to_numpy(target_q_torch.reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        target_q = _nstep_return(rew, end_flag, target_q, indices, dt, gamma, n_step)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        i64 = np.array([[0, 1]], dtype=np.int64)
        i642 = np.array([0, 1], dtype=np.int64)
        _gae_return(f64, f64, f64, b, i642, 0.1, 0.1)
        _gae_return(f32, f32, f64, b, i642, 0.1, 0.1)
        _nstep_return(f64, b, f32.reshape(-1, 1), i64, i642 ,0.1, 1)
