import argparse
from datetime import datetime as time
from multiprocessing import Manager, Pool

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

plt.rcParams['figure.figsize'] = (2 * 4.5, 2 * 4)
plt.rcParams['figure.dpi'] = 60
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.frameon'] = True

# This script doesn't use any heated gridworld classes, but implements the same mechanic (made for debug).
# Global:
x_0 = 20
size = 40  # 2 * x_0, "real size" is size + 1, should be non-even
reward_step = -1
reward_target = 0
cutoff_time = 500


@jit(nopython=True)
def policy(Q_val, pos, eps):
    if np.random.uniform(0, 1) < 1 - eps:
        if Q_val[pos, 0] == Q_val[pos, 1]:
            return np.random.choice(2)
        else:
            return Q_val[pos].argmax()
    else:
        return np.random.choice(2)


@jit(nopython=True)
def do_action(pos, act, T, drift):
    acts = [act]
    if pos > x_0:
        acts.extend(np.random.choice(2, size=T))
    if pos > 0.75 * size and drift != 0:
        if np.random.uniform(0, 1) < drift:
            acts.append(0)
    new_pos = pos
    reward = reward_step
    for act in acts:
        if act == 0:
            new_pos -= 1
            if new_pos == 0:
                reward = reward_target
                return new_pos, reward
        if act == 1:
            new_pos += 1
            if new_pos == size:
                reward = reward_target
                return new_pos, reward
    return new_pos, reward


def run_world(agent, args, times_to_target, episodes_lock, played_episodes,
              common_policy, right_policy, policy_lock,
              densities, dens_lock,
              right_trend, performance, epsilon_log, trend_lock):
    algorithm = args.algorithm
    a = args.learning_rate
    mu = args.discounting
    eps = args.epsilon
    eps_factor = args.eps_factor
    learning_time = args.learning_time
    check_t = args.check_time
    T = args.T
    drift = args.drift
    greedy = args.greedy

    Q_val = np.zeros((size + 1, 2))
    double_Q_val = np.zeros((size + 1, 2))
    pos = x_0
    time_to_target = 0
    last_score = 0
    if algorithm == "MC":  # $\pi_R$ policy
        learning_time = 0
        Q_val[:x_0, 0] = 1.0  # if x < x_0 then left
        Q_val[x_0:, 1] = 1.0  # if x >= x_0 then right
    # ------------------- learning
    for t in range(learning_time):
        time_to_target += 1

        if algorithm == "Q_learning":
            act = policy(Q_val, pos, eps)
            new_pos, reward = do_action(pos, act, T, drift)
            Q_val[pos, act] += a * (reward + mu * Q_val[new_pos].max() - Q_val[pos, act])

        if algorithm == "Double_Q_learning":
            avg_Q = Q_val + double_Q_val
            act = policy(avg_Q, pos, eps)
            new_pos, reward = do_action(pos, act, T, drift)

            if np.random.choice(2) == 0:
                a_act = np.argmax(Q_val[new_pos, :])
                Q_val[pos, act] += a * (reward + mu * double_Q_val[new_pos, a_act].max() - Q_val[pos, act])
            else:
                b_act = np.argmax(double_Q_val[new_pos, :])
                double_Q_val[pos, act] += a * (reward + mu * Q_val[new_pos, b_act].max() - double_Q_val[pos, act])

        pos = new_pos

        if (t % check_t) == 0:
            with trend_lock:
                if np.isnan(performance[t]):
                    performance[t] = last_score
                    epsilon_log[t] = eps
                else:
                    performance[t] += last_score
                    epsilon_log[t] += eps
                if Q_val[x_0, 0] < Q_val[x_0, 1]:  # 1 = right action
                    if np.isnan(right_trend[t]):
                        right_trend[t] = 1
                    else:
                        right_trend[t] += 1

        if pos == 0 or pos == size:
            last_score = time_to_target
            time_to_target = 0
            pos = x_0
            eps = eps * eps_factor
            with episodes_lock:
                played_episodes.value += 1

    # ------------------- check
    Q_val = (Q_val + double_Q_val) / 2

    with policy_lock:
        for k in range(size + 1):
            if Q_val[k, 0] < Q_val[k, 1]:  # 1 - right action
                common_policy[k] += 1
            elif Q_val[k, 0] > Q_val[k, 1]:  # 0 - left action
                common_policy[k] -= 1
            if Q_val[k, 0] < Q_val[k, 1]:
                right_policy[k] += 1
    pos = x_0
    time_to_target = 0
    density = np.zeros(size + 1)

    while True:
        if greedy:
            act = policy(Q_val, pos, 0)
        else:
            act = policy(Q_val, pos, eps)

        pos, _ = do_action(pos, act, T, drift)
        time_to_target += 1
        density[pos] += 1

        if pos == 0 or pos == size:
            with dens_lock:
                for k in range(0, size + 1):
                    densities[k] += density[k]
            times_to_target[agent] = time_to_target
            return

        if time_to_target > cutoff_time:
            times_to_target[agent] = np.nan
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", help="default: Double_Q_learning",
                        choices=["Q_learning", "Double_Q_learning", "MC"],
                        default="Double_Q_learning")
    parser.add_argument("--learning_rate", help="default: 0.1", type=float, default=0.1)
    parser.add_argument("--learning_time", help="number of timeframes, default: 100_000", type=int, default=100_000)
    parser.add_argument("--T", help="temperature of the heated region, default: 3", type=int, choices=[0, 1, 2, 3],
                        default=3)
    parser.add_argument("--drift", help="drift probability, from 0.0 to 1.0, default: 0.3", type=float, default=0.3)
    parser.add_argument("--discounting", help="discounting constant, default: 0.9", type=float, default=0.9)
    parser.add_argument("--epsilon", help="epsilon-greediness, default: 1.0", type=float, default=1.0)
    parser.add_argument("--eps_factor", help="multiply eps by this factor after every played episode, default: 0.999",
                        type=float,
                        default=0.999, )
    parser.add_argument("--agents", help="independent learning agents, default: 12", type=int, default=12)
    parser.add_argument("--processes", help="for multiprocessing pool, default: 4", type=int, default=4)
    parser.add_argument("--check_time", help="check performance every ... timeframes, default: 2_000", type=int,
                        default=2_000)
    parser.add_argument("--greedy", help="use greedy policy when checking the final score, default: True", type=bool,
                        default=True)
    args = parser.parse_args()
    algorithm = args.algorithm
    a = args.learning_rate
    mu = args.discounting
    eps = args.epsilon
    learning_time = args.learning_time if algorithm != "MC" else 0
    check_t = args.check_time
    T = args.T
    drift = args.drift
    agents = args.agents
    processes = args.processes
    greedy = 'greedy' if args.greedy else 'non-greedy'

    print(f"MULTIPROCESSING HEATMAP 1D, processes: {processes}")
    print(args.__dict__)
    print(f"    started: {time.now().strftime('%H-%M-%S')}")
    with Manager() as manager, Pool(processes=processes) as pool:
        times_to_target = manager.list(np.zeros(agents, dtype=int))

        common_policy = manager.list(np.zeros(size + 1, dtype=int))
        right_policy = manager.list(np.zeros(size + 1, dtype=int))
        policy_lock = manager.Lock()

        densities = manager.list(np.zeros(size + 1, dtype=int))
        dens_lock = manager.Lock()

        right_trend = manager.list(np.full(learning_time, np.nan))
        trend_lock = manager.Lock()

        episodes_lock = manager.Lock()
        played_episodes = manager.Value('d', 0.0)

        performance = manager.list(np.full(learning_time, np.nan))
        epsilon_log = manager.list(np.full(learning_time, np.nan))
        perf_lock = manager.Lock()

        pool_args = [(agent, args, times_to_target, episodes_lock, played_episodes,
                      common_policy, right_policy, policy_lock,
                      densities, dens_lock,
                      right_trend, performance, epsilon_log, trend_lock) for agent in range(agents)]
        pool.starmap(run_world, pool_args)
        policy_txt = ""
        for i, policy in enumerate(common_policy):
            if i == x_0:
                policy_txt += "_"
            if policy == 0:
                policy_txt += "0"
            elif policy > 0:
                policy_txt += "R"
            elif policy < 0:
                policy_txt += "L"
            if i == x_0:
                policy_txt += "_"
        performance = np.array(performance) / agents
        epsilon_log = np.array(epsilon_log) / agents
        right_trend = np.array(right_trend) / agents * 100
        for t in range(0, learning_time):
            if np.isnan(right_trend[t]):
                performance[t] = performance[t - 1]
                epsilon_log[t] = epsilon_log[t - 1]
                right_trend[t] = right_trend[t - 1]
        print(f"    finished: {time.now().strftime('%H-%M-%S')}")
        print("--------------RESULT---------------")
        header = '\n'.join([
            r'learning time: %.1e' % learning_time,
            r'$\alpha$: %s' % args.learning_rate,
            r'$\gamma$: %s' % args.discounting,
            r'$\epsilon$: %s' % args.epsilon,
            r'$\epsilon$ decrease: $%s$' % args.eps_factor,
            f'mean for T=0: {min(size - x_0, x_0)}',
            r'mean: %.2f' % np.nanmean(times_to_target),
            r'std: %.2f' % np.nanstd(times_to_target),
            r'played episodes: %i' % (played_episodes.value / agents),
            r'agents: %i' % agents,
            r'failed agents (t>500): %i' % np.count_nonzero(np.isnan(times_to_target)),
            f'R actions, x_0: {right_policy[x_0]}',
            f'R actions, x_0-1: {right_policy[x_0 - 1]}',
            f'R actions, x_0-2: {right_policy[x_0 - 2]}',
        ])
        print(header)
        print("POLICY: " + str(policy_txt))
        print("-----------------------------------")

        fig, axs = plt.subplots(2, 2)  # rows, cols
        fig.suptitle('%s, %s states, 2 actions' % (algorithm, size + 1))
        (ax0, ax1, ax2, ax3) = axs.reshape(4)

        ax0.set_title('Path density')
        density = np.reshape(densities, (size + 1, 1))
        ax0.imshow(density.T, interpolation='nearest', origin='lower', aspect='auto')
        ax0.set_ylim(-1.3, 1.3)
        ax0.text(x_0, 0.9, '$T_{left}=0$       $T_{right}=%s$' % T, ha='center', fontsize=15)
        ax0.text(x_0, 0.6, 'START', ha='center', fontsize=10)
        ax0.text(0.5, 0.6, f'END', ha='center', fontsize=10)
        ax0.text(size - 0.5, 0.6, f'END', ha='center', fontsize=10)
        ax0.text(x_0, -1.3, f'common policy\n{policy_txt}', ha='center', family='monospace', fontsize=9)
        ax0.set_axis_off()

        if drift != 0:
            c = 'black'
            ax0.text(int(size * 3 / 4) + 2, -0.75, f"drift={drift}", ha="center", fontsize=12)
            ax0.annotate("", xy=(0.75 * size, 0), xytext=(size, 0),
                         arrowprops=dict(arrowstyle="->", mutation_scale=30))

        ax1.set_title(f'FP time distribution ({greedy})')
        bbox = {'facecolor': 'white', 'alpha': 0.6, 'linewidth': 0}
        ax1.hist(times_to_target, density=True, bins=np.arange(0, 100, 1), align='left', alpha=0.6)
        ax1.axvline(min(size - x_0, x_0), ls="--", color="tab:red")
        ax1.text(0.49, 0.95, header, transform=ax1.transAxes, va='top', ha='left', bbox=bbox, fontsize=10)
        ax1.set_xlabel('first-passage time')

        ax2.set_title("Right actions in $x_0$, %")
        ax2.plot(right_trend)
        ax2.set_ylim(-3, 103)
        ax2.grid(True, which='both', ls=':')
        ax2.set_xlabel('timeframes')

        ax3.set_title(f'Check epsilon: every {check_t} ticks')
        ax3.plot(epsilon_log, c='tab:red', label='Epsilon')
        ax3.grid(True, which='both', ls=':')
        ax3.legend(loc='upper right')
        ax3.set_xlabel('timeframes')

        header += f'\nfinal try: {greedy}'
        header += f'\nr_step: {reward_step}, r_target: {reward_target}'
        header += f'\ncheck performance every {check_t} steps'
        header += f'\nT_left: 0, T_right: {T}'
        header += f'\ndrift (x > 3/4): {drift}'
        header += f'\ncommon policy (by cell): {policy_txt}'

        date = time.now().strftime("%y.%m.%d %H-%M")
        filename = f"output/{date} Heatmap1D {algorithm} T={T} a={a} drift={drift}"

        np.savetxt(filename + " density.txt", density, fmt="%i", header=header)
        np.savetxt(filename + " fp_times.txt", times_to_target, header=header)
        # np.savetxt(filename + " policy (R).txt", right_policy, fmt="%i", header=header)
        np.savetxt(filename + " policy_trend.txt", right_trend, header=header)
        np.savetxt(filename + " performance.txt", performance, header=header)
        np.savetxt(filename + " epsilon_log.txt", epsilon_log, header=header)
        fig.savefig(filename + ".png", dpi=200, bbox_inches="tight")
        print(f'output data:')
        print(f'"{filename}.png"')
        print(f'"{filename} density.txt"')
        print(f'"{filename} fp_times.txt"')
        print(f'"{filename} policy_trend.txt"')
        print(f'"{filename} performance.txt"')
        print(f'"{filename} epsilon_log.txt"')
        plt.show()
        print('... saving figures can take some time')
        print('END PROGRAM')
