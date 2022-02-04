import argparse
import multiprocessing
from datetime import datetime as time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from heatedgridworld.Policy import GridPolicy
from heatedgridworld.State import GridState, GridStateRegion, HeatmapRegion
from heatedgridworld.World import HeatmapGridWorld

plt.rcParams['figure.figsize'] = (2 * 4.5, 2 * 4)
plt.rcParams['figure.dpi'] = 60
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['legend.fancybox'] = False
plt.rcParams['legend.frameon'] = True

# Global: size of grid and center position
size_x = 10
x_c = int(size_x / 2)
cutoff_time = 500


def run_world(agent, args, targets, obstacles, heated_regions, cold_region):
    try:
        algorithm = args.algorithm
        a = args.learning_rate
        mu = args.discounting
        eps = args.eps
        eps_factor = args.eps_factor
        learning_time = args.learning_time
        check_t = args.check_time
        tries = args.tries
        greedy = args.greedy
        # --------------- init world: ------------------
        world = HeatmapGridWorld(size_x, size_x, bound="reflect", heat_regions=heated_regions,
                                 targets=targets, step_reward=-1, target_reward=100, obstacles=obstacles)
        policy = GridPolicy(eps=eps, act_size=world.act_size)
        pos = GridState(x=0, y=0)
        new_pos = GridState(x=None, y=None)
        Q_val = np.zeros((size_x, size_x, world.act_size))
        agent_density = np.zeros((size_x, size_x))
        epsilon_log = np.full(learning_time, np.nan)
        # performance = np.full(learning_time, np.nan)
        # time_to_target = 0
        # last_time_to_target = 0
        played_episodes = 0

        double_Q_val = np.copy(Q_val)  # "Double_Q_learning"
        act = policy.get_action(Q_val, pos)  # "SARSA"

        # ----------------- learning --------------------
        for t in range(0, learning_time):
            # time_to_target += 1
            target_reached = False

            if algorithm == "Q_learning":
                act = policy.get_action(Q_val, pos)
                target_reached, reward = world.do_action(pos, new_pos, act)
                Q_val[pos.x, pos.y, act] += a * (
                        reward + mu * Q_val[new_pos.x, new_pos.y].max() - Q_val[pos.x, pos.y, act])

            if algorithm == "Double_Q_learning":
                avg_Q = Q_val + double_Q_val
                act = policy.get_action(avg_Q, pos)
                target_reached, reward = world.do_action(pos, new_pos, act)

                if world.rng.choice(2) == 0:
                    a_act = np.argmax(Q_val[new_pos.x, new_pos.y, :])
                    Q_val[pos.x, pos.y, act] += \
                        a * (reward + mu * double_Q_val[new_pos.x, new_pos.y, a_act].max() - Q_val[pos.x, pos.y, act])
                else:
                    b_act = np.argmax(double_Q_val[new_pos.x, new_pos.y, :])
                    double_Q_val[pos.x, pos.y, act] += \
                        a * (reward + mu * Q_val[new_pos.x, new_pos.y, b_act].max() - double_Q_val[pos.x, pos.y, act])

            if algorithm == "SARSA":
                target_reached, reward = world.do_action(pos, new_pos, act)
                new_act = policy.get_action(Q_val, pos)
                Q_val[pos.x, pos.y, act] += a * (
                        reward + mu * Q_val[new_pos.x, new_pos.y, new_act] - Q_val[pos.x, pos.y, act])
                act = new_act

            if algorithm == "Expected_SARSA":
                act = policy.get_action(Q_val, pos)
                target_reached, reward = world.do_action(pos, new_pos, act)

                argsmax = policy.get_argsmax(Q_val, new_pos)
                expectation = np.sum(eps / world.act_size * Q_val[new_pos.x, new_pos.y]) \
                              + np.sum((1 - eps) / len(argsmax) * Q_val[new_pos.x, new_pos.y][argsmax])

                Q_val[pos.x, pos.y, act] += a * (reward + mu * expectation - Q_val[pos.x, pos.y, act])

            pos.set(new_pos.x, new_pos.y)

            if (t % check_t) == 0:
                # performance[t] = last_time_to_target
                epsilon_log[t] = eps
            if target_reached:
                # last_time_to_target = time_to_target
                # time_to_target = 0
                played_episodes += 1
                eps = eps * eps_factor
                policy.type_prob = [1 - eps, eps]
                pos.reset()

        # ----------------- resulting score --------------------
        Q_val = (Q_val + double_Q_val) / 2
        pos.reset()
        scores = []
        time_to_target = 0
        agent_try = 0
        in_cold_region = 0

        while agent_try < tries:
            time_to_target += 1
            if greedy:
                act = policy.get_greedy_action(Q_val, pos)
            else:
                act = policy.get_action(Q_val, pos)
            shift, reward = world.do_action(pos, new_pos, act)
            pos.set(new_pos.x, new_pos.y)
            agent_density[pos.x, pos.y] += 1

            if cold_region.contains(pos):
                in_cold_region = 1

            if reward > 0:
                scores.append(time_to_target)
                pos.reset()
                agent_try += 1
                time_to_target = 0

            if time_to_target > cutoff_time:
                scores.append(np.nan)
                agent_try += 1

        return scores, agent_density, epsilon_log, world, pos, in_cold_region, played_episodes
    except KeyboardInterrupt:
        print(f"<child process> {agent} agent - cancelled")
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", help="default: Q_learning",
                        choices=["Q_learning", "Expected_SARSA", "Double_Q_learning", "SARSA"], default="Q_learning")
    parser.add_argument("--learning_rate", help="default: 0.1", type=float, default=0.1)
    parser.add_argument("--learning_time", help="number of timeframes, default: 50_000", type=int, default=50_000)
    parser.add_argument("--T", help="temperature of the heated region, default: 3", type=int, choices=[0, 1, 2, 3],
                        default=3)
    parser.add_argument("--discounting", help="discounting constant, default: 0.9", type=float, default=0.9)
    parser.add_argument("--eps", help="epsilon-greediness, default: 0.1", type=float, default=0.1, )
    parser.add_argument("--eps_factor", help="multiply eps by this factor after every played episode, default: 1.0",
                        type=float,
                        default=1.0, )
    parser.add_argument("--agents", help="independent learning agents, default: 50", type=int, default=50, )
    parser.add_argument("--processes", help="for multiprocessing pool, default: 4", type=int, default=4)
    parser.add_argument("--check_time", help="check eps every ... timeframes, default: 1_000", type=int, default=1_000)
    parser.add_argument("--tries", help="number of episodes for the final score, default: 1", type=int, default=1)
    parser.add_argument("--greedy", help="use greedy policy when checking the final score, default: True", type=bool,
                        default=True)
    args = parser.parse_args()
    algorithm = args.algorithm
    a = args.learning_rate
    mu = args.discounting
    eps = args.eps
    learning_time = args.learning_time
    check_t = args.check_time
    tries = args.tries
    T = args.T
    agents = args.agents
    processes = args.processes
    greedy = 'greedy' if args.greedy else 'non-greedy'

    # ------------- init HeatmapRegions for 10x10 grid -------------
    targets = (GridState(x=size_x - 1, y=size_x - 1),)
    obstacles = (GridStateRegion([[x_c, x_c], [x_c, x_c + 3]]),
                 GridStateRegion([[x_c, x_c], [x_c + 3, x_c]]),)
    # lower right quarter
    heated_regions = (HeatmapRegion([[x_c, 0], [size_x - 1, x_c - 1]], T),)

    # # different locations 1
    # heated_regions = (HeatmapRegion([[x_c, 2], [size_x - 3, x_c - 1]], T),)
    # # different locations 2
    # heated_regions = (HeatmapRegion([[1, 4], [2, 5]], T),)

    # small region near target for agents' counting
    cold_region = HeatmapRegion([[x_c, size_x - 1], [size_x - 4, size_x - 1]], 0)
    # visualisation only
    world_region = HeatmapRegion([[0, 0], [size_x - 1, size_x - 1]], 0)

    print(f"MULTIPROCESSING HEATMAP, processes: {processes}")
    print(args.__dict__)
    print(f"    started: {time.now().strftime('%H-%M-%S')}")
    with multiprocessing.Pool(processes) as pool:
        try:
            pool_args = [(agent, args, targets, obstacles, heated_regions, cold_region) for agent in range(agents)]
            result = list(pool.starmap(run_world, pool_args))
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
            print("Ctrl-C: CANCELLED BY USER")
            exit()
    scores = []
    agent_density = np.zeros((size_x, size_x))
    epsilon_log = None
    cold_agents = 0
    played_episodes = 0
    for agent_result in result:
        for score in agent_result[0]:
            scores.append(score)
        agent_density += agent_result[1]
        if epsilon_log is None:
            epsilon_log = agent_result[2] / agents
        else:
            epsilon_log += agent_result[2] / agents
        cold_agents += agent_result[5]
        played_episodes += agent_result[6] / agents

    for t in range(0, learning_time):
        if np.isnan(epsilon_log[t]):
            epsilon_log[t] = epsilon_log[t - 1]
    world = result[0][3]
    pos = result[0][4]
    print(f"    finished: {time.now().strftime('%H-%M-%S')}")
    print("--------------RESULT---------------")
    header = '\n'.join((
        r'learning time: %.1e' % args.learning_time,
        r'$\alpha: %s$' % args.learning_rate,
        r'$\gamma: %s$' % args.discounting,
        r'$\epsilon: %s$' % args.eps,
        r'$\epsilon$ factor: $%s$' % args.eps_factor,
        r'mean for T=0: %.1f' % (2 * size_x - 2),
        r'mean: %.1f' % np.nanmean(scores),
        r'std: %.1f' % np.nanstd(scores),
        r'played episodes: %i' % played_episodes,
        r'agents: %i' % agents,
        r'failed agents (t>500): %i' % np.count_nonzero(np.isnan(scores)),
        r'in heated region: %i' % (agents - cold_agents),
    ))
    print(header)
    print("-----------------------------------")

    fig, axs = plt.subplots(2, 2)  # rows, cols
    fig.suptitle('%s, %s' % (algorithm, world.get_name()))
    (ax0, ax1, ax2, ax3) = axs.reshape(4)

    ax0.set_title('Heatmap Regions')
    heated_region = heated_regions[0]
    target = targets[0]
    legend_patches = [mpatches.Patch(color=world_region.color, label='T=%s' % world_region.heat_lvl),
                      mpatches.Patch(color=heated_region.color, label='T=%s' % heated_region.heat_lvl),
                      mpatches.Patch(color='grey', label='obstacle'),
                      mpatches.Patch(color='#f0f0f0', label="for counting\n'cold' agents"),
                      ]
    ax0.add_patch(patches.Rectangle(world_region.start_point, world_region.width, world_region.height,
                                    linewidth=1, edgecolor=world_region.color, facecolor=world_region.color))
    ax0.add_patch(patches.Rectangle(heated_region.start_point, heated_region.width, heated_region.height,
                                    linewidth=1, edgecolor=heated_region.color, facecolor=heated_region.color))
    for obstacle in world.obstacles:
        ax0.add_patch(patches.Rectangle(obstacle.start_point, obstacle.width, obstacle.height,
                                        linewidth=0, edgecolor=None, facecolor='grey'))
    ax0.add_patch(patches.Rectangle(cold_region.start_point, cold_region.width, cold_region.height,
                                    linewidth=1, edgecolor='#f0f0f0', facecolor=cold_region.color))
    for i in range(0, size_x + 1):
        grid_x = [-0.5, size_x - 1 + 0.5]
        grid_y = [i - 0.5, i - 0.5]
        ax0.plot(grid_x, grid_y, color='#ffffff', linestyle='dotted')
        grid_x = [i - 0.5, i - 0.5]
        grid_y = [-0.5, size_x - 1 + 0.5]
        ax0.plot(grid_x, grid_y, color='#ffffff', linestyle='dotted')
    ax0.set_ylim(-1, size_x + 1)
    ax0.set_xlim(-2, size_x - 0.5)
    ax0.text(pos.init_x - 0.5, pos.init_y - 0.2, 'START', color='#ffffff', fontsize=8)
    ax0.text(targets[0].init_x - 0.5, targets[0].init_y - 0.2, ' END', color='#ffffff', fontsize=8)
    ax0.legend(handles=legend_patches, title='Legend', loc='upper right', bbox_to_anchor=(0.07, 1.02))
    ax0.set_axis_off()

    ax1.set_title('Path density')
    ax1.imshow(np.transpose(agent_density), interpolation='nearest', origin='lower', aspect='auto')
    ax1.set_ylim(-1, size_x + 1)
    ax1.set_xlim(-2, size_x - 0.5)
    ax1.text(-0.5, size_x, '%s, over last %s tries' % (greedy, tries))
    ax1.text(pos.init_x - 0.5, pos.init_y - 0.2, 'START', color='#ffffff', fontsize=8)
    ax1.text(targets[0].init_x - 0.5, targets[0].init_y - 0.2, ' END', color='#ffffff', fontsize=8)
    ax1.set_axis_off()

    ax2.set_title(f'FP time distribution ({greedy})')
    bbox = {'facecolor': 'white', 'alpha': 0.6, 'linewidth': 0}
    ax2.hist(scores, bins=np.arange(10, 50, 1), density=True, alpha=0.6)
    ax2.text(0.49, 0.95, header, transform=ax2.transAxes, va='top', ha='left', fontsize=10, bbox=bbox)
    ax2.axvline(18.5, ls="--", color="tab:red", alpha=0.6)
    ax2.set_ylim(0, 0.4)
    ax2.set_xlabel('first-passage time')

    ax3.set_title(f'Check epsilon: every {check_t} ticks')
    ax3.plot(epsilon_log, c='tab:red', label='Epsilon')
    ax3.set_xlabel('timeframes')
    ax3.set_yscale('log')
    ax3.grid(True, which='both', ls=':')
    ax3.legend(loc='upper right')

    header += f'\nfinal try: {greedy}'
    header += f'\ntries: {args.tries}'
    header += f'\nr_step: {world.step_reward}, r_target: {world.target_reward}'
    header += f'\ncheck eps every {args.check_time} steps'

    date = time.now().strftime("%y.%m.%d %H-%M")
    filename = f"output/{date} Heatmap2D {algorithm} T={T} a={a}"

    np.savetxt(filename + ' epsilon_log.txt', epsilon_log, header=header)
    np.savetxt(filename + ' density.txt', agent_density, header=header)
    np.savetxt(filename + ' fp_times.txt', scores, header=header)
    fig.savefig(filename + '.png', dpi=200, bbox_inches='tight')
    # fig.savefig(filename + '.pdf', bbox_inches='tight')
    print(f'output data:')
    print(f'"{filename}.png"')
    # print(f'"{filename}.pdf"')
    print(f'"{filename} density.txt"')
    print(f'"{filename} fp_times.txt"')
    print(f'"{filename} epsilon_log.txt"')
    plt.show()
    print('... saving figures can take some time')
    print('END PROGRAM')
