import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

def plotrewfunc_power_velocity():
    fname = './power_velocity/run_vp_600_0025to025_a102_a202_b106_tb-tag-EpRewMean.csv'
    data = pd.read_csv('./' + fname)
    # data = data[['Step', 'Value']]
    # data = data.set_index('Step')

    data = data.rename(index=str, columns={"Step": "Update"})

    # data.set_index("Step", drop=True, inplace=True)

    #ax = data.plot(x='Update', y='Value', legend=False, figsize=(8, 5))
    ax = data.plot(x='Update', y='Value', legend=False)
    ax.set_aspect(aspect=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Mean episode reward")

    data.plot(x='Update', y='Value', kind='line', ax=ax, legend=False
              # xlim=[0, None], ylim=[0.0, None],
              # ylabel='mean episode reward',
              # xlabel='update'
              )

    # plt.show()
    plt.savefig('./power_velocity/Locomotion_EpRewMean_2.pdf', dpi=300, bbox_inches='tight')
    #plt.close('all')
    plt.clf()

def plotrewfunc_target_tracking():
    fname = './target_tracking/run_tt_test6b_3000_tb-tag-EpRewMean.csv'
    data = pd.read_csv('./' + fname)
    # data = data[['Step', 'Value']]
    # data = data.set_index('Step')

    data = data.rename(index=str, columns={"Step": "Update"})

    # data.set_index("Step", drop=True, inplace=True)

    ax = data.plot(x='Update', y='Value', legend=False, figsize=(8,5))
    #ax = data.plot(x='Update', y='Value', legend=False)
    #ax.set_aspect(aspect=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Mean episode reward")

    data.plot(x='Update', y='Value', kind='line', ax=ax, legend=False
              # xlim=[0, None], ylim=[0.0, None],
              # ylabel='mean episode reward',
              # xlabel='update'
              )

    # plt.show()
    plt.savefig('./target_tracking/TargetTracking_EpRewMean_2.pdf', dpi=300, bbox_inches='tight')
    #plt.close('all')
    plt.clf()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    plotrewfunc_power_velocity()
    plotrewfunc_target_tracking()

if __name__ == '__main__':
    main()

