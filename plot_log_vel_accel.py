#!/usr/bin/env python

import math, sys, os, numpy as np, time, threading, logging, matplotlib.pyplot as plt
import pdb

# Custom start and stop timesteps
START = 0
STOP = 100

def plot_data(data):
    fig = plt.figure(figsize=(15,7))
    t = data[START:STOP,0]
    plt.subplot(511)
    plt.plot(t,data[START:STOP,1], alpha=0.7, label='Deep Guidance X')
    plt.plot(t,data[START:STOP,2], alpha=0.7, label='Deep Guidance Y')
    plt.grid();plt.legend()


    # # plt.plot(t,data[:,2], alpha=0.7, label='Deep Guidance X')
    # plt.plot(t,data[:,3], alpha=0.7, label='Y')
    # # plt.plot(t,data[:,4], alpha=0.7, label='Z')
    # # plt.plot(t,data[:,5],  'b', alpha=0.7, label='Follower X')
    # plt.plot(t,data[:,6],  'r', alpha=0.7, label='Y')
    # # plt.plot(t,data[:,7],  'g', alpha=0.7, label='Z')
    # plt.grid();plt.legend()


    plt.subplot(512)
    plt.plot(t,data[START:STOP,5],  '--b', alpha=0.7, label='Follower X')
    plt.plot(t,data[START:STOP,6], '--r', alpha=0.7, label='Follower Y')
    plt.plot(t,data[START:STOP,7], '--g', alpha=0.7, label='Follower Z')
    # plt.plot(t,data[:,],  'b', alpha=0.7, label='Follower X')
    # plt.plot(t,data[:,6],  'r', alpha=0.7, label='Y')
    # plt.plot(t,data[:,7],  'g', alpha=0.7, label='Z')
    plt.grid();plt.legend()

    plt.subplot(513)
    plt.plot(t,data[START:STOP,8], alpha=0.7, label='Target X')
    plt.plot(t,data[START:STOP,9], alpha=0.7, label='Target Y')
    plt.plot(t,data[START:STOP,10], alpha=0.7, label='Target Z')
    plt.grid();plt.legend()
    

    # plt.subplot(313)
    # plt.plot(t,data[:,9], alpha=0.7, label='Target X')
    # plt.plot(t,data[:,10], alpha=0.7, label='Y')
    # plt.plot(t,data[:,11], alpha=0.7, label='Z')
    # plt.grid();plt.legend()

    # plt.subplot(514)
    # plt.plot(data[:,8:13],label='past_actions', alpha=0.3)
    # plt.grid();plt.legend()
    
    # plt.subplot(515)
    # plt.plot(data[:,14:], label='runway_state', alpha = 0.3)
    # plt.grid()#;plt.legend()

    
    
    # plt.grid();plt.legend()

    # plt.subplot(413)
    # plt.plot(data[:,17], label='Throttle', alpha=0.8)
    # plt.plot(data[:,15]*57.3, label='Angle of Attack')
    # plt.plot(data[:,16], label='Flap Deflection')
    # plt.ylim([-30,80])
    # plt.grid();plt.legend()

    # plt.subplot(414)
    
    # plt.plot(data[:,21]*100, label='Throttle sum err x100')
    # plt.plot(data[:,19]*57.3, label='Pitch sum error')
    # plt.grid();plt.legend()
    # plt.tight_layout()


    # fig = plt.figure(figsize=(10,9))
    # plt.subplot(411)
    # plt.plot(data[:,7],label='F_x')
    # plt.plot(data[:,8],label='F_y')
    # plt.plot(data[:,9],label='F_z')
    # plt.grid();plt.legend()
    
    # plt.subplot(412)
    # plt.plot(data[:,10],label='M_x')
    # plt.plot(data[:,11],label='M_y')
    # plt.plot(data[:,12],label='M_z')
    # plt.grid();plt.legend()

    # plt.subplot(413)
    # rho = 1.225
    # coeff = 120.0
    # offset = 0.52
    # plt.plot(np.sqrt( ((data[:,24]-offset)*coeff)/(0.5*rho) ), label='Airspeed')
    # plt.grid();plt.legend()
    # plt.plot(data[:,18]*57.3, label='Angle of Attack Ref')
    
    # plt.plot(data[:,17], label='Throttle')
    # plt.grid();plt.legend()

    # plt.subplot(413)
    # plt.plot(data[:,16], label='Flap Deflection')
    # plt.grid();plt.legend()



    # plt.subplot(414)
    # plt.plot(data[:,20], label='Lift sum err')
    # plt.plot(data[:,21], label='Drag sum err')
    # plt.plot(data[:,19]*57.3, label='AoA sum error')
    # plt.grid();plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    log_filename = sys.argv[1] if len(sys.argv)>1 else '/tmp/log'
    data = np.load(log_filename)
    plot_data(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()