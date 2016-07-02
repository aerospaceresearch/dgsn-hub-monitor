import os
import sys
import platform
#import requests
#from os import path
path_separator = os.sep
import time
import numpy as np

from scipy import fft
import matplotlib, scipy.fftpack
from pylab import plt

def edge_detection(inp):
    kernel_x = [[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]

    kernel_y = [[-1.0, -2.0, -1.0],
              [0.0, 0.0, 0.0],
              [1.0, 2.0, 1.0]]


    out = inp
    #print(len(inp), len(inp[0]))
    out1 = []
    for i in range(len(inp)):
        p2 = []
        p3 = []

        i2 = i + 1
        if i + 1 >= len(inp):
            i2 = 0

        for j in range(len(inp[i])):
            j2 = j + 1
            if j + 1 >= len(inp[i]):
                j2 = 0

            Gx = (inp[i-1][j-1] * kernel_x[0][0] + inp[i-1][j] * kernel_x[0][1] + inp[i-1][j2] * kernel_x[0][2])
            Gx = Gx + (inp[i][j-1] * kernel_x[1][0] + inp[i][j] * kernel_x[1][1] + inp[i][j2] * kernel_x[1][2])
            Gx = Gx + (inp[i2][j-1] * kernel_x[2][0] + inp[i2][j] * kernel_x[2][1] + inp[i2][j2] * kernel_x[2][2])


            #p2.append(Gx)

            Gy = (inp[i-1][j-1] * kernel_y[0][0] + inp[i-1][j] * kernel_y[0][1] + inp[i-1][j2] * kernel_y[0][2])
            Gy = Gy + (inp[i][j-1] * kernel_y[1][0] + inp[i][j] * kernel_y[1][1] + inp[i][j2] * kernel_y[1][2])
            Gy = Gy + (inp[i2][j-1] * kernel_y[2][0] + inp[i2][j] * kernel_y[2][1] + inp[i2][j2] * kernel_y[2][2])
            p2.append((Gx**2 + Gy**2)**0.5)
        out1.append(p2)

    return out1

def meaning(inp):
    out = []
    for j in range(len(inp[0])):
        sum = 0
        for i in range(len(inp)):
            sum = sum + inp[i][j]
        out.append(sum/len(inp))
    return out

def maxing(inp):
    out = []
    for j in range(len(inp[0])):
        max = 0
        for i in range(len(inp)):
            if max < inp[i][j]:
                max = inp[i][j]
        out.append(max)
    return out

def substract(inp, threshold):
    out = np.zeros((len(inp), len(inp[0])))
    for i in range(len(inp)):
        for j in range(len(inp[i])):
            out[i][j] = inp[i][j] - threshold[j]

    return out

def jump(subst, threshold):
    out = np.zeros((len(subst), len(subst[0])))
    for i in range(len(subst)):
        for j in range(len(subst[i])):
            if (subst[i][j] + threshold[j]) / threshold[j] > 1.5:
                out[i][j] = subst[i][j]
    return out


def signal(inp):
    #threshold = (np.mean(inp) + np.max(inp))/1.4
    #mean1 = meaning(inp)
    mean1 = np.mean(inp)
    max1 = np.max(inp)

    out = np.zeros((len(inp), len(inp[0])))
    for i in range(len(inp)):
        for j in range(len(inp[i])):
            if inp[i][j] > (mean1 + max1) / 3.0 and inp[i][j] / mean1 > 3.8:
                out[i][j] = inp[i][j]
            else:
                out[i][j] = 0.0

    return out

def centerofgravity(inp):
    out = np.zeros((len(inp), len(inp[0])))
    for i in range(len(inp)):
        Msum = 0
        mx_sum = 0
        for j in range(len(inp[i])):
            Msum = Msum + inp[i][j]
            mx_sum = mx_sum + inp[i][j] * (float(j))

        if Msum > 0:
            #print(i,j, Msum, mx_sum)
            x = int(np.round(mx_sum / Msum))
            #print(x)
            out[i][x] = Msum


    return out

def signal_distance(inp, factor):
    signal = 0
    counter = 30 * 60 / factor
    counter0 = counter
    lines = np.zeros(len(inp))
    jumps = 0
    jumpposition =[]
    jumpposition1 =[]
    for i in range(len(inp)):
        line = np.max(inp[i][:])
        if line > 3.0:
            lines[i] = 1

    for i in range(len(lines)-1):
        if lines[i] > lines[i+1]:
            #print("sss", i)
            #jumps += 1
            jumpposition.append(i)

        if lines[i] < lines[i+1]:
            #print("sss", i)
            #jumps += 1
            jumpposition1.append(i)

    for i in range(len(jumpposition)-1):
        if jumpposition[i+1] - jumpposition[i] < counter0:# and jumpposition[i] - jumpposition1[i] > 2.0:

            jumps += 1
    #print("jumps", jumps)
    return jumps

def make_fft(file, window, samplerate, df, every, iq_stream):

    result_of_fft = []

    counting_until_every_reached = 0

    adc_offset = -127
    # bringing the signal per kernel down around the average reduces the dc peak at f=0hz!
    # either by offset or directly by casting as uint int :)

    for slice in range(0 , len(iq_stream) , window*2):
        iq_stream_slice = (adc_offset + iq_stream[slice: slice + window * 2: 2]) + 1j * (adc_offset + iq_stream[slice + 1: slice + window * 2: 2])

        # if wav, i and q is different than it is for binary format
        # iq_stream_slice = (offset + test[s+1: s+window*2: 2]) + 1j*(offset + test[s: s+window*2: 2])

        iq_stream_slice_intensity = np.abs(fft(iq_stream_slice))
        del iq_stream_slice

        counting_until_every_reached += 1
        if slice == 0:
            iq_stream_slice_overlay = iq_stream_slice_intensity# / np.max(g)
        else:
            if len(iq_stream_slice_intensity) == window: # in case the recorded stream cannot be fully parted
                iq_stream_slice_overlay = iq_stream_slice_overlay + iq_stream_slice_intensity# / np.max(g)

        if counting_until_every_reached >= every:#reduce amount of plot here
            #iq_stream_slice_overlay = iq_stream_slice_overlay
            yplot = 1.0/window * (np.fft.fftshift(iq_stream_slice_overlay))
            #print(yplot)
            yplot = yplot / every
            #print(yplot)
            #print(np.max(yplot), np.mean(yplot))
            result_of_fft.append(np.log(yplot))
            counting_until_every_reached = 0
            iq_stream_slice_overlay = 0

        del iq_stream_slice_intensity

    return result_of_fft

def signal_substraction(file, window, samplerate, df, every, result_of_fft):

    # we asume, that our satellite signals are not contineously received
    # so we do a substraction on frequency level. the signal intensity of each frequency for the window kernel
    # is substracted from the overall, longtime average of this frequency.
    # this allows to see fluctuations better, because they are adding to the average.
    # so we can find a superposed signal easier this way.
    threshold = meaning(result_of_fft)
    signal_lowered = substract(result_of_fft, threshold)
    print(time.time(), "lowering done")

    v = np.zeros((len(signal_lowered), window))
    #w = np.zeros((len(signal_lowered), window))

    bandwidth = int(window / (2*2*2*2*2*2*2*2*2))
    #print("band", bandwidth*df)

    for j in range(0, window, bandwidth):
        #print(j)

        u = []
        for kk in range(len(signal_lowered)):
            u.append(signal_lowered[kk][j:j+bandwidth])
            #u = u/np.max(u)

        #print("test", len(u), len(u[0]))
        '''
        plt.imshow(u, interpolation='nearest')
        #plt.imshow(result)
        plt.gca().invert_yaxis()
        #plt.gca().invert_xaxis()
        # #plt.gca().set_xticks(xf)
        plt.show()
        '''

        edged = signal(edge_detection(u))
        #print(np.max(edged))
        #graved = centerofgravity(edged)
        #signal_strength.append(np.sum(edged))


        for k in range(len(edged)):
            for l in range(len(edged[k])):
                v[k][j+l] = edged[k][l]# + signaaaal

    print(time.time(), "edging done")

    w3 = []
    filename3 = file+"_all.png"
    for lll in range(len(v)):
        w3.append(signal_lowered[lll])

    plt.imshow(w3, interpolation='nearest')
    #plt.imshow(result)
    plt.gca().invert_yaxis()
    #plt.figure(figsize=(1200, 800))
    plt.savefig(filename3, format='png', dpi=500)
    #plt.savefig(filename, format='png', dpi=1000)
    #plt.show()

    del w3

    return v


def detect_signal(file, window, samplerate, df, every, v, xf, f_center):


    ##### detect
    # loading in sdr data

    # loading in the database what frequencies are known
    db_satname = ["NOAA19", "NOAA15", "NOAA18", "ISS", "ISS APRS"]
    db_f = [137100000.0, 137620000.0, 137912500.0, 145800000.0, 145825000.0]
    db_f_band = [24000.0, 24000.0, 24000.0, 6000.0, 6000.0]

    f = []
    f_band = []
    for kkk in range(len(db_f)):
        if db_f[kkk] >= f_center - samplerate and db_f[kkk] <= f_center + samplerate:
            f.append(db_f[kkk])
            f_band.append(db_f_band[kkk])

    print(f)


    start = []
    bandrange = []
    for kkkk in range(len(f)):
        start.append(int((f[kkkk] - f_center) / df + len(xf)/2.0))
        #print(start[-1], (0), df, len(xf)/2.0)
        bandrange.append(int(f_band[kkkk]/df))
        #print("band", bandrange[-1])
        #print("test", start[-1], df, bandrange[-1])

    print(time.time(), "graphing start")
    foundknownsignal = []
    foundknownsignal1 = []
    for j in range(len(f)):
        w1 = []
        w2 = []
        for ll in range(len(v)):
            w1.append(v[ll][start[j] - bandrange[j]: start[j] + bandrange[j]])
            w2.append(v[ll][start[j] - (bandrange[j]+int(60000/df)): start[j] + (bandrange[j]+int(60000/df))])


        foundknownsignal.append(signal_distance(w1, every))
        foundknownsignal1.append(int(np.sum(w1)))
        filename = file+"_"+str(f[j])+"_sig"+str(foundknownsignal[j])+"_"+str(foundknownsignal1[j])+".png"
        filename1 = file+"_"+str(f[j])+"x_sig"+str(foundknownsignal[j])+"_"+str(foundknownsignal1[j])+".png"

        plt.imshow(w1, interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.savefig(filename, format='png')
        #plt.show()

        plt.imshow(w2, interpolation='nearest')
        #plt.imshow(result)
        plt.gca().invert_yaxis()
        plt.savefig(filename1, format='png', dpi=100)
        #plt.show()

    print(time.time(), "graphing end")

    '''
    #signal_strength = signal_strength / np.max(signal_strength)
    #signal_strength1 = signal_strength1 / np.max(signal_strength1)
    print(len(signal_strength))
    print(signal_strength)
    #plt.plot(signal_strength)
    plt.plot(xf, frr)
    plt.show()
    '''
    return foundknownsignal, foundknownsignal1

def monitor(a, a1, temp, iq_stream):
    print(time.time(), "monitoring start")
    file = temp[0]+"_"+temp[1]+"_"+temp[2]
    file_sdr = a+path_separator+a1[1]+path_separator+temp[0]+"_"+temp[1]+"_"+temp[2]+".npy"
    if os.path.exists(file_sdr):
        sdrmeta = np.load(file_sdr)
        device_number = int(sdrmeta[0])
        center_frequency = int(sdrmeta[1])
        samplerate = int(sdrmeta[2])
        gain = sdrmeta[3] # can be "auto" inside
        nsamples = int(sdrmeta[4])
        freq_correction = int(sdrmeta[5])
        user_hash = sdrmeta[6]
        print("exists",file_sdr)
    else:
        samplerate = 2048000
        center_frequency = 137900000.0

    window = 2048*2*2
    #http://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    # sample spacing
    T = 1.0 / samplerate
    x = np.linspace(0.0, window*T, window)

    xf = np.fft.fftfreq(window, T)
    xf = np.fft.fftshift(xf)

    df = xf[1]-xf[0]
    every = (len(iq_stream) / (samplerate * 2.0)) * 8192.0 / (window)
    print(every)

    signal_fft = make_fft(file, window, samplerate, df, every, iq_stream)
    print(time.time(), "fft done")
    signal_lowered = signal_substraction(file, window, samplerate, df, every, signal_fft)
    print(time.time(), "substraction done")
    return detect_signal(file, window, samplerate, df, every, signal_lowered, xf, center_frequency)


def Task2(folder, subfolders):
        a = folder
        print(a)
        a1 = subfolders
        #b = samplerate

        end = 0
        counter = 0
        resetter = 1000
        while end == 0 and counter < resetter:
            #print(a
            for root, dirs, files in os.walk(a+"/"+a1[0]):
                for file in files:
                    process = 0
                    #print(file)
                    if file.find("tmp")==-1 and file.find("coded")==-1:

                        temp = (file.split(".")[0]).split("_")
                        #print("test", temp)

                        if os.path.exists(a+"/"+a1[4]+"/"+file.split(".")[0]+"_monitor.npy")\
                                or os.path.exists(a+"/"+a1[4]+"/"+file.split(".")[0]+"_monitor_blocked.npy"):
                            process = 1
                            #print(a+"/"+a1[2]+"/"+file)
                            #print("file exists", a+"/"+a1[2]+"/"+temp[0]+"_"+temp[1]+"_"+temp[2]+"_coded.npy")

                        if process == 0:
                            # blocking the file
                            np.save(a+"/"+a1[4]+"/"+temp[0]+"_"+temp[1]+"_"+temp[2]+"_monitor_blocked.npy",
                                    a+"/"+a1[4]+"/"+temp[0]+"_"+temp[1]+"_"+temp[2]+"_monitor.npy")


                            #print("ggggggg", file
                            time.sleep(3)
                            # loading in data
                            splittingfile = file.split("_")
                            #test1 = np.load(a+path_separator+a1[0]+path_separator+splittingfile[0]+"_"+splittingfile[1]+"_"+splittingfile[2])
                            test1 = np.memmap(a+path_separator+a1[0]+path_separator+splittingfile[0]+"_"+splittingfile[1]+"_"+splittingfile[2])
                            print(a+path_separator+a1[0]+path_separator+splittingfile[0]+"_"+splittingfile[1]+"_"+splittingfile[2])

                            satsignal = monitor(a, a1, temp, test1)
                            if len(satsignal[0]) == 0:
                                satsignal = [0,0]# really, nooothing was found :(


                            #if os.path.exists(a+"/"+a1[4]+"/"+file.split(".")[0]+"_monitor.npy") == False:
                            #    np.save(a+"/"+a1[4]+"/"+file.split(".")[0]+"_monitor.npy", "dummydata")


                            # removing the blocker file and the original IQ file,
                            # because it is nothing interesting inside, perhaps!
                            if os.path.exists(a+"/"+a1[4]+"/"+file.split(".")[0]+"_monitor_blocked.npy") == True and np.max(satsignal[0]) < 6:
                                del test1
                                os.remove(a+path_separator+a1[0]+path_separator+splittingfile[0]+"_"+splittingfile[1]+
                                          "_"+splittingfile[2]) #remove iq file
                                os.remove(a+"/"+a1[4]+"/"+file.split(".")[0]+"_monitor_blocked.npy",)
                                #remove the blocker file
                                os.remove(a+path_separator+a1[1]+path_separator+temp[0]+"_"+temp[1]+"_"+temp[2]+".npy")
                                #remove the sdrmeta data file

                            counter = 0

            counter += 1
            time.sleep(1)
            if counter % 10 == 0:
                print("task2", counter)

        programme_ends = 1
        return

if __name__ == '__main__':
    print("you are using", platform.system(), platform.release(), os.name)

    # creating the central shared dgsn-node-data for all programs on the nodes
    #######################################
    pathname = os.path.abspath(os.path.dirname(sys.argv[0]))
    pathname_all = ""
    for i in range(len(pathname.split(path_separator))-2): # creating the folders two folder levels above
        pathname_all = pathname_all + pathname.split(path_separator)[i] + path_separator
    pathname_save = pathname_all + "dgsn-node-data"

    # creating the dump folder for files and the needed data folders
    #######################################
    if not os.path.exists(pathname_save):
        os.makedirs(pathname_save)

    folder = pathname_save + path_separator + "rec"
    subfolders = ["iq", "sdr", "gapped", "coded", "monitor"]
    if not os.path.exists(folder):
        os.makedirs(folder)

    if os.path.exists(folder):
        for i in range(len(subfolders)):
            if not os.path.exists(folder + path_separator + subfolders[i]):
                os.makedirs(folder + path_separator + subfolders[i])


    Task2(folder, subfolders)