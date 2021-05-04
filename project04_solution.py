import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt
import math

f_0 = 30e6  # carrier frequency [Hz]
f_s = 200e6  # sampling frequency [sample / sec]
f_CA = 1.023e6  # C/A code chipping rate [chip / sec]
f_P = 10.23e6  # P code chipping rate [chip / sec]
BOC_fund = 1.023e6  # BOC fundamental frequency [Hz]
BOC_m = 10  # BOC(m,n)  # M-code = 10, E1-b = 1
BOC_n = 5   # BOC(m.n)  # M-code = 5,  E1-b = 1
t_max = 0.1  # simulation time [sec]
filt_length = 500  # for post FFT filtering

BOC_f_subc = BOC_m * BOC_fund  # BOC subcarrier rate
BOC_f_chip = BOC_n * BOC_fund  # chip rate

time_array = np.arange(0, t_max, 1/f_s)

carrier = np.sin(2*np.pi*f_0 * time_array)

CA_code = -2.0 * np.round(np.random.rand(int(np.ceil(f_CA * t_max)))) + 1
CA_chip_num = np.floor(time_array * f_CA).astype('int')
# sampled_CA = CA_code(CA_chip_num)  # error:  numpy.ndarray object is not callable
sampled_CA = CA_code[CA_chip_num]

P_code = -2.0 * np.round(np.random.rand(int(np.ceil(f_P * t_max)))) + 1
P_chip_num = np.floor(time_array * f_P).astype('int')
# sampled_CA = CA_code(CA_chip_num)  # error:  numpy.ndarray object is not callable
sampled_P = P_code[P_chip_num]

carrier_CA = carrier * sampled_CA
carrier_P = 1/math.sqrt(2.0) * carrier * sampled_P

BOC_subc = np.array([1.0, -1.0])
BOC_subc_num = np.mod(np.floor(time_array * 2*BOC_f_subc).astype('int'), 2)
sampled_BOC_subc = BOC_subc[BOC_subc_num]

BOC_chips = -2.0 * np.round(np.random.rand(int(np.ceil(BOC_f_chip * t_max)))) + 1
BOC_chip_num = np.floor(time_array * BOC_f_chip).astype('int')
sampled_BOC_chip = BOC_chips[BOC_chip_num]

carrier_BOC = carrier * sampled_BOC_subc * sampled_BOC_chip

fft_carrier = np.fft.fft(carrier)/len(carrier)  # normalize by length
fft_carrier = abs(fft_carrier[range(int(len(carrier)/2))])

fft_carrier_CA = np.fft.fft(carrier_CA)/len(carrier_CA)  # normalize by length
fft_carrier_CA = abs(fft_carrier_CA[range(int(len(carrier_CA)/2))])
filt_fft_carrier_CA = spsig.filtfilt(np.ones(filt_length), 1, fft_carrier_CA)  # window filter

fft_carrier_P = np.fft.fft(carrier_P)/len(carrier_P)  # normalize by length
fft_carrier_P = abs(fft_carrier_P[range(int(len(carrier_P)/2))])
filt_fft_carrier_P = spsig.filtfilt(np.ones(filt_length), 1, fft_carrier_P)  # window filter

fft_carrier_BOC = np.fft.fft(carrier_BOC)/len(carrier_BOC)  # normalize by length
fft_carrier_BOC = abs(fft_carrier_BOC[range(int(len(carrier_BOC)/2))])
filt_fft_carrier_BOC = spsig.filtfilt(np.ones(filt_length), 1, fft_carrier_BOC)  # window filter

# freq axis values
len_time = len(time_array)
values = np.arange(int(len_time/2))
timePeriod = len_time / f_s
freq_axis = values / timePeriod

plt.figure()
plt.plot(freq_axis/1e6, fft_carrier)
plt.grid()
plt.xlabel('Frequency [MHz]')
plt.ylabel('Relative power [W/Hz]')
plt.xlim([15, 45])

plt.figure()
plt.plot(freq_axis/1e6, filt_fft_carrier_P, label='P code')
plt.plot(freq_axis/1e6, filt_fft_carrier_CA, label='C/A Code')
plt.plot(freq_axis/1e6, filt_fft_carrier_BOC, label='M Code')
plt.grid()
plt.xlabel('Frequency [MHz]')
plt.ylabel('Relative Power Spectral Density [W/Hz] (Linear Scale)')
# plt.ylim([4, 14])
plt.xlim([10, 50])
plt.legend()

plt.figure()
plt.semilogy(freq_axis/1e6, filt_fft_carrier_P, label='P code')
plt.semilogy(freq_axis/1e6, filt_fft_carrier_CA, label='C/A Code')
plt.semilogy(freq_axis/1e6, filt_fft_carrier_BOC, label='M Code')
plt.grid()
plt.xlabel('Frequency [MHz]')
plt.ylabel('Relative Power Spectral Density [W/Hz] (Log Scale)')
plt.ylim([10, 400])
plt.xlim([10, 50])
plt.legend()

plt.show()
